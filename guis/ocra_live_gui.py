#!/usr/bin/env python3
"""ORCA-Fusion BT live viewer using the user's patched pylablib DCAM backend.

Install GUI dependencies into your EXISTING camera environment:
    python -m pip install PyQt5 pyqtgraph astropy
Run:
    python orca_live_gui.py
    python orca_live_gui.py --demo     # synthetic images, no camera access

Prerequisite for hardware: existing dcamapi4_lib.py edit replacing
"dcamapi.dll" with "libdcamapi.so". This script does not modify installations.

Live view samples newest frames at up to 20 Hz. Record N frames uses a separate
finite acquisition, validates consecutive indices/stamps, and saves a FITS cube.
Recording uses RAM (maximum 512 MiB raw cube; peak memory can exceed 3x this).
Cancel discards the unfinished sequence. Only complete sequences are saved.
Recording uses the currently APPLIED settings, not uncommitted GUI edits.
Settings are applied with acquisition stopped, then streaming resumes if active.
Camera settings are read back, but are not restored to pre-launch values on exit.
Sensor mode and triggering are set to AREA / internal on connection.
API references:
https://pylablib.readthedocs.io/en/latest/devices/DCAM.html
https://pyqtgraph.readthedocs.io/en/latest/api_reference/widgets/imageview.html
"""

# --- THESE MUST BE THE VERY FIRST LINES ---
import pylablib
pylablib.par["devices/only_windows_dlls"] = False
pylablib.par["devices/dlls/dcamapi"] = "/usr/local/lib"

import argparse
import os
import tempfile
import queue
import sys
import threading
import time
import traceback
from datetime import datetime, timezone

import numpy as np
from PyQt5 import QtCore, QtWidgets
import pyqtgraph as pg

pg.setConfigOptions(imageAxisOrder="row-major")


class DemoCamera:
    """Small synthetic camera for checking the GUI without hardware."""
    def __init__(self):
        self.roi = (0, 512, 0, 512, 1, 1)
        self.exposure = 50e-6
        self.speed = 3
        self.running = False
        self.last = 0

    def get_roi(self):
        return self.roi

    def get_detector_size(self):
        return (512, 512)

    def get_exposure(self):
        return self.exposure

    def set_exposure(self, value):
        self.exposure = max(18e-6, value)

    def set_roi(self, x0=0, x1=None, y0=0, y1=None, hbin=1, vbin=1):
        self.roi = (x0, x1 or 512, y0, y1 or 512, hbin, hbin)

    def set_attribute_value(self, name, value):
        if name == "READOUT SPEED":
            self.speed = value

    def start_acquisition(self, **kwargs):
        self.running = True
        self.last = time.monotonic()

    def clear_acquisition(self):
        self.running = False

    def close(self):
        self.clear_acquisition()

    def read_newest_image(self):
        now = time.monotonic()
        if not self.running or now - self.last < max(0.03, self.exposure):
            return None
        self.last = now
        x0, x1, y0, y1, b, _ = self.roi
        yy, xx = np.mgrid[y0:y1:b, x0:x1:b]
        spot = np.exp(-((xx - 256 - 60*np.sin(now))**2 + (yy - 256)**2)/4000)
        image = 100 + self.exposure/50e-6*15000*spot
        image += np.random.default_rng().normal(0, 20, image.shape)
        return np.clip(image, 0, 65535).astype(np.uint16)


class CameraThread(QtCore.QThread):
    recording = QtCore.pyqtSignal(bool)
    progress = QtCore.pyqtSignal(int, int)
    settings = QtCore.pyqtSignal(object)
    status = QtCore.pyqtSignal(str)
    failure = QtCore.pyqtSignal(str)

    def __init__(self, index=0, demo=False):
        super().__init__()
        self.index, self.demo = index, demo
        self.commands = queue.Queue()
        self.lock = threading.Lock()
        self.latest = None
        self.cam = None
        self.live = False
        self.single = False
        self.serial = 0
        self.actual = {}
        self.cancel_record = threading.Event()

    def publish_settings(self):
        c = self.cam
        if self.demo:
            model = "DEMO — synthetic camera"
            speeds = [(v, str(v)) for v in (1, 2, 3)]
            speed = c.speed
            bins = [1, 2, 4]
        else:
            model = str(c.get_device_info())
            attr = c.get_attribute("READOUT SPEED", error_on_missing=False)
            speeds, speed = [], None
            if attr is not None:
                attr.update_limits()
                speed = int(attr.get_value())
                if attr.writable:
                    ids = (attr.ivalues if attr.kind == "enum" else
                           range(int(attr.min), int(attr.max)+1, max(1, int(attr.step))))
                    speeds = [(int(v), f"{v}: {attr.ilabels[v]}" if v in attr.ilabels
                               else str(v)) for v in ids]
            bins = list(c.get_attribute("BINNING").ivalues)
            if not bins:
                bins = [int(c.get_roi()[4])]
        self.actual = dict(model=model, roi=tuple(c.get_roi()),
                           detector=c.get_detector_size(), exposure=c.get_exposure(),
                           speeds=speeds, speed=speed, bins=bins)
        self.settings.emit(self.actual.copy())

    def stop_capture(self):
        self.live = self.single = False
        self.cam.clear_acquisition()

    def begin(self, single=False):
        self.cam.clear_acquisition()
        self.cam.start_acquisition(mode="sequence", nframes=8)
        self.single, self.live = single, not single
        self.deadline = time.monotonic() + max(5, self.cam.get_exposure()+3)
        self.status.emit("Waiting for one frame…" if single else "Live")

    def record_sequence(self, values):
        """Finite buffered acquisition; only publish a file after N validated frames."""
        from astropy.io import fits
        count, path = values["count"], values["path"]
        resume = self.live
        self.stop_capture()
        self.recording.emit(True)
        temporary = None
        try:
            self.publish_settings()
            settings = self.actual.copy()
            x0, x1, y0, y1, bx, by = settings["roi"]
            shape = ((y1-y0)//by, (x1-x0)//bx) if self.demo else self.cam.get_data_dimensions()
            # This instrument produces uint16; abort explicitly on a different pixel type.
            raw_bytes = count * int(np.prod(shape)) * 2
            if raw_bytes > 512 * 1024**2:
                raise ValueError("Sequence exceeds the 512 MiB raw-cube limit. Reduce N or ROI. "
                                 "Camera buffers and FITS writing also require memory.")
            cube = np.empty((count, *shape), dtype=np.uint16)
            received = 0
            stamps = []
            previous_stamp = None
            host_start = datetime.now(timezone.utc).isoformat()
            self.cam.start_acquisition(mode="snap", nframes=count)
            deadline = time.monotonic()+max(5, settings["exposure"]+3)
            next_progress = 0
            self.status.emit(f"Recording {count} frames…")
            self.progress.emit(0, count)
            while received < count:
                if self.cancel_record.is_set() or self.isInterruptionRequested():
                    self.status.emit("Recording cancelled — unfinished sequence discarded")
                    return
                if self.demo:
                    frame = self.cam.read_newest_image()
                    frames = [] if frame is None else [frame]
                    infos = [None]*len(frames)
                else:
                    available = self.cam.get_new_images_range()
                    if available is None:
                        frames, infos = [], []
                    else:
                        if available[0] != received:
                            raise RuntimeError(f"Missing frame: expected index {received}, "
                                               f"next available is {available[0]}. No file saved.")
                        end = min(available[1], received+8, count)
                        if end <= received:
                            frames, infos = [], []
                        else:
                            frames, infos, actual_range = self.cam.read_multiple_images(
                                rng=(received, end), missing_frame="none",
                                return_info=True, return_rng=True)
                            if (tuple(actual_range) != (received, end) or len(frames) != end-received
                                    or len(infos) != len(frames)):
                                raise RuntimeError("Incomplete frame range. No file saved.")
                for frame, info in zip(frames, infos):
                    if frame is None or frame.shape != tuple(shape) or frame.dtype != np.uint16:
                        raise RuntimeError("Missing frame or unexpected image format. No file saved.")
                    index = getattr(info, "frame_index", received)
                    if index != received:
                        raise RuntimeError(f"Unexpected frame index {index}; expected {received}.")
                    stamp = getattr(info, "framestamp", None)
                    if stamp is not None:
                        stamp = int(stamp)
                        if previous_stamp is not None and (stamp-previous_stamp) % (2**32) != 1:
                            raise RuntimeError("Gap in camera frame stamps. No file saved.")
                        previous_stamp = stamp
                    stamps.append(-1 if stamp is None else stamp)
                    cube[received] = frame
                    received += 1
                    deadline = time.monotonic()+max(5, settings["exposure"]+3)
                if frames and (time.monotonic() >= next_progress or received == count):
                    self.progress.emit(received, count)
                    next_progress = time.monotonic()+0.1
                    self.serial += 1
                    with self.lock:
                        self.latest = (self.serial, cube[received-1].copy(), settings.copy(),
                                       datetime.now(timezone.utc).isoformat())
                if not frames:
                    if time.monotonic() > deadline:
                        raise TimeoutError(f"Recording timed out at {received}/{count}. No file saved.")
                    self.msleep(10)
            self.cam.clear_acquisition()
            if self.cancel_record.is_set() or self.isInterruptionRequested():
                self.status.emit("Recording cancelled — sequence discarded")
                return
            host_end = datetime.now(timezone.utc).isoformat()
            self.status.emit("Writing FITS cube…")
            header = fits.Header()
            header["EXPTIME"] = (settings["exposure"], "Exposure per frame [s]")
            header["NFRAMES"] = (count, "Number of recorded frames")
            header["NREQ"] = (count, "Number of requested frames")
            header["COMPLETE"] = (True, "All requested frames acquired")
            header["BUNIT"] = "ADU"
            header["CAMERA"] = "".join(c if 32 <= ord(c) < 127 else "?"
                                       for c in settings["model"])
            header["RDSPEED"] = (-1 if settings["speed"] is None else settings["speed"])
            for name, value in zip(("XSTART", "XEND", "YSTART", "YEND", "XBINNING", "YBINNING"),
                                   settings["roi"]):
                header[name] = int(value)
            header["SENSMODE"] = "AREA"
            header["TRIGSRC"] = "INTERNAL"
            header["HOSTBEG"] = (host_start, "Host UTC before capture")
            header["HOSTEND"] = (host_end, "Host UTC after capture")
            header["STAMPCHK"] = (previous_stamp is not None, "Camera frame stamps checked for gaps")
            header["DEMO"] = self.demo
            header.add_comment("ROI coordinates are unbinned; XEND/YEND are exclusive.")
            header.add_comment("HOSTBEG/HOSTEND are host times, not hardware exposure timestamps.")
            table = fits.BinTableHDU.from_columns([
                fits.Column(name="FRAME_INDEX", format="K", array=np.arange(count)),
                fits.Column(name="FRAME_STAMP", format="K", array=np.asarray(stamps, dtype=np.int64)),
            ], name="FRAMEINFO")
            table.header.add_comment("FRAME_STAMP=-1 means unavailable; FRAME_INDEX is zero-based.")
            # Write beside the destination; replace only after the complete write succeeds.
            fd, temporary = tempfile.mkstemp(prefix=".orca_", suffix=".fits", dir=os.path.dirname(path))
            os.close(fd)
            fits.HDUList([fits.PrimaryHDU(cube, header=header), table]).writeto(
                temporary, overwrite=True, checksum=True)
            if self.cancel_record.is_set() or self.isInterruptionRequested():
                self.status.emit("Recording cancelled — output discarded")
                return
            os.replace(temporary, path)
            temporary = None
            self.status.emit(f"Saved {count} frames: {path}")
        finally:
            if temporary is not None:
                os.unlink(temporary)
            self.stop_capture()
            self.recording.emit(False)
            if resume and not self.isInterruptionRequested():
                self.begin()

    def run(self):
        try:
            if self.demo:
                self.cam = DemoCamera()
            else:
                import pylablib
                pylablib.par["devices/only_windows_dlls"] = False
                pylablib.par["devices/dlls/dcamapi"] = "/usr/local/lib"
                
                from pylablib.devices import DCAM
                self.cam = DCAM.DCAMCamera(self.index)
                self.cam.clear_acquisition()
                self.cam.set_trigger_mode("int")
                sensor = self.cam.get_attribute("SENSOR MODE", error_on_missing=False)
                if sensor is not None:
                    area = next((v for label, v in sensor.labels.items()
                                 if label.upper() == "AREA"), None)
                    if area is None:
                        raise RuntimeError("AREA sensor mode is unavailable.")
                    if sensor.get_value() != area:
                        self.cam.set_attribute_value("SENSOR MODE", area)
            self.publish_settings()
            self.status.emit("Connected — stopped")
            while not self.isInterruptionRequested():
                try:
                    command, values = self.commands.get(timeout=0.05)
                except queue.Empty:
                    command, values = None, None
                try:
                    if command == "record":
                        self.record_sequence(values)
                    elif command == "stop":
                        self.stop_capture()
                        self.status.emit("Stopped")
                    elif command in ("start", "snap"):
                        self.begin(single=command == "snap")
                    elif command == "apply":
                        resume = self.live
                        self.stop_capture()
                        if values["speed"] is not None:
                            self.cam.set_attribute_value("READOUT SPEED", values["speed"])
                        x, y, w, h = values["roi"]
                        b = values["bin"]
                        self.cam.set_roi(x, x+w, y, y+h, hbin=b, vbin=b)
                        self.cam.set_exposure(values["exposure"])
                        self.publish_settings()
                        if resume:
                            self.begin()
                        else:
                            self.status.emit("Settings applied — stopped")
                    if self.live or self.single:
                        frame = self.cam.read_newest_image()
                        if frame is not None:
                            self.serial += 1
                            packet = (self.serial, np.array(frame, copy=True),
                                      self.actual.copy(), datetime.now(timezone.utc).isoformat())
                            with self.lock:
                                self.latest = packet  # One-slot mailbox: never queue old images.
                            self.deadline = time.monotonic()+max(5, self.cam.get_exposure()+3)
                            if self.single:
                                self.stop_capture()
                                self.status.emit("Single frame acquired — stopped")
                        elif time.monotonic() > self.deadline:
                            raise TimeoutError("No new frame received. Check the camera and USB connection.")
                except Exception:
                    self.recording.emit(False)
                    details = traceback.format_exc()
                    try:
                        self.stop_capture()
                        self.publish_settings()  # Show actual state after a partial setting failure.
                    except Exception:
                        details += "\n" + traceback.format_exc()
                    self.failure.emit(details)
                    self.status.emit("Error — stopped")
        except Exception:
            self.failure.emit(traceback.format_exc())
        finally:
            if self.cam is not None:
                try:
                    self.cam.close()
                except Exception:
                    self.failure.emit(traceback.format_exc())


class Window(QtWidgets.QMainWindow):
    def __init__(self, index=0, demo=False):
        super().__init__()
        self.setWindowTitle("ORCA-Fusion BT — Live view" + (" [DEMO]" if demo else ""))
        self.resize(1220, 860)
        self.packet = None
        self.last_serial = -1
        self.last_shape = None
        self.closing = False
        self.display_count = 0
        self.rate_start = time.monotonic()
        self.detector = (2304, 2304)
        self.record_busy = False
        self.applied = None
        self.worker = CameraThread(index, demo)

        root = QtWidgets.QWidget()
        self.setCentralWidget(root)
        layout = QtWidgets.QHBoxLayout(root)
        panel = QtWidgets.QWidget()
        panel.setMinimumWidth(330)
        side = QtWidgets.QVBoxLayout(panel)
        self.identity = QtWidgets.QLabel("Connecting…")
        self.identity.setWordWrap(True)
        side.addWidget(self.identity)
        self.controls = QtWidgets.QGroupBox("Camera settings")
        form = QtWidgets.QFormLayout(self.controls)
        self.exposure = QtWidgets.QDoubleSpinBox()
        self.exposure.setDecimals(3)
        self.exposure.setRange(0.001, 1e9)
        self.exposure.setSingleStep(10)
        self.exposure.setSuffix(" µs")
        self.exposure.setKeyboardTracking(False)
        form.addRow("Exposure", self.exposure)
        self.speed = QtWidgets.QComboBox()
        form.addRow("Readout speed ID", self.speed)
        self.binning = QtWidgets.QComboBox()
        form.addRow("Binning", self.binning)
        self.roi = []
        for label in ("X start", "Y start", "Width", "Height"):
            spin = QtWidgets.QSpinBox()
            spin.setRange(0 if "start" in label else 1, 2304)
            spin.setKeyboardTracking(False)
            self.roi.append(spin)
            form.addRow(label, spin)
        presets = QtWidgets.QWidget()
        buttons = QtWidgets.QHBoxLayout(presets)
        buttons.setContentsMargins(0, 0, 0, 0)
        for label, size in (("Full", None), ("512 centre", 512)):
            button = QtWidgets.QPushButton(label)
            button.clicked.connect(lambda checked=False, s=size: self.preset(s))
            buttons.addWidget(button)
        form.addRow(presets)
        self.apply_button = QtWidgets.QPushButton("Apply settings")
        self.apply_button.clicked.connect(self.apply_settings)
        form.addRow(self.apply_button)
        note = QtWidgets.QLabel("ROI is in unbinned detector pixels. Applied values are read back.\nAREA mode • internal trigger")
        note.setWordWrap(True)
        form.addRow(note)
        side.addWidget(self.controls)

        self.capture_controls = QtWidgets.QWidget()
        row = QtWidgets.QHBoxLayout(self.capture_controls)
        row.setContentsMargins(0, 0, 0, 0)
        for text, command in (("Start", "start"), ("Stop", "stop"), ("Snap", "snap")):
            button = QtWidgets.QPushButton(text)
            button.clicked.connect(lambda checked=False, cmd=command: self.send(cmd))
            row.addWidget(button)
        side.addWidget(self.capture_controls)
        self.auto = QtWidgets.QCheckBox("Auto contrast")
        self.auto.setChecked(True)
        side.addWidget(self.auto)
        self.save_button = QtWidgets.QPushButton("Save displayed frame (.npz)")
        self.save_button.setEnabled(False)
        self.save_button.clicked.connect(self.save_frame)
        side.addWidget(self.save_button)
        self.record_group = QtWidgets.QGroupBox("Record a new FITS sequence")
        record_form = QtWidgets.QFormLayout(self.record_group)
        self.frame_count = QtWidgets.QSpinBox()
        self.frame_count.setRange(1, 100000)
        self.frame_count.setValue(10)
        record_form.addRow("Number of frames", self.frame_count)
        self.record_button = QtWidgets.QPushButton("Record N frames…")
        self.record_button.clicked.connect(self.start_recording)
        record_form.addRow(self.record_button)
        self.cancel_button = QtWidgets.QPushButton("Cancel recording")
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self.worker.cancel_record.set)
        record_form.addRow(self.cancel_button)
        self.record_progress = QtWidgets.QProgressBar()
        self.record_progress.setRange(0, 10)
        self.record_progress.setValue(0)
        record_form.addRow(self.record_progress)
        hint = QtWidgets.QLabel("Uses applied settings. Cancel discards the sequence. "
                               "Raw cube limit: 512 MiB; extra RAM is needed for buffers/writing.")
        hint.setWordWrap(True)
        record_form.addRow(hint)
        side.addWidget(self.record_group)
        self.record_button.setEnabled(False)
        self.stats = QtWidgets.QLabel("No frame yet")
        self.stats.setWordWrap(True)
        side.addWidget(self.stats)
        self.log = QtWidgets.QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumHeight(170)
        side.addWidget(self.log)
        side.addStretch()
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(panel)
        scroll.setMinimumWidth(365)
        scroll.setMaximumWidth(390)
        layout.addWidget(scroll)
        self.view = pg.ImageView()
        self.view.ui.roiBtn.hide()
        self.view.ui.menuBtn.hide()
        layout.addWidget(self.view, 1)
        self.controls.setEnabled(False)
        self.capture_controls.setEnabled(False)
        self.worker.recording.connect(self.set_recording)
        self.worker.progress.connect(self.record_progress_received)
        self.worker.settings.connect(self.settings_received)
        self.worker.status.connect(self.show_status)
        self.worker.failure.connect(self.show_error)
        self.worker.finished.connect(self.finished)
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.display_latest)
        self.timer.start(50)
        self.worker.start()

    def show_status(self, message):
        self.statusBar().showMessage(message)
        self.log.appendPlainText(message)

    def show_error(self, message):
        print(message, file=sys.stderr)
        self.log.appendPlainText(message)

    def settings_received(self, values):
        self.applied = values.copy()
        self.identity.setText(values["model"])
        self.detector = values["detector"]
        self.exposure.setValue(values["exposure"]*1e6)
        self.speed.clear()
        for code, label in values["speeds"]:
            self.speed.addItem(label, code)
        self.speed.setCurrentIndex(self.speed.findData(values["speed"]))
        self.speed.setEnabled(bool(values["speeds"]))
        self.binning.clear()
        for b in values["bins"]:
            self.binning.addItem(f"{b} × {b}", b)
        x0, x1, y0, y1, b, _ = values["roi"]
        self.binning.setCurrentIndex(self.binning.findData(b))
        for spin, limit in zip(self.roi, self.detector*2):
            spin.setMaximum(limit)
        for spin, value in zip(self.roi, (x0, y0, x1-x0, y1-y0)):
            spin.setValue(value)
        if not self.closing and not self.record_busy:
            self.controls.setEnabled(True)
            self.capture_controls.setEnabled(True)
            self.record_button.setEnabled(True)

    def preset(self, size):
        width, height = self.detector
        w, h = (width, height) if size is None else (min(size, width), min(size, height))
        for spin, value in zip(self.roi, ((width-w)//2, (height-h)//2, w, h)):
            spin.setValue(value)

    def send(self, command, values=None):
        self.worker.commands.put((command, values))

    def apply_settings(self):
        x, y, w, h = [spin.value() for spin in self.roi]
        if x+w > self.detector[0] or y+h > self.detector[1]:
            QtWidgets.QMessageBox.warning(self, "ROI", "ROI extends beyond the detector.")
            return
        self.controls.setEnabled(False)
        self.capture_controls.setEnabled(False)
        self.record_button.setEnabled(False)
        self.send("apply", dict(exposure=self.exposure.value()*1e-6,
                                speed=self.speed.currentData(), bin=self.binning.currentData(),
                                roi=(x, y, w, h)))

    def display_latest(self):
        with self.worker.lock:
            packet = self.worker.latest
        if packet is None or packet[0] == self.last_serial:
            return
        self.packet = packet
        self.last_serial, frame, settings, timestamp = packet
        changed = frame.shape != self.last_shape
        self.view.setImage(frame, autoRange=changed, autoLevels=self.auto.isChecked(),
                           autoHistogramRange=changed)
        self.last_shape = frame.shape
        self.save_button.setEnabled(True)
        self.display_count += 1
        elapsed = time.monotonic()-self.rate_start
        if elapsed >= 1:
            # Statistics describe all pixels; full-scale count is not a detector saturation calibration.
            full_scale = np.iinfo(frame.dtype).max
            self.stats.setText(
                f"{frame.shape[1]} × {frame.shape[0]} | {frame.dtype}\n"
                f"Exposure: {settings['exposure']*1e6:.3f} µs\n"
                f"Min / max: {frame.min()} / {frame.max()} ADU\n"
                f"Mean: {frame.mean():.1f} ADU\n"
                f"At digital full scale: {100*np.count_nonzero(frame == full_scale)/frame.size:.3f}%\n"
                f"Display updates: {self.display_count/elapsed:.1f}/s (not camera FPS)")
            self.rate_start, self.display_count = time.monotonic(), 0

    def set_recording(self, busy):
        self.record_busy = busy
        enabled = not busy and not self.closing and self.worker.isRunning()
        self.controls.setEnabled(enabled)
        self.capture_controls.setEnabled(enabled)
        self.record_button.setEnabled(enabled)
        self.frame_count.setEnabled(enabled)
        self.cancel_button.setEnabled(busy and not self.closing)

    def record_progress_received(self, done, total):
        self.record_progress.setRange(0, total)
        self.record_progress.setValue(done)
        self.record_progress.setFormat(f"{done} / {total} frames")

    def start_recording(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save new FITS sequence", "orca_sequence.fits", "FITS cube (*.fits)")
        if not path:
            return
        if not path.lower().endswith(".fits"):
            path += ".fits"
            if os.path.exists(path) and QtWidgets.QMessageBox.question(
                self, "Replace file?", f"Replace {path}?",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No) != QtWidgets.QMessageBox.Yes:
                return
        self.worker.cancel_record.clear()
        self.set_recording(True)
        self.send("record", {"count": self.frame_count.value(), "path": os.path.abspath(path)})

    def save_frame(self):
        # Hold this packet before opening the dialog so its metadata and image stay paired.
        packet = self.packet
        if packet is None:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save frame", "orca_frame.npz", "NumPy archive (*.npz)")
        if not path:
            return
        if not path.lower().endswith(".npz"):
            path += ".npz"
        _, frame, settings, timestamp = packet
        try:
            np.savez(path, frame=frame, exposure_s=settings["exposure"],
                     roi=np.asarray(settings["roi"]),
                     readout_speed_id=-1 if settings["speed"] is None else settings["speed"],
                     camera=settings["model"], host_received_utc=timestamp)
            self.show_status(f"Saved {path}")
        except Exception:
            self.show_error(traceback.format_exc())

    def finished(self):
        self.record_button.setEnabled(False)
        self.cancel_button.setEnabled(False)
        self.controls.setEnabled(False)
        self.capture_controls.setEnabled(False)
        self.statusBar().showMessage("Camera closed")
        if self.closing:
            self.close()

    def closeEvent(self, event):
        if self.worker.isRunning():
            self.closing = True
            self.controls.setEnabled(False)
            self.capture_controls.setEnabled(False)
            self.record_button.setEnabled(False)
            self.cancel_button.setEnabled(False)
            self.worker.cancel_record.set()
            self.worker.requestInterruption()
            self.statusBar().showMessage("Closing camera…")
            event.ignore()  # Keep Qt alive until the camera-owning thread has closed it.
        else:
            self.timer.stop()
            event.accept()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0, help="DCAM camera index (default: 0)")
    parser.add_argument("--demo", action="store_true", help="Use synthetic images")
    args = parser.parse_args()
    app = QtWidgets.QApplication(sys.argv[:1])
    window = Window(args.camera, args.demo)
    window.show()
    sys.exit(app.exec_())


"""
# e.g. to read in the saved frames : 

from astropy.io import fits
import matplotlib.pyplot as plt

with fits.open("orca_sequence.fits") as hdul:
    hdul.info()
    cube = hdul[0].data
    header = hdul[0].header
    print(cube.shape)  # (N, rows, columns)
    print(header)

    plt.imshow(cube[0], cmap="gray")
    plt.colorbar(label="ADU")
    plt.show()


"""

