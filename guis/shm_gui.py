#!/usr/bin/env python3
"""Standalone compact Hamamatsu DCAM GUI with hardware ROI and SHM reconnection.

Reviewed against device/dcam_cmd.c, dcamlib.c and server_common/servif.h.

Dependencies: PyQt5, pyqtgraph, numpy, ImageStreamIOWrap.

ROI conventions:
  - The manual centre and size, and dragged rectangle, are in *current streamed*
    image coordinates, x=column, y=row (0-based).
  - Sensor origin = current stream's SENSOR origin + selected image origin.
  - C server offset is measured relative to the centred ROI, not the sensor edge.
  - Cropping automatically syncs an unconfirmed sensor origin using server `st`.
  - An advanced manual origin override remains available if explicitly confirmed.
  - Selecting a rectangle never changes SHM; pressing Crop sends hardware commands.
  - ImageStreamIO [X,Y] arrays are converted once to NumPy [Y,X] for display.

This script cannot itself resize the server-side ImageStreamIO image or DCAM
buffers. The device-specific C command handler must do so correctly.
"""

import re
import socket
import subprocess
import sys

import numpy as np
import pyqtgraph as pg
from PyQt5.QtCore import Qt, QThread, QTimer, pyqtSignal
from PyQt5.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout,
    QGroupBox, QHBoxLayout, QLabel, QLineEdit, QMainWindow, QMessageBox,
    QPlainTextEdit, QPushButton, QScrollArea, QSpinBox, QToolButton,
    QVBoxLayout, QWidget,
)

import ImageStreamIOWrap

# Update only if the actual camera sensor is different.
SENSOR_WIDTH = 2304
SENSOR_HEIGHT = 2304
# ORCA-Fusion BT, normal AREA readout: H/V sizes and positions are 4-pixel
# multiples. If running PROGRESSIVE / lightsheet, set X_STEP = 1, Y_STEP = 4.
X_STEP = 4             # ROI left/top-position step (normal AREA mode)
Y_STEP = 4
# The current C protocol applies `size` with a temporarily centred ROI.
# Since (2304 - size) / 2 must also be divisible by 4, a multiple of 8
# is conservative for both dimensions until the server has an atomic ROI API.
X_SIZE_STEP = 8
Y_SIZE_STEP = 8
STREAM_NAME = "dcam0_data"
SERVER_HOST = "127.0.0.1"
SERVER_PORT = 18010
DEVICE_NAME = "dcam0"
IMAGE_REFRESH_MS = 30
TMUX_REFRESH_MS = 1000
POWER_GAMMA = 0.5


def roi_from_center(cx, cy, width, height):
    """Integer half-open ROI (x, y, w, h), with centre in edge coordinates."""
    w, h = int(width), int(height)
    if w <= 0 or h <= 0:
        raise ValueError("ROI width and height must be positive")
    x = int(np.floor(float(cx) - w / 2.0 + 0.5))
    y = int(np.floor(float(cy) - h / 2.0 + 0.5))
    return x, y, w, h


def validate_sensor_roi(x, y, w, h):
    if x < 0 or y < 0 or w <= 0 or h <= 0:
        raise ValueError("ROI offset must be nonnegative and size positive")
    if x + w > SENSOR_WIDTH or y + h > SENSOR_HEIGHT:
        raise ValueError(
            f"Requested hardware ROI ({x}, {y}, {w}, {h}) lies outside "
            f"the {SENSOR_WIDTH} x {SENSOR_HEIGHT} sensor"
        )
    if x % X_STEP or w % X_SIZE_STEP or y % Y_STEP or h % Y_SIZE_STEP:
        raise ValueError(
            f"ROI X/Y origin must be multiples of {X_STEP}/{Y_STEP} pixels, "
            f"and width/height multiples of {X_SIZE_STEP}/{Y_SIZE_STEP}. "
            "The size restriction also avoids an unaligned centred "
            "intermediate ROI in the current C server."
        )


def snap_sensor_roi(x, y, w, h):
    """Expand to outward hardware-grid boundaries (half-open rectangle)."""
    x0 = (x // X_STEP) * X_STEP
    y0 = (y // Y_STEP) * Y_STEP
    # Expand width/height to size multiples, not just the position grid.
    wnew = ((x + w - x0 + X_SIZE_STEP - 1) // X_SIZE_STEP) * X_SIZE_STEP
    hnew = ((y + h - y0 + Y_SIZE_STEP - 1) // Y_SIZE_STEP) * Y_SIZE_STEP
    # Shift the expanded ROI inward near the sensor edge without excluding
    # any requested pixel; positions remain multiples of X_STEP/Y_STEP.
    if x0 + wnew > SENSOR_WIDTH:
        x0 = ((SENSOR_WIDTH - wnew) // X_STEP) * X_STEP
    if y0 + hnew > SENSOR_HEIGHT:
        y0 = ((SENSOR_HEIGHT - hnew) // Y_STEP) * Y_STEP
    return x0, y0, wnew, hnew


def server_offsets_from_sensor_roi(x, y, w, h):
    """Convert absolute 0-based sensor ROI to the DCAM server's centred offset.

    The C server calculates HPOS=(hfullsize-hsize)/2 + xoffset,
    and analogously for VPOS. Thus its `offset` is NOT the sensor HPOS.
    """
    validate_sensor_roi(x, y, w, h)
    center_x = (SENSOR_WIDTH - w) // 2
    center_y = (SENSOR_HEIGHT - h) // 2
    return x - center_x, y - center_y


def sensor_origin_from_server_offset(server_x, server_y, w, h):
    """Inverse of server_offsets_from_sensor_roi (uses sensor full size)."""
    return ((SENSOR_WIDTH - w) // 2 + server_x,
            (SENSOR_HEIGHT - h) // 2 + server_y)


def parse_server_status(reply):
    """Read `dcam0 st` output. None if disconnected or no valid dimensions.

    Returns {'width','height','server_x','server_y','sensor_x','sensor_y',
             'capture'}, where the two offsets are server-centred coordinates.
    """
    fields = {}
    for line in reply.splitlines():
        match = re.match(r"\s*(Format size|Image size|Offset|Capture)\s*:\s*\[\s*(.*?)\s*\]", line, re.I)
        if match:
            fields[match.group(1).lower()] = match.group(2).strip()
    try:
        fmt = re.fullmatch(r"(\d+)\s*x\s*(\d+)", fields['format size'], re.I)
        im = re.fullmatch(r"(\d+)\s*x\s*(\d+)", fields['image size'], re.I)
        off = re.fullmatch(r"([+-]?\d+)\s*,\s*([+-]?\d+)", fields['offset'])
        if not all((fmt, im, off)):
            return None
        full_w, full_h = map(int, fmt.groups())
        w, h = map(int, im.groups())
        ox, oy = map(int, off.groups())
        # The script's hardware geometry is specific to this camera.
        if (full_w, full_h) != (SENSOR_WIDTH, SENSOR_HEIGHT):
            raise ValueError(f"Server reports sensor {full_w}x{full_h}, GUI expects {SENSOR_WIDTH}x{SENSOR_HEIGHT}")
        sx, sy = sensor_origin_from_server_offset(ox, oy, w, h)
        return dict(width=w, height=h, server_x=ox, server_y=oy,
                    sensor_x=sx, sensor_y=sy,
                    capture=fields.get('capture', ''))
    except (KeyError, ValueError):
        return None


def transform_display(frame, mode, buffer=None):
    """Display-only transform; never writes back to SHM or the raw frame."""
    if mode == "linear":
        return frame, buffer
    if buffer is None or buffer.shape != frame.shape:
        buffer = np.empty(frame.shape, dtype=np.float32)
    np.maximum(frame, 0, out=buffer)
    if mode == "log":
        np.log1p(buffer, out=buffer)
        buffer *= np.float32(1.0 / np.log(10.0))
    elif mode == "power":
        np.power(buffer, POWER_GAMMA, out=buffer)
    else:
        raise ValueError(f"Unknown display mode: {mode}")
    return buffer, buffer


def read_camera_frame(image):
    """Read ImageStreamIO into canonical NumPy image coordinates [y, x].

    ImageStreamIO stores dimensions as (X, Y) and its Python buffer binding
    exposes those axes in that order using Fortran-style strides (X-fastest).
    Consequently np.array(image) has shape (width, height). Transposing
    its *copied* array yields (height, width) with rows/columns corresponding
    to the detector and to the row-major pyqtgraph ImageItem.

    This function is deliberately used in both the reconnect and the live
    image paths. Copying occurs BEFORE the transpose to avoid retaining a
    view of the server's possibly replaced SHM mapping.
    """
    raw_xy = np.array(image, copy=True)
    if raw_xy.ndim != 2:
        raise ValueError(f"Expected 2D ImageStreamIO data, got {raw_xy.shape}")
    return raw_xy.T


def tcp_command(command, timeout=3.0):
    """Execute one C-server command; wait for the actual SERVER_EOR (\r).

    Source: server_common/servif.h defines SERVER_EOR as "\r" and errors
    start with "Error: ". A command with no output still returns EOR and is
    successful; a timeout or disconnection without EOR is not successful.
    """
    wire = f"{DEVICE_NAME} {command}\r".encode("ascii")
    received = bytearray()
    with socket.create_connection((SERVER_HOST, SERVER_PORT), timeout=timeout) as sock:
        sock.settimeout(timeout)
        sock.sendall(wire)
        while b"\r" not in received:
            chunk = sock.recv(4096)
            if not chunk:
                raise RuntimeError(f"Server closed before completing '{command}' (no EOR)")
            received.extend(chunk)
            if len(received) > 65536:
                raise RuntimeError(f"Response to '{command}' exceeds 64 KiB")
    payload = bytes(received).split(b"\r", 1)[0]
    reply = payload.decode("utf-8", errors="replace").strip()
    if any(line.lstrip().startswith("Error:") for line in reply.splitlines()):
        raise RuntimeError(f"'{command}' rejected: {reply}")
    return reply  # empty response is normal for successful commands


class CommandSequenceWorker(QThread):
    completed = pyqtSignal(bool, str, object)

    def __init__(self, commands, expected=None, parent=None):
        super().__init__(parent)
        self.commands = tuple(commands)
        self.expected = expected  # tuple (sensor_x, sensor_y, width, height)

    def run(self):
        log = []
        observed = None
        try:
            for cmd in self.commands:
                try:
                    reply = tcp_command(cmd)
                except RuntimeError as exc:
                    # The C server rejects `stop` if the camera is already
                    # idle. For a reconfiguration that is a safe no-op.
                    if cmd == 'stop' and 'is not running' in str(exc):
                        reply = '(already stopped)'
                    else:
                        raise
                log.append(f"> {cmd}\n{reply if reply else '(OK)'}")
                if cmd == 'st':
                    observed = parse_server_status(reply)
                    if observed is None:
                        raise RuntimeError('Cannot parse camera ROI from `st` response')
            if self.expected is not None:
                x, y, w, h = self.expected
                actual = (observed['sensor_x'], observed['sensor_y'],
                          observed['width'], observed['height'])
                if actual != self.expected:
                    raise RuntimeError(f'Camera reports ROI {actual}, expected {(x, y, w, h)}')
                if not observed['capture'].startswith('BUSY'):
                    raise RuntimeError(f"Camera is not capturing after run: {observed['capture']}")
        except Exception as exc:
            self.completed.emit(False, "\n".join(log + [f"ERROR: {exc}"]), observed)
            return
        self.completed.emit(True, "\n".join(log), observed)


class RoiViewBox(pg.ViewBox):
    """Left-drag selects image rectangle when draw mode is enabled."""

    dragRectangle = pyqtSignal(float, float, float, float, bool)

    def __init__(self):
        super().__init__(enableMenu=False)
        self.draw_mode = False

    def mouseDragEvent(self, event, axis=None):
        if self.draw_mode and event.button() == Qt.LeftButton:
            event.accept()
            p0 = self.mapToView(event.buttonDownPos())
            p1 = self.mapToView(event.pos())
            x0, x1 = sorted((p0.x(), p1.x()))
            y0, y1 = sorted((p0.y(), p1.y()))
            self.dragRectangle.emit(x0, y0, x1, y1, bool(event.isFinish()))
            return
        super().mouseDragEvent(event, axis=axis)


class ServerCameraGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Hamamatsu DCAM — Camera & ROI")
        self.resize(1350, 820)

        self.cam = None
        self.shm_connected = False
        self.latest_frame = None
        self._current_shape = None
        self._display_buffer = None
        self._last_mouse_position = None
        self._roi_setting_widgets = False
        self._roi_box = None
        self._worker = None
        self._pending_target = None  # (origin_x, origin_y, width, height)
        self._reconnect_expect_shape = None
        self._reconnect_attempts = 0
        self._reconnect_error = ""

        self.setup_ui()
        self.start_timers()
        self.begin_reconnect()

    def setup_ui(self):
        # Single-file GUI: no launcher or imports from older GUI versions.
        root = QWidget()
        self.setCentralWidget(root)
        outer = QHBoxLayout(root)

        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setMinimumWidth(750) #345)
        left_scroll.setMaximumWidth(820)#410)
        left_panel = QWidget()
        left = QVBoxLayout(left_panel)
        left_scroll.setWidget(left_panel)
        outer.addWidget(left_scroll)

        # Camera state controls remain unchanged.
        state_group = QGroupBox("Camera")
        state_row = QHBoxLayout(state_group)
        self.btn_open = QPushButton("Open")
        self.btn_close = QPushButton("Close")
        self.btn_run = QPushButton("Live")
        self.btn_stop = QPushButton("Stop")
        for btn, cmd in ((self.btn_open, "open"), (self.btn_close, "close"),
                         (self.btn_run, "run"), (self.btn_stop, "stop")):
            btn.clicked.connect(lambda _checked=False, c=cmd: self.send_basic(c))
            state_row.addWidget(btn)
        left.addWidget(state_group)

        # Exposure is independent from ROI size/offset. The latter are
        # specified in ONE place, below, rather than duplicated here.
        exposure_group = QGroupBox("Exposure")
        exposure_row = QHBoxLayout(exposure_group)
        exposure_row.addWidget(QLabel("Seconds:"))
        self.inp_exp = QLineEdit("0.0001")
        self.inp_exp.setToolTip("Camera exposure time in seconds")
        exposure_row.addWidget(self.inp_exp, 1)
        self.btn_apply = QPushButton("Set exposure")
        self.btn_apply.clicked.connect(self.apply_settings)
        exposure_row.addWidget(self.btn_apply)
        left.addWidget(exposure_group)

        roi_group = QGroupBox("Hardware ROI")
        roi_form = QFormLayout(roi_group)
        self.roi_cx = QDoubleSpinBox()
        self.roi_cy = QDoubleSpinBox()
        self.roi_w = QSpinBox()
        self.roi_h = QSpinBox()
        for spin, upper in ((self.roi_cx, SENSOR_WIDTH),
                            (self.roi_cy, SENSOR_HEIGHT)):
            spin.setRange(0, upper)
            spin.setDecimals(1)
            spin.setSingleStep(1.0)
        self.roi_w.setRange(1, SENSOR_WIDTH)
        self.roi_h.setRange(1, SENSOR_HEIGHT)
        self.roi_w.setValue(SENSOR_WIDTH)
        self.roi_h.setValue(SENSOR_HEIGHT)
        self.roi_cx.setValue(SENSOR_WIDTH / 2)
        self.roi_cy.setValue(SENSOR_HEIGHT / 2)

        # Two short rows instead of four long rows.
        centre_row = QHBoxLayout()
        centre_row.addWidget(QLabel("X"))
        centre_row.addWidget(self.roi_cx, 1)
        centre_row.addWidget(QLabel("Y"))
        centre_row.addWidget(self.roi_cy, 1)
        roi_form.addRow("Centre (px):", centre_row)
        size_row = QHBoxLayout()
        size_row.addWidget(QLabel("W"))
        size_row.addWidget(self.roi_w, 1)
        size_row.addWidget(QLabel("H"))
        size_row.addWidget(self.roi_h, 1)
        roi_form.addRow("Size (px):", size_row)
        for spin in (self.roi_cx, self.roi_cy, self.roi_w, self.roi_h):
            spin.valueChanged.connect(self.preview_roi_from_inputs)

        # Editing ROI fields already updates the overlay, so a separate
        # Preview button would be redundant.
        action_row = QHBoxLayout()
        self.btn_draw = QPushButton("Draw ROI")
        self.btn_draw.setCheckable(True)
        self.btn_draw.setToolTip("Drag a rectangle over the live image")
        self.btn_draw.toggled.connect(self.toggle_drawing)
        self.btn_crop = QPushButton("Crop")
        self.btn_crop.clicked.connect(self.crop_roi)
        self.btn_full = QPushButton("Full sensor")
        self.btn_full.clicked.connect(self.reset_full_sensor)
        action_row.addWidget(self.btn_draw)
        action_row.addWidget(self.btn_crop)
        action_row.addWidget(self.btn_full)
        roi_form.addRow(action_row)

        self.roi_preview_label = QLabel("No selection")
        self.roi_preview_label.setWordWrap(True)
        roi_form.addRow(self.roi_preview_label)

        utility_row = QHBoxLayout()
        self.btn_sync = QPushButton("Sync camera")
        self.btn_sync.setToolTip("Read hardware ROI origin from camera status (st)")
        self.btn_sync.clicked.connect(self.sync_server_status)
        self.btn_reconnect = QPushButton("Reconnect SHM")
        self.btn_reconnect.clicked.connect(self.force_reconnect)
        utility_row.addWidget(self.btn_sync)
        utility_row.addWidget(self.btn_reconnect)
        roi_form.addRow(utility_row)

        # Manual stream-origin settings are still available when automatic
        # status sync is unavailable, but no longer occupy the main UI.
        self.advanced_toggle = QToolButton()
        self.advanced_toggle.setText("Advanced: stream origin")
        self.advanced_toggle.setCheckable(True)
        self.advanced_toggle.setArrowType(Qt.RightArrow)
        self.advanced_toggle.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        roi_form.addRow(self.advanced_toggle)
        self.advanced_origin = QWidget()
        origin_form = QFormLayout(self.advanced_origin)
        origin_form.setContentsMargins(0, 0, 0, 0)
        self.origin_x = QSpinBox()
        self.origin_y = QSpinBox()
        self.origin_x.setRange(0, SENSOR_WIDTH - 1)
        self.origin_y.setRange(0, SENSOR_HEIGHT - 1)
        origin_form.addRow("Sensor origin X:", self.origin_x)
        origin_form.addRow("Sensor origin Y:", self.origin_y)
        self.origin_confirmed = QCheckBox("Confirm stream origin")
        origin_form.addRow(self.origin_confirmed)
        self.origin_x.valueChanged.connect(self.origin_changed)
        self.origin_y.valueChanged.connect(self.origin_changed)
        roi_form.addRow(self.advanced_origin)
        self.advanced_toggle.toggled.connect(self.set_advanced_visible)
        self.advanced_origin.hide()
        left.addWidget(roi_group)

        # Keep the existing colormaps, contrast and nonlinear scaling.
        display_group = QGroupBox("Image Display")
        display_form = QFormLayout(display_group)
        self.combo_cmap = QComboBox()
        self.combo_cmap.addItems(["viridis", "cividis", "magma", "inferno", "plasma"])
        self.combo_cmap.currentTextChanged.connect(self.change_colormap)
        display_form.addRow("Colormap:", self.combo_cmap)
        self.chk_auto = QCheckBox("Auto contrast")
        self.chk_auto.setChecked(True)
        display_form.addRow(self.chk_auto)
        self.chk_log = QCheckBox("Log scale")
        self.chk_power = QCheckBox(f"Power-law (gamma={POWER_GAMMA:g})")
        self.chk_log.toggled.connect(self.log_toggled)
        self.chk_power.toggled.connect(self.power_toggled)
        display_form.addRow(self.chk_log)
        display_form.addRow(self.chk_power)
        left.addWidget(display_group)

        log_group = QGroupBox("Camera Server Log")
        log_layout = QVBoxLayout(log_group)
        self.tmux_log = QPlainTextEdit()
        self.tmux_log.setReadOnly(True)
        self.tmux_log.setMinimumHeight(120)
        self.tmux_log.setStyleSheet(
            "background-color: #1e1e1e; color: #00ff00; font-family: monospace;"
        )
        log_layout.addWidget(self.tmux_log)
        left.addWidget(log_group, 1)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(right, 1)
        self.view_box = RoiViewBox()
        image_item = pg.ImageItem(
            np.zeros((2, 2), dtype=np.uint8), axisOrder="row-major"
        )
        self.image_view = pg.ImageView(view=self.view_box, imageItem=image_item)
        self.image_view.ui.roiBtn.hide()
        self.image_view.ui.menuBtn.hide()
        self.image_view.setColorMap(pg.colormap.get("viridis"))
        self.view_box.dragRectangle.connect(self.on_drag_rectangle)
        right_layout.addWidget(self.image_view, 1)

        self._roi_box = pg.RectROI(
            [0, 0], [10, 10], pen=pg.mkPen("y", width=2), movable=True,
        )
        self._roi_box.addScaleHandle([1, 1], [0, 0])
        self._roi_box.addScaleHandle([0, 0], [1, 1])
        self._roi_box.setZValue(100)
        self.view_box.addItem(self._roi_box)
        self._roi_box.sigRegionChangeFinished.connect(self.roi_box_moved)
        self._roi_box.hide()

        self.pixel_label = QLabel("Hover over an image pixel for its raw value")
        self.pixel_label.setStyleSheet("font-family: monospace; padding: 5px;")
        right_layout.addWidget(self.pixel_label)
        scene = self.image_view.getImageItem().scene()
        self.mouse_proxy = pg.SignalProxy(
            scene.sigMouseMoved, rateLimit=30, slot=self.on_mouse_moved
        )

    def start_timers(self):
        self.img_timer = QTimer(self)
        self.img_timer.timeout.connect(self.update_frame)
        self.tmux_timer = QTimer(self)
        self.tmux_timer.timeout.connect(self.update_tmux_log)
        self.tmux_timer.start(TMUX_REFRESH_MS)
        self.reconnect_timer = QTimer(self)
        self.reconnect_timer.timeout.connect(self.try_reconnect)

    def set_busy(self, busy):
        for widget in (self.btn_open, self.btn_close, self.btn_run,
                       self.btn_stop, self.btn_apply, self.btn_crop,
                       self.btn_full, self.btn_reconnect, self.btn_sync):
            widget.setEnabled(not busy)

    def status(self, message, timeout_ms=10000):
        self.statusBar().showMessage(message, timeout_ms)
        print(message)

    def origin_changed(self, _value):
        self.origin_confirmed.setChecked(False)
        self.preview_label_refresh()

    def _set_origin(self, x, y, verified):
        self.origin_x.blockSignals(True)
        self.origin_y.blockSignals(True)
        self.origin_x.setValue(x)
        self.origin_y.setValue(y)
        self.origin_x.blockSignals(False)
        self.origin_y.blockSignals(False)
        self.origin_confirmed.setChecked(verified)
        self.preview_label_refresh()

    def _set_roi_fields(self, x, y, w, h, show=True):
        self._roi_setting_widgets = True
        fields = ((self.roi_cx, x + w / 2.0),
                  (self.roi_cy, y + h / 2.0),
                  (self.roi_w, w), (self.roi_h, h))
        for widget, value in fields:
            widget.blockSignals(True)
            widget.setValue(value)
            widget.blockSignals(False)
        self._roi_setting_widgets = False
        if show:
            self.preview_roi_from_inputs()
        else:
            self._roi_box.hide()
            self.preview_label_refresh()

    def get_view_roi(self):
        return roi_from_center(
            self.roi_cx.value(), self.roi_cy.value(),
            self.roi_w.value(), self.roi_h.value(),
        )

    def preview_label_refresh(self):
        x, y, w, h = self.get_view_roi()
        sx, sy = x + self.origin_x.value(), y + self.origin_y.value()
        if self.origin_confirmed.isChecked():
            self.roi_preview_label.setText(
                f"Selected: {w} × {h} px | Sensor ({sx}, {sy})"
            )
        else:
            self.roi_preview_label.setText(
                f"Selected: {w} × {h} px | Origin checked automatically on Crop"
            )

    def preview_roi_from_inputs(self, *_args):
        if self._roi_setting_widgets or self._roi_box is None:
            return
        x, y, w, h = self.get_view_roi()
        self._roi_setting_widgets = True
        try:
            self._roi_box.setPos((x, y), update=False)
            self._roi_box.setSize((w, h), update=True)
            self._roi_box.show()
        finally:
            self._roi_setting_widgets = False
        self.preview_label_refresh()

    def roi_box_moved(self, _roi):
        if self._roi_setting_widgets or self.latest_frame is None:
            return
        pos = self._roi_box.pos()
        size = self._roi_box.size()
        self._set_roi_fields(
            int(np.floor(pos.x())), int(np.floor(pos.y())),
            max(1, int(np.ceil(size.x()))), max(1, int(np.ceil(size.y()))),
        )

    def toggle_drawing(self, enabled):
        self.view_box.draw_mode = enabled
        # Disable the ROI and its scale handles while ViewBox handles drags.
        self._roi_box.setEnabled(not enabled)
        self.btn_draw.setText(
            "Drawing…" if enabled else "Draw ROI"
        )
        self.status("Drag over the image to select an ROI" if enabled else "Draw mode off", 3000)

    def on_drag_rectangle(self, x0, y0, x1, y1, finished):
        if self.latest_frame is None:
            return
        h, w = self.latest_frame.shape
        x0 = max(0, min(w, int(np.floor(x0))))
        y0 = max(0, min(h, int(np.floor(y0))))
        x1 = max(0, min(w, int(np.ceil(x1))))
        y1 = max(0, min(h, int(np.ceil(y1))))
        if x1 > x0 and y1 > y0:
            self._set_roi_fields(x0, y0, x1 - x0, y1 - y0)
        if finished:
            self.btn_draw.setChecked(False)

    def validate_view_roi(self):
        if self.latest_frame is None:
            raise ValueError("No live image is available for image-relative selection")
        x, y, w, h = self.get_view_roi()
        frame_h, frame_w = self.latest_frame.shape
        if x < 0 or y < 0 or x + w > frame_w or y + h > frame_h:
            raise ValueError(
                f"Selected ROI ({x}, {y}, {w}, {h}) is outside the "
                f"current {frame_w} x {frame_h} streamed image"
            )
        return x, y, w, h

    def crop_roi(self):
        try:
            x, y, w, h = self.validate_view_roi()
            # Cropping is a hardware operation: never infer the sensor origin
            # from image indices. If it has not been confirmed already, query
            # the camera's `st` command, compare geometry to live SHM, and
            # fill the origin fields automatically. A matching full-frame SHM
            # image is NOT, on its own, sufficient evidence of its origin.
            if not self.origin_confirmed.isChecked():
                try:
                    self._read_verified_camera_status()
                except Exception as exc:
                    raise ValueError(
                        "Cannot verify the current camera ROI automatically. "
                        f"{exc}\n\nCheck that the camera is Live, then "
                        "try Sync camera or Reconnect SHM."
                    ) from exc
            sensor_x = x + self.origin_x.value()
            sensor_y = y + self.origin_y.value()
            sx, sy, sw, sh = snap_sensor_roi(sensor_x, sensor_y, w, h)
            validate_sensor_roi(sx, sy, sw, sh)
            # In the normal case the current SHM ROI is itself on the 4px grid.
            # Do not silently expand a drawn crop outside the current image.
            img_x, img_y = sx - self.origin_x.value(), sy - self.origin_y.value()
            frame_h, frame_w = self.latest_frame.shape
            if (img_x < 0 or img_y < 0 or img_x + sw > frame_w or
                    img_y + sh > frame_h):
                raise ValueError(
                    "Snapping to the hardware pixel grid would extend outside "
                    "the current frame. Check the stream's sensor origin."
                )
        except ValueError as exc:
            QMessageBox.warning(self, "Invalid crop", str(exc))
            return
        if (sx, sy, sw, sh) != (sensor_x, sensor_y, w, h):
            self._set_roi_fields(img_x, img_y, sw, sh)
            self.status(
                f"ROI aligned to hardware step: X={sx}, Y={sy}, "
                f"W={sw}, H={sh}"
            )
        self.apply_hardware_roi(sx, sy, sw, sh)

    def reset_full_sensor(self):
        self.apply_hardware_roi(0, 0, SENSOR_WIDTH, SENSOR_HEIGHT)

    def apply_hardware_roi(self, x, y, w, h, exposure=None):
        try:
            validate_sensor_roi(x, y, w, h)
        except ValueError as exc:
            QMessageBox.warning(self, "Invalid hardware ROI", str(exc))
            return
        msg = (
            f"Set hardware ROI to sensor X={x}, Y={y}, W={w}, H={h}?\n\n"
            "Acquisition will stop, the ROI will be changed, and acquisition "
            "will restart. If the server fails to resize its buffers and SHM, "
            "the GUI will report a dimension mismatch."
        )
        if QMessageBox.question(
            self, "Apply hardware ROI", msg,
            QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes,
        ) != QMessageBox.Yes:
            return
        # First offset=0 avoids temporary out-of-sensor geometry when
        # transitioning from a previous large or displaced ROI.
        commands = ["stop"]
        if exposure is not None:
            commands.append(f"exp {exposure:.9g}")
        # The server offset is *relative to the centred sensor ROI*.
        # See dcamlib_SetImageOffset(), not an absolute sensor top-left.
        offx, offy = server_offsets_from_sensor_roi(x, y, w, h)
        commands.extend([
            "offset 0 0", f"size {w} {h}", f"offset {offx} {offy}",
            "run", "st"
        ])
        self._pending_target = (x, y, w, h)
        self.suspend_stream()
        self.set_busy(True)
        self.status(f"Configuring sensor ROI X={x}, Y={y}, W={w}, H={h}…")
        self._worker = CommandSequenceWorker(
            commands, expected=(x, y, w, h), parent=self
        )
        self._worker.completed.connect(self.roi_commands_finished)
        self._worker.start()

    def roi_commands_finished(self, ok, log, observed):
        print("Camera configuration transcript:\n" + log)
        self.set_busy(False)
        if not ok:
            self.status("ROI command failed; server may be stopped. See terminal details.", 20000)
            QMessageBox.critical(self, "Hardware ROI command failed", log)
            self._pending_target = None
            self.begin_reconnect()  # may recover an unchanged/stopped SHM
            return
        x, y, w, h = self._pending_target
        self.status(f"Hardware ROI verified by `st`; checking SHM for {w} x {h}…")
        self.begin_reconnect(expected_shape=(h, w))

    def _parse_int_pair(self, raw, label):
        tokens = raw.replace(",", " ").split()
        if len(tokens) != 2:
            raise ValueError(f"{label} requires two integers, e.g. '512 512'")
        try:
            return tuple(int(v) for v in tokens)
        except ValueError:
            raise ValueError(f"{label} must contain integers") from None

    def apply_settings(self):
        """Change exposure only, retaining the current hardware ROI.

        The existing C-server workflow stops capture before changing hardware
        properties and restarts it afterwards. This uses the same sequence,
        but never sends any size/offset command.
        """
        try:
            exp = float(self.inp_exp.text())
            if not np.isfinite(exp) or exp <= 0:
                raise ValueError("Exposure must be a finite positive number")
        except ValueError as exc:
            QMessageBox.warning(self, "Invalid exposure", str(exc))
            return

        # Reattach after stop/run, just as for hardware ROI changes.
        self._exposure_expected_shape = (
            self.latest_frame.shape if self.latest_frame is not None else None
        )
        self.suspend_stream()
        self.set_busy(True)
        self.status(f"Setting exposure to {exp:.9g} s…")
        self._worker = CommandSequenceWorker(
            ["stop", f"exp {exp:.9g}", "run", "st"], parent=self
        )
        self._worker.completed.connect(self.exposure_commands_finished)
        self._worker.start()

    def exposure_commands_finished(self, ok, log, observed):
        print("Exposure configuration transcript:\n" + log)
        self.set_busy(False)
        if not ok:
            self.status("Exposure command failed; check server log", 20000)
            QMessageBox.critical(self, "Exposure update failed", log)
            self.begin_reconnect()
            return
        # The camera's ROI stays unchanged. Verify that its SHM shape does too.
        expected = self._exposure_expected_shape
        if expected is not None and observed is not None:
            reported = (observed["height"], observed["width"])
            if reported != expected:
                self.status(
                    f"Camera image size unexpectedly changed: {reported[::-1]}",
                    20000,
                )
                QMessageBox.warning(
                    self, "Camera geometry changed",
                    f"Expected W×H={expected[::-1]}, camera reports W×H={reported[::-1]}.",
                )
                self.begin_reconnect()
                return
        self.status("Exposure updated; hardware ROI unchanged")
        self.begin_reconnect(expected_shape=expected)

    def set_advanced_visible(self, visible):
        self.advanced_origin.setVisible(visible)
        self.advanced_toggle.setArrowType(
            Qt.DownArrow if visible else Qt.RightArrow
        )

    def _read_verified_camera_status(self):
        """Determine the live SHM frame's absolute sensor origin using `st`.

        Refuse mismatched or idle camera geometry rather than assigning an
        origin to stale SHM data. This is called on demand by Crop and Sync.
        """
        if not self.shm_connected or self.latest_frame is None:
            raise RuntimeError("No live SHM frame is available")

        reply = tcp_command("st")
        info = parse_server_status(reply)
        if info is None:
            raise RuntimeError("Unable to parse the camera status (st) response")

        fh, fw = self.latest_frame.shape
        if (info["width"], info["height"]) != (fw, fh):
            raise RuntimeError(
                f"Camera reports {info['width']} x {info['height']}, "
                f"but SHM is {fw} x {fh}"
            )
        if not info["capture"].upper().startswith("BUSY"):
            raise RuntimeError(
                f"Camera is not actively capturing (status: "
                f"{info['capture'] or 'unknown'})"
            )

        x, y = info["sensor_x"], info["sensor_y"]
        if not (0 <= x and 0 <= y and x + fw <= SENSOR_WIDTH
                and y + fh <= SENSOR_HEIGHT):
            raise RuntimeError(
                f"Camera-reported sensor origin ({x}, {y}) and size "
                f"{fw} x {fh} exceed sensor bounds"
            )

        self._set_origin(x, y, True)
        return info

    def sync_server_status(self):
        """Read camera status and verify the sensor origin against live SHM."""
        try:
            info = self._read_verified_camera_status()
            self.status(
                f"Camera synced: sensor X={info['sensor_x']}, "
                f"Y={info['sensor_y']}, "
                f"W={info['width']}, H={info['height']}"
            )
        except Exception as exc:
            self.origin_confirmed.setChecked(False)
            self.preview_label_refresh()
            QMessageBox.warning(self, "Cannot sync camera", str(exc))

    def send_basic(self, cmd):
        if cmd in ("open", "close", "run"):
            self.suspend_stream()
            self._set_origin(self.origin_x.value(), self.origin_y.value(), False)
        elif cmd == "stop":
            self.img_timer.stop()  # freeze the last copied frame
        succeeded = False
        try:
            response = tcp_command(cmd)
            succeeded = True
            self.status(f"{cmd}: {response}")
        except Exception as exc:
            QMessageBox.critical(self, "Server command error", str(exc))
        finally:
            # Do not reattach to a stale ImageStreamIO image after Close.
            if cmd in ("open", "run"):
                self.begin_reconnect()
            elif cmd == "stop" and not succeeded and self.shm_connected:
                self.img_timer.start(IMAGE_REFRESH_MS)

    def close_stream(self):
        old = self.cam
        self.cam = None
        self.shm_connected = False
        if old is not None and callable(getattr(old, "close", None)):
            try:
                old.close()
            except Exception as exc:
                print(f"Nonfatal SHM close warning: {exc}")

    def suspend_stream(self):
        self.img_timer.stop()
        self.reconnect_timer.stop()
        self.latest_frame = None
        self._display_buffer = None
        self._current_shape = None
        self._last_mouse_position = None
        # Clear the ImageItem's array before releasing the underlying SHM mapping.
        self.image_view.setImage(
            np.zeros((2, 2), dtype=np.uint8),
            autoLevels=True, autoRange=False,
        )
        self._roi_box.hide()
        self.close_stream()

    def force_reconnect(self):
        self.suspend_stream()
        self._set_origin(self.origin_x.value(), self.origin_y.value(), False)
        self.begin_reconnect()

    def begin_reconnect(self, expected_shape=None):
        self._reconnect_expect_shape = expected_shape
        self._reconnect_attempts = 0
        self._reconnect_error = ""
        self.reconnect_timer.start(250)
        self.try_reconnect()

    def try_reconnect(self):
        if self.shm_connected:
            self.reconnect_timer.stop()
            return
        self._reconnect_attempts += 1
        candidate = None
        try:
            candidate = ImageStreamIOWrap.Image()
            candidate.open(STREAM_NAME)
            # Copy is intentional: do not leave an ImageItem pointing at
            # possibly replaced/unmapped shared memory.
            frame = read_camera_frame(candidate)
            if frame.ndim != 2:
                raise ValueError(f"SHM image is {frame.ndim}D, expected 2D")
            if (self._reconnect_expect_shape is not None and
                    frame.shape != self._reconnect_expect_shape):
                raise ValueError(
                    f"SHM is {frame.shape[::-1]} instead of requested "
                    f"{self._reconnect_expect_shape[::-1]}"
                )
        except Exception as exc:
            self._reconnect_error = str(exc)
            if candidate is not None and callable(getattr(candidate, "close", None)):
                try:
                    candidate.close()
                except Exception:
                    pass
            if self._reconnect_attempts >= 24:  # approximately 6 s
                self.reconnect_timer.stop()
                self.status(
                    "SHM reconnect/resize not verified: " + self._reconnect_error,
                    30000,
                )
                self._pending_target = None
                self._reconnect_expect_shape = None
            return

        self.reconnect_timer.stop()
        self.cam = candidate
        self.shm_connected = True
        self.img_timer.start(IMAGE_REFRESH_MS)
        self.update_frame(initial=True)
        if self._pending_target is not None:
            x, y, w, h = self._pending_target
            self._set_origin(x, y, True)
            self.status(
                f"SHM dimensions verified at {w} x {h}. "
                "Origin matches the camera's reported `st` geometry.",
                20000,
            )
            self._pending_target = None
        else:
            self.status(f"SHM connected: {frame.shape[1]} x {frame.shape[0]}")
        self._reconnect_expect_shape = None

    def current_display_mode(self):
        if self.chk_log.isChecked():
            return "log"
        if self.chk_power.isChecked():
            return "power"
        return "linear"

    def log_toggled(self, on):
        if on:
            self.chk_power.blockSignals(True)
            self.chk_power.setChecked(False)
            self.chk_power.blockSignals(False)
        self.refresh_display_mode()

    def power_toggled(self, on):
        if on:
            self.chk_log.blockSignals(True)
            self.chk_log.setChecked(False)
            self.chk_log.blockSignals(False)
        self.refresh_display_mode()

    def refresh_display_mode(self):
        if self.shm_connected:
            self.update_frame(reset_levels=True)

    def change_colormap(self, name):
        self.image_view.setColorMap(pg.colormap.get(name))

    def update_frame(self, initial=False, reset_levels=False):
        if not self.shm_connected or self.cam is None:
            return
        try:
            frame = read_camera_frame(self.cam)
            if frame.ndim != 2:
                raise ValueError(f"Expected a 2D camera frame, got {frame.shape}")
            shape_changed = frame.shape != self._current_shape
            self.latest_frame = frame
            if shape_changed:
                self._current_shape = frame.shape
                self._display_buffer = None
            display, self._display_buffer = transform_display(
                frame, self.current_display_mode(), self._display_buffer
            )
            self.image_view.setImage(
                display,
                autoLevels=(self.chk_auto.isChecked() or initial or
                            reset_levels or shape_changed),
                autoRange=initial or shape_changed,
                autoHistogramRange=initial or reset_levels or shape_changed,
            )
            if shape_changed:
                h, w = frame.shape
                self._set_roi_fields(0, 0, w, h, show=False)
                if not initial:  # external ROI geometry may have changed
                    self._set_origin(self.origin_x.value(), self.origin_y.value(), False)
            if self._last_mouse_position is not None:
                self.update_pixel_label(self._last_mouse_position)
        except Exception as exc:
            self.status(f"Image read error: {exc}", 2000)
            # If the server replaced/unlinked SHM, an old mapping may be stale.
            self.suspend_stream()
            self.begin_reconnect()

    def on_mouse_moved(self, events):
        scene_pos = events[0]
        self._last_mouse_position = scene_pos
        self.update_pixel_label(scene_pos)

    def update_pixel_label(self, scene_pos):
        frame = self.latest_frame
        if frame is None:
            return
        image_item = self.image_view.getImageItem()
        pos = image_item.mapFromScene(scene_pos)
        x, y = int(np.floor(pos.x())), int(np.floor(pos.y()))
        h, w = frame.shape
        if 0 <= x < w and 0 <= y < h:
            value = frame[y, x]
            sensor_x, sensor_y = x + self.origin_x.value(), y + self.origin_y.value()
            qualifier = "" if self.origin_confirmed.isChecked() else " (assumed)"
            self.pixel_label.setText(
                f"Image x={x}, y={y}  |  Sensor x={sensor_x}, y={sensor_y}"
                f"{qualifier}  |  Raw value={value}"
            )
        else:
            self.pixel_label.setText("Outside image")

    def update_tmux_log(self):
        try:
            result = subprocess.run(
                ["tmux", "capture-pane", "-p", "-t", "dcam0"],
                capture_output=True, text=True, timeout=0.5,
            )
            if result.returncode == 0:
                output = result.stdout.strip()
                if self.tmux_log.toPlainText() != output:
                    self.tmux_log.setPlainText(output)
                    scrollbar = self.tmux_log.verticalScrollBar()
                    scrollbar.setValue(scrollbar.maximum())
            else:
                self.tmux_log.setPlainText(
                    "[!] Tmux session 'dcam0' not found or server is offline."
                )
        except Exception as exc:
            self.tmux_log.setPlainText(f"Error querying tmux: {exc}")

    def closeEvent(self, event):
        self.img_timer.stop()
        self.reconnect_timer.stop()
        if self._worker is not None and self._worker.isRunning():
            QMessageBox.warning(
                self, "Camera command in progress",
                "Camera configuration is still in progress. Please close afterward.",
            )
            event.ignore()
            return
        self.suspend_stream()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ServerCameraGUI()
    window.show()
    sys.exit(app.exec_())
