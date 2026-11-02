#!/usr/bin/env python3
"""PyQtGraph SLM Fourier-mode and virtual-actuator poke controller.

Run: python slm_pyqtgraph_gui.py --screen 1
     python slm_pyqtgraph_gui.py --simulate --width 1920 --height 1200
"""
import argparse
import sys
import numpy as np
from PyQt5 import QtCore, QtWidgets
import pyqtgraph as pg
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

def make_pattern(shape, mode, baseline, amplitude, fx=0, fy=0, phase=0,
                 nx=16, ny=16, ix=0, iy=0):
    h, w = shape
    if mode == 'Flat':
        return np.full(shape, baseline, dtype=np.uint8)
    if mode == 'Fourier':
        # Pixel centres: frequencies in cycles across the full active display.
        x = (np.arange(w, dtype=np.float32) + 0.5) / w
        y = (np.arange(h, dtype=np.float32) + 0.5) / h
        angle = (2 * np.pi * (fx * x[None, :] + fy * y[:, None])
                 + np.deg2rad(phase))
        values = baseline + amplitude * np.cos(angle)
    else:
        values = np.full(shape, float(baseline), dtype=np.float32)
        x0, x1 = int(ix * w / nx), int((ix + 1) * w / nx)
        y0, y1 = int(iy * h / ny), int((iy + 1) * h / ny)
        values[y0:y1, x0:x1] += amplitude
    return np.clip(np.rint(values), 0, 255).astype(np.uint8)


class Controller(QtWidgets.QMainWindow):
    def __init__(self, slm, shape):
        super().__init__()
        self.slm, self.shape = slm, shape
        self.setWindowTitle('SLM Fourier / Actuator Poke Controller')
        self.resize(1150, 740)
        root = QtWidgets.QWidget()
        self.setCentralWidget(root)
        layout = QtWidgets.QHBoxLayout(root)
        panel = QtWidgets.QWidget()
        panel.setMaximumWidth(350)
        form = QtWidgets.QFormLayout(panel)
        layout.addWidget(panel)
        self.preview = pg.ImageView()
        self.preview.ui.roiBtn.hide()
        self.preview.ui.menuBtn.hide()
        layout.addWidget(self.preview, 1)
        self.mode = QtWidgets.QComboBox()
        self.mode.addItems(['Fourier', 'Poke', 'Flat'])
        form.addRow('Pattern', self.mode)
        self.base = self.spin(0, 255, 127, 1)
        self.amp = self.spin(-255, 255, 50, 1)
        form.addRow('Baseline [8-bit]', self.base)
        form.addRow('Amplitude [8-bit]', self.amp)
        self.fx = self.spin(-500, 500, 4, 0.25)
        self.fy = self.spin(-500, 500, 0, 0.25)
        self.phase = self.spin(-360, 360, 0, 5)
        form.addRow('fx [cycles / width]', self.fx)
        form.addRow('fy [cycles / height]', self.fy)
        form.addRow('Phase [degrees]', self.phase)
        self.nx = self.ispin(1, shape[1], 16)
        self.ny = self.ispin(1, shape[0], 16)
        self.ix = self.ispin(0, 15, 0)
        self.iy = self.ispin(0, 15, 0)
        form.addRow('Grid Nx', self.nx)
        form.addRow('Grid Ny', self.ny)
        form.addRow('Poke x index (0-based)', self.ix)
        form.addRow('Poke y index (0-based)', self.iy)
        self.auto = QtWidgets.QCheckBox('Auto apply to SLM')
        self.auto.setChecked(True)
        form.addRow(self.auto)
        apply_btn = QtWidgets.QPushButton('Apply pattern')
        flat_btn = QtWidgets.QPushButton('Flat (zero)')
        form.addRow(apply_btn)
        form.addRow(flat_btn)
        self.status = QtWidgets.QLabel('')
        self.status.setWordWrap(True)
        form.addRow(self.status)
        self.timer = QtCore.QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.setInterval(80)
        self.timer.timeout.connect(self.update_pattern)
        self.nx.valueChanged.connect(self.update_index_limits)
        self.ny.valueChanged.connect(self.update_index_limits)
        for widget in [self.mode, self.base, self.amp, self.fx, self.fy,
                       self.phase, self.nx, self.ny, self.ix, self.iy]:
            if isinstance(widget, QtWidgets.QComboBox):
                widget.currentIndexChanged.connect(self.schedule_update)
            else:
                widget.valueChanged.connect(self.schedule_update)
        self.auto.toggled.connect(self.schedule_update)
        apply_btn.clicked.connect(lambda: self.update_pattern(force=True))
        flat_btn.clicked.connect(self.zero)
        self.update_pattern(force=True)

    @staticmethod
    def spin(low, high, value, step):
        w = QtWidgets.QDoubleSpinBox()
        w.setRange(low, high)
        w.setDecimals(2)
        w.setSingleStep(step)
        w.setValue(value)
        return w

    @staticmethod
    def ispin(low, high, value):
        w = QtWidgets.QSpinBox()
        w.setRange(low, high)
        w.setValue(min(value, high))
        return w

    def update_index_limits(self):
        self.ix.setMaximum(self.nx.value() - 1)
        self.iy.setMaximum(self.ny.value() - 1)

    def schedule_update(self, *args):
        self.timer.start()

    def update_pattern(self, force=False):
        arr = make_pattern(self.shape, self.mode.currentText(),
                           self.base.value(), self.amp.value(),
                           self.fx.value(), self.fy.value(), self.phase.value(),
                           self.nx.value(), self.ny.value(),
                           self.ix.value(), self.iy.value())
        # PyQtGraph axisOrder='row-major' means numpy (y,x) is displayed correctly.
        self.preview.setImage(arr, autoLevels=False, levels=(0, 255), autoRange=False)
        if force or self.auto.isChecked():
            try:
                if self.slm is not None:
                    self.slm.show(arr)
                self.status.setText(f'{self.shape[1]} x {self.shape[0]} | applied'
                                    + (' (simulation)' if self.slm is None else ''))
            except Exception as exc:
                self.status.setText(f'SLM output failed: {exc}')
        else:
            self.status.setText('Preview only; press Apply to send')

    def zero(self):
        self.mode.setCurrentText('Flat')
        self.base.setValue(0)
        self.update_pattern(force=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--screen', default='1', help='Monitor index or output name')
    parser.add_argument('--simulate', action='store_true')
    parser.add_argument('--width', type=int, default=1920)
    parser.add_argument('--height', type=int, default=1200)
    args = parser.parse_args()
    pg.setConfigOptions(imageAxisOrder='row-major')
    app = QtWidgets.QApplication(sys.argv[:1])
    slm = None
    try:
        if not args.simulate:
            from slmdisplay import SLM
            screen = int(args.screen) if args.screen.lstrip('-').isdigit() else args.screen
            slm = SLM(screen=screen)
            slm.__enter__()
            shape = (slm.height, slm.width)
        else:
            shape = (args.height, args.width)
        window = Controller(slm, shape)
        window.show()
        return app.exec_()
    finally:
        if slm is not None:
            slm.__exit__(None, None, None)


if __name__ == '__main__':
    sys.exit(main())