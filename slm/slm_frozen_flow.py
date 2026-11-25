#!/usr/bin/env python3
"""Frozen-flow Kolmogorov turbulence for an 8-bit SLM.

Requires numpy, PyQt5 and your existing slmdisplay module. No turbulence package.
All spatial parameters are in SLM pixels; phase is in radians at --wavelength-nm.

Example:
    python3 slm_frozen_flow.py --r0 150 --speed 25 --angle 30 --duration 60
    python3 slm_frozen_flow.py --r0 150 --speed 25 --duration 0  # Ctrl+C
    python3 slm_frozen_flow.py --dry-run --duration 3 --diagnostics
"""
import argparse
import time
import numpy as np

# ----------------------- User defaults -----------------------
R0_PIXELS = 150.0          # Fried parameter at WAVELENGTH_NM
SPEED_PIXELS_S = 25.0     # frozen screen translation on the SLM
ANGLE_DEG = 0.0           # +x is right, +y is downward on the SLM array
DURATION_S = 30.0         # 0 = continuous (Ctrl+C to stop)
FPS = 20.0                # requested commands per second; actual rate may be lower
WAVELENGTH_NM = 633.0     # wavelength at which r0 and phase are specified
SLM_2PI_NM = 635.0        # nominal wavelength for 0..255 = 2*pi reflected phase
BASELINE = 127            # command corresponding to zero phase
SCREEN_FACTOR = 1.5      # periodic screen size / SLM size (per dimension)
L0_PIXELS = 0.0          # outer scale; 0 = Kolmogorov (box-limited)
SEED = 7
SLM_SCREEN = '1'
# -------------------------------------------------------------


def next_power_of_two(n):
    return 1 << (int(np.ceil(np.log2(n))))


def phase_screen(ny, nx, r0, l0, seed):
    """Real phase screen, rad, from the spatial phase power spectral density.

    Fourier frequency f is in cycles/SLM-pixel. The 2D phase PSD is
       W_phi(f) = 0.023 r0^(-5/3) (f^2 + L0^(-2))^(-11/6).
    For L0=0, use f^(-11/3) and set DC to zero.
    Multiplying FFT(white noise) by sqrt(W * df_x * df_y * Npix)
    gives the correctly normalised phase variance in the inverse FFT.
    """
    rng = np.random.default_rng(seed)
    fy = np.fft.fftfreq(ny).astype(np.float32)[:, None]
    fx = np.fft.rfftfreq(nx).astype(np.float32)[None, :]
    f2 = fx * fx + fy * fy
    if l0 > 0:
        f2 += (1.0 / l0) ** 2
    f2[0, 0] = np.inf                 # zero piston; no singularity
    psd = 0.023 * r0 ** (-5.0 / 3.0) * f2 ** (-11.0 / 6.0)
    white = rng.standard_normal((ny, nx), dtype=np.float32)
    spectrum = np.fft.rfft2(white)
    spectrum *= np.sqrt(psd * (1.0 / nx) * (1.0 / ny) * (nx * ny))
    phase = np.fft.irfft2(spectrum, s=(ny, nx)).astype(np.float32)
    return phase


def shifted_crop(screen, x, y, width, height):
    """Periodically extract a subpixel-shifted crop using bilinear interpolation."""
    ny, nx = screen.shape
    ix, iy = int(np.floor(x)), int(np.floor(y))
    ax, ay = x - ix, y - iy
    xx = (np.arange(width) + ix) % nx
    yy = (np.arange(height) + iy) % ny
    xx1 = (xx + 1) % nx
    yy1 = (yy + 1) % ny
    a = screen[np.ix_(yy, xx)]
    b = screen[np.ix_(yy, xx1)]
    c = screen[np.ix_(yy1, xx)]
    d = screen[np.ix_(yy1, xx1)]
    return ((1-ay)*((1-ax)*a + ax*b) + ay*((1-ax)*c + ax*d))


def phase_to_command(phase, wavelength_nm, slm_2pi_nm, baseline):
    """Convert desired reflected phase (radians) into nominal 8-bit grey values.

    Assumes 255 grey steps correspond to 2*pi at slm_2pi_nm and phase
    scales inversely with illumination wavelength. This is NOT a measured LUT.
    """
    steps = phase * (255.0 / (2.0 * np.pi)) * (wavelength_nm / slm_2pi_nm)
    return (np.rint(steps).astype(np.int32) + baseline).__mod__(256).astype(np.uint8)


def diagnostics(screen, r0, l0):
    print('Separation [px]  measured D_phi [rad^2]  infinite-Kolm. D_phi [rad^2]')
    for separation in (2, 4, 8, 16, 32, 64, 128):
        if separation >= min(screen.shape) // 4:
            break
        delta_x = screen[:, separation:] - screen[:, :-separation]
        delta_y = screen[separation:, :] - screen[:-separation, :]
        measured = 0.5 * (np.mean(delta_x**2) + np.mean(delta_y**2))
        theory = 6.88 * (separation / r0) ** (5.0 / 3.0)
        print(f'{separation:>15}  {measured:>22.4g}  {theory:>28.4g}')
    if l0 > 0:
        print('Theory column assumes infinite L0; finite L0 reduces large-scale power.')
    print('Finite periodic screen and bilinear sampling also modify structure function.')


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--r0', type=float, default=R0_PIXELS, help='Fried r0 [SLM pixels]')
    p.add_argument('--speed', type=float, default=SPEED_PIXELS_S, help='wind speed [SLM pixels/s]')
    p.add_argument('--angle', type=float, default=ANGLE_DEG, help='motion angle [deg], +x right, +y down')
    p.add_argument('--duration', type=float, default=DURATION_S, help='seconds; 0 = continuous')
    p.add_argument('--fps', type=float, default=FPS)
    p.add_argument('--wavelength-nm', type=float, default=WAVELENGTH_NM)
    p.add_argument('--slm-2pi-nm', type=float, default=SLM_2PI_NM)
    p.add_argument('--baseline', type=int, default=BASELINE)
    p.add_argument('--screen-factor', type=float, default=SCREEN_FACTOR)
    p.add_argument('--L0', type=float, default=L0_PIXELS, help='outer scale [SLM pixels]; 0 = infinite')
    p.add_argument('--seed', type=int, default=SEED)
    p.add_argument('--screen', default=SLM_SCREEN, help='SLM display number')
    p.add_argument('--diagnostics', action='store_true', help='print phase structure function check')
    p.add_argument('--dry-run', action='store_true', help='run without SLM; assume 1920 x 1152')
    args = p.parse_args()
    values = [args.r0, args.speed, args.angle, args.duration, args.fps,
              args.wavelength_nm, args.slm_2pi_nm, args.screen_factor, args.L0]
    if not all(np.isfinite(values)) or args.r0 <= 0 or args.speed < 0 or args.duration < 0 \
            or args.fps <= 0 or args.wavelength_nm <= 0 or args.slm_2pi_nm <= 0 \
            or args.screen_factor < 1 or args.L0 < 0 or not 0 <= args.baseline <= 255:
        p.error('Invalid parameters (r0/fps/wavelengths > 0, factor >= 1, remaining nonnegative).')

    app = slm = None
    entered = False
    if args.dry_run:
        width, height = 1920, 1152
    else:
        from PyQt5 import QtWidgets
        from slmdisplay import SLM
        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        display = int(args.screen) if args.screen.lstrip('-').isdigit() else args.screen
        slm = SLM(screen=display)
        slm.__enter__()
        entered = True
        width, height = slm.width, slm.height

    flat = np.full((height, width), args.baseline, dtype=np.uint8)
    try:
        nx = next_power_of_two(np.ceil(width * args.screen_factor))
        ny = next_power_of_two(np.ceil(height * args.screen_factor))
        print(f'SLM: {width} x {height}; phase screen: {nx} x {ny} (periodic)')
        print(f'r0={args.r0:g} px, wind={args.speed:g} px/s at {args.angle:g} deg, '
              f'FPS target={args.fps:g}, duration={args.duration:g} s (0 = continuous)')
        screen = phase_screen(ny, nx, args.r0, args.L0, args.seed)
        if args.diagnostics:
            diagnostics(screen, args.r0, args.L0)
        theta = np.deg2rad(args.angle)
        vx, vy = args.speed * np.cos(theta), args.speed * np.sin(theta)
        if args.speed > 0:
            repeat_time = min(nx / abs(vx) if abs(vx) > 1e-10 else np.inf,
                              ny / abs(vy) if abs(vy) > 1e-10 else np.inf)
            print(f'Periodic screen coordinate wrap may occur after ~{repeat_time:.1f} s; '
                  'the field repeats exactly only after both axes realign.')
        print('Sending turbulence; press Ctrl+C to stop. SLM will return to baseline.')
        start = time.monotonic()
        frame = 0
        while True:
            elapsed = time.monotonic() - start
            if args.duration > 0 and elapsed >= args.duration and frame:
                break
            # Work from elapsed wall time so missed frame deadlines do not slow the wind.
            phase = shifted_crop(screen, vx * elapsed, vy * elapsed, width, height)
            command = phase_to_command(phase, args.wavelength_nm,
                                       args.slm_2pi_nm, args.baseline)
            if slm is not None:
                slm.show(command)
                app.processEvents()
            frame += 1
            if frame % max(1, int(args.fps * 5)) == 0:
                print(f'  t={elapsed:7.1f} s  updates={frame}  '
                      f'command range={int(command.min())}..{int(command.max())}', flush=True)
            deadline = start + frame / args.fps
            delay = deadline - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            else:
                # Skip missed deadlines instead of accumulating lag.
                frame = max(frame, int((time.monotonic() - start) * args.fps))
        print(f'Finished: {frame} nominal frame slots, {time.monotonic()-start:.2f} s.')
    except KeyboardInterrupt:
        print('\nStopped by user.')
    finally:
        if entered:
            try:
                slm.show(flat)
                app.processEvents()
            finally:
                try:
                    slm.__exit__(None, None, None)
                except FileNotFoundError as exc:
                    if not (exc.filename and str(exc.filename).startswith('/psm_')):
                        raise
            print('SLM restored to flat baseline.')


if __name__ == '__main__':
    main()
