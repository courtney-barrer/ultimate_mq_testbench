#!/usr/bin/env python3
"""Acquire SLM modal probes and flat references from a MILK DCAM stream.

Examples:
  python3 build_IM_v2.py --basis fourier --fmin 15 --fmax 50 \
      --nmodes 20 --amp-min -40 --amp-max 40 --namp 5 \
      --nframes 10 --settle 0.3 --output fourier_highfreq.fits

  python3 build_IM_v2.py --basis zonal_128 --nmodes 10 \
      --amp-min -40 --amp-max 40 --namp 3 \
      --nframes 5 --settle 0.3 --output zonal_test.fits

One FLAT_CMD extension stores the uniform 8-bit SLM command. At the start
of EACH mode, a FLAT_FRAMES extension contains N camera frames for that
flat command. SLM_CMD and CAM_FRAMES extensions hold all normal probes.
"""

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from astropy.io import fits
from PyQt5 import QtWidgets

import ImageStreamIOWrap
from slmdisplay import SLM


def utc_now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]


def make_modes(basis, nmodes, shape, fmin=0.0, fmax=None):
    """Return ordered mode parameter dicts; Fourier frequencies are cycles/SLM."""
    h, w = shape
    modes = []

    if basis.lower() == "fourier":
        if not np.isfinite(fmin) or fmin < 0:
            raise ValueError("--fmin must be finite and >= 0")
        if fmax is not None and (
            not np.isfinite(fmax) or fmax < fmin or fmax <= 0
        ):
            raise ValueError("--fmax must be finite, positive, and >= --fmin")

        # Generate unique real Fourier wavevectors in the upper half-plane.
        # Both cosine (phase=0) and sine (phase=90 deg) are included.
        kmax = max(2, int(np.ceil(fmin)), int(np.ceil(np.sqrt(nmodes))))
        if fmax is not None:
            kmax = max(kmax, int(np.ceil(fmax)))

        while True:
            freqs = [
                (fx, fy)
                for fy in range(kmax + 1)
                for fx in range(-kmax, kmax + 1)
                if (fy > 0 or fx > 0)
                and fmin <= np.hypot(fx, fy)
                and (fmax is None or np.hypot(fx, fy) <= fmax)
            ]
            freqs.sort(key=lambda p: (p[0]**2 + p[1]**2, p[1], p[0]))

            # When fmax is omitted, make sure no excluded wavevector
            # could have a *lower* radial frequency than the selected ones.
            enough = 2 * len(freqs) >= nmodes
            if fmax is not None:
                break
            if enough:
                cutoff_radius = np.hypot(*freqs[(nmodes - 1) // 2])
                if cutoff_radius <= kmax:
                    break
            kmax *= 2

        if fmax is not None:
            # Spread requested modes across the specified radial band,
            # rather than taking only modes immediately above fmin.
            n_wavevectors = (nmodes + 1) // 2
            radii = np.array([np.hypot(fx, fy) for fx, fy in freqs])
            selected = []
            used = set()
            for target in np.linspace(fmin, fmax, n_wavevectors):
                for idx in np.argsort(np.abs(radii - target), kind="stable"):
                    if int(idx) not in used:
                        used.add(int(idx))
                        selected.append(freqs[int(idx)])
                        break
            freqs = selected

        for fx, fy in freqs:
            for phase in (0, 90):
                modes.append(dict(fx=fx, fy=fy, phase=phase))

    elif basis.lower().startswith("zonal_"):
        b = int(basis.split("_", 1)[1])
        if b < 1 or b > min(h, w):
            raise ValueError("Invalid zonal binning factor")

        nx, ny = w // b, h // b
        ox, oy = (w - nx * b) // 2, (h - ny * b) // 2
        cells = [(ix, iy) for iy in range(ny) for ix in range(nx)]
        cells.sort(key=lambda p: (
            (p[0] - (nx - 1)/2)**2 + (p[1] - (ny - 1)/2)**2,
            p[1], p[0]
        ))
        for ix, iy in cells:
            modes.append(dict(x0=ox + ix*b, y0=oy + iy*b, binning=b))
    else:
        raise ValueError("Basis must be 'fourier' or 'zonal_<pixels>'")

    if nmodes > len(modes):
        raise ValueError(
            f"Requested {nmodes} modes, but {len(modes)} are available "
            "in the specified frequency range/grid"
        )
    return modes[:nmodes]


def mode_map(shape, mode):
    """Return a dimensionless SLM mode with absolute amplitude <= 1."""
    h, w = shape
    if "binning" in mode:
        array = np.zeros(shape, dtype=np.float32)
        x0, y0, b = mode["x0"], mode["y0"], mode["binning"]
        array[y0:y0+b, x0:x0+b] = 1.0
        return array

    x = (np.arange(w, dtype=np.float32) + 0.5) / w
    y = (np.arange(h, dtype=np.float32) + 0.5) / h
    angle = (
        2 * np.pi * (mode["fx"] * x[None, :] + mode["fy"] * y[:, None])
        + np.deg2rad(mode["phase"])
    )
    return np.cos(angle).astype(np.float32)


def next_frame(cam, previous_cnt, timeout):
    """Poll MILK until a distinct, stable frame is available."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        count = int(cam.md.cnt0)
        if count != previous_cnt and not cam.md.write:
            frame = np.array(cam, copy=True)
            if int(cam.md.cnt0) == count and not cam.md.write:
                return count, frame
        time.sleep(0.001)
    raise TimeoutError("No new stable DCAM frame received")


def acquire_frames(cam, nframes, timeout):
    """Discard one new post-settling frame, then capture nframes distinct frames."""
    count = int(cam.md.cnt0)
    count, _ = next_frame(cam, count, timeout)

    cube = None
    first_count = None
    missed = 0
    for frame_index in range(nframes):
        previous_count = count
        count, frame = next_frame(cam, count, timeout)
        if cube is None:
            cube = np.empty((nframes, *frame.shape), dtype=frame.dtype)
            first_count = count
        else:
            if frame.shape != cube.shape[1:] or frame.dtype != cube.dtype:
                raise RuntimeError("Camera frame format changed during acquisition")
        cube[frame_index] = frame
        missed += max(0, count - previous_count - 1)

    return cube, first_count, count, missed


def acquire_after_slm(slm, app, command, settle, cam, nframes, timeout):
    slm.show(command)
    app.processEvents()  # Flush the Qt display update.
    time.sleep(settle)
    app.processEvents()
    return acquire_frames(cam, nframes, timeout)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--basis", required=True,
                        help="'fourier' or 'zonal_<pixel binning>'")
    parser.add_argument("--nmodes", type=int, required=True)
    parser.add_argument("--amp-min", type=float, required=True)
    parser.add_argument("--amp-max", type=float, required=True)
    parser.add_argument("--namp", type=int, required=True)
    parser.add_argument("--nframes", type=int, default=10)
    parser.add_argument("--settle", type=float, default=0.2,
                        help="SLM settling delay [s]")
    parser.add_argument("--output", required=True)
    parser.add_argument("--fmin", type=float, default=0.0,
                        help="Minimum radial Fourier frequency [cycles/full SLM]")
    parser.add_argument("--fmax", type=float, default=None,
                        help="Maximum radial Fourier frequency [cycles/full SLM]")
    parser.add_argument("--baseline", type=int, default=127)
    parser.add_argument("--screen", default="1")
    parser.add_argument("--shm", default="dcam0_data")
    parser.add_argument("--timeout", type=float, default=5.0)
    args = parser.parse_args()

    if min(args.nmodes, args.namp, args.nframes) < 1:
        parser.error("nmodes, namp and nframes must be positive")
    if not np.isfinite(args.settle) or args.settle < 0:
        parser.error("settle must be finite and nonnegative")
    if not np.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("timeout must be finite and positive")
    if not np.isfinite(args.amp_min) or not np.isfinite(args.amp_max):
        parser.error("amplitude bounds must be finite")
    if args.amp_max < args.amp_min:
        parser.error("amp-max must be >= amp-min")
    if not 0 <= args.baseline <= 255:
        parser.error("baseline must be between 0 and 255")
    if Path(args.output).exists():
        parser.error(f"Output already exists: {args.output}")

    amps = np.linspace(args.amp_min, args.amp_max, args.namp)
    cam = ImageStreamIOWrap.Image()
    cam.open(args.shm)  # DCAM server must already be running.

    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(sys.argv[:1])
    screen = (int(args.screen)
              if args.screen.lstrip("-").isdigit() else args.screen)

    slm = SLM(screen=screen)
    slm.__enter__()
    shape = (slm.height, slm.width)
    flat = np.full(shape, args.baseline, dtype=np.uint8)

    try:
        modes = make_modes(args.basis, args.nmodes, shape,
                           fmin=args.fmin, fmax=args.fmax)

        header = fits.Header()
        header["DATE"] = utc_now()
        header["BASIS"] = args.basis
        header["NMODE"] = len(modes)
        header["NAMP"] = args.namp
        header["NFRAME"] = args.nframes
        header["NFLAT"] = 1  # One dedicated flat acquisition per mode.
        header["AMPMIN"] = args.amp_min
        header["AMPMAX"] = args.amp_max
        header["SETTLE"] = args.settle
        header["BASE"] = args.baseline
        header["STREAM"] = args.shm
        header["SLMW"] = shape[1]
        header["SLMH"] = shape[0]
        header["WVLNM"] = 633.0
        header["PIXPUM"] = 6.5
        if args.basis.lower() == "fourier":
            header["FMIN"] = args.fmin
            if args.fmax is not None:
                header["FMAX"] = args.fmax

        fits.PrimaryHDU(header=header).writeto(args.output)

        # The same uniform SLM image is used for each dedicated flat.
        flat_hdr = fits.Header()
        flat_hdr["EXTNAME"] = "FLAT_CMD"
        flat_hdr["EXTVER"] = 1
        flat_hdr["BASE"] = args.baseline
        fits.append(args.output, flat, header=flat_hdr)

        total = len(modes) * len(amps)
        step = 0
        for mi, mode in enumerate(modes):
            # Explicit reference for this mode, regardless of amplitude grid.
            flat_date = utc_now()
            cube, first, last, missed = acquire_after_slm(
                slm, app, flat, args.settle, cam, args.nframes, args.timeout
            )
            fh = fits.Header()
            fh["EXTNAME"] = "FLAT_FRAMES"
            fh["EXTVER"] = mi + 1
            fh["MODEID"] = mi
            fh["DATE-OBS"] = flat_date
            fh["BASE"] = args.baseline
            fh["CNTFIRST"] = first
            fh["CNTLAST"] = last
            fh["DROPPED"] = missed
            fits.append(args.output, cube, header=fh)
            print(f"[flat mode {mi}] frames={args.nframes} missed={missed}",
                  flush=True)

            basis_image = mode_map(shape, mode)
            for ai, amp in enumerate(amps):
                values = args.baseline + amp * basis_image
                if values.min() < 0 or values.max() > 255:
                    raise ValueError(
                        f"SLM command would clip at mode {mi}, amplitude {amp:g}"
                    )
                command = np.rint(values).astype(np.uint8)

                probe_date = utc_now()
                cube, first, last, missed = acquire_after_slm(
                    slm, app, command, args.settle, cam,
                    args.nframes, args.timeout
                )

                common = fits.Header()
                common["CMDID"] = step
                common["MODEID"] = mi
                common["AMPIDX"] = ai
                common["AMPL"] = float(amp)
                for key, value in mode.items():
                    common[key.upper()[:8]] = value

                ch = common.copy()
                ch["EXTNAME"] = "SLM_CMD"
                ch["EXTVER"] = step + 1
                fits.append(args.output, command, header=ch)

                ch = common.copy()
                ch["EXTNAME"] = "CAM_FRAMES"
                ch["EXTVER"] = step + 1
                ch["DATE-OBS"] = probe_date
                ch["CNTFIRST"] = first
                ch["CNTLAST"] = last
                ch["DROPPED"] = missed
                fits.append(args.output, cube, header=ch)

                step += 1
                print(f"[{step}/{total}] mode={mi} amp={amp:.2f} "
                      f"frames={args.nframes} missed={missed}", flush=True)

        print(f"Saved: {args.output}")
    finally:
        try:
            slm.show(flat)
            app.processEvents()
        finally:
            try:
                slm.__exit__(None, None, None)
            except FileNotFoundError as exc:
                # slmdisplay may double-unlink its multiprocessing SHM.
                # This avoids a traceback AFTER FITS was saved; the underlying
                # resource_tracker warning should be fixed in slmdisplay.
                print(f"SLM cleanup warning: {exc}", file=sys.stderr)
            if hasattr(cam, "close"):
                cam.close()


if __name__ == "__main__":
    main()
