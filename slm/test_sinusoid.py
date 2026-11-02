"""Show a sinusoidal grating with a given period (in pixels) on the SLM.

While it runs, type a new period (and optionally an angle) to update the grating,
or press Enter on an empty line to quit.
"""

import argparse

import numpy as np

from slmdisplay import SLM


def sinusoid(shape, period_px, angle_deg=0.0):
    """uint8 grating 127.5 * (1 + sin(2 pi r / period)), r along angle_deg."""
    height, width = shape
    y, x = np.mgrid[0:height, 0:width]
    theta = np.deg2rad(angle_deg)
    r = x * np.cos(theta) + y * np.sin(theta)
    return np.round(127.5 * (1 + np.sin(2 * np.pi * r / period_px))).astype(np.uint8)


def positive_float(text):
    value = float(text)
    if value <= 0:
        raise argparse.ArgumentTypeError(f"must be positive, got {text}")
    return value


def screen_arg(text):
    """OS monitor index (as in xrandr --listmonitors) or output name such as HDMI-A-0."""
    return int(text) if text.lstrip("-").isdigit() else text


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "period", nargs="?", type=positive_float, default=50.0,
        help="grating period in pixels (default: 50)",
    )
    parser.add_argument(
        "angle", nargs="?", type=float, default=0.0,
        help="direction of the grating vector in degrees: 0 gives vertical fringes "
             "(varying along x), 90 horizontal fringes (default: 0)",
    )
    parser.add_argument(
        "--screen", type=screen_arg, default=1,
        help="SLM monitor index as listed by `xrandr --listmonitors` (0 = primary), "
             "or output name, e.g. HDMI-A-0 (default: 1)",
    )
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    period, angle = args.period, args.angle

    with SLM(screen=args.screen) as slm:
        while True:
            slm.show(sinusoid(slm.shape, period, angle))
            print(f"Showing period {period:g} px, angle {angle:g} deg on {slm.width}x{slm.height}.")
            line = input("New 'PERIOD [ANGLE]', or Enter to quit: ").split()
            if not line:
                break
            try:
                new = parser.parse_args(line, namespace=argparse.Namespace(screen=args.screen))
            except SystemExit:
                continue  # argparse already printed the error
            period, angle = new.period, new.angle


if __name__ == "__main__":
    main()
