"""Continuously sweep about the current position using the ZeroMQ service.

Angles are MOTOR SHAFT degrees, not reflected-beam angles. Assumes the
established 200-full-step/revolution driver setting. No hardware changes.
"""
import argparse
import math
import signal
import sys
import time

import zmq
from zmq_client import MirrorClient

DEGREES_PER_STEP = 1.8


def endpoints(center, amplitude):
    if not math.isfinite(amplitude) or amplitude <= 0:
        raise ValueError("Amplitude must be finite and positive")
    steps = math.floor(amplitude / DEGREES_PER_STEP + 0.5)
    if not 1 <= steps <= 1000000:
        raise ValueError("Amplitude must round to 1–1000000 full steps")
    return center - steps, center + steps, steps * DEGREES_PER_STEP


def scan(motor, amplitude=3.6, speed=10, dwell=0.5, cycles=0, zero=False):
    """Run until interrupted (cycles=0), or for a finite number of cycles."""
    if type(speed) is not int or not 10 <= speed <= 200:
        raise ValueError("Speed must be an integer from 10 to 200 steps/s")
    if not math.isfinite(dwell) or dwell < 0:
        raise ValueError("Dwell must be finite and nonnegative")
    if type(cycles) is not int or cycles < 0:
        raise ValueError("Cycles must be a nonnegative integer")
    endpoints(0, amplitude)  # Validate before making any changes.
    state = motor.status()
    if state["moving"]:
        raise RuntimeError("Motor is already moving; this example will not take over")
    if zero:
        motor.zero()
        state = motor.status()
    if not state["referenced"]:
        raise RuntimeError("Set a reference first, or use --zero to declare the current physical position zero")
    center = state["position"]
    low, high, actual = endpoints(center, amplitude)
    if low < state["lower"] or high > state["upper"]:
        raise ValueError(f"Scan {low}..{high} exceeds configured limits {state['lower']}..{state['upper']}")
    print(f"Centre: {center} steps. Requested amplitude: ±{amplitude:g}°; actual: ±{actual:g}°.", flush=True)
    print(f"Targets: {low} and {high} steps; speed {speed} steps/s; endpoint pause {dwell:g} s.", flush=True)
    print("Motor power must be ON. Ctrl+C stops pulses; no automatic return to centre.", flush=True)
    motor.set_speed(speed)
    attempted_move = False
    try:
        cycle = 0
        while cycles == 0 or cycle < cycles:
            for target in (high, low):
                # Mark before sending: a timeout could still mean the move was accepted.
                attempted_move = True
                job = motor.goto(target)
                motor.wait(job, timeout=(high - low) / 10 + 10)
                print(f"Reached commanded position {target} steps", flush=True)
                time.sleep(dwell)
            cycle += 1
    finally:
        if attempted_move:
            try:
                motor.stop()
                print("STOP acknowledged. Motor may remain energised.", flush=True)
            except Exception as error:
                print(f"STOP could not be confirmed: {error}. Switch off 12 V if needed.", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default="tcp://127.0.0.1:5555")
    parser.add_argument("--amplitude", type=float, default=3.6, help="Degrees EACH SIDE of current position (default 3.6)")
    parser.add_argument("--speed", type=int, default=10, help="Steps/second, 10–200 (default 10)")
    parser.add_argument("--dwell", type=float, default=0.5, help="Seconds paused at each end (default 0.5)")
    parser.add_argument("--cycles", type=int, default=0, help="Full back-and-forth cycles; 0 runs continuously")
    parser.add_argument("--zero", action="store_true", help="Explicitly declare current position zero before scanning")
    args = parser.parse_args()

    def interrupted(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupted)
    try:
        with MirrorClient(args.endpoint) as motor:
            scan(motor, args.amplitude, args.speed, args.dwell, args.cycles, args.zero)
    except KeyboardInterrupt:
        print("Scan ended.")
    except (RuntimeError, ValueError, TimeoutError, zmq.ZMQError) as error:
        print(f"Scan ended: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
