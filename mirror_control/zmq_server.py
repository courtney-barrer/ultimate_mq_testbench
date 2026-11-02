"""Single USB owner, local JSON request/reply API. No automatic movement."""
import argparse
from collections import OrderedDict
import json
import signal
import time
import uuid

import serial
import zmq


class Bridge:
    def __init__(self, port):
        self.port = port
        self.state = None
        self.active = None
        self.jobs = OrderedDict()

    def event(self, line):
        if line.startswith("READY"):
            raise RuntimeError("Arduino reset; restart service and establish zero again")
        if line.startswith("STATE "):
            values = list(map(int, line.split()[1:]))
            if len(values) != 7:
                raise RuntimeError("Invalid Arduino state")
            self.state = dict(zip(("position", "target", "moving", "referenced", "lower", "upper", "speed"), values))
            self.state["moving"] = bool(self.state["moving"])
            self.state["referenced"] = bool(self.state["referenced"])
        if self.active and (line == "DONE" or line.startswith("STOPPED")):
            self.jobs[self.active]["status"] = "done" if line == "DONE" else "stopped"
            self.jobs[self.active]["reason"] = line
            self.active = None

    def transact(self, command):
        # Consume already-buffered completion events before issuing a new command.
        while self.port.in_waiting:
            self.event(self.read_line())
        self.port.write((command + "\n").encode("ascii"))
        acknowledged = command == "STATUS"
        deadline = time.monotonic() + 0.8
        while time.monotonic() < deadline:
            line = self.read_line()
            if not line:
                continue
            self.event(line)
            if line.startswith("ERR "):
                return line[4:]
            if line == "OK" or line == "OK ZERO" or line == "STOPPED":
                acknowledged = True
            if acknowledged and line.startswith("STATE "):
                return None
        raise RuntimeError("Arduino response timeout; command outcome uncertain; do not retry movement blindly")

    def read_line(self):
        # Firmware writes short complete lines; accumulate across serial timeouts.
        if not hasattr(self, "partial"):
            self.partial = bytearray()
        chunk = self.port.read_until(b"\n", 256)
        self.partial.extend(chunk)
        if len(self.partial) > 512:
            raise RuntimeError("Oversized serial response")
        if not self.partial.endswith(b"\n"):
            return ""
        result = self.partial.decode("ascii").strip()
        self.partial.clear()
        return result

    def handle(self, request):
        if not isinstance(request, dict):
            return {"ok": False, "error": "Request must be a JSON object"}
        op = request.get("op")
        fields = {"status": (), "zero": (), "stop": (), "move": ("steps",),
                  "goto": ("position",), "speed": ("value",), "limits": ("lower", "upper"),
                  "job": ("id",)}
        if not isinstance(op, str) or op not in fields:
            return {"ok": False, "error": "Unknown operation"}
        if set(request) != {"op", *fields[op]}:
            return {"ok": False, "error": "Missing or unexpected fields"}
        if op == "job":
            if not isinstance(request["id"], str) or request["id"] not in self.jobs:
                return {"ok": False, "error": "Unknown/expired job; service may have restarted"}
            self.transact("STATUS")
            return {"ok": True, "job": dict(self.jobs[request["id"]]), "state": self.state}
        args = [request[name] for name in fields[op]]
        if any(type(n) is not int or abs(n) > 1000000 for n in args):
            return {"ok": False, "error": "Arguments must be integers within +/-1000000"}
        self.transact("STATUS")
        if op in ("move", "goto", "zero", "speed", "limits") and self.state["moving"]:
            return {"ok": False, "error": "BUSY", "state": self.state}
        command = {"limits": "LIMIT"}.get(op, op.upper())
        job_id = None
        if op in ("move", "goto"):
            job_id = uuid.uuid4().hex
            self.active = job_id
            self.jobs[job_id] = {"id": job_id, "status": "running"}
        error = self.transact(" ".join([command, *map(str, args)]))
        if error:
            if job_id:
                self.jobs.pop(job_id)
                self.active = None
            return {"ok": False, "error": error, "state": self.state}
        while len(self.jobs) > 128:
            self.jobs.popitem(last=False)
        result = {"ok": True, "state": self.state}
        if job_id:
            result["job_id"] = job_id
        return result


def serve(bridge, endpoint, stopping, ready=None):
    context = zmq.Context()
    sock = context.socket(zmq.REP)
    sock.setsockopt(zmq.LINGER, 0)
    sock.setsockopt(zmq.MAXMSGSIZE, 4096)
    try:
        sock.bind(endpoint)
        if ready:
            ready.set()
        print(f"Listening on {endpoint}; no movement until commanded", flush=True)
        last_poll = 0
        while not stopping():
            if sock.poll(50):
                raw = sock.recv()
                try:
                    request = json.loads(raw)
                except (ValueError, UnicodeError):
                    sock.send_json({"ok": False, "error": "Invalid JSON"})
                    continue
                try:
                    response = bridge.handle(request)
                except (RuntimeError, serial.SerialException, OSError, ValueError) as error:
                    sock.send_json({"ok": False, "error": str(error)})
                    raise
                sock.send_json(response)
            if time.monotonic() - last_poll >= 0.25:
                bridge.transact("STATUS")
                last_poll = time.monotonic()
    finally:
        try:
            bridge.port.write(b"STOP\n")
        except (serial.SerialException, OSError):
            pass
        sock.close()
        context.term()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", required=True, help="UNO port, e.g. /dev/ttyACM0")
    parser.add_argument("--endpoint", default="tcp://127.0.0.1:5555")
    args = parser.parse_args()
    if not args.endpoint.startswith("tcp://127.0.0.1:"):
        parser.error("This version supports local-computer clients only (tcp://127.0.0.1:PORT)")
    stopped = False

    def stop(signum, frame):
        nonlocal stopped
        stopped = True

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    with serial.Serial(args.port, 115200, timeout=0.05, write_timeout=0.2, exclusive=True) as port:
        time.sleep(2.5)  # UNO bootloader/reset on opening USB.
        port.reset_input_buffer()
        bridge = Bridge(port)
        bridge.transact("STOP")
        serve(bridge, args.endpoint, lambda: stopped)


if __name__ == "__main__":
    main()
