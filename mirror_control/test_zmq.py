"""Integration tests using real ZMQ and a simulated serial Arduino, no motor."""
import os
import pty
import select
import threading
import time
import unittest
import uuid

import serial
import zmq
from zmq_server import Bridge, serve
from zmq_client import MirrorClient


class FakeArduino:
    def __init__(self, fd):
        self.fd = fd
        self.pos = self.target = 0
        self.reference = False
        self.moving = False
        self.quit = False
        self.finish_at = 0
        self.commands = []

    def emit(self, line):
        os.write(self.fd, (line + "\n").encode())

    def state(self):
        self.emit(f"STATE {self.pos} {self.target} {int(self.moving)} {int(self.reference)} -100 100 50")

    def run(self):
        buffer = b""
        while not self.quit:
            if self.moving and time.monotonic() >= self.finish_at:
                self.pos = self.target
                self.moving = False
                self.emit("DONE")
                self.state()
            if not select.select([self.fd], [], [], 0.01)[0]:
                continue
            buffer += os.read(self.fd, 4096)
            while b"\n" in buffer:
                raw, buffer = buffer.split(b"\n", 1)
                command = raw.decode().split()
                self.commands.append(command)
                op = command[0]
                if op == "STATUS":
                    self.state()
                    continue
                if op == "STOP":
                    self.moving = False
                    self.target = self.pos
                    self.emit("STOPPED")
                elif op == "ZERO":
                    self.pos = self.target = 0
                    self.reference = True
                    self.emit("OK ZERO")
                elif op in ("MOVE", "GOTO"):
                    if not self.reference:
                        self.emit("ERR ZERO_REQUIRED")
                        continue
                    target = int(command[1]) + (self.pos if op == "MOVE" else 0)
                    if not -100 <= target <= 100:
                        self.emit("ERR OUT_OF_RANGE")
                        continue
                    self.target = target
                    self.moving = self.target != self.pos
                    self.finish_at = time.monotonic() + 0.5
                    if not self.moving:
                        self.emit("DONE")
                    self.emit("OK")
                else:
                    self.emit("OK")
                self.state()


class Integration(unittest.TestCase):
    def test_clients(self):
        master, slave = pty.openpty()
        port = serial.Serial(os.ttyname(slave), 115200, timeout=0.02)
        fake = FakeArduino(master)
        emulation = threading.Thread(target=fake.run)
        emulation.start()
        bridge = Bridge(port)
        stopped, ready = threading.Event(), threading.Event()
        endpoint = f"ipc:///tmp/mirror-zmq-test-{uuid.uuid4().hex}"
        errors = []

        def server():
            try:
                serve(bridge, endpoint, stopped.is_set, ready)
            except Exception as error:
                errors.append(error)

        worker = threading.Thread(target=server)
        worker.start()
        try:
            self.assertTrue(ready.wait(2))
            with MirrorClient(endpoint) as a, MirrorClient(endpoint) as b:
                self.assertFalse(a.status()["referenced"])
                with self.assertRaisesRegex(RuntimeError, "ZERO_REQUIRED"):
                    a.move(1)
                a.zero()
                job = a.move(10)
                with self.assertRaisesRegex(RuntimeError, "BUSY"):
                    b.move(1)
                self.assertEqual(a.wait(job)["job"]["status"], "done")
                self.assertEqual(b.status()["position"], 10)
                job2 = b.goto(-10)
                a.stop()
                with self.assertRaisesRegex(RuntimeError, "did not complete"):
                    b.wait(job2)
                self.assertEqual(a.wait(job)["job"]["status"], "done")
                a.wait(a.goto(10))  # Already there: DONE precedes OK.
                with self.assertRaisesRegex(RuntimeError, "OUT_OF_RANGE"):
                    a.goto(101)
                with self.assertRaisesRegex(RuntimeError, "integers"):
                    a.move(True)
                with self.assertRaisesRegex(RuntimeError, "unexpected"):
                    a.request("zero", extra="MOVE 10")
                with self.assertRaisesRegex(RuntimeError, "Unknown/expired"):
                    a.request("job", id="missing")
                time.sleep(0.6)
                self.assertGreater(sum(c == ["STATUS"] for c in fake.commands), 4)
                context = zmq.Context()
                with context.socket(zmq.REQ) as raw:
                    raw.connect(endpoint)
                    raw.send(b"not JSON")
                    self.assertTrue(raw.poll(1000))
                    self.assertFalse(raw.recv_json()["ok"])
                context.term()
                a.status()  # Service survives malformed requests.
            with MirrorClient("ipc:///tmp/missing-mirror-server", timeout_ms=50) as missing:
                with self.assertRaises(TimeoutError):
                    missing.move(1)
        finally:
            stopped.set()
            worker.join(3)
            fake.quit = True
            emulation.join(2)
            port.close()
            os.close(master)
            os.close(slave)
        self.assertFalse(worker.is_alive())
        self.assertEqual(errors, [])


if __name__ == "__main__":
    unittest.main()
