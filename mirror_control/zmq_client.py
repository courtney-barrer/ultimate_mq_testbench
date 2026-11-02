"""Import MirrorClient in each Python process that needs motor access."""
import time
import zmq


class MirrorClient:
    def __init__(self, endpoint="tcp://127.0.0.1:5555", timeout_ms=2500):
        self.endpoint = endpoint
        self.timeout_ms = timeout_ms
        self.context = zmq.Context()

    def request(self, op, **fields):
        # A fresh socket avoids REQ state errors after a timeout. Never retry
        # automatically: a timed-out MOVE may already have been accepted.
        with self.context.socket(zmq.REQ) as socket:
            socket.setsockopt(zmq.LINGER, 0)
            socket.setsockopt(zmq.SNDTIMEO, self.timeout_ms)
            socket.connect(self.endpoint)
            socket.send_json({"op": op, **fields})
            if not socket.poll(self.timeout_ms):
                raise TimeoutError("No reply; movement outcome may be unknown. Check status or stop; do not blindly retry.")
            result = socket.recv_json()
        if not result.get("ok"):
            raise RuntimeError(result.get("error", "Request failed"))
        return result

    def status(self):
        return self.request("status")["state"]

    def zero(self):
        return self.request("zero")

    def stop(self):
        return self.request("stop")

    def set_speed(self, steps_per_second):
        return self.request("speed", value=steps_per_second)

    def set_limits(self, lower, upper):
        return self.request("limits", lower=lower, upper=upper)

    def move(self, steps):
        return self.request("move", steps=steps)["job_id"]

    def goto(self, position):
        return self.request("goto", position=position)["job_id"]

    def wait(self, job_id, timeout=30):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            result = self.request("job", id=job_id)
            job = result["job"]
            if job["status"] == "done":
                return result
            if job["status"] != "running":
                raise RuntimeError(f"Movement did not complete: {job}")
            time.sleep(0.1)
        raise TimeoutError("Wait expired; motion may still be running. Call stop() if desired.")

    def close(self):
        self.context.term()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


if __name__ == "__main__":
    with MirrorClient() as motor:
        print(motor.status())  # Read-only demonstration.
