"""Write single-channel 8-bit images to an SLM connected as a video output.

Each pixel value (0-255) is one discrete drive voltage level, written identically
to the R, G and B channels of the SLM's video input.

The SLM is not treated as a monitor. The image is drawn on a write-only surface
that the window manager never sees, that no mouse or keyboard event can reach, and
that the mouse pointer cannot move onto. It is owned by a separate process (see
_worker), so nothing GUI-related runs in the calling script.

Mapping from the old slmpy API:
    SLMdisplay(monitor=1, isImageLock=True, alwaysTop=False)
        -> SLM(screen=1)
    slm.getSize()                -> slm.width, slm.height  (or slm.shape)
    slm.updateArray(img)         -> slm.show(img)          (isImageLock=True)
                                    slm.show(img, wait=False)  (isImageLock=False)
    updateArray(..., sleep=0.2)  -> slm.show(img, settle=0.2)
    slm.close()                  -> slm.close()  or use ``with SLM(...) as slm:``
"""

import atexit
import builtins
import json
import os
import socket
import subprocess
import sys
import time
from multiprocessing.connection import Connection
from multiprocessing.shared_memory import SharedMemory

import numpy as np

from . import _protocol as P
from ._protocol import ScreenInfo

_RESIZE_MODES = ("error", "nearest")


class DisplayProcessError(RuntimeError):
    """The display process died or stopped answering."""


def _check_image(image, shape, resize):
    """Raise TypeError/ValueError unless ``image`` can be shown on a screen of ``shape``."""
    if not isinstance(image, np.ndarray):
        raise TypeError(f"image must be a numpy array, got {type(image).__name__}")
    if image.dtype != np.uint8:
        raise TypeError(f"image must have dtype uint8, got {image.dtype}")
    if image.ndim != 2:
        raise ValueError(
            f"image must be 2-D (height, width) single-channel, got shape {image.shape}"
        )
    if image.shape == shape:
        return
    if resize == "error":
        raise ValueError(
            f"image shape {image.shape} does not match the SLM shape {shape}; "
            "pass resize='nearest' to SLM() to stretch smaller images"
        )
    h, w = image.shape
    if not (0 < h <= shape[0] and 0 < w <= shape[1]):
        raise ValueError(f"image shape {image.shape} must fit within {shape}")


def _spawn_worker(args):
    """Start the display process; return (process, parent-side Connection)."""
    parent_sock, child_sock = socket.socketpair()
    with child_sock:
        child_fd = child_sock.fileno()
        proc = subprocess.Popen(
            [sys.executable, "-m", "slmdisplay._worker", str(child_fd), json.dumps(args)],
            pass_fds=(child_fd,),
        )
    return proc, Connection(parent_sock.detach())


def _raise_child_error(msg):
    _, exc_name, text = msg
    exc_type = getattr(builtins, exc_name, None)
    if not (isinstance(exc_type, type) and issubclass(exc_type, Exception)):
        exc_type = DisplayProcessError
        text = f"{exc_name}: {text}"
    raise exc_type(text)


def _recv(conn, proc, timeout):
    """Receive one message, raising if the display process is gone or silent."""
    try:
        if conn.poll(timeout):
            return conn.recv()
    except (EOFError, ConnectionResetError):
        pass
    else:
        if proc.poll() is None:
            raise DisplayProcessError(f"no reply from display process within {timeout} s")
    raise DisplayProcessError(f"display process exited (code {proc.wait()})")


def list_screens(timeout=10.0):
    """Return the monitors, indexed as in ``xrandr --listmonitors``."""
    _require_x11()
    proc, conn = _spawn_worker({"mode": "list"})
    try:
        msg = _recv(conn, proc, timeout)
        if msg[0] == P.ERROR:
            _raise_child_error(msg)
        return msg[1]
    finally:
        conn.close()
        proc.wait(timeout)


def _require_x11():
    if os.environ.get("XDG_SESSION_TYPE") == "wayland":
        raise RuntimeError("slmdisplay needs an X11 session (Wayland session detected)")
    if not os.environ.get("DISPLAY"):
        raise RuntimeError("slmdisplay needs an X11 session (DISPLAY is not set)")


class SLM:
    """Write-only image output on the SLM's video connector.

    Parameters
    ----------
    screen : int or str, default 1
        OS monitor index, as listed by ``xrandr --listmonitors`` and
        ``list_screens()`` (normally 0 is the primary screen), or an output
        name such as ``"HDMI-A-0"``.
    resize : {"error", "nearest"}, default "error"
        What to do with images smaller than the screen. "error" rejects them;
        "nearest" stretches them to fill the screen with nearest-neighbour
        sampling (no interpolated values). Use it only with integer factors.
    neutral_output : bool, default True
        Set an identity colour matrix and a linear gamma ramp on the SLM output
        (and print what was changed), so pixel values reach the SLM unaltered.
    timeout : float, default 10.0
        Seconds to wait for the display process to start or answer.
    """

    def __init__(self, screen=1, resize="error", neutral_output=True, timeout=10.0):
        if resize not in _RESIZE_MODES:
            raise ValueError(f"resize must be one of {_RESIZE_MODES}, got {resize!r}")
        _require_x11()
        self.resize = resize
        self.timeout = timeout
        self._closed = False
        self._shm = None
        self._seq = 0
        self._displayed_seq = 0

        self._proc, self._conn = _spawn_worker(
            {"mode": "display", "screen": screen, "neutral_output": neutral_output}
        )
        try:
            msg = self._recv()
            if msg[0] == P.ERROR:
                _raise_child_error(msg)
            info = msg[1]
            self.width, self.height = info["width"], info["height"]
            self.screens = info["screens"]
            self._window_id = info["window_id"]
            self._shm = SharedMemory(create=True, size=self.width * self.height)
            self._conn.send((P.ATTACH, self._shm.name))
        except BaseException:
            self.close()
            raise
        atexit.register(self.close)

    @property
    def shape(self):
        """(height, width) of the screen, i.e. the shape ``show`` expects."""
        return (self.height, self.width)

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.close()

    def __repr__(self):
        state = "closed" if self._closed else f"{self.width}x{self.height}"
        return f"<SLM {state}>"

    def _recv(self):
        return _recv(self._conn, self._proc, self.timeout)

    def _wait_for(self, tag, seq=None):
        """Read messages until one tagged ``tag`` (and covering ``seq``) arrives."""
        while True:
            msg = self._recv()
            if msg[0] == P.DISPLAYED:
                self._displayed_seq = msg[1]
            if msg[0] == tag and (seq is None or msg[1] >= seq):
                return msg

    def _check_open(self):
        if self._closed:
            raise RuntimeError("SLM is closed")

    def show(self, image, wait=True, settle=0.0):
        """Display a (height, width) uint8 array; one value per pixel voltage level.

        Parameters
        ----------
        image : numpy.ndarray
            2-D uint8 array. Row 0 is the top of the screen.
        wait : bool, default True
            Block until the frame has been swapped onto the screen (vsync).
            With False, return once the display process has its own copy, so
            the next frame can be computed while this one is swapped in. The
            rate is still capped at the refresh rate, and a frame may now and
            then be replaced by the next one before it reaches the screen.
        settle : float, default 0.0
            Extra seconds to sleep after the frame is displayed, e.g. for the
            liquid-crystal response time.

        Returns
        -------
        int
            Sequence number of this frame.
        """
        self._check_open()
        _check_image(image, self.shape, self.resize)
        h, w = image.shape
        np.ndarray((h, w), dtype=np.uint8, buffer=self._shm.buf)[...] = image
        self._seq += 1
        self._conn.send((P.FRAME, self._seq, h, w))
        # The shared buffer may be rewritten only once the frame has been copied out.
        self._wait_for(P.UPLOADED, self._seq)
        if wait and self._displayed_seq < self._seq:
            self._wait_for(P.DISPLAYED, self._seq)
        if settle > 0:
            time.sleep(settle)
        return self._seq

    def clear(self, value=0, wait=True):
        """Set every pixel to ``value``."""
        return self.show(np.full(self.shape, value, dtype=np.uint8), wait=wait)

    def read(self):
        """Return the image currently on the SLM as a (height, width) uint8 array.

        Read back from the surface's framebuffer, i.e. what is sent to the output
        (before the output's colour matrix/gamma, which ``neutral_output`` keeps neutral).
        """
        self._check_open()
        self._conn.send((P.READBACK,))
        pixels = self._wait_for(P.PIXELS)[1]
        image = pixels[..., 0]
        if not (np.array_equal(image, pixels[..., 1]) and np.array_equal(image, pixels[..., 2])):
            raise DisplayProcessError("framebuffer channels differ; the surface is not greyscale")
        return np.ascontiguousarray(image)

    def close(self):
        """Remove the surface and stop the display process. Safe to call twice."""
        if self._closed:
            return
        self._closed = True
        atexit.unregister(self.close)
        try:
            self._conn.send((P.QUIT,))
        except OSError:
            pass
        try:
            self._proc.wait(self.timeout)
        except subprocess.TimeoutExpired:
            self._proc.kill()
            self._proc.wait()
        self._conn.close()
        if self._shm is not None:
            self._shm.close()
            self._shm.unlink()
