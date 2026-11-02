"""Messages exchanged between the SLM object and its display process.

Every message is a tuple whose first element is one of the tags below.

Parent -> child:
    (ATTACH, shm_name)           shared memory to read frames from (after READY)
    (FRAME, seq, height, width)  new image waiting in shared memory
    (READBACK,)                  return the pixels currently on screen
    (QUIT,)                      close the window and exit

Child -> parent:
    (READY, info)                window is up; info is a dict (see _worker)
    (SCREENS, screens)           reply in list-only mode
    (ERROR, exc_name, message)   startup failed; exc_name is a builtin exception
    (UPLOADED, seq)              frame copied out of shared memory
    (DISPLAYED, seq, t)          frame seq (and all before it) swapped to screen
    (PIXELS, array)              reply to READBACK, (H, W, 3) uint8
"""

from dataclasses import dataclass

ATTACH = "attach"
FRAME = "frame"
READBACK = "readback"
QUIT = "quit"

READY = "ready"
SCREENS = "screens"
ERROR = "error"
UPLOADED = "uploaded"
DISPLAYED = "displayed"
PIXELS = "pixels"


@dataclass(frozen=True)
class ScreenInfo:
    """A monitor as reported by the X server.

    ``index`` is the OS monitor number, as listed by ``xrandr --listmonitors``
    (normally 0 is the primary screen).
    """

    index: int
    name: str | None
    x: int
    y: int
    width: int
    height: int
    primary: bool = False
