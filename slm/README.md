# slmdisplay

Write 8-bit phase masks to a spatial light modulator (SLM) connected as a video output (HDMI/DP).
This is a modern rewrite of [slmPy](https://github.com/wavefrontshaping/slmPy) (kept in `slmPy/` for reference).

Each image is a single-channel `uint8` array of shape `(height, width)`. Every value (0–255) is one
discrete drive voltage level. It is written unchanged to the R, G and B channels of the video signal,
with no interpolation, dithering or colour handling.

The SLM is **not** treated as a monitor. Writing images (and reading back the current one) is the
only interaction. The mouse can't move onto it, no window or focus ever lands on it, and your
keyboard and mouse stay on the main screen.

## Install
to set this up within the ultimate_mq_testbench repository go to slm folder and do
```bash
pip install -e .
```

## Usage

```python
import numpy as np
from slmdisplay import SLM, list_screens

print(list_screens())    # [ScreenInfo(index=0, name='eDP', ..., primary=True), ScreenInfo(index=1, name='HDMI-A-0', ...)]

with SLM(screen=1) as slm:          # OS monitor index from `xrandr --listmonitors`, or SLM(screen="HDMI-A-0")
    print(slm.shape)                # (height, width), e.g. (1200, 1920)
    x = np.arange(slm.width)
    grating = np.broadcast_to((x % 50 * 255 // 50).astype(np.uint8), slm.shape)
    slm.show(grating)               # returns once the frame is on the SLM
    slm.show(grating, settle=0.05)  # ...and then waits 50 ms for the liquid crystal
    current = slm.read()            # (height, width) uint8: what is on the SLM now
    slm.clear()                     # all zeros
```

- `show(image, wait=True)` blocks until the frame has been swapped onto the output (vsync), so in a
  measurement loop you know the SLM has the mask before you trigger a camera. It does not
  include the liquid-crystal response time; use `settle=` for that.
- `show(image, wait=False)` returns once the display process has its own copy. That lets you
  compute the next mask in parallel. The rate is still capped at the refresh rate.
- `read()` returns the image currently being sent, read back from the framebuffer.
- `SLM(resize="nearest")` accepts images smaller than the SLM and stretches them with
  nearest-neighbour sampling (use integer factors, e.g. half resolution → 2×2 blocks).
  The default `resize="error"` rejects any shape mismatch.
- Non-`uint8`, RGB/3-D or wrongly shaped arrays raise `TypeError`/`ValueError`. Nothing is
  cast or rescaled silently.

Without a `with` block, call `slm.close()` yourself. It is also called automatically at interpreter exit.
The image stays on the SLM only while the `SLM` is open.

## How it works

A separate process (`python -m slmdisplay._worker`) owns everything on the SLM side, so your
script, Jupyter kernel or matplotlib never touch a GUI toolkit. That process:

- Covers the SLM's screen area with an **override-redirect** X surface. The window manager never
  sees it, so it can't be focused, moved, or listed in alt-tab or the dock.
- Gives that surface an **empty input shape**, so no mouse or keyboard event can ever reach it.
- Puts **XFixes pointer barriers** around the SLM's screen area, so the mouse can't move onto it.
  If the pointer is already there, it can leave but not come back.
- Sets an **identity colour matrix and a linear gamma ramp** on the SLM output (see below).
- Takes frames through shared memory, uploads them to a single-channel OpenGL texture, draws them
  1:1 with `GL_NEAREST`, swaps on vsync and acknowledges.

When the process exits (on `close()`, at the end of your script, or if your script crashes or is
killed), the surface and the barriers disappear with it.

## Colour pipeline

Any colour transform on the output would change the drive voltages. By default, `SLM()`
(`neutral_output=True`):

- sets the output's colour matrix (CTM) to identity and its gamma ramp to linear;
- reads both back and raises if they didn't stick;
- prints what it changed to stderr.

On this machine the Meadowlark output had a non-identity CTM: the first `SLM()` reset it to identity.

GNOME's colour manager has an auto-generated ICC profile assigned to the Meadowlark. To stop it
from re-applying a profile later, remove it under **Settings → Color → MEADOWLARK**.

What we can verify is the X/RandR configuration and the framebuffer contents. The electrical signal
on the HDMI cable itself can't be checked from software.

Coming from slmPy:

| slmPy | slmdisplay |
| --- | --- |
| `SLMdisplay(monitor=1, isImageLock=True, alwaysTop=False)` | `SLM(screen=1)` |
| `slm.getSize()` → `(resX, resY)` | `slm.width`, `slm.height`, `slm.shape` |
| `slm.updateArray(img)` (`isImageLock=True`) | `slm.show(img)` |
| `isImageLock=False` | `slm.show(img, wait=False)` |
| `updateArray(img, sleep=0.2)` | `slm.show(img, settle=0.2)` |
| — | `slm.read()` |
| 3-channel colour arrays | not supported, single channel only |
| `Client` / `listen_port` network mode | not ported |

## Setup notes

- Requires an **X11** session (Linux). Override-redirect surfaces and pointer barriers don't exist
  under Wayland.
- Set the SLM output to its native resolution (`xrandr`) and turn off screen blanking:
  `xset s off -dpms`.

### Recovering from a stuck surface

The surface goes away when your script exits or is killed. If a display process is ever left behind,
run `pkill -f slmdisplay._worker`.

## Tests

```bash
uv run pytest -m "not display"     # no screen needed
SLMDISPLAY_TEST_SCREEN=1 SLMDISPLAY_TEST_REFRESH=60 uv run pytest   # on the SLM, incl. vsync timing
```

The display tests read the drawn pixels back and check they equal the input exactly. They also
check that the surface is override-redirect and input-transparent, that nothing grabs input or takes
focus, and that the output's colour pipeline is neutral. The test that moves the pointer against the
barriers only runs with `SLMDISPLAY_TEST_ALLOW_POINTER=1`; use it on a nested X server
(e.g. `Xephyr :99 -screen 640x400`, then `DISPLAY=:99`).
