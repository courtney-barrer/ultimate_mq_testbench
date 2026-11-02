"""Display process: owns the pyglet window and runs its loop on its own main thread.

Started by slmdisplay.display as ``python -m slmdisplay._worker FD ARGS_JSON``,
where FD is one end of a socket pair wrapped as a multiprocessing Connection.
Nothing GUI-related ever touches the user's process; pyglet is only imported here.
"""

import ctypes
import json
import sys
import time
from multiprocessing.connection import Connection
from multiprocessing.shared_memory import SharedMemory

import numpy as np

from . import _protocol as P

# Imported once pyglet's options are set (see _import_pyglet).
_x11 = None

_VERTEX_SRC = """#version 330 core
out vec2 uv;
void main() {
    // Corners (0,0) (1,0) (0,1) (1,1) as a triangle strip; no vertex buffer needed.
    vec2 p = vec2(float(gl_VertexID & 1), float((gl_VertexID >> 1) & 1));
    uv = vec2(p.x, 1.0 - p.y);  // numpy row 0 is the top of the screen
    gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}
"""

_FRAGMENT_SRC = """#version 330 core
uniform sampler2D image;
in vec2 uv;
out vec4 color;
void main() {
    float v = texture(image, uv).r;
    color = vec4(v, v, v, 1.0);
}
"""

# Idle wait for a new command when nothing needs redrawing.
_POLL_INTERVAL = 0.002


def _import_pyglet():
    global _x11
    import pyglet

    pyglet.options.shadow_window = False
    pyglet.options.debug_gl = False
    from . import _x11
    return pyglet


def _list_screens(pyglet):
    """Return (pyglet screens, ScreenInfos), both ordered by OS monitor index.

    The OS index is the order of the X server's RandR monitor list, as shown by
    ``xrandr --listmonitors``. Each monitor is matched to the pyglet screen that
    drives it (by RandR output, else by geometry). Falls back to pyglet's own
    order if the server reports no monitors.
    """
    display = pyglet.display.get_display()
    screens = display.get_screens()
    ordered = []
    for m in _x11.monitors(display._display):
        for s in screens:
            same_output = getattr(s, "output_id", None) in m["outputs"]
            same_area = (s.x, s.y, s.width, s.height) == (m["x"], m["y"], m["width"], m["height"])
            if s not in [o for o, _ in ordered] and (same_output or same_area):
                ordered.append((s, m["primary"]))
                break
    if len(ordered) != len(screens):
        ordered = [(s, False) for s in screens]
    return [s for s, _ in ordered], [
        P.ScreenInfo(i, getattr(s, "name", None), s.x, s.y, s.width, s.height, primary)
        for i, (s, primary) in enumerate(ordered)
    ]


def _pick_screen(screens, infos, screen):
    if isinstance(screen, str):
        for s, info in zip(screens, infos):
            if info.name == screen:
                return s
        names = [info.name for info in infos]
        raise ValueError(f"no screen named {screen!r}; available: {names}")
    if not 0 <= screen < len(screens):
        raise ValueError(
            f"screen index {screen} out of range; {len(screens)} screen(s) available "
            "(see xrandr --listmonitors)"
        )
    return screens[screen]


def _send_error(conn, exc):
    conn.send((P.ERROR, type(exc).__name__, str(exc)))


def list_screens_main(conn):
    """Report the screens and exit."""
    try:
        pyglet = _import_pyglet()
        conn.send((P.SCREENS, _list_screens(pyglet)[1]))
    except Exception as exc:
        _send_error(conn, exc)


def _make_window_class(pyglet):
    """pyglet Window that never grabs input or sets keyboard focus.

    pyglet's X11 fullscreen mode confines the pointer to the window
    (XGrabPointer in _update_exclusivity) and focuses it on creation
    (activate). Both are private pyglet methods, hence the pyglet<2.2 pin in
    pyproject.toml.
    """

    class SLMWindow(pyglet.window.Window):
        def _update_exclusivity(self):
            pass

        def activate(self):
            pass

    return SLMWindow


class _Renderer:
    """Write-only surface covering one monitor, drawing a uint8 texture with GL_NEAREST.

    It is an override-redirect window with an empty input shape: the window
    manager never manages or focuses it, and no input event can reach it.
    """

    def __init__(self, pyglet, screen):
        from pyglet import gl
        from pyglet.graphics.shader import Shader, ShaderProgram

        self.gl = gl
        config = gl.Config(
            double_buffer=True, red_size=8, green_size=8, blue_size=8, depth_size=0
        )
        self.window = _make_window_class(pyglet)(
            fullscreen=True,
            screen=screen,
            vsync=True,
            config=config,
            caption="SLM",
            visible=False,
        )
        self.dpy = self.window._x_display
        _x11.make_overlay(self.dpy, self.window._window, self.window._view)
        self.window.set_visible(True)

        self.needs_redraw = True
        self.window.push_handlers(
            on_expose=self._request_redraw,
            on_resize=lambda width, height: self._request_redraw(),
        )

        self.program = ShaderProgram(
            Shader(_VERTEX_SRC, "vertex"), Shader(_FRAGMENT_SRC, "fragment")
        )
        self.program.use()
        self.program["image"] = 0

        self.vao = gl.GLuint()
        gl.glGenVertexArrays(1, ctypes.byref(self.vao))
        gl.glBindVertexArray(self.vao)

        self.texture = gl.GLuint()
        gl.glGenTextures(1, ctypes.byref(self.texture))
        gl.glActiveTexture(gl.GL_TEXTURE0)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.texture)
        for param, value in (
            (gl.GL_TEXTURE_MIN_FILTER, gl.GL_NEAREST),
            (gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST),
            (gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE),
            (gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE),
        ):
            gl.glTexParameteri(gl.GL_TEXTURE_2D, param, value)
        gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, 1)
        gl.glPixelStorei(gl.GL_PACK_ALIGNMENT, 1)
        # Dithering could perturb the exact 8-bit levels.
        gl.glDisable(gl.GL_DITHER)
        gl.glDisable(gl.GL_BLEND)
        self.texture_shape = None

    def _request_redraw(self):
        self.needs_redraw = True

    @property
    def size(self):
        return self.window.get_framebuffer_size()

    def upload(self, pixels, height, width):
        gl = self.gl
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.texture)
        if self.texture_shape != (height, width):
            gl.glTexImage2D(
                gl.GL_TEXTURE_2D, 0, gl.GL_R8, width, height, 0,
                gl.GL_RED, gl.GL_UNSIGNED_BYTE, pixels,
            )
            self.texture_shape = (height, width)
        else:
            gl.glTexSubImage2D(
                gl.GL_TEXTURE_2D, 0, 0, 0, width, height,
                gl.GL_RED, gl.GL_UNSIGNED_BYTE, pixels,
            )
        self.needs_redraw = True

    def draw(self):
        gl = self.gl
        width, height = self.size
        gl.glViewport(0, 0, width, height)
        gl.glClearColor(0.0, 0.0, 0.0, 1.0)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)
        if self.texture_shape is not None:
            self.program.use()
            gl.glBindVertexArray(self.vao)
            gl.glDrawArrays(gl.GL_TRIANGLE_STRIP, 0, 4)

    def present(self):
        """Draw, swap on vsync and wait until the GPU is done."""
        self.draw()
        self.window.flip()
        self.gl.glFinish()
        self.needs_redraw = False
        return time.perf_counter()

    def read_pixels(self):
        """Redraw into the back buffer and read it as (H, W, 3), row 0 at the top."""
        gl = self.gl
        self.draw()
        width, height = self.size
        out = np.empty((height, width, 3), dtype=np.uint8)
        gl.glReadBuffer(gl.GL_BACK)
        gl.glReadPixels(
            0, 0, width, height, gl.GL_RGB, gl.GL_UNSIGNED_BYTE, out.ctypes.data
        )
        return out[::-1].copy()

    def close(self):
        self.window.close()


def _log(message):
    print(f"slmdisplay: {message}", file=sys.stderr, flush=True)


def display_main(conn, screen, neutral_output):
    """Open the surface, then serve commands until QUIT or the parent goes away."""
    try:
        pyglet = _import_pyglet()
        screens, infos = _list_screens(pyglet)
        target = _pick_screen(screens, infos, screen)
        info = infos[screens.index(target)]
        renderer = _Renderer(pyglet, target)
        dpy = renderer.dpy
        if neutral_output:
            if hasattr(target, "output_id"):
                for message in _x11.neutralize_output(
                    dpy, target.output_id, target.crtc_id, info.name
                ):
                    _log(message)
            else:
                _log(f"screen {info.name!r} has no RandR output; colour pipeline not checked")
        # Barriers live as long as this process's X connection.
        _x11.fence_pointer(dpy, info.x, info.y, info.width, info.height)
        px, py = _x11.pointer_position(dpy)
        if info.x <= px < info.x + info.width and info.y <= py < info.y + info.height:
            _log("the mouse pointer is on the SLM screen; move it off (it cannot come back)")
    except Exception as exc:
        _send_error(conn, exc)
        return

    try:
        width, height = renderer.size
        info = {
            "width": width,
            "height": height,
            "screens": infos,
            "window_id": getattr(renderer.window, "_window", None),
        }
        conn.send((P.READY, info))
        tag, shm_name = conn.recv()
        assert tag == P.ATTACH, tag
        shm = SharedMemory(name=shm_name) #, track=False) # use track for python3.13, not for 3.12
    except (EOFError, BrokenPipeError, ConnectionResetError):
        renderer.close()
        return
    pixels = (ctypes.c_ubyte * shm.size).from_buffer(shm.buf)

    newest_seq = None
    try:
        while True:
            renderer.window.dispatch_events()
            timeout = 0 if renderer.needs_redraw else _POLL_INTERVAL
            while conn.poll(timeout):
                timeout = 0
                msg = conn.recv()
                tag = msg[0]
                if tag == P.FRAME:
                    _, seq, h, w = msg
                    renderer.upload(pixels, h, w)
                    newest_seq = seq
                    conn.send((P.UPLOADED, seq))
                elif tag == P.READBACK:
                    conn.send((P.PIXELS, renderer.read_pixels()))
                elif tag == P.QUIT:
                    return
            if renderer.needs_redraw:
                t = renderer.present()
                if newest_seq is not None:
                    conn.send((P.DISPLAYED, newest_seq, t))
                    newest_seq = None
    except (EOFError, BrokenPipeError, ConnectionResetError):
        pass  # parent went away
    finally:
        del pixels
        shm.close()
        renderer.close()


def main(argv):
    conn = Connection(int(argv[1]))
    args = json.loads(argv[2])
    try:
        if args["mode"] == "list":
            list_screens_main(conn)
        else:
            display_main(conn, args["screen"], args["neutral_output"])
    finally:
        conn.close()


if __name__ == "__main__":
    main(sys.argv)
