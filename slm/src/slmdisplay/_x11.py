"""X11 helpers that turn the pyglet window into a write-only surface on the SLM.

- make_overlay: override-redirect (the window manager never sees it, so it can
  never be focused, moved or listed) and an empty input shape (no pointer or
  keyboard event is ever delivered to it).
- fence_pointer: XFixes pointer barriers that stop the mouse from entering the
  SLM monitor. They belong to the calling X connection and vanish with it.
- neutralize_output: identity colour matrix (CTM) and linear gamma ramp on the
  SLM output, so pixel values reach the device unchanged.

Only imported by the display process. Reuses pyglet's Xlib bindings and loads
libXfixes/libXrandr for the few calls pyglet does not bind.
"""

import ctypes
from ctypes import POINTER, Structure, byref, c_int, c_long, c_ubyte, c_ulong, c_ushort

from pyglet.libs.x11 import xlib, xsync

_xfixes = ctypes.CDLL("libXfixes.so.3")
_xrandr = ctypes.CDLL("libXrandr.so.2")

_Display = POINTER(xlib.Display)

# X11/extensions/xfixeswire.h
BARRIER_POSITIVE_X = 1 << 0
BARRIER_POSITIVE_Y = 1 << 1
BARRIER_NEGATIVE_X = 1 << 2
BARRIER_NEGATIVE_Y = 1 << 3

_xfixes.XFixesQueryVersion.argtypes = [_Display, POINTER(c_int), POINTER(c_int)]
_xfixes.XFixesQueryVersion.restype = c_int
_xfixes.XFixesCreatePointerBarrier.argtypes = [
    _Display, c_ulong, c_int, c_int, c_int, c_int, c_int, c_int, POINTER(c_int),
]
_xfixes.XFixesCreatePointerBarrier.restype = c_ulong


class XRRCrtcGamma(Structure):
    _fields_ = [
        ("size", c_int),
        ("red", POINTER(c_ushort)),
        ("green", POINTER(c_ushort)),
        ("blue", POINTER(c_ushort)),
    ]


_xrandr.XRRGetOutputProperty.argtypes = [
    _Display, c_ulong, c_ulong, c_long, c_long, c_int, c_int, c_ulong,
    POINTER(c_ulong), POINTER(c_int), POINTER(c_ulong), POINTER(c_ulong),
    POINTER(POINTER(c_ubyte)),
]
_xrandr.XRRGetOutputProperty.restype = c_int
_xrandr.XRRChangeOutputProperty.argtypes = [
    _Display, c_ulong, c_ulong, c_ulong, c_int, c_int, POINTER(c_ubyte), c_int,
]
_xrandr.XRRChangeOutputProperty.restype = None
_xrandr.XRRGetCrtcGammaSize.argtypes = [_Display, c_ulong]
_xrandr.XRRGetCrtcGammaSize.restype = c_int
_xrandr.XRRAllocGamma.argtypes = [c_int]
_xrandr.XRRAllocGamma.restype = POINTER(XRRCrtcGamma)
_xrandr.XRRGetCrtcGamma.argtypes = [_Display, c_ulong]
_xrandr.XRRGetCrtcGamma.restype = POINTER(XRRCrtcGamma)
_xrandr.XRRSetCrtcGamma.argtypes = [_Display, c_ulong, POINTER(XRRCrtcGamma)]
_xrandr.XRRSetCrtcGamma.restype = None
_xrandr.XRRFreeGamma.argtypes = [POINTER(XRRCrtcGamma)]
_xrandr.XRRFreeGamma.restype = None

class XRRMonitorInfo(Structure):
    _fields_ = [
        ("name", c_ulong),
        ("primary", c_int),
        ("automatic", c_int),
        ("noutput", c_int),
        ("x", c_int),
        ("y", c_int),
        ("width", c_int),
        ("height", c_int),
        ("mwidth", c_int),
        ("mheight", c_int),
        ("outputs", POINTER(c_ulong)),
    ]


# pyglet binds XGetAtomName returning bytes, which leaks the Xlib string; this one can be freed.
_get_atom_name = ctypes.CDLL("libX11.so.6").XGetAtomName
_get_atom_name.argtypes = [_Display, c_ulong]
_get_atom_name.restype = ctypes.c_void_p

_xrandr.XRRGetMonitors.argtypes = [_Display, c_ulong, c_int, POINTER(c_int)]
_xrandr.XRRGetMonitors.restype = POINTER(XRRMonitorInfo)
_xrandr.XRRFreeMonitors.argtypes = [POINTER(XRRMonitorInfo)]
_xrandr.XRRFreeMonitors.restype = None

# 3x3 identity in DRM's S31.32 sign-magnitude format: (low 32 bits, high 32 bits) per entry.
IDENTITY_CTM = [0, 1, 0, 0, 0, 0,
                0, 0, 0, 1, 0, 0,
                0, 0, 0, 0, 0, 1]


def make_overlay(dpy, window, view=None):
    """Make an unmapped window override-redirect and transparent to all input."""
    attrs = xlib.XSetWindowAttributes()
    attrs.override_redirect = True
    xlib.XChangeWindowAttributes(dpy, window, xlib.CWOverrideRedirect, byref(attrs))
    for win in (window, view):
        if win:
            region = xlib.XCreateRegion()
            xsync.XShapeCombineRegion(dpy, win, xsync.ShapeInput, 0, 0, region, xsync.ShapeSet)
            xlib.XDestroyRegion(region)


def fence_pointer(dpy, x, y, width, height):
    """Block the pointer from entering the rectangle; leaving it stays possible."""
    major, minor = c_int(5), c_int(0)
    _xfixes.XFixesQueryVersion(dpy, byref(major), byref(minor))
    if major.value < 5:
        raise RuntimeError(f"XFixes {major.value}.{minor.value} has no pointer barriers")
    root = xlib.XDefaultRootWindow(dpy)
    right, bottom = x + width, y + height
    edges = [
        (x, y, x, bottom, BARRIER_NEGATIVE_X),          # left: may only move left (out)
        (right, y, right, bottom, BARRIER_POSITIVE_X),  # right: may only move right (out)
        (x, y, right, y, BARRIER_NEGATIVE_Y),           # top: may only move up (out)
        (x, bottom, right, bottom, BARRIER_POSITIVE_Y), # bottom: may only move down (out)
    ]
    barriers = [
        _xfixes.XFixesCreatePointerBarrier(dpy, root, x1, y1, x2, y2, allowed, 0, None)
        for x1, y1, x2, y2, allowed in edges
    ]
    xlib.XSync(dpy, False)
    return barriers


def monitors(dpy):
    """Active monitors in the X server's order, i.e. the index ``xrandr --listmonitors`` shows.

    Returns a list of dicts with name, primary, x, y, width, height and outputs.
    """
    count = c_int()
    info = _xrandr.XRRGetMonitors(dpy, xlib.XDefaultRootWindow(dpy), True, byref(count))
    if not info:
        return []
    try:
        result = []
        for m in info[: count.value]:
            name = _get_atom_name(dpy, m.name)
            result.append({
                "name": ctypes.cast(name, ctypes.c_char_p).value.decode(),
                "primary": bool(m.primary),
                "x": m.x, "y": m.y, "width": m.width, "height": m.height,
                "outputs": list(m.outputs[: m.noutput]),
            })
            xlib.XFree(name)
        return result
    finally:
        _xrandr.XRRFreeMonitors(info)


def pointer_position(dpy):
    """Root coordinates of the pointer."""
    root = xlib.XDefaultRootWindow(dpy)
    root_ret, child = xlib.Window(), xlib.Window()
    rx, ry, wx, wy = c_int(), c_int(), c_int(), c_int()
    mask = ctypes.c_uint()
    xlib.XQueryPointer(dpy, root, byref(root_ret), byref(child),
                       byref(rx), byref(ry), byref(wx), byref(wy), byref(mask))
    return rx.value, ry.value


def get_output_property(dpy, output, name):
    """Return (type_atom, format, values) of an output property, or None if absent."""
    atom = xlib.XInternAtom(dpy, name.encode(), True)
    if not atom:
        return None
    actual_type, actual_format = c_ulong(), c_int()
    nitems, bytes_after = c_ulong(), c_ulong()
    data = POINTER(c_ubyte)()
    status = _xrandr.XRRGetOutputProperty(
        dpy, output, atom, 0, 1024, False, False, 0,  # AnyPropertyType
        byref(actual_type), byref(actual_format), byref(nitems), byref(bytes_after), byref(data),
    )
    if status != 0 or not actual_type.value:
        return None
    try:
        # Xlib hands format-32 data back as C longs.
        item = {8: ctypes.c_ubyte, 16: ctypes.c_short, 32: c_long}[actual_format.value]
        values = list(ctypes.cast(data, POINTER(item))[: nitems.value])
    finally:
        xlib.XFree(data)
    if actual_format.value == 32:
        # Normalize to the 32-bit signed values xrandr prints.
        values = [ctypes.c_int32(v).value for v in values]
    return actual_type.value, actual_format.value, values


def set_identity_ctm(dpy, output):
    """Set the output's CTM to identity. Return the previous value, or None if no CTM."""
    current = get_output_property(dpy, output, "CTM")
    if current is None:
        return None
    prop_type, _, old = current
    data = (c_long * len(IDENTITY_CTM))(*IDENTITY_CTM)
    atom = xlib.XInternAtom(dpy, b"CTM", False)
    _xrandr.XRRChangeOutputProperty(
        dpy, output, atom, prop_type, 32, xlib.PropModeReplace,
        ctypes.cast(data, POINTER(c_ubyte)), len(IDENTITY_CTM),
    )
    xlib.XSync(dpy, False)
    return old


def linear_ramp(size):
    return [i * 65535 // (size - 1) for i in range(size)]


def get_gamma(dpy, crtc):
    """Return the CRTC's (red, green, blue) gamma ramps as lists."""
    gamma = _xrandr.XRRGetCrtcGamma(dpy, crtc)
    try:
        g = gamma.contents
        return tuple(list(getattr(g, c)[: g.size]) for c in ("red", "green", "blue"))
    finally:
        _xrandr.XRRFreeGamma(gamma)


def set_linear_gamma(dpy, crtc):
    """Load a linear gamma ramp on the CRTC. Return True if it was not linear before."""
    size = _xrandr.XRRGetCrtcGammaSize(dpy, crtc)
    if size < 2:
        return False
    ramp = linear_ramp(size)
    changed = any(channel != ramp for channel in get_gamma(dpy, crtc))
    gamma = _xrandr.XRRAllocGamma(size)
    try:
        for channel in ("red", "green", "blue"):
            arr = getattr(gamma.contents, channel)
            for i, v in enumerate(ramp):
                arr[i] = v
        _xrandr.XRRSetCrtcGamma(dpy, crtc, gamma)
    finally:
        _xrandr.XRRFreeGamma(gamma)
    xlib.XSync(dpy, False)
    return changed


def output_is_neutral(dpy, output, crtc):
    """True if the output's CTM (if any) is identity and its gamma ramp is linear."""
    ctm = get_output_property(dpy, output, "CTM")
    if ctm is not None and ctm[2] != IDENTITY_CTM:
        return False
    size = _xrandr.XRRGetCrtcGammaSize(dpy, crtc)
    return size < 2 or all(channel == linear_ramp(size) for channel in get_gamma(dpy, crtc))


def neutralize_output(dpy, output, crtc, name):
    """Force identity CTM and linear gamma on one output; return messages describing changes."""
    messages = []
    old_ctm = set_identity_ctm(dpy, output)
    if old_ctm is not None and old_ctm != IDENTITY_CTM:
        messages.append(f"reset colour matrix (CTM) on {name} to identity (was {old_ctm})")
    if set_linear_gamma(dpy, crtc):
        messages.append(f"reset gamma ramp on {name} to linear")
    if not output_is_neutral(dpy, output, crtc):
        raise RuntimeError(f"could not make the colour pipeline of {name} neutral")
    return messages


def input_shape_rectangle_count(dpy, window):
    """Number of rectangles in the window's input shape (0 = transparent to input)."""
    xext = ctypes.CDLL("libXext.so.6")
    xext.XShapeGetRectangles.argtypes = [_Display, c_ulong, c_int, POINTER(c_int), POINTER(c_int)]
    xext.XShapeGetRectangles.restype = ctypes.c_void_p
    count, ordering = c_int(), c_int()
    rects = xext.XShapeGetRectangles(dpy, window, xsync.ShapeInput, byref(count), byref(ordering))
    if rects:
        xlib.XFree(rects)
    return count.value
