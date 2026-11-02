"""Display single-channel 8-bit phase masks on a spatial light modulator."""

from ._protocol import ScreenInfo
from .display import SLM, DisplayProcessError, list_screens

__version__ = "0.1.0"

__all__ = ["SLM", "ScreenInfo", "DisplayProcessError", "list_screens", "__version__"]
