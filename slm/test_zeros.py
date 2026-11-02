"""Hold an all-zero image on the SLM until Enter is pressed."""

import numpy as np

from slmdisplay import SLM

with SLM(screen=1) as slm:
    image = np.zeros((slm.height, slm.width), dtype=np.uint8)
    slm.show(image)
    input(f"SLM ({slm.width}x{slm.height}) is all zeros. Press Enter to close...")
