#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import pylablib

# Requires your existing edit in dcamapi4_lib.py:
# "dcamapi.dll" -> "libdcamapi.so"
pylablib.par["devices/only_windows_dlls"] = False
pylablib.par["devices/dlls/dcamapi"] = "/usr/local/lib"

from pylablib.devices import DCAM


# -------------------------------------------------------------------------
# User settings
# -------------------------------------------------------------------------
CAMERA_INDEX = 0
EXPOSURES_S = [1e-5, 5e-5, 1e-4]
ROI_SIZE = 512
N_FRAMES = 10

# First run: leave this as None to retain the current readout speed.
# The script prints the available numeric IDs and manufacturer labels.
# Set this to one of those IDs on a subsequent run.
READOUT_SPEED_ID = None


# -------------------------------------------------------------------------
# 1. Connect and inspect the camera
# -------------------------------------------------------------------------
cam = DCAM.DCAMCamera(CAMERA_INDEX)

try:
    print("\nCAMERA")
    print(cam.get_device_info())

    # Internal triggering: acquisition does not require an external signal.
    cam.set_trigger_mode("int")

    print("\nINITIAL SETTINGS")
    print("Exposure [s]:", cam.get_exposure())
    print("ROI:", cam.get_roi())

    # Generic DCAM properties supplement pylablib's convenience methods.
    # A missing property returns None here.
    for name in [
        "READOUT SPEED",
        "SENSOR MODE",
        "BINNING",
        "TRIGGER SOURCE",
    ]:
        attr = cam.get_attribute(name, error_on_missing=False)
        if attr is not None:
            if attr.readable:
                print(f"{name}: {attr.get_value(enum_as_str=True)}")
            if attr.kind == "enum":
                print("  Available {ID: label}:", attr.ilabels)

    # Uncomment to inspect all readable property values:
    # print(cam.get_all_attribute_values(enum_as_str=True))


    # ---------------------------------------------------------------------
    # 2. Inspect and optionally change readout speed
    # ---------------------------------------------------------------------
    print("\nREADOUT SPEED")

    readout = cam.get_attribute(
        "READOUT SPEED", error_on_missing=False
    )

    if readout is None:
        print("This camera does not expose READOUT SPEED.")

    else:
        print("Available {ID: label}:", readout.ilabels)
        print("Current:", readout.get_value(enum_as_str=True))

        if READOUT_SPEED_ID is not None:
            if not readout.writable:
                raise RuntimeError("READOUT SPEED is not writable.")

            if READOUT_SPEED_ID not in readout.ivalues:
                raise ValueError(
                    f"Choose a readout ID from {readout.ilabels}"
                )

            # Change settings while acquisition is stopped.
            cam.clear_acquisition()
            cam.set_attribute_value(
                "READOUT SPEED", READOUT_SPEED_ID
            )

            print("Selected:", readout.get_value(enum_as_str=True))

    # SENSOR MODE is a separate property from READOUT SPEED.
    # Do not change it blindly: it can alter acquisition behaviour.


    # ---------------------------------------------------------------------
    # 3. Select the full detector with 1x1 binning
    # ---------------------------------------------------------------------
    cam.clear_acquisition()
    cam.set_roi(hbin=1, vbin=1)

    full_roi = cam.get_roi()
    print("\nFULL-DETECTOR ROI:", full_roi)

    # ROI tuple:
    # (horizontal_start, horizontal_end,
    #  vertical_start, vertical_end, horizontal_bin, vertical_bin)
    #
    # End coordinates are exclusive.
    # NumPy images are indexed frame[row, column] = frame[y, x].


    # ---------------------------------------------------------------------
    # 4. Change exposure and capture a frame at each setting
    # ---------------------------------------------------------------------
    print("\nEXPOSURE TEST")

    exposure_images = []
    actual_exposures = []

    for requested_exposure in EXPOSURES_S:
        cam.clear_acquisition()
        cam.set_exposure(requested_exposure)

        # Always read back: the camera may quantize the requested value.
        actual_exposure = cam.get_exposure()
        frame = cam.snap(timeout=5.0).copy()

        exposure_images.append(frame)
        actual_exposures.append(actual_exposure)

        print(
            f"Requested={requested_exposure:.6f} s, "
            f"actual={actual_exposure:.6f} s, "
            f"shape={frame.shape}, dtype={frame.dtype}, "
            f"mean={frame.mean():.2f}, max={frame.max()}"
        )


    # ---------------------------------------------------------------------
    # 5. Set a centred ROI
    # ---------------------------------------------------------------------
    cam.clear_acquisition()

    x0, x1, y0, y1, _, _ = full_roi
    width = min(ROI_SIZE, x1 - x0)
    height = min(ROI_SIZE, y1 - y0)

    left = x0 + (x1 - x0 - width) // 2
    top = y0 + (y1 - y0 - height) // 2

    cam.set_roi(
        left, left + width,
        top, top + height,
        hbin=1, vbin=1,
    )

    # Camera alignment constraints may adjust the requested boundaries.
    print("\nACTUAL ROI:", cam.get_roi())

    cam.set_exposure(1e-4)
    roi_frame = cam.snap(timeout=5.0).copy()
    print("ROI frame shape:", roi_frame.shape)


    # ---------------------------------------------------------------------
    # 6. Acquire a short sequence at the current ROI and exposure
    # ---------------------------------------------------------------------
    cam.clear_acquisition()

    frames = np.asarray(
        cam.grab(nframes=N_FRAMES, frame_timeout=5.0)
    ).copy()

    print("\nSEQUENCE")
    print("Array shape [frame, row, column]:", frames.shape)
    print("Pixel dtype:", frames.dtype)

    # Float conversion prevents integer overflow when combining frames.
    mean_frame = frames.astype(np.float64).mean(axis=0)

    # Optional saving; preserves the original pixel values:
    # np.save("orca_sequence.npy", frames)
    # np.save("orca_mean.npy", mean_frame)

    # Optional binning example, if supported by your camera:
    # cam.clear_acquisition()
    # cam.set_roi(hbin=2, vbin=2)
    # print("Actual binned ROI:", cam.get_roi())
    # binned_frame = cam.snap(timeout=5.0).copy()

finally:
    # Also releases acquisition resources if an exception occurs.
    cam.close()


# -------------------------------------------------------------------------
# 7. Display after releasing the camera
# -------------------------------------------------------------------------
# Use a shared intensity scale to compare the exposures.
vmin = min(float(im.min()) for im in exposure_images)
vmax = max(float(im.max()) for im in exposure_images)

fig, axes = plt.subplots(
    1, len(exposure_images),
    figsize=(12, 4),
    constrained_layout=True,
    squeeze=False,
)

for ax, im, exposure in zip(
    axes[0], exposure_images, actual_exposures
):
    image = ax.imshow(im, cmap="gray", vmin=vmin, vmax=vmax)
    ax.set_title(f"Exposure: {exposure * 1000:.3f} ms")
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")

fig.colorbar(image, ax=list(axes[0]), label="Pixel value [ADU]")

fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
image = ax.imshow(mean_frame, cmap="gray")
ax.set_title(f"ROI average of {len(frames)} frames")
ax.set_xlabel("Column")
ax.set_ylabel("Row")
fig.colorbar(image, ax=ax, label="Mean pixel value [ADU]")

plt.show()