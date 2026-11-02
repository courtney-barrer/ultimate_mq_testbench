from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt

filename = "fourier_highfreq.fits"
display_bin = 4
plt.close("all")


def camera_average(hdu, binning=4):
    """Average exposures and spatial blocks, reading one strip at a time."""
    height, width = hdu.shape[-2:]
    nframes = hdu.shape[0] if len(hdu.shape) == 3 else 1

    result = np.empty(
        ((height + binning - 1) // binning,
         (width + binning - 1) // binning),
        dtype=np.float32,
    )
    starts = np.arange(0, width, binning)
    counts = np.minimum(binning, width - starts)

    for row, y in enumerate(range(0, height, binning)):
        end = min(y + binning, height)
        strip = np.zeros((end - y, width), dtype=np.float64)

        for frame in range(nframes):
            if len(hdu.shape) == 3:
                strip += hdu.section[frame, y:end, :]
            else:
                strip += hdu.section[y:end, :]

        profile = (strip / nframes).mean(axis=0)
        result[row] = np.add.reduceat(profile, starts) / counts

    return result, nframes


with fits.open(filename, mode="readonly", memmap=False) as hd:
    commands = [
        (i, h) for i, h in enumerate(hd)
        if h.name.strip().upper() == "SLM_CMD"
    ]

    best = {}

    for camera_index, camera in enumerate(hd):
        if camera.name.strip().upper() != "CAM_FRAMES":
            continue

        cmdid = camera.header.get("CMDID")
        matches = [
            (i, h) for i, h in commands
            if cmdid is not None and h.header.get("CMDID") == cmdid
        ]

        if not matches:
            extver = camera.header.get("EXTVER")
            matches = [
                (i, h) for i, h in commands
                if extver is not None
                and h.header.get("EXTVER") == extver
            ]

        if len(matches) != 1:
            print(f"Skipping HDU {camera_index}: ambiguous/missing command")
            continue

        command_index, command = matches[0]
        header = dict(command.header)
        header.update(dict(camera.header))

        if not all(k in header for k in ("MODEID", "AMPL", "FX", "FY")):
            continue

        mode = int(header["MODEID"])
        amplitude = float(header["AMPL"])
        fx, fy = float(header["FX"]), float(header["FY"])

        if amplitude == 0 or not np.isfinite([amplitude, fx, fy]).all():
            continue

        entry = dict(
            command_index=command_index,
            camera_index=camera_index,
            mode=mode,
            amplitude=amplitude,
            fx=fx,
            fy=fy,
            phase=header.get("PHASE"),
            frequency=np.hypot(fx, fy),
        )

        # Largest absolute amplitude per mode; prefer positive on ties.
        previous = best.get(mode)
        if previous is None or (abs(amplitude), amplitude) > (
            abs(previous["amplitude"]), previous["amplitude"]
        ):
            best[mode] = entry

    chosen = sorted(
        best.values(),
        key=lambda item: (-item["frequency"], item["mode"]),
    )[:3]

    if not chosen:
        raise ValueError("No matched nonzero Fourier probes found.")

    print("File:", filename)
    for item in chosen:
        print(
            f"Mode {item['mode']}: "
            f"(fx, fy)=({item['fx']:g}, {item['fy']:g}), "
            f"amplitude={item['amplitude']:+g}, "
            f"camera HDU={item['camera_index']}"
        )

    fig, axes = plt.subplots(
        len(chosen), 2,
        figsize=(12, 3.3 * len(chosen)),
        squeeze=False,
        gridspec_kw={"wspace": 0.015, "hspace": 0.035},
    )
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.94)
    fig.text(0.25, 0.97, "SLM command", ha="center", fontsize=13)
    fig.text(0.75, 0.97, "Mean camera image", ha="center", fontsize=13)

    for row, item in enumerate(chosen):
        command = hd[item["command_index"]]
        camera = hd[item["camera_index"]]

        command_image = command.section[::display_bin, ::display_bin]
        mean_image, nframes = camera_average(camera, display_bin)

        axes[row, 0].imshow(
            command_image,
            cmap="gray",
            vmin=0, vmax=255,
            origin="upper",
            aspect="auto",
            interpolation="nearest",
        )

        finite = mean_image[np.isfinite(mean_image)]
        lo, hi = (
            np.percentile(finite, [1, 99])
            if finite.size else (0, 1)
        )

        axes[row, 1].imshow(
            mean_image,
            cmap="viridis",
            vmin=lo, vmax=max(hi, lo + 1),
            origin="upper",
            aspect="auto",
            interpolation="nearest",
        )

        label = (
            f"Mode {item['mode']}  "
            f"f=({item['fx']:g}, {item['fy']:g})  "
            f"A={item['amplitude']:+g}"
        )
        if item["phase"] is not None:
            label += f"  phase={item['phase']}°"

        for ax in axes[row]:
            ax.set_axis_off()

        for ax, text in (
            (axes[row, 0], label),
            (axes[row, 1],
             f"HDU {item['camera_index']} · mean of {nframes} frames"),
        ):
            ax.text(
                0.02, 0.97, text,
                transform=ax.transAxes,
                va="top",
                color="white",
                fontsize=10,
                bbox=dict(
                    facecolor="black", alpha=0.65,
                    edgecolor="none", pad=3,
                ),
            )

plt.show()