"""Figures for the oblique-registration talk."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tifffile
from PIL import Image, ImageFilter

FIG = Path(__file__).parent / "figures"
RAW = FIG / "raw"
ATLAS = Path(__file__).parents[1] / "notebooks/reference_images/hindbrain/mapzebrain/T_AVG_HuCD.tif"

# fish32 configuration
Z_TOP, Z_BOT = 242, 327
Y_TOP, Y_BOT = 250, 850
Y_LO, Y_HI = 100, 900  # atlas-y actually spanned by the plane (from the §5 QC plot)

INK = "#1B1B1F"
ACCENT = "#E4572E"
COOL = "#2E86AB"
MUTED = "#8A8A93"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.edgecolor": INK,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "savefig.dpi": 220,
    "figure.facecolor": "white",
})


def z_of_y(y):
    t = (np.asarray(y, float) - Y_TOP) / (Y_BOT - Y_TOP)
    return Z_TOP + t * (Z_BOT - Z_TOP)


def sagittal_cut():
    """Atlas sagittal view with the oblique cut drawn through it."""
    vol = tifffile.memmap(ATLAS)  # (z, y, x)
    nz, ny, nx = vol.shape
    sag = np.max(np.asarray(vol[:, :, nx // 2 - 60:nx // 2 + 60], dtype=np.float32), axis=2)
    lo, hi = np.percentile(sag, [2, 99.5])

    for name, annotate in (("fig_sagittal_cut", True), ("fig_hinge_anchors", "anchors")):
        fig, ax = plt.subplots(figsize=(9, 3.6))
        ax.imshow(sag, cmap="gray", vmin=lo, vmax=hi, aspect="auto",
                  extent=[0, ny, nz, 0], interpolation="bilinear")

        yy = np.array([Y_LO, Y_HI])
        ax.plot(yy, z_of_y(yy), color=ACCENT, lw=3, solid_capstyle="round",
                label="acquisition plane (oblique cut)")
        ax.axhline(z_of_y(Y_LO), color=COOL, lw=1.6, ls="--", alpha=.9,
                   label="a single flat atlas z-slice")

        if annotate == "anchors":
            for y, z, dx, ha in ((Y_TOP, Z_TOP, -14, "right"), (Y_BOT, Z_BOT, 16, "left")):
                ax.plot([y], [z], "o", ms=11, mfc="white", mec=ACCENT, mew=3, zorder=5)
                ax.annotate(f"y={y} → z={z}", (y, z), xytext=(dx, -30),
                            textcoords="offset points", ha=ha, va="top",
                            fontsize=12, fontweight="bold", color=ACCENT,
                            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=ACCENT, lw=1.5))
        else:
            bb = dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=.9)
            ax.annotate(f"z = {Z_TOP}", (Y_LO + 40, z_of_y(Y_LO + 40)), xytext=(6, -36),
                        textcoords="offset points", fontsize=13, fontweight="bold",
                        color=ACCENT, bbox=bb)
            ax.annotate(f"z = {Z_BOT}", (Y_HI - 40, z_of_y(Y_HI - 40)), xytext=(-52, 28),
                        textcoords="offset points", fontsize=13, fontweight="bold",
                        color=ACCENT, bbox=dict(boxstyle="round,pad=0.25", fc="white",
                                                ec="none", alpha=.9))
            ax.annotate("", xy=(ny + 40, Z_TOP), xytext=(ny + 40, Z_BOT),
                        arrowprops=dict(arrowstyle="<->", color=ACCENT, lw=2.2))
            ax.text(ny + 62, (Z_TOP + Z_BOT) / 2, f"{Z_BOT - Z_TOP} µm of depth\nin ONE plane",
                    fontsize=12.5, fontweight="bold", color=ACCENT, va="center")

        ax.set_xlim(0, ny + (30 if annotate == "anchors" else 260))
        ax.set_ylim(430, 110)
        ax.set_xlabel("atlas y  (rostral → caudal)", fontsize=12)
        ax.set_ylabel("atlas z (µm)", fontsize=12)
        ax.legend(loc="lower left", fontsize=10.5, framealpha=.93)
        fig.tight_layout()
        fig.savefig(FIG / f"{name}.png", bbox_inches="tight")
        plt.close(fig)
        print("wrote", name)


def staircase():
    """The argument: piecewise-constant masks vs the continuous ramp."""
    y = np.linspace(Y_LO, Y_HI, 800)
    z = z_of_y(y)

    def steps(n):
        edges = np.linspace(Y_LO, Y_HI, n + 1)
        out = np.empty_like(y)
        for i in range(n):
            m = (y >= edges[i]) & (y <= edges[i + 1])
            out[m] = z_of_y((edges[i] + edges[i + 1]) / 2)
        return out

    # --- slide 8: staircase vs ramp
    fig, (ax, axe) = plt.subplots(2, 1, figsize=(9, 5.6), sharex=True,
                                  gridspec_kw={"height_ratios": [2.4, 1]})
    ax.plot(y, steps(3), color=COOL, lw=3, label="masked bands (3 pieces) — what you register to")
    ax.plot(y, z, color=ACCENT, lw=3, ls="--", label="true depth of the plane")
    ax.set_ylabel("atlas z (µm)", fontsize=12)
    ax.legend(fontsize=11, loc="upper left", bbox_to_anchor=(0.01, 0.99), framealpha=.95)
    ax.set_ylim(205, 375)
    ax.grid(alpha=.18)

    for n, c in ((2, "#9AA0AC"), (3, COOL), (5, "#7FB2CC")):
        axe.plot(y, np.abs(steps(n) - z), color=c, lw=2.4 if n == 3 else 1.8,
                 label=f"{n} bands  (max {np.abs(steps(n)-z).max():.0f} µm)")
    axe.axhline(7, color=INK, lw=1.4, ls=":")
    axe.text(Y_LO + 15, 8.6, "one soma ≈ 7 µm", fontsize=10.5, style="italic")
    axe.set_ylabel("depth error (µm)", fontsize=12)
    axe.set_xlabel("position along the plane  (atlas y)", fontsize=12)
    axe.set_ylim(0, 44)
    axe.legend(fontsize=10, ncol=3, loc="upper center", framealpha=.95)
    axe.grid(alpha=.18)
    fig.tight_layout()
    fig.savefig(FIG / "fig_staircase.png", bbox_inches="tight")
    plt.close(fig)
    print("wrote fig_staircase")

    # --- slide 10: same axes, ramp only
    fig, ax = plt.subplots(figsize=(9, 4.0))
    ax.plot(y, steps(3), color="#D6D9E0", lw=3, label="masked bands (discarded)")
    ax.plot(y, z, color=ACCENT, lw=4, label="model the ramp directly")
    ax.set_ylabel("atlas z (µm)", fontsize=12)
    ax.set_xlabel("position along the plane  (atlas y)", fontsize=12)
    ax.legend(fontsize=11, loc="upper left", framealpha=.93)
    ax.grid(alpha=.18)
    fig.tight_layout()
    fig.savefig(FIG / "fig_ramp.png", bbox_inches="tight")
    plt.close(fig)
    print("wrote fig_ramp")


def _sagittal():
    vol = tifffile.memmap(ATLAS)
    nz, ny, nx = vol.shape
    sag = np.max(np.asarray(vol[:, :, nx // 2 - 60:nx // 2 + 60], dtype=np.float32), axis=2)
    return sag, nz, ny


def hinge_dark():
    """Anchors figure on a dark ground, for the dark hinge slide."""
    sag, nz, ny = _sagittal()
    lo, hi = np.percentile(sag, [2, 99.5])
    fig, ax = plt.subplots(figsize=(9, 3.4))
    fig.patch.set_facecolor("#12151C")
    ax.set_facecolor("#12151C")
    ax.imshow(sag, cmap="gray", vmin=lo, vmax=hi, aspect="auto",
              extent=[0, ny, nz, 0], interpolation="bilinear")
    yy = np.array([Y_LO, Y_HI])
    ax.plot(yy, z_of_y(yy), color=ACCENT, lw=3.2, solid_capstyle="round")
    for y, z, dx, ha in ((Y_TOP, Z_TOP, -14, "right"), (Y_BOT, Z_BOT, 16, "left")):
        ax.plot([y], [z], "o", ms=11, mfc="#12151C", mec=ACCENT, mew=3, zorder=5)
        ax.annotate(f"y={y} → z={z}", (y, z), xytext=(dx, -30), textcoords="offset points",
                    ha=ha, va="top", fontsize=12.5, fontweight="bold", color="white",
                    bbox=dict(boxstyle="round,pad=0.35", fc=ACCENT, ec="none"))
    ax.set_xlim(0, ny + 30)
    ax.set_ylim(430, 110)
    ax.set_xlabel("atlas y  (rostral → caudal)", fontsize=12, color="#C9CFDA")
    ax.set_ylabel("atlas z (µm)", fontsize=12, color="#C9CFDA")
    ax.tick_params(colors="#8C93A1")
    for sp in ax.spines.values():
        sp.set_color("#39414F")
    fig.tight_layout()
    fig.savefig(FIG / "fig_hinge_anchors_dark.png", bbox_inches="tight", facecolor="#12151C")
    plt.close(fig)
    print("wrote fig_hinge_anchors_dark")


def plane_stack():
    """11 parallel oblique planes, 20 µm apart — one affine, shifted in z."""
    sag, nz, ny = _sagittal()
    lo, hi = np.percentile(sag, [2, 99.5])
    fig, ax = plt.subplots(figsize=(9, 4.0))
    ax.imshow(sag, cmap="gray", vmin=lo, vmax=hi, aspect="auto",
              extent=[0, ny, nz, 0], interpolation="bilinear")
    yy = np.array([Y_LO, Y_HI])
    for k in range(-5, 6):
        z = z_of_y(yy) + 20 * k
        if k == 0:
            ax.plot(yy, z, color=ACCENT, lw=3.2, zorder=4)
        else:
            ax.plot(yy, z, color="#7FB2CC", lw=1.7, alpha=.95, zorder=3)
    ax.annotate("", xy=(Y_HI + 55, z_of_y(Y_HI)), xytext=(Y_HI + 55, z_of_y(Y_HI) + 20),
                arrowprops=dict(arrowstyle="<->", color=INK, lw=1.8))
    ax.text(Y_HI + 72, z_of_y(Y_HI) + 10, "Δz = 20 µm\n= 20 atlas voxels",
            fontsize=11.5, fontweight="bold", color=INK, va="center",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none"))
    ax.annotate("reference plane", (Y_LO + 60, z_of_y(Y_LO + 60)), xytext=(10, -34),
                textcoords="offset points", fontsize=11.5, fontweight="bold", color=ACCENT,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=ACCENT, lw=1.4),
                arrowprops=dict(arrowstyle="->", color=ACCENT, lw=1.6))
    ax.set_xlim(-40, ny + 400)
    ax.set_ylim(z_of_y(Y_HI) + 125, z_of_y(Y_LO) - 130)
    ax.set_xlabel("atlas y  (rostral → caudal)", fontsize=12)
    ax.set_ylabel("atlas z (µm)", fontsize=12)
    ax.set_title("11 planes, one affine, one tilt — only z shifts",
                 fontsize=13.5, fontweight="bold", pad=10)
    fig.tight_layout()
    fig.savefig(FIG / "fig_plane_stack.png", bbox_inches="tight")
    plt.close(fig)
    print("wrote fig_plane_stack")


def crops():
    """Panel crops out of the notebook QC figures."""
    def crop(src, box, out, pad=6, trim_white=False, trim_panel=False):
        im = Image.open(RAW / src)
        w, h = im.size
        x0, y0, x1, y1 = [int(round(v * s)) for v, s in zip(box, (w, h, w, h))]
        im = im.crop((max(0, x0 - pad), max(0, y0 - pad),
                      min(w, x1 + pad), min(h, y1 + pad)))
        if trim_white:
            rs, cs = np.nonzero(np.asarray(im.convert("L")) < 245)
            im = im.crop((cs.min(), rs.min(), cs.max() + 1, rs.max() + 1))
        if trim_panel:
            # keep only the dark image block, dropping matplotlib's own panel title
            a = np.asarray(im.convert("L")).astype(float)
            r = np.nonzero(a.mean(1) < 128)[0]
            c = np.nonzero(a.mean(0) < 128)[0]
            im = im.crop((c.min(), r.min(), c.max() + 1, r.max() + 1))
        im.save(FIG / out)
        print("wrote", out, Image.open(FIG / out).size)

    # cell13: row 1 = oblique | axis-aligned | experiment ; row 2 = atlas z242 | z327
    crop("cell13_0.png", (0.00, 0.00, 0.62, 0.52), "fig_oblique_vs_axis.png")
    # slide 15 wants the two image blocks bare — the notebook draws a grid on one
    # panel's axes and not the other, which reads as a difference in the data
    src13 = Image.open(RAW / "cell13_0.png").convert("RGB")
    # this panel carries a yellow reference grid the other one lacks; the grid is
    # yellow-only, so the blue channel drops its colour and a 3px median erases
    # the remaining 1px alpha residue without touching the anatomy
    ob = src13.crop((67, 32, 278, 369)).split()[2].filter(ImageFilter.MedianFilter(3))
    Image.merge("RGB", (ob,) * 3).save(FIG / "fig_cut_oblique.png")
    src13.crop((534, 32, 744, 368)).save(FIG / "fig_cut_axis.png")
    print("wrote fig_cut_oblique.png / fig_cut_axis.png")
    # piecewise masking: the notebook renders 8 panels; keep the 3 that carry the
    # argument (masked band, composite, merge) and label them on the slide instead
    src34 = Image.open(RAW / "cell34_0.png").convert("RGB")
    for name, c0 in (("mask", 82), ("composite", 827), ("merge", 1200)):
        src34.crop((c0, 79, c0 + 206, 408)).save(FIG / f"fig_piece_{name}.png")
    print("wrote fig_piece_mask/composite/merge.png")
    crop("cell13_0.png", (0.02, 0.545, 0.19, 0.99), "fig_atlas_ztop.png", pad=0, trim_panel=True)
    crop("cell13_0.png", (0.36, 0.545, 0.51, 0.99), "fig_atlas_zbot.png", pad=0, trim_panel=True)
    crop("cell13_0.png", (0.64, 0.02, 0.85, 0.47), "fig_experiment.png", pad=0, trim_panel=True)
    # title hero: the Elastix-warped overlay panel only (already on black)
    crop("cell19_0.png", (0.60, 0.09, 0.93, 1.00), "fig_hero.png", pad=0, trim_white=True)
    # ROI region colouring: plane 0 only (image blocks, no baked-in titles)
    src = Image.open(RAW / "cell40_0.png").convert("RGB")
    left, right = src.crop((22, 81, 259, 459)), src.crop((336, 81, 573, 459))
    gap = 14
    pair = Image.new("RGB", (left.width + gap + right.width, left.height), "white")
    pair.paste(left, (0, 0)); pair.paste(right, (left.width + gap, 0))
    pair.save(FIG / "fig_rois_crop.png")
    print("wrote fig_rois_crop.png", pair.size)
    # band pick: drop the notebook's own suptitle above the axes
    crop("cell09_0.png", (0.0, 0.055, 1.0, 1.0), "fig_band_pick.png", pad=0)
    # pre/post-Elastix overlay: keep the two image blocks only, label them on the slide
    crop("cell19_0.png", (0.075, 0.055, 0.400, 0.985), "fig_pre_elastix.png", pad=0)
    crop("cell19_0.png", (0.632, 0.055, 0.957, 0.985), "fig_post_elastix.png", pad=0)

    for src, out in (("cell05_0.png", "fig_std_plane.png"),
                     ("cell17_0.png", "fig_qc_merge.png"),
                     ("cell19_0.png", "fig_qc_overlay.png"),
                     ("cell19_1.png", "fig_depth_vs_y.png"),
                     ("cell34_0.png", "fig_piecewise.png"),
                     ("cell40_0.png", "fig_rois.png")):
        Image.open(RAW / src).save(FIG / out)
        print("wrote", out)


def all_planes():
    """cell25 is an 11-row x 3-col strip; transpose it to 3 rows x 11 columns
    so it fits a landscape slide and each panel stays legible."""
    src = Image.open(RAW / "cell25_0.png").convert("RGB")
    cols = [(10, 197), (334, 521), (658, 845)]
    rows = [(80, 379), (416, 714), (752, 1050), (1087, 1386), (1423, 1721),
            (1759, 2057), (2094, 2392), (2430, 2728), (2765, 3064),
            (3101, 3399), (3437, 3735)]
    ph, gap = 298, 5
    tiles = [[src.crop((c0, r0, c1, r0 + ph)) for r0, _ in rows] for c0, c1 in cols]
    tw, th = tiles[0][0].size
    n = len(rows)
    out = Image.new("RGB", (n * tw + (n - 1) * gap, 3 * th + 2 * gap), "white")
    for r, row in enumerate(tiles):
        for i, t in enumerate(row):
            out.paste(t, (i * (tw + gap), r * (th + gap)))
    out.save(FIG / "fig_all_planes.png")
    print("wrote fig_all_planes.png", out.size)


if __name__ == "__main__":
    crops()
    all_planes()
    staircase()
    sagittal_cut()
    hinge_dark()
    plane_stack()
