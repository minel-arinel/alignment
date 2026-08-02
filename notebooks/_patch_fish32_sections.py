#!/usr/bin/env python3
"""Fish32: row-band sections + atlas z 245/285/295/318 (no TIFF masks)."""
import json
from pathlib import Path

nb_path = Path(__file__).resolve().parent / "brain_alignment_fish32.ipynb"
nb = json.loads(nb_path.read_text())


def set_cell(idx, source, cell_type=None):
    if cell_type:
        nb["cells"][idx]["cell_type"] = cell_type
    lines = source.splitlines(keepends=True)
    if lines and not lines[-1].endswith("\n"):
        lines[-1] += "\n"
    nb["cells"][idx]["source"] = lines


set_cell(
    0,
    "".join(nb["cells"][0]["source"]).replace(
        "# Piecewise registration: TIFF masks on experiment + atlas (blackout; optional elastix CLI masks).",
        "# Piecewise registration: row sections on experiment + atlas z per section (row blackout; no TIFF masks).",
    ),
)

set_cell(
    5,
    "".join(nb["cells"][5]["source"]).replace(
        "Re-run the **piecewise region masks** cell below after changing this angle.",
        "Re-run the **experiment section split** cell below after changing this angle.",
    ),
)

set_cell(
    7,
    """### Split experiment into sections (row edges, no TIFF masks)

**Fish32** uses horizontal section boundaries on embedded **`plot_img`** (1024², row **0** = top).

Edit **`EXPERIMENT_SECTION_ROW_EDGES`**: strictly increasing `[y0, y1, …, yN]` with **`y0 = 0`**, **`yN = 1024`**.

Stores **`EXPERIMENT_SECTIONS`** for piecewise registration. Atlas z indices are set in the next cell.
""",
    "markdown",
)

set_cell(
    8,
    r'''# Requires: `plot_img` from fine-rotation cell above.

# --- fish32 plane 0: row boundaries on embedded 1024² (row 0 = top) ---
EXPERIMENT_SECTION_ROW_EDGES = [
    0,
    220,
    525,
    790,
    1024,
]

EXPERIMENT_SECTION_NAMES = None

_h, _w = plot_img.shape[:2]
EXPERIMENT_SECTION_ROW_EDGES = [int(y) for y in EXPERIMENT_SECTION_ROW_EDGES]

if len(EXPERIMENT_SECTION_ROW_EDGES) < 2:
    raise ValueError("EXPERIMENT_SECTION_ROW_EDGES needs at least [0, height]")
if EXPERIMENT_SECTION_ROW_EDGES[0] != 0 or EXPERIMENT_SECTION_ROW_EDGES[-1] != _h:
    raise ValueError(
        f"Edges must start at 0 and end at image height {_h}. "
        f"Got [{EXPERIMENT_SECTION_ROW_EDGES[0]}, …, {EXPERIMENT_SECTION_ROW_EDGES[-1]}]"
    )
if any(
    EXPERIMENT_SECTION_ROW_EDGES[i] >= EXPERIMENT_SECTION_ROW_EDGES[i + 1]
    for i in range(len(EXPERIMENT_SECTION_ROW_EDGES) - 1)
):
    raise ValueError("EXPERIMENT_SECTION_ROW_EDGES must be strictly increasing")

_n_sections = len(EXPERIMENT_SECTION_ROW_EDGES) - 1
if EXPERIMENT_SECTION_NAMES is not None:
    if len(EXPERIMENT_SECTION_NAMES) != _n_sections:
        raise ValueError(
            f"EXPERIMENT_SECTION_NAMES length {len(EXPERIMENT_SECTION_NAMES)} != {_n_sections} sections"
        )
    _names = list(EXPERIMENT_SECTION_NAMES)
else:
    _names = [f"section_{i}" for i in range(_n_sections)]

EXPERIMENT_SECTIONS = []
for _i in range(_n_sections):
    EXPERIMENT_SECTIONS.append(
        {
            "name": str(_names[_i]),
            "row_start": int(EXPERIMENT_SECTION_ROW_EDGES[_i]),
            "row_end": int(EXPERIMENT_SECTION_ROW_EDGES[_i + 1]),
        }
    )

print(f"{_n_sections} sections on embedded experiment ({_h}×{_w}):")
for _b in EXPERIMENT_SECTIONS:
    print(f"  {_b['name']}: rows [{_b['row_start']}, {_b['row_end']})")

fig, ax = plt.subplots(1, 1, figsize=(8, 8))
ax.imshow(plot_img, cmap="gray", vmin=target_vmin, vmax=target_vmax)
for _y in EXPERIMENT_SECTION_ROW_EDGES[1:-1]:
    ax.axhline(_y - 0.5, color="yellow", linewidth=0.8, alpha=0.95)
for _b in EXPERIMENT_SECTIONS:
    _yc = (_b["row_start"] + _b["row_end"]) / 2.0
    ax.text(
        _w * 0.02,
        _yc,
        _b["name"],
        color="yellow",
        fontsize=9,
        va="center",
        ha="left",
    )
ax.set_title(f"experiment plane {selected_plane} — {_n_sections} sections")
ax.axis("off")
plt.tight_layout()
plt.show()
''',
)

set_cell(
    9,
    """### Load mapZebrain atlas slices (one z per section)

Requires **`EXPERIMENT_SECTIONS`** from the section-split cell above.

**Region → atlas z** (edit **`MAPZEBRAIN_Z_PER_SECTION`** if needed):

| Region | Section | z |
|--------|---------|---|
| 1 | `section_0` (top) | 245 |
| 2 | `section_1` | 285 |
| 3 | `section_2` | 295 |
| 4 | `section_3` (bottom) | 318 |

Stores **`mapzebrain_by_section`** for piecewise registration.
""",
    "markdown",
)

src10 = "".join(nb["cells"][10]["source"])
src10 = src10.replace(
    "# Requires: EXPERIMENT_SECTIONS / MAPZEBRAIN_Z_PER_SECTION from the mask cell above.",
    "# Requires: EXPERIMENT_SECTIONS from the section-split cell above.",
)
src10 = src10.replace(
    "if 'EXPERIMENT_SECTIONS' not in globals() or 'MAPZEBRAIN_Z_PER_SECTION' not in globals():\n    raise RuntimeError('Run the piecewise mask cell above first.')",
    "if 'EXPERIMENT_SECTIONS' not in globals():\n    raise RuntimeError('Run the experiment section-split cell above first.')",
)
if "MAPZEBRAIN_Z_PER_SECTION = [" not in src10:
    src10 = src10.replace(
        "mapzebrain_img_path = Path(r'./reference_images/hindbrain/mapzebrain/T_AVG_HuCD.tif')\n\n",
        "mapzebrain_img_path = Path(r'./reference_images/hindbrain/mapzebrain/T_AVG_HuCD.tif')\n\n"
        "MAPZEBRAIN_Z_PER_SECTION = [\n"
        "    245,  # region 1 — section_0 (top)\n"
        "    285,  # region 2 — section_1\n"
        "    295,  # region 3 — section_2\n"
        "    318,  # region 4 — section_3 (bottom)\n"
        "]\n\n",
    )
else:
    import re

    src10 = re.sub(
        r"MAPZEBRAIN_Z_PER_SECTION = \[[\s\S]*?\]\n",
        "MAPZEBRAIN_Z_PER_SECTION = [\n"
        "    245,  # region 1 — section_0 (top)\n"
        "    285,  # region 2 — section_1\n"
        "    295,  # region 3 — section_2\n"
        "    318,  # region 4 — section_3 (bottom)\n"
        "]\n",
        src10,
        count=1,
    )
src10 = src10.replace(
    "_n_sections = len(EXPERIMENT_SECTIONS)\nif len(MAPZEBRAIN_Z_PER_SECTION) != _n_sections:",
    "_n_sections = len(EXPERIMENT_SECTIONS)\nMAPZEBRAIN_Z_PER_SECTION = [int(z) for z in MAPZEBRAIN_Z_PER_SECTION]\nif len(MAPZEBRAIN_Z_PER_SECTION) != _n_sections:",
)
src10 = src10.replace("mask rows", "rows")
src10 = src10.replace("experiment mask rows", "experiment rows")
set_cell(10, src10)

# Atlas metric: match experiment row bands (no TIFF masks)
sections_nb = json.loads(
    (Path(__file__).resolve().parent / "brain_alignment_from_tiff_sections.ipynb").read_text()
)
src11 = "".join(sections_nb["cells"][11]["source"]).replace(
    'ATLAS_METRIC_ROW_MODE = "manual"',
    'ATLAS_METRIC_ROW_MODE = "match_experiment_section"',
)
set_cell(11, src11)

set_cell(
    14,
    """### Alignment (piecewise only)

This notebook uses **piecewise** registration (one Elastix run per experiment row section + atlas z). There is no separate global single-slice `register_image2` step.

Continue to **Piecewise alignment** below after the unaligned preview.
""",
    "markdown",
)

set_cell(
    15,
    """## Piecewise alignment (row sections → atlas z)

One Elastix run per section; experiment and atlas images are zeroed outside each section's row band (no TIFF masks).
""",
    "markdown",
)

set_cell(
    16,
    """### Piecewise config (sections + atlas z)

**`REGION_SECTIONS`** is built from **`EXPERIMENT_SECTIONS`** + **`MAPZEBRAIN_Z_PER_SECTION`**.
""",
    "markdown",
)

set_cell(17, "".join(sections_nb["cells"][17]["source"]))
set_cell(18, "".join(sections_nb["cells"][18]["source"]))

set_cell(
    19,
    """### Run piecewise Elastix (one transform folder per section)

Blacks out experiment row section + matching atlas ROI, then runs Elastix.
""",
    "markdown",
)

set_cell(20, "".join(sections_nb["cells"][20]["source"]))
set_cell(21, "".join(sections_nb["cells"][21]["source"]))
set_cell(22, "".join(sections_nb["cells"][22]["source"]))

src23 = "".join(nb["cells"][23]["source"])
src23 = src23.replace(
    "section_0 = top experiment rows → first z in MAPZEBRAIN_Z_PER_SECTION",
    "section_0 (top) → z=245 … section_3 (bottom) → z=318",
)
set_cell(23, src23)

nb_path.write_text(json.dumps(nb, indent=1))
print(f"Updated {nb_path}")
