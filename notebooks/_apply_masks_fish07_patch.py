#!/usr/bin/env python3
"""Apply fish07 plane3 piecewise mask layout to brain_alignment_from_tiff_masks.ipynb."""
import json
import re
from pathlib import Path

nb_path = Path(__file__).resolve().parent / "brain_alignment_from_tiff_masks.ipynb"
nb = json.loads(nb_path.read_text())


def set_cell(idx, source, cell_type=None):
    if cell_type:
        nb["cells"][idx]["cell_type"] = cell_type
    lines = source.splitlines(keepends=True)
    if lines and not lines[-1].endswith("\n"):
        lines[-1] += "\n"
    nb["cells"][idx]["source"] = lines


src0 = "".join(nb["cells"][0]["source"])
src0 = src0.replace(
    "# Piecewise registration uses SimpleITK Elastix + image blackout (no CLI masks required).",
    "# Piecewise registration: TIFF masks on experiment + atlas (blackout; optional elastix CLI masks).",
)
set_cell(0, src0)

src2 = "".join(nb["cells"][2]["source"])
src2 = src2.replace(
    "fish_name = 'elavl3H2BGCaMP6s_7dpf_DOI_50ugml_32_20251007'",
    "fish_name = 'elavl3H2BGCaMP6s_8dpf_DOI_0ugml_7_20250924'",
)
src2 = src2.replace("selected_plane = 0", "selected_plane = 3")
set_cell(2, src2)

set_cell(
    5,
    "".join(nb["cells"][5]["source"]).replace(
        "Re-run the **registration masks** cell below after changing this angle.",
        "Re-run the **piecewise region masks** cell below after changing this angle.",
    ),
)

set_cell(
    7,
    """### Piecewise regions from TIFF masks (experiment + atlas)

**Fish07 plane 3** uses a custom layout:

1. **Top (`section_0`)** — mask TIFF `fish07_plane3_mask1.tif`
2. **Upper middle (`section_1`)** — embedded rows `[0, 660)` minus the top mask
3. **Lower middle (`section_2`)** — rows `[660, 885)`
4. **Bottom (`section_3`)** — rows `[885, 1024)`

Row index 0 = top of `plot_img` (1024²). Atlas masks for row bands use the same row intervals on each z-slice embed canvas (full width); region 0 uses the experiment mask row span on the atlas.

Stores **`experiment_masks_by_section`**, **`EXPERIMENT_SECTIONS`**, **`PIECEWISE_REGION_SPECS`**, **`MAPZEBRAIN_Z_PER_SECTION`**.
""",
    "markdown",
)

set_cell(
    8,
    r'''# Requires: `plot_img`, `experiment_std_scaled`, `EXPERIMENT_CW_DEGREES` from cells above.

# --- fish07 plane 3: experiment regions on embedded 1024² (row 0 = top) ---
_REGION1_MASK_PATH = Path(
    "/Volumes/Kobi/DOI_animals_good/elavl3H2BGCaMP6s_8dpf_DOI_0ugml_7_20250924"
    "/mcorr_tiff_outputs/fish07_plane3_mask1.tif"
)
_ROW_BAND_MID = 660    # section_1 vs section_2 boundary
_ROW_BAND_LOWER = 885  # section_2 vs section_3 boundary

# Atlas z per region (tune if needed)
MAPZEBRAIN_Z_PER_SECTION = [245, 285, 295, 318]

_h, _w = int(plot_img.shape[0]), int(plot_img.shape[1])
if (_h, _w) != (1024, 1024):
    raise ValueError(f"Expected plot_img 1024×1024, got {plot_img.shape[:2]}")

if not _REGION1_MASK_PATH.is_file():
    raise FileNotFoundError(f"Top-region mask not found: {_REGION1_MASK_PATH}")

_cw = float(globals().get("EXPERIMENT_CW_DEGREES", 0) or 0)
_mask_top = sitkalignment.prepare_experiment_mask_for_plot_img(
    _REGION1_MASK_PATH,
    plot_img,
    experiment_std_scaled=experiment_std_scaled,
    cw_degrees=_cw,
)
_band_upper = sitkalignment.registration_mask_embedded_rectangle(
    plot_img.shape, 0, _ROW_BAND_MID, 0, _w
)
_mask_upper_mid = (
    _band_upper.astype(np.uint8) * (1 - _mask_top.astype(np.uint8))
).astype(np.uint8)
_mask_lower_mid = sitkalignment.registration_mask_embedded_rectangle(
    plot_img.shape, _ROW_BAND_MID, _ROW_BAND_LOWER, 0, _w
)
_mask_bottom = sitkalignment.registration_mask_embedded_rectangle(
    plot_img.shape, _ROW_BAND_LOWER, _h, 0, _w
)

experiment_masks_by_section = {
    "section_0": _mask_top,
    "section_1": _mask_upper_mid,
    "section_2": _mask_lower_mid,
    "section_3": _mask_bottom,
}

PIECEWISE_REGION_SPECS = [
    {
        "name": "section_0",
        "mapzebrain_z": MAPZEBRAIN_Z_PER_SECTION[0],
        "experiment_mask": _REGION1_MASK_PATH,
        "atlas_mask": None,
    },
    {
        "name": "section_1",
        "mapzebrain_z": MAPZEBRAIN_Z_PER_SECTION[1],
        "experiment_mask": None,
        "atlas_mask": None,
        "atlas_row_band": (0, _ROW_BAND_MID),
    },
    {
        "name": "section_2",
        "mapzebrain_z": MAPZEBRAIN_Z_PER_SECTION[2],
        "experiment_mask": None,
        "atlas_mask": None,
        "atlas_row_band": (_ROW_BAND_MID, _ROW_BAND_LOWER),
    },
    {
        "name": "section_3",
        "mapzebrain_z": MAPZEBRAIN_Z_PER_SECTION[3],
        "experiment_mask": None,
        "atlas_mask": None,
        "atlas_row_band": (_ROW_BAND_LOWER, _h),
    },
]

EXPERIMENT_SECTIONS = []
for _spec in PIECEWISE_REGION_SPECS:
    _name = str(_spec["name"])
    _mask = experiment_masks_by_section[_name]
    if int(_mask.sum()) == 0:
        raise ValueError(f"Empty experiment mask for {_name!r}")
    _rs, _re = sitkalignment.mask_embedded_row_bounds(_mask)
    EXPERIMENT_SECTIONS.append(
        {
            "name": _name,
            "row_start": _rs,
            "row_end": _re,
            "experiment_mask": _spec.get("experiment_mask"),
        }
    )

sitkalignment.validate_piecewise_region_masks(experiment_masks_by_section, warn_overlap=True)

_n = len(EXPERIMENT_SECTIONS)
print(f"{_n} piecewise regions on embedded experiment ({_h}×{_w}):")
for _b in EXPERIMENT_SECTIONS:
    _m = experiment_masks_by_section[_b["name"]]
    _path = _b.get("experiment_mask")
    _path_s = _path.name if _path is not None else "(row band / composite)"
    print(
        f"  {_b['name']}: {_path_s}, {int(_m.sum())} px, "
        f"rows [{_b['row_start']}, {_b['row_end']})"
    )
print(
    f"  bands: top=mask, upper_mid=[0,{_ROW_BAND_MID})−top, "
    f"lower_mid=[{_ROW_BAND_MID},{_ROW_BAND_LOWER}), bottom=[{_ROW_BAND_LOWER},{_h})"
)

fig, axes = plt.subplots(1, 2, figsize=(14, 7))
axes[0].imshow(plot_img, cmap="gray", vmin=target_vmin, vmax=target_vmax)
axes[0].set_title(f"experiment plane {selected_plane}")
for _y in (_ROW_BAND_MID, _ROW_BAND_LOWER):
    axes[0].axhline(_y, color="cyan", lw=0.8, alpha=0.7, ls="--")
axes[0].axis("off")

_overlay = np.zeros((*plot_img.shape, 3), dtype=np.float32)
_cmap = plt.cm.tab10(np.linspace(0, 1, max(_n, 1)))
for _i, _b in enumerate(EXPERIMENT_SECTIONS):
    _col = _cmap[_i % len(_cmap)]
    _overlay[experiment_masks_by_section[_b["name"]] > 0] = _col[:3]
axes[1].imshow(plot_img, cmap="gray", vmin=target_vmin, vmax=target_vmax)
axes[1].imshow(_overlay, alpha=0.45)
axes[1].set_title("experiment region masks")
for _y in (_ROW_BAND_MID, _ROW_BAND_LOWER):
    axes[1].axhline(_y, color="white", lw=0.8, alpha=0.6, ls="--")
axes[1].axis("off")
plt.tight_layout()
plt.show()
''',
)

set_cell(
    9,
    """### Load mapZebrain atlas slices (one z per region)

Requires **`PIECEWISE_REGION_SPECS`** / **`EXPERIMENT_SECTIONS`** from the mask cell above.
""",
    "markdown",
)

src10 = "".join(nb["cells"][10]["source"])
src10 = src10.replace(
    "# Requires: EXPERIMENT_SECTIONS from the custom section cell above.\n# One atlas z per section (same order as EXPERIMENT_SECTIONS: top experiment section first).",
    "# Requires: EXPERIMENT_SECTIONS / MAPZEBRAIN_Z_PER_SECTION from the mask cell above.",
)
src10 = src10.replace(
    "if 'EXPERIMENT_SECTIONS' not in globals():\n    raise RuntimeError('Run the EXPERIMENT_SECTIONS cell above first.')",
    "if 'EXPERIMENT_SECTIONS' not in globals() or 'MAPZEBRAIN_Z_PER_SECTION' not in globals():\n    raise RuntimeError('Run the piecewise mask cell above first.')",
)
src10 = re.sub(r"MAPZEBRAIN_Z_PER_SECTION = \[[\s\S]*?\]\n\n", "", src10, count=1)
src10 = src10.replace(
    "_n_sections = len(EXPERIMENT_SECTIONS)\nMAPZEBRAIN_Z_PER_SECTION = [int(z) for z in MAPZEBRAIN_Z_PER_SECTION]\nif len(MAPZEBRAIN_Z_PER_SECTION) != _n_sections:",
    "_n_sections = len(EXPERIMENT_SECTIONS)\nif len(MAPZEBRAIN_Z_PER_SECTION) != _n_sections:",
)
src10 = src10.replace(
    '        f"rows [{_b[\'row_start\']}, {_b[\'row_end\']})"',
    '        f"mask rows [{_b[\'row_start\']}, {_b[\'row_end\']})"',
)
src10 = src10.replace(
    '    print(f"  {_b[\'name\']}: experiment rows [{_b[\'row_start\']}, {_b[\'row_end\']}) -> z={_z}")',
    '    print(f"  {_b[\'name\']}: experiment mask rows [{_b[\'row_start\']}, {_b[\'row_end\']}) -> z={_z}")',
)
src10 = src10.replace(
    "# Preview grid (50 px grid lines; y labels every 200 px; same frame as ATLAS_METRIC_REGIONS).",
    "# Preview grid (50 px grid lines; y labels every 200 px; embedded atlas frame).",
)
set_cell(10, src10)

set_cell(
    11,
    r'''# Requires: mapzebrain_by_section, experiment_masks_by_section, PIECEWISE_REGION_SPECS.

ATLAS_MOVING_BLACKOUT = True

for _spec in PIECEWISE_REGION_SPECS:
    _name = str(_spec["name"])
    if _name not in mapzebrain_by_section:
        raise KeyError(f"{_name!r} missing from mapzebrain_by_section")
    _emb = mapzebrain_by_section[_name]["embedded"]
    _native_shape = mapzebrain_by_section[_name]["native_shape"]
    _h_at, _w_at = _emb.shape[:2]

    _atlas_path = _spec.get("atlas_mask")
    if _atlas_path is not None:
        _atlas_path = Path(_atlas_path)
        if not _atlas_path.is_file():
            raise FileNotFoundError(f"atlas_mask for {_name!r}: {_atlas_path}")
        _mask = sitkalignment.prepare_atlas_mask_for_embedded_slice(
            _atlas_path, _emb.shape, native_shape=_native_shape
        )
        mapzebrain_by_section[_name]["atlas_mask_path"] = _atlas_path
    elif _name == "section_0":
        _exp_m = experiment_masks_by_section[_name]
        _rs, _re = sitkalignment.mask_embedded_row_bounds(_exp_m)
        _mask = sitkalignment.registration_mask_embedded_rectangle(
            _emb.shape, _rs, _re, 0, _w_at
        )
        mapzebrain_by_section[_name]["atlas_mask_path"] = None
    elif "atlas_row_band" in _spec:
        _rs, _re = (int(_spec["atlas_row_band"][0]), int(_spec["atlas_row_band"][1]))
        _mask = sitkalignment.registration_mask_embedded_rectangle(
            _emb.shape, _rs, _re, 0, _w_at
        )
        mapzebrain_by_section[_name]["atlas_mask_path"] = None
    else:
        raise ValueError(f"No atlas_mask or atlas_row_band for {_name!r}")

    mapzebrain_by_section[_name]["moving_metric_mask"] = _mask
    _b = next(b for b in EXPERIMENT_SECTIONS if b["name"] == _name)
    print(
        f"{_name} z={mapzebrain_by_section[_name]['mapzebrain_z']}: "
        f"experiment {int(experiment_masks_by_section[_name].sum())} px "
        f"[{_b['row_start']}, {_b['row_end']}) | atlas {int(_mask.sum())} px"
    )

_n = len(EXPERIMENT_SECTIONS)
fig, axes = plt.subplots(_n, 2, figsize=(10, 3.5 * _n))
if _n == 1:
    axes = np.asarray([axes])

for _i, _b in enumerate(EXPERIMENT_SECTIONS):
    _name = _b["name"]
    ax_exp = axes[_i, 0]
    ax_exp.imshow(plot_img, cmap="gray", vmin=target_vmin, vmax=target_vmax)
    _em = experiment_masks_by_section[_name] > 0
    ax_exp.imshow(np.ma.masked_where(~_em, _em), cmap="Greens", alpha=0.35, vmin=0, vmax=1)
    ax_exp.set_title(f"experiment — {_name} ({int(_em.sum())} px)")
    ax_exp.axis("off")

    ax_at = axes[_i, 1]
    _mask = mapzebrain_by_section[_name]["moving_metric_mask"]
    _atlas_show = (
        sitkalignment.apply_moving_blackout(
            mapzebrain_by_section[_name]["embedded"], _mask
        )
        if ATLAS_MOVING_BLACKOUT
        else mapzebrain_by_section[_name]["embedded"]
    )
    ax_at.imshow(_atlas_show, cmap="gray", vmax=700)
    ax_at.imshow(np.ma.masked_where(_mask == 0, _mask), cmap="Greens", alpha=0.35, vmin=0, vmax=1)
    ax_at.set_title(f"atlas z={mapzebrain_by_section[_name]['mapzebrain_z']}")
    ax_at.axis("off")

plt.suptitle("Green = metric mask voxels", y=1.01)
plt.tight_layout()
plt.show()
''',
)

set_cell(
    14,
    "".join(nb["cells"][14]["source"]).replace(
        "one Elastix run per experiment row section + atlas z",
        "one Elastix run per mask-defined region + atlas z",
    ),
)

set_cell(
    15,
    """## Piecewise alignment (mask-limited regions)

One Elastix run per region; fixed/moving images blacked out outside each mask.
""",
    "markdown",
)

set_cell(
    16,
    """### Piecewise config (regions + atlas z)

**`REGION_SECTIONS`** is built from **`PIECEWISE_REGION_SPECS`** and mask-derived row bounds.
""",
    "markdown",
)

set_cell(
    17,
    '''# --- Piecewise config ---
USE_PIECEWISE_ALIGNMENT = True
PIECEWISE_SCALE_PENALTY = 150
PIECEWISE_ITERATIONS = (10000, 500)
PIECEWISE_SAVE_TAG = f"alignment_rev_plane{selected_plane}_piecewise"

if 'PIECEWISE_REGION_SPECS' not in globals() or 'EXPERIMENT_SECTIONS' not in globals():
    raise RuntimeError('Run the mask + mapZebrain cells first.')
if 'mapzebrain_by_section' not in globals():
    raise RuntimeError('Run the mapZebrain load cell.')
if 'moving_metric_mask' not in next(iter(mapzebrain_by_section.values())):
    raise RuntimeError('Run the atlas mask cell.')

REGION_SECTIONS = []
for _section, _z in zip(EXPERIMENT_SECTIONS, MAPZEBRAIN_Z_PER_SECTION):
    _name = str(_section['name'])
    _emp = _section.get('experiment_mask')
    REGION_SECTIONS.append(
        {
            'name': _name,
            'row_start': int(_section['row_start']),
            'row_end': int(_section['row_end']),
            'mapzebrain_z': int(_z),
            'experiment_mask_path': Path(_emp) if _emp is not None else None,
            'atlas_mask_path': mapzebrain_by_section[_name].get('atlas_mask_path'),
        }
    )


def validate_region_sections(sections, masks_by_name, canvas_shape):
    names = set()
    covered = np.zeros(canvas_shape[:2], dtype=bool)
    for b in sections:
        name = str(b["name"])
        if name in names:
            raise ValueError(f"Duplicate region name: {name}")
        names.add(name)
        m = np.asarray(masks_by_name[name], dtype=bool)
        overlap = covered & m
        if overlap.any():
            raise ValueError(f"Overlapping mask pixels for {name!r} ({int(overlap.sum())} px)")
        covered |= m
    gap = int((~covered).sum())
    if gap:
        print(f"[piecewise] {gap} px not in any region mask")

if "plot_img" not in globals():
    raise RuntimeError("Run the experiment pipeline cells first.")

validate_region_sections(REGION_SECTIONS, experiment_masks_by_section, plot_img.shape)
print(f"Configured {len(REGION_SECTIONS)} piecewise regions:")
for _b in REGION_SECTIONS:
    print(
        f"  {_b['name']}: mask rows [{_b['row_start']}, {_b['row_end']}), "
        f"atlas z={_b['mapzebrain_z']}"
    )
''',
)

src18 = "".join(nb["cells"][18]["source"]).replace("section boundaries", "region masks")
src18 = src18.replace(
    "for _i, _b in enumerate(REGION_SECTIONS):\n    rs, re = int(_b[\"row_start\"]), int(_b[\"row_end\"])\n    _col = _cmap[_i % len(_cmap)]\n    axes[0].axhline(rs, color=_col, lw=1.2, alpha=0.9)\n    axes[0].axhline(re, color=_col, lw=1.2, alpha=0.9)\n    axes[0].text(\n        8,\n        (rs + re) // 2,\n        _b[\"name\"],\n        color=_col,\n        fontsize=10,\n        va=\"center\",\n        bbox=dict(boxstyle=\"round,pad=0.2\", fc=\"black\", alpha=0.45),\n    )\n    _overlay[rs:re, :, :] = _col[:3]",
    "for _i, _b in enumerate(REGION_SECTIONS):\n    _col = _cmap[_i % len(_cmap)]\n    _m = experiment_masks_by_section[_b[\"name\"]] > 0\n    _overlay[_m] = _col[:3]\n    rs, re = int(_b[\"row_start\"]), int(_b[\"row_end\"])\n    axes[0].text(8, (rs + re) // 2, _b[\"name\"], color=_col, fontsize=10, va=\"center\", bbox=dict(boxstyle=\"round,pad=0.2\", fc=\"black\", alpha=0.45))",
)
set_cell(18, src18)

set_cell(
    19,
    """### Run piecewise Elastix (one transform folder per region)

Blacks out voxels outside each region's experiment and atlas masks, then runs Elastix.
""",
    "markdown",
)

src20 = "".join(nb["cells"][20]["source"])
old_block = """    _section_mask = piecewise_section_mask(
        plot_img.shape, _section["row_start"], _section["row_end"]
    )
    _fixed_mask = (_global_fixed.astype(np.uint8) * _section_mask).astype(np.uint8)
    if _fixed_mask.sum() == 0:
        raise ValueError(f"Empty fixed mask for section {_name}")

    # Rebuild moving mask from ROI so edits to ATLAS_METRIC_REGIONS are not stale.
    _roi = _section.get("atlas_metric_roi") or mapzebrain_by_section[_name].get("atlas_metric_roi")
    _frame = _section.get("atlas_metric_coord_frame") or mapzebrain_by_section[_name].get(
        "atlas_metric_coord_frame", "embedded"
    )
    if _roi is None:
        raise ValueError(f"No atlas_metric_roi for section {_name!r} — run atlas metric cell")
    _moving_mask = sitkalignment.moving_metric_mask_from_roi(
        _moving.shape,
        _native_shape,
        _roi,
        coord_frame=_frame,
    )
    _n_mask = int(_moving_mask.sum())
    _n_total = int(_moving_mask.size)
    _fixed_px = int(_fixed_mask.sum())
    print(
        f"  {_name} fixed metric: {_fixed_px} px (experiment rows "
        f"[{_section['row_start']}, {_section['row_end']}))"
    )
    print(
        f"  {_name} moving metric: {_n_mask}/{_n_total} px "
        f"({100.0 * _n_mask / _n_total:.1f}% of slice) atlas rows "
        f"[{_roi['row_start']}, {_roi['row_end']})"
    )
    if _moving_mask.sum() == 0:
        raise ValueError(f"Empty moving mask for section {_name}")

    _blackout_moving = bool(globals().get("ATLAS_MOVING_BLACKOUT", True))
    if _blackout_moving:
        _reg_moving = sitkalignment.apply_registration_blackout(_reg_moving, _moving_mask)
        print(f"  {_name} moving blackout: outside atlas ROI set to 0")

    _reg_fixed = sitkalignment.apply_registration_blackout(
        np.asarray(_reg_plot_piece, dtype=np.float64), _fixed_mask
    )
    print(f"  {_name} fixed blackout: outside experiment row section set to 0")

    _save = exp_folder.joinpath(f"{PIECEWISE_SAVE_TAG}_{_name}")
    print(
        f"Registering section {_name!r}: rows [{_section['row_start']}, {_section['row_end']}), "
        f"z={_z} -> {_save}"
    )"""

new_block = """    if _name not in experiment_masks_by_section:
        raise KeyError(f"Section {_name!r} missing from experiment_masks_by_section")
    _fixed_mask = (
        _global_fixed.astype(np.uint8) * experiment_masks_by_section[_name].astype(np.uint8)
    ).astype(np.uint8)
    if _fixed_mask.sum() == 0:
        raise ValueError(f"Empty fixed mask for section {_name}")

    _moving_mask = np.asarray(
        mapzebrain_by_section[_name]["moving_metric_mask"], dtype=np.uint8
    )
    _n_mask = int(_moving_mask.sum())
    _n_total = int(_moving_mask.size)
    _fixed_px = int(_fixed_mask.sum())
    print(
        f"  {_name} fixed metric: {_fixed_px} px (experiment mask, "
        f"row span [{_section['row_start']}, {_section['row_end']}))"
    )
    print(
        f"  {_name} moving metric: {_n_mask}/{_n_total} px "
        f"({100.0 * _n_mask / _n_total:.1f}% of slice)"
    )
    if _moving_mask.sum() == 0:
        raise ValueError(f"Empty moving mask for section {_name}")

    _blackout_moving = bool(globals().get("ATLAS_MOVING_BLACKOUT", True))
    if _blackout_moving:
        _reg_moving = sitkalignment.apply_registration_blackout(_reg_moving, _moving_mask)
        print(f"  {_name} moving blackout: outside atlas mask set to 0")

    _reg_fixed = sitkalignment.apply_registration_blackout(
        np.asarray(_reg_plot_piece, dtype=np.float64), _fixed_mask
    )
    print(f"  {_name} fixed blackout: outside experiment mask set to 0")

    _save = exp_folder.joinpath(f"{PIECEWISE_SAVE_TAG}_{_name}")
    print(
        f"Registering region {_name!r}: mask rows [{_section['row_start']}, {_section['row_end']}), "
        f"z={_z} -> {_save}"
    )"""

if old_block not in src20:
    raise SystemExit("cell 20 old_block not found")
src20 = src20.replace(old_block, new_block)
src20 = src20.replace(
    'raise RuntimeError("Run experiment + section cells first (defines plot_img).")',
    'raise RuntimeError("Run experiment + mask cells first (defines plot_img).")',
)
src20 = src20.replace(
    'raise RuntimeError("Run the atlas metric regions cell first.")',
    'raise RuntimeError("Run the atlas mask cell first.")',
)
src20 = src20.replace(
    'if "REGION_SECTIONS" not in globals():\n    raise RuntimeError("Run the piecewise config cell (defines REGION_SECTIONS).")',
    'if "REGION_SECTIONS" not in globals():\n    raise RuntimeError("Run the piecewise config cell.")\nif "experiment_masks_by_section" not in globals():\n    raise RuntimeError("Run the piecewise mask cell.")',
)
set_cell(20, src20)

src21 = "".join(nb["cells"][21]["source"]).replace(
    'axes[_i, 0].imshow(plot_img, cmap="gray", vmin=target_vmin, vmax=target_vmax)\n    axes[_i, 0].axhline(rs, color="yellow", lw=1.0, alpha=0.9)\n    axes[_i, 0].axhline(re, color="yellow", lw=1.0, alpha=0.9)\n    axes[_i, 0].set_title(f"experiment — metric rows [{rs}, {re})")',
    'axes[_i, 0].imshow(plot_img, cmap="gray", vmin=target_vmin, vmax=target_vmax)\n    _fm = experiment_masks_by_section[_name] > 0\n    axes[_i, 0].imshow(np.ma.masked_where(~_fm, _fm), cmap="Greens", alpha=0.35, vmin=0, vmax=1)\n    axes[_i, 0].set_title(f"experiment — mask ({int(_fm.sum())} px)")',
)
set_cell(21, src21)

set_cell(22, "".join(nb["cells"][22]["source"]).replace("no section lines", "stitched by region masks"))

nb_path.write_text(json.dumps(nb, indent=1))
json.loads(nb_path.read_text())
print("patched", nb_path)
