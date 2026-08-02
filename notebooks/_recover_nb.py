#!/usr/bin/env python3
"""Repair truncated brain_alignment_from_tiff.ipynb and add multi-region count print + intact viz cell."""
import json
import textwrap
from pathlib import Path

path = Path("/Users/elysiaye/alignment/notebooks/brain_alignment_from_tiff.ipynb")
lines = path.read_text().splitlines(keepends=True)

# Keep lines 1..44815 (indices 0..44814) — through _n = ...
head = lines[:44815]
# Rest from closing bracket of source array
rest = lines[44816:]  # starts with "      ]\n"

fixed_tail = [
    '        "print(f\\"fish {_fish_id}: {_n} / {len(_ish)} ROIs with ≥1 region (in-memory only; see FISH_ROI_REGIONS)\\")\\n",\n'
    .replace("{_n}", "{_n}")
    .replace("_ish", "_fish"),
    '        "_n_multi = int(_fish[REGION_COLUMN].map(lambda v: isinstance(v, list) and len(v) > 1).sum())\\n",\n',
    '        "print(f\\"fish {_fish_id}: {_n_multi} ROIs with >1 region label (multiple masks hit)\\")\\n",\n',
]

# Fix typo: I had _ish in replace - correct line manually
fixed_tail[0] = (
    '        "print(f\\"fish {_fish_id}: {_n} / {len(_fish)} ROIs with ≥1 region '
    '(in-memory only; see FISH_ROI_REGIONS)\\")\\n",\n'
)

# Find first line in rest that is exactly "    },\n" after the cell's closing — keep rest through first complete cell end after ]
# rest[0] is "      ]\n", rest[1] is "    },\n", rest[2] is "    {\n" — drop from rest[2] onward (broken next cell)
if len(rest) < 2 or not rest[0].strip().startswith("]"):
    raise SystemExit(f"unexpected rest[0]: {rest[0][:80]!r}")

after_cell18 = rest[:2]  # "      ]\n", "    },\n"

viz_source = textwrap.dedent(
    r'''
# Color-coded ROIs: experiment vs atlas for each plane in EXPERIMENT_ATLAS_QC_PAIRS (no HDF writes).
import matplotlib.patches as mpatches
if "FISH_ROI_REGIONS" not in globals():
    raise RuntimeError("Run the region-assignment cell above first (defines FISH_ROI_REGIONS).")
if "target_img_by_plane" not in globals() or not target_img_by_plane:
    raise RuntimeError("Run the memmap cell (target_img_by_plane).")
if "atlas_slice_embedded" not in globals():
    raise RuntimeError("Run the mapping cell (atlas_slice_embedded).")
if "com_to_target_xy" not in globals():
    raise RuntimeError("com_to_target_xy missing — run an earlier setup cell.")
def _unpack_com_v(c):
    v = np.asarray(c, dtype=float).ravel()
    if v.size < 2:
        raise ValueError(f"COM must have at least 2 values, got {c!r}")
    return float(v[0]), float(v[1])
REGION_COLUMN = "region_aligned_mapzebrain"
NA_LABEL = "N/A"
def _primary_region(val):
    if val is None:
        return NA_LABEL
    if isinstance(val, float) and np.isnan(val):
        return NA_LABEL
    if isinstance(val, list):
        return val[0] if val else NA_LABEL
    return str(val)
fr = FISH_ROI_REGIONS.copy()
fr["__lab"] = fr[REGION_COLUMN].map(_primary_region)
_all_labels = sorted(set(fr["__lab"].tolist()), key=lambda s: (s != NA_LABEL, s))
if NA_LABEL not in _all_labels:
    _all_labels.append(NA_LABEL)
_na_rgb = (0.55, 0.55, 0.58)
_lab2rgb = {NA_LABEL: _na_rgb}
_i = 0
for _lb in _all_labels:
    if _lb == NA_LABEL:
        continue
    _lab2rgb[_lb] = plt.cm.tab20(_i % 20)[:3]
    _i += 1
_tvmin = float(globals().get("target_vmin", 0))
_tvmax = float(globals().get("target_vmax", 1000))
_svmax = float(globals().get("source_vmax", 1000))
_evmin = float(globals().get("QC_EXPERIMENT_VMIN", _tvmin))
_evmax = float(globals().get("QC_EXPERIMENT_VMAX", min(_tvmax, 380.0)))
_qc_rows_v = [(int(ep), int(mz)) for ep, mz in EXPERIMENT_ATLAS_QC_PAIRS if ep in target_img_by_plane]
if not _qc_rows_v:
    raise ValueError("No QC rows with memmap targets.")
_n = len(_qc_rows_v)
_fig, _axes = plt.subplots(_n, 2, figsize=(14, 3.6 * _n))
if _n == 1:
    _axes = np.asarray([_axes])
for _ri, (_pl, _mz) in enumerate(_qc_rows_v):
    _sub = fr[fr["plane"] == _pl]
    _tgt = target_img_by_plane[_pl]
    _src = atlas_slice_embedded(_mz)
    _ok = _sub["com_mapzebrain_xy"].notna()
    _subo = _sub.loc[_ok]
    _labs = _subo["__lab"].tolist()
    _xy_e = np.array([com_to_target_xy(*_unpack_com_v(c)) for c in _subo["com"]], dtype=float)
    _xy_m = np.array([tuple(_subo.loc[i, "com_mapzebrain_xy"]) for i in _subo.index], dtype=float)
    _cols = np.array([_lab2rgb.get(lb, _na_rgb) for lb in _labs])
    _axl, _axr = _axes[_ri, 0], _axes[_ri, 1]
    _axl.imshow(_tgt, cmap="gray", vmin=_evmin, vmax=_evmax)
    if len(_xy_e):
        _axl.scatter(_xy_e[:, 0], _xy_e[:, 1], c=_cols, s=6, alpha=0.92, linewidths=0)
    _axl.set_title(f"experiment plane {_pl}  n={len(_sub)}")
    _axl.axis("off")
    _axr.imshow(_src, cmap="gray", vmin=0, vmax=_svmax)
    if len(_xy_m):
        _axr.scatter(_xy_m[:, 0], _xy_m[:, 1], c=_cols, s=6, alpha=0.92, linewidths=0)
    _axr.set_title(f"atlas z={_mz}  (primary region label for color)")
    _axr.axis("off")
_lab_counts = fr["__lab"].value_counts()
_handles = [
    mpatches.Patch(
        color=_lab2rgb[l],
        label=f"{l}  [{int(_lab_counts.get(l, 0))}]",
    )
    for l in _all_labels
    if l in _lab2rgb
]
_fig.legend(handles=_handles, loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8, title="region [n ROIs]")
plt.suptitle("ROIs color-coded by primary region (first mask hit); " + NA_LABEL + " = gray", y=1.01)
plt.tight_layout()
plt.subplots_adjust(right=0.82)
plt.show()
'''
).strip()

viz_cell = {
    "cell_type": "code",
    "execution_count": None,
    "id": "viz_regions_recovered",
    "metadata": {},
    "outputs": [],
    "source": [ln + "\n" for ln in viz_source.splitlines()],
}

nb_partial = "".join(head + fixed_tail + after_cell18)
# nb_partial ends with "    },\n" — need to insert comma and next cell
# after_cell18 ends cell 18; next should be comma + viz cell + close cells array

footer = """    ,
"""  # WRONG

# Build valid JSON by parsing up to end of cells array
prefix = "".join(head + fixed_tail + after_cell18)
if not prefix.rstrip().endswith("},"):
    raise SystemExit("prefix should end with cell close },")

# Remove trailing comma issue: after_cell18 is `      ]\n` and `    },\n` — need comma between cells
text = "".join(head + fixed_tail + after_cell18)
# text ends with `    },\n` — next is `    {\n` from broken — we replace tail from `    {\n` of broken cell

# Find position of broken third line in original rest
broken_start = "".join(lines[44818:])  # from `    {\n` of bad cell
# Instead: assemble full notebook as dict
nb = json.loads(
    prefix.rstrip().rstrip(",")
    + ","
    + json.dumps(viz_cell)[1:-1]
    + "]}"
)  # broken

# Simpler: use json.dumps for whole notebook
cells_json = prefix + json.dumps([viz_cell])[1:-1]  # dumps wraps [...]

# prefix ends with `    },\n` — cells array needs comma: `    },\n    {viz...}\n  ]`

full = "".join(head + fixed_tail + after_cell18).rstrip("\n")
if not full.endswith("}"):
    pass
# full currently: ... `    },` without newline at end? after_cell18[1] includes newline

full = "".join(head + fixed_tail + after_cell18)
# insert comma after last }
if not full.endswith(",\n"):
    if full.endswith("}\n"):
        full = full[:-2] + "},\n"

full_cells_inner = full.split('"cells": [', 1)[1]
# too fragile

# --- Robust: json.loads only the cells array we build from scratch using nbformat structure
import uuid

cell18_close = json.loads(
    "[" + json.dumps({"cell_type": "code", "execution_count": None, "id": "recovered18", "metadata": {}, "outputs": [], "source": [l.rstrip("\n") for l in head[44700:44815]]})[:0] + "]"
)

# Abandon: read original good notebook from git not available.

# Manual assembly of tail string
part_a = "".join(head + fixed_tail + after_cell18)  # ends with `    },\n`
viz_json = json.dumps(viz_cell, indent=2)
# indent viz to match file (2 spaces base) — embed as next cell
viz_lines = textwrap.indent(viz_json, "    ")
# join: part_a should end with closing of cell — add comma
blob = part_a.rstrip() + ",\n" + viz_json + "\n  ],\n"
blob += """  "metadata": {
    "kernelspec": {
      "display_name": "Python 3",
      "language": "python",
      "name": "python3"
    },
    "language_info": {
      "name": "python",
      "version": "3.11.0"
    }
  },
  "nbformat": 4,
  "nbformat_minor": 5
}
"""

# part_a must be valid start of JSON from `{` - actually file starts with `{\n  "cells": [\n`
# head includes from line 1 - line 1 is `{\n`

# Verify part_a parses as incomplete - use json.loads(part_a + " null]}")

# Easiest: load with json from string built from cells only
cells = []
# read original notebook cells 0..17 from backup - we don't have backup

# Use nbformat read from partial: write part_a to temp, manually impossible

# **Use subprocess git show HEAD:path** if in git
import subprocess

repo = Path("/Users/elysiaye/alignment")
try:
    good = subprocess.check_output(
        ["git", "-C", str(repo), "show", "HEAD:notebooks/brain_alignment_from_tiff.ipynb"],
        stderr=subprocess.DEVNULL,
    )
    nb = json.loads(good)
    print("recovered from git HEAD")
except Exception as e:
    print("git failed", e)
    raise SystemExit(1)

# Patch cell 18 source in recovered nb
for ci, cell in enumerate(nb["cells"]):
    src = "".join(cell.get("source", []))
    if "FISH_ROI_REGIONS = _fish.copy()" in src and "MASKS_FOLDER" in src:
        old = 'print(f"fish {_fish_id}: {_n} / {len(_fish)} ROIs with ≥1 region (in-memory only; see FISH_ROI_REGIONS)")\n'
        new = (
            old
            + "_n_multi = int(_fish[REGION_COLUMN].map(lambda v: isinstance(v, list) and len(v) > 1).sum())\n"
            + 'print(f"fish {_fish_id}: {_n_multi} ROIs with >1 region label (multiple masks hit)")\n'
        )
        if old not in src:
            # try with comma variant
            old2 = old.replace(")\n", ")\\n")  # no
            raise SystemExit("old print not found in cell " + str(ci))
        src2 = src.replace(old, new)
        nb["cells"][ci]["source"] = [ln + "\n" for ln in src2.splitlines()]
        print("patched cell", ci)
        break
else:
    raise SystemExit("region cell not found in git version")

# Clear all outputs to avoid bloat
for cell in nb["cells"]:
    cell["outputs"] = []
    cell["execution_count"] = None

path.write_text(json.dumps(nb, indent=2) + "\n")
print("wrote", path, "cells", len(nb["cells"]))
