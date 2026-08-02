#!/usr/bin/env python3
"""
Thoroughly explore temporal_pack_compact.h5: how it is laid out on disk and what
each table holds — WITHOUT blindly loading the ~6 GB /compact_temporal frame
(whose trace arrays are pickled and may not unpickle under a mismatched numpy).

Layers of inspection (each independent — run all or comment out what you don't need):
  1. h5py raw view      — true HDF5 groups/datasets/shapes/dtypes + pandas attrs.
  2. pandas store meta  — keys, storage format (fixed/table), nrows, columns. No data load.
  3. side table peek     — /com_aligned loaded fully (small) with dtypes + sample rows.
  4. main table peek     — /compact_temporal column list + ONE guarded sample row,
                           reporting the shape/dtype of each array-valued cell.

Usage:
  python scripts/explore_temporal_pack.py /path/to/temporal_pack_compact.h5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

MAIN_KEY = "compact_temporal"   # upstream-built per-ROI table (fixed format, big)
SIDE_KEY = "com_aligned"        # written by the alignment notebooks (small)


# ---------------------------------------------------------------- 1. h5py raw view
def raw_h5py_view(path: Path) -> None:
    """The literal HDF5 tree: every group/dataset, its shape/dtype, and the
    pandas-written attributes (pandas_type, encoding, block layout, etc.).
    This is what a pandas DataFrame REALLY looks like on disk."""
    import h5py

    print("=" * 78)
    print(f"1. RAW HDF5 STRUCTURE (h5py): {path}")
    print("=" * 78)

    _PANDAS_ATTRS = {  # attrs worth surfacing; they explain the on-disk encoding
        "pandas_type", "pandas_version", "encoding", "nblocks",
        "ndim", "axis0_variety", "axis1_variety",
        "block0_items_variety", "table_type", "index_cols",
    }

    def visit(name: str, obj) -> None:
        indent = "  " * name.count("/")
        if isinstance(obj, h5py.Group):
            print(f"{indent}{name}/   (Group)")
        else:  # Dataset
            print(f"{indent}{name}   shape={obj.shape} dtype={obj.dtype}")
        # surface only the informative pandas attrs to avoid noise
        for k, v in obj.attrs.items():
            if k in _PANDAS_ATTRS or k.endswith("_kind") or k.endswith("_dtype"):
                if isinstance(v, bytes):
                    v = v.decode("utf-8", "replace")
                print(f"{indent}    @{k} = {v!r}")

    with h5py.File(str(path), "r") as f:
        # top-level group attrs (per pandas key: pandas_type tells fixed vs table)
        for top in f.keys():
            grp = f[top]
            ptype = grp.attrs.get("pandas_type", b"")
            if isinstance(ptype, bytes):
                ptype = ptype.decode()
            print(f"\n/{top}   pandas_type={ptype!r}")
            grp.visititems(visit)
    print()


# ------------------------------------------------------- 2. pandas store metadata
def pandas_store_meta(path: Path) -> None:
    """Keys + per-key storage format, row count, and column names — pulled from
    the storer WITHOUT materializing the frame."""
    import pandas as pd

    print("=" * 78)
    print(f"2. PANDAS HDFStore METADATA: {path}")
    print("=" * 78)
    with pd.HDFStore(str(path), mode="r") as store:
        keys = list(store.keys())
        print(f"keys ({len(keys)}): {keys}\n")
        for key in keys:
            st = store.get_storer(key)
            fmt = "table" if getattr(st, "is_table", False) else "fixed"
            nrows = getattr(st, "nrows", "?")
            # columns: for fixed frames the storer exposes them without a data read
            try:
                ncols = st.ncols
            except Exception:
                ncols = "?"
            print(f"{key}: format={fmt}  nrows={nrows}  ncols={ncols}")
            # axis0 holds the column labels for a fixed frame
            try:
                attrs = st.group._v_attrs
                print(f"    storer attrs: "
                      f"pandas_type={getattr(attrs, 'pandas_type', '?')!r}")
            except Exception:
                pass
            print()


# --------------------------------------------------------- 3. side table full peek
def peek_side_table(path: Path, n: int = 5) -> None:
    """/com_aligned is small — load it fully, show dtypes and a few real rows.
    This is the table the alignment notebooks WRITE (per-ROI atlas mapping)."""
    import pandas as pd

    print("=" * 78)
    print(f"3. SIDE TABLE  /{SIDE_KEY}  (notebook-written, small)")
    print("=" * 78)
    try:
        side = pd.read_hdf(path, key=SIDE_KEY)
    except KeyError:
        print(f"  /{SIDE_KEY} not present (no fish aligned yet).\n")
        return

    print(f"rows={len(side)}  columns={list(side.columns)}\n")
    print("dtypes:")
    print(side.dtypes.to_string())
    print("\nper-fish row counts (fish_id -> rows):")
    if "fish_id" in side.columns:
        print(side["fish_id"].value_counts().sort_index().to_string())
    print(f"\nfirst {n} rows:")
    with pd.option_context("display.max_colwidth", 60, "display.width", 200):
        print(side.head(n).to_string())
    # describe the shape of the tuple/list-valued cells
    print("\narray/list-valued columns (type + shape of row 0):")
    for c in side.columns:
        v = side[c].iloc[0] if len(side) else None
        if isinstance(v, (list, tuple)):
            print(f"  {c}: {type(v).__name__} len={len(v)}  e.g. {v}")
    print()


# --------------------------------------------------------- 4. main table peek
def peek_main_table(path: Path) -> None:
    """/compact_temporal is big and its trace cells are pickled. Read ONLY the
    column labels (axis0) and a SINGLE row, reporting each cell's shape/dtype so
    you learn the schema without unpickling millions of arrays (or hitting the
    numpy-version pickle error on a full load)."""
    import numpy as np
    import pandas as pd
    import tables as tb

    print("=" * 78)
    print(f"4. MAIN TABLE  /{MAIN_KEY}  (upstream-built, big; trace cells pickled)")
    print("=" * 78)

    # Column labels live in axis0 of a fixed-format frame — cheap to read.
    try:
        with tb.open_file(str(path), "r") as hf:
            grp = hf.get_node(f"/{MAIN_KEY}")
            axis0 = hf.get_node(f"/{MAIN_KEY}/axis0").read()
            cols = [c.decode() if isinstance(c, bytes) else c for c in axis0]
            nrows = hf.get_node(f"/{MAIN_KEY}/axis1").shape[0]
            print(f"nrows={nrows}  ncols={len(cols)}")
            print(f"columns: {cols}\n")
            # block layout: which columns are numeric vs object (pickled)
            for node in grp:
                nm = node._v_name
                if nm.endswith("_items"):
                    items = [c.decode() if isinstance(c, bytes) else c
                             for c in node.read()]
                    vals = nm.replace("_items", "_values")
                    try:
                        vnode = hf.get_node(grp, vals)
                        kind = getattr(vnode.atom, "kind", "?")
                        print(f"  {nm}: cols={items}  -> values kind={kind}")
                    except Exception:
                        print(f"  {nm}: cols={items}")
            print()
    except Exception as exc:  # noqa: BLE001
        print(f"  could not read raw layout: {exc}\n")

    # ONE guarded sample row. read_hdf can slice a fixed frame by start/stop.
    print("Single-row sample (start=0, stop=1) — guarded against pickle/numpy errors:")
    try:
        one = pd.read_hdf(path, key=MAIN_KEY, start=0, stop=1)
        row = one.iloc[0]
        for c in one.columns:
            v = row[c]
            if isinstance(v, np.ndarray):
                print(f"  {c}: ndarray shape={v.shape} dtype={v.dtype}  "
                      f"head={np.asarray(v).ravel()[:4]}")
            elif isinstance(v, (list, tuple)):
                print(f"  {c}: {type(v).__name__} len={len(v)}  e.g. {v[:4]}")
            else:
                print(f"  {c}: {type(v).__name__} = {v!r}")
    except ModuleNotFoundError as exc:
        print(f"  [pickle/numpy mismatch] {exc}")
        print("  -> trace arrays were pickled under a different numpy; read in a")
        print("     matching env, or use the raw block layout above for the schema.")
    except Exception as exc:  # noqa: BLE001
        print(f"  [could not sample] {type(exc).__name__}: {exc}")
    print()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("h5_path", type=Path, nargs="?",
                   default=Path("temporal_pack_compact.h5"))
    p.add_argument("--skip-main", action="store_true",
                   help="skip the big /compact_temporal sample row")
    args = p.parse_args(argv)

    path = args.h5_path.expanduser().resolve()
    if not path.is_file():
        print(f"error: file not found: {path}", file=sys.stderr)
        return 1

    raw_h5py_view(path)
    pandas_store_meta(path)
    peek_side_table(path)
    if not args.skip_main:
        peek_main_table(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
