#!/usr/bin/env python3
"""
Print the structure of temporal_pack_compact.h5 (or any HDF5 file).

Uses PyTables (tables), which matches the stack used by pandas.read_hdf.

Usage:
  python scripts/inspect_temporal_pack_h5.py /path/to/temporal_pack_compact.h5
  python scripts/inspect_temporal_pack_h5.py /path/to/temporal_pack_compact.h5 --pandas
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _attrs_lines(node) -> list[str]:
    out: list[str] = []
    for name in node.attrs._f_list():
        try:
            val = node.attrs[name]
        except Exception as exc:  # noqa: BLE001
            val = f"<unreadable: {exc}>"
        out.append(f"      @{name} = {val!r}")
    return out


def print_pytables_tree(path: Path) -> None:
    import tables as tb

    print(f"=== PyTables / HDF5 tree: {path} ===\n")
    with tb.open_file(str(path), mode="r") as hf:
        print(f"File title: {hf.title!r}\n")
        for node in hf.walk_nodes():
            depth = node._v_depth
            indent = "  " * depth
            cls = node.__class__.__name__
            line = f"{indent}{node._v_pathname}  ({cls})"

            if isinstance(node, tb.Table):
                nrows = node.nrows
                cols = ", ".join(node.colnames)
                print(f"{line}\n{indent}  rows={nrows}  columns: {cols}")
            elif isinstance(node, tb.Array):
                print(f"{line}\n{indent}  shape={node.shape}  dtype={node.atom.dtype}")
            elif isinstance(node, tb.Group):
                print(line)
            else:
                print(line)

            al = _attrs_lines(node)
            if al:
                for a in al:
                    print(a)
        print()


def print_pandas_store(path: Path) -> None:
    import pandas as pd

    print(f"=== pandas HDFStore metadata: {path} ===\n")
    with pd.HDFStore(str(path), mode="r") as store:
        keys = list(store.keys())
        print(f"keys ({len(keys)}): {keys}\n")
        for key in keys:
            st = store.get_storer(key)
            print(f"{key}")
            print(f"  type: {type(st).__name__}")
            if hasattr(st, "table") and st.table is not None:
                t = st.table
                print(f"  table.nrows: {t.nrows}")
                print(f"  table.colnames: {t.colnames}")
            if hasattr(st, "attrs") and st.attrs is not None:
                for an in sorted(st.attrs.keys()):
                    print(f"  attr {an}: {st.attrs[an]!r}")
            print()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "h5_path",
        type=Path,
        nargs="?",
        default=Path("temporal_pack_compact.h5"),
        help="Path to HDF5 file (default: ./temporal_pack_compact.h5)",
    )
    p.add_argument(
        "--pandas",
        action="store_true",
        help="Also print pandas HDFStore keys and storer metadata",
    )
    args = p.parse_args(argv)

    path: Path = args.h5_path.expanduser().resolve()
    if not path.is_file():
        print(f"error: file not found: {path}", file=sys.stderr)
        return 1

    print_pytables_tree(path)
    if args.pandas:
        print_pandas_store(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
