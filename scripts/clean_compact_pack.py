#!/usr/bin/env python3
"""
Rewrite temporal_pack_compact.h5 into a fresh file that:
  * drops the alignment-injected columns from /compact_temporal
    (com_aligned_mapzebrain, single_region, region_aligned_mapzebrain),
  * eliminates all dead/unaccounted space (fresh mode="w" write),
  * preserves the /com_aligned side table as-is.

MUST run in an env whose numpy matches what pickled the trace block
(here: the `alignment` env, numpy 2.x). Non-destructive: original is untouched.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

SRC = Path(sys.argv[1] if len(sys.argv) > 1
           else "/Volumes/Kobi/DOI_animals_good/temporal_pack_compact.h5")
DST = Path(sys.argv[2] if len(sys.argv) > 2
           else SRC.with_name("temporal_pack_compact_clean.h5"))

DROP_COLS = ["com_aligned_mapzebrain", "single_region", "region_aligned_mapzebrain"]

print(f"src: {SRC}  ({SRC.stat().st_size/1e9:.3f} GB)")
print(f"dst: {DST}")

# --- main table: read (unpickles traces), drop alignment cols, fresh write ---
base = pd.read_hdf(SRC, key="compact_temporal")
print(f"\n/compact_temporal: {len(base)} rows, columns before: {list(base.columns)}")
present = [c for c in DROP_COLS if c in base.columns]
base = base.drop(columns=present, errors="ignore")
print(f"dropped: {present}")
print(f"columns after: {list(base.columns)}")

base.to_hdf(DST, key="compact_temporal", mode="w", format="fixed")
print("wrote /compact_temporal -> fresh file")

# --- side table: copy verbatim if present ---
try:
    side = pd.read_hdf(SRC, key="com_aligned")
    side.to_hdf(DST, key="com_aligned", mode="a", format="fixed")
    print(f"copied /com_aligned: {len(side)} rows")
except KeyError:
    print("/com_aligned not present — skipped")

print(f"\nDONE. dst size: {DST.stat().st_size/1e9:.3f} GB "
      f"(was {SRC.stat().st_size/1e9:.3f} GB)")
print("Original is untouched. Verify, then swap manually:")
print(f"  mv {SRC} {SRC.with_suffix('.bloated.bak.h5')}")
print(f"  mv {DST} {SRC}")
