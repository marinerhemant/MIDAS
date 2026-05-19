"""Convert the legacy FF parity fixture (ref_dataset/) to the unified-format
fixture (ref_dataset_unified/) consumed by midas_indexer.

Legacy FF format:
- Spots.bin: (N, 9) float64
- Data.bin: flat int32 spotRows
- nData.bin: flat int32 (count, offset) pairs, offset in int32 units
- no positions.csv

Unified PF-format:
- Spots.bin: (N, 10) float64 — col 9 = ScanNr (0 for FF)
- Data.bin: flat int64 (spotRow, scannrobs) pairs; scannrobs=0 for FF
- nData.bin: flat int64 (count, offset) pairs, offset in PAIR units
- positions.csv: single line "0.000000\n"

The offset numerically stays the same: each old int32 maps 1-to-1 to a new
(spotRow, 0) pair, so the bin's logical offset index is identical.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np

SRC = Path(__file__).parent / "ref_dataset"
DST = Path(__file__).parent / "ref_dataset_unified"


def main() -> None:
    DST.mkdir(parents=True, exist_ok=True)
    (DST / "golden").mkdir(parents=True, exist_ok=True)

    # 1. Spots.bin: 9 cols → 10 cols (append ScanNr=0)
    spots = np.fromfile(SRC / "Spots.bin", dtype=np.float64).reshape(-1, 9)
    spots10 = np.zeros((spots.shape[0], 10), dtype=np.float64)
    spots10[:, :9] = spots
    spots10.tofile(DST / "Spots.bin")
    print(f"Spots.bin: {spots.shape} → {spots10.shape}")

    # 2. Data.bin: int32 flat → int64 (spotRow, 0) pairs
    data_i32 = np.fromfile(SRC / "Data.bin", dtype=np.int32)
    data_pairs = np.zeros((data_i32.size, 2), dtype=np.int64)
    data_pairs[:, 0] = data_i32.astype(np.int64)
    data_pairs.tofile(DST / "Data.bin")
    print(f"Data.bin: {data_i32.shape} int32 → {data_pairs.shape} int64 pairs")

    # 3. nData.bin: int32 flat (count, offset) → int64 (count, offset).
    # Offsets stay numerically equal because each old int slot maps 1:1 to
    # a new pair slot.
    ndata_i32 = np.fromfile(SRC / "nData.bin", dtype=np.int32)
    ndata_i64 = ndata_i32.astype(np.int64)
    ndata_i64.tofile(DST / "nData.bin")
    print(f"nData.bin: {ndata_i32.shape} int32 → {ndata_i64.shape} int64")

    # 4. positions.csv: single FF voxel at y=0
    (DST / "positions.csv").write_text("0.000000\n")
    print("positions.csv: single line '0.000000'")

    # 5. Copy CSV + golden + paramstest verbatim
    for fn in ("SpotsToIndex.csv", "hkls.csv"):
        shutil.copy2(SRC / fn, DST / fn)
        print(f"copied {fn}")

    for fn in ("IndexBest.bin", "IndexBestFull.bin"):
        shutil.copy2(SRC / "golden" / fn, DST / "golden" / fn)
        print(f"copied golden/{fn}")

    # 6. paramstest.txt: rewrite OutputFolder so the unified binary writes
    # into ref_dataset_unified/midas/ instead of into the legacy dir.
    src_params = (SRC / "paramstest.txt").read_text()
    legacy_out = "/Users/hsharma/opt/MIDAS/packages/midas_index/tests/data/ref_dataset/midas"
    unified_out = str((DST / "midas").resolve())
    dst_params = src_params.replace(legacy_out, unified_out)
    # Also write IDsFileName so the unified main can find SpotsToIndex.csv
    if "IDsFileName" not in dst_params:
        dst_params = dst_params.rstrip("\n") + "\nIDsFileName SpotsToIndex.csv\n"
    (DST / "paramstest.txt").write_text(dst_params)
    (DST / "midas").mkdir(exist_ok=True)
    print(f"paramstest.txt: OutputFolder → {unified_out}")


if __name__ == "__main__":
    main()
