"""Convert the pipeline's PF/unified binned files to the legacy FF C layout.

The FF pipeline runs midas_transforms' VOXEL binner, so its Spots.bin /
Data.bin / nData.bin are all in the PF (unified) format that
midas_index/c_src/IndexerUnified.c reads. FF_HEDM/src/IndexerOMP.c reads a
narrower, older layout. THREE independent widths differ — converting only one
of them (as a first attempt did) leaves the indexer reading garbage:

  Spots.bin   PF (N,10) float64  ->  FF (N,9) float64      drop col 9 = ScanNr
  nData.bin   PF (B,2)  int64    ->  FF (B,2) int32        (count, offset)
  Data.bin    PF (T,2)  int64    ->  FF (T,)  int32        drop col 1 = ScanNr

Symptom of getting nData wrong: IndexerOMP reads ``ndata[Pos*2]`` as int32
out of an int64 array, so a bin lookup lands on the wrong bin and frequently
reads an OFFSET as a COUNT — up to 220,925 instead of <=24 here. The inner
loop then scans ~10^4x too many rows, which is why the indexer ran >420 s
instead of the expected ~1 spot/s/core, and matched nothing.

Verifies the interpretation before writing: counts must sum to the number of
Data entries, and offsets must be non-decreasing.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

N_RING_BINS = 5          # HighestRingNo, from the paramstest RingNumbers
N_ETA_BINS = 3600        # ceil(360 / EtaBinSize=0.1)
N_OME_BINS = 3600        # ceil(360 / OmeBinSize=0.1)


def main() -> int:
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    dst.mkdir(parents=True, exist_ok=True)
    total_bins = N_RING_BINS * N_ETA_BINS * N_OME_BINS

    # ---- Spots.bin : (N,10) f8 -> (N,9) f8 --------------------------------
    sp = np.fromfile(src / "Spots.bin", dtype=np.float64)
    assert sp.size % 10 == 0, f"Spots.bin not 10-wide ({sp.size} doubles)"
    sp = sp.reshape(-1, 10)
    scan = np.unique(sp[:, 9])
    assert scan.size == 1, f"ScanNr is not constant ({scan[:5]}) — not a single-scan FF run"
    np.ascontiguousarray(sp[:, :9]).tofile(dst / "Spots.bin")
    print(f"Spots.bin : {sp.shape[0]} spots, 10 -> 9 cols (dropped ScanNr={scan[0]:.0f})")

    # ---- nData.bin : (B,2) i8 -> (B,2) i4 ---------------------------------
    nd = np.fromfile(src / "nData.bin", dtype=np.int64)
    assert nd.size == total_bins * 2, (
        f"nData.bin has {nd.size} int64 for {total_bins} bins — expected "
        f"{total_bins * 2}. Check N_RING/ETA/OME_BINS against the paramstest."
    )
    nd = nd.reshape(-1, 2)
    counts, offsets = nd[:, 0], nd[:, 1]
    assert np.all(np.diff(offsets) >= 0), "offsets are not non-decreasing"

    # ---- Data.bin : (T,2) i8 (rowno, scanno) -> (T,) i4 -------------------
    da = np.fromfile(src / "Data.bin", dtype=np.int64)
    assert da.size % 2 == 0, "Data.bin not int64-pair shaped"
    da = da.reshape(-1, 2)
    assert da.shape[0] == counts.sum(), (
        f"Data.bin has {da.shape[0]} pairs but counts sum to {counts.sum()}"
    )
    assert da[:, 0].min() >= 0 and da[:, 0].max() < sp.shape[0], (
        f"Data col 0 range [{da[:,0].min()}, {da[:,0].max()}] is not a valid "
        f"spot-row index into {sp.shape[0]} spots"
    )
    dscan = np.unique(da[:, 1])
    assert dscan.size == 1, f"Data col 1 (ScanNr) is not constant: {dscan[:5]}"

    for name, arr in (("count", counts), ("offset", offsets),
                      ("row", da[:, 0])):
        assert arr.max() <= np.iinfo(np.int32).max, f"{name} overflows int32"

    nd32 = np.empty((total_bins, 2), dtype=np.int32)
    nd32[:, 0] = counts
    nd32[:, 1] = offsets
    nd32.tofile(dst / "nData.bin")
    np.ascontiguousarray(da[:, 0].astype(np.int32)).tofile(dst / "Data.bin")

    print(f"nData.bin : {total_bins} bins, int64 -> int32 pairs "
          f"(count max {counts.max()}, sum {counts.sum()})")
    print(f"Data.bin  : {da.shape[0]} entries, int64 pairs -> int32 rows "
          f"(dropped ScanNr={dscan[0]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
