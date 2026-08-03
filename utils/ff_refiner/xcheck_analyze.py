"""Correct analysis of the six-way refiner crosscheck.

Two bugs in the first pass, both worth recording:

  * orientation was compared as a raw matrix angle, which reported a median of
    exactly 120.000 deg for EVERY pair -- the cubic symmetry angle. Two
    symmetry-equivalent orientations are the same orientation. Use
    midas_stress.orientation.misorientation_om_batch (canonical; returns
    RADIANS) instead of rolling one.
  * every row was compared, including seeds the refiner never filled, which
    are identically zero in both files and drove the median to 0.00.
"""
from pathlib import Path
import numpy as np

R = Path("/gdata/dm/1ID/2026/pokharel_jul26/analysis/au3_cubes_ff_000008")
F = np.load(R / "refiner_crosscheck/fits.npy", allow_pickle=True).item()
C_ORIENT, C_POS, C_LAT = slice(1, 10), slice(11, 14), slice(15, 21)
SG = 225

tags = list(F)
n = min(F[t].shape[0] for t in tags)
print("rows per implementation:", {t: F[t].shape[0] for t in tags}, "-> using", n)

# Which rows did EVERY implementation actually fill?
filled = np.ones(n, bool)
for t in tags:
    a = F[t][:n]
    filled &= (np.abs(a[:, C_ORIENT]).sum(axis=1) > 0) & np.isfinite(a).all(axis=1)
print(f"rows filled by ALL implementations: {int(filled.sum())} of {n}")
for t in tags:
    a = F[t][:n]
    own = (np.abs(a[:, C_ORIENT]).sum(axis=1) > 0)
    print(f"  {t:12s} filled {int(own.sum()):4d}")

from midas_stress.orientation import misorientation_om_batch

print(f"\nPairwise agreement over the {int(filled.sum())} commonly-filled seeds")
print(f"  {'pair':26s} {'|dpos| um  med / p95 / max':>34s} "
      f"{'miso deg med / max':>22s} {'|da| A max':>12s}")
for i, ta in enumerate(tags):
    for tb in tags[i + 1:]:
        A, B = F[ta][:n][filled], F[tb][:n][filled]
        dp = np.linalg.norm(A[:, C_POS] - B[:, C_POS], axis=1)
        mis = np.degrees(misorientation_om_batch(
            A[:, C_ORIENT], B[:, C_ORIENT], SG))
        da = np.abs(A[:, C_LAT.start] - B[:, C_LAT.start])
        print(f"  {ta+' vs '+tb:26s} {np.median(dp):9.3f} "
              f"{np.percentile(dp,95):9.3f} {dp.max():9.3f} "
              f"{np.median(mis):10.4f} {mis.max():9.4f} {da.max():11.2e}")
