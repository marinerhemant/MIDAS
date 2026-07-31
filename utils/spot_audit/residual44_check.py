"""Are the 44 isolated 'credible' spots a real population or a noise tail?

A genuine small-grain population persists as the SNR cut is raised: the count
falls slowly and the implied size distribution keeps a mode. A noise tail
thresholded at SNR 5 collapses toward zero and piles tightly just above
whatever cut you chose -- the distribution is made BY the threshold.
"""
from pathlib import Path
import numpy as np, pandas as pd

R = Path("/gdata/dm/1ID/2026/pokharel_jul26/analysis/au3_cubes_ff_000008")
m = pd.read_csv(R / "spot_noise_audit/spot_audit_snr.csv")
idx = m["indexed"].to_numpy().astype(bool)
ring = m["RingNumber"].to_numpy().astype(int)
snr = m["own_snr"].to_numpy()
I = m[idx]
iy, iz, io = I["YRawPx"].to_numpy(), I["ZRawPx"].to_numpy(), I["Omega"].to_numpy()
wrap = lambda x: (x + 180.0) % 360.0 - 180.0

base = ~idx & (ring > 0) & np.isfinite(snr)
d0 = m[base]
dy = d0["YRawPx"].to_numpy()[:, None] - iy[None, :]
dz = d0["ZRawPx"].to_numpy()[:, None] - iz[None, :]
do = np.abs(wrap(d0["Omega"].to_numpy()[:, None] - io[None, :]))
dist = np.sqrt(dy**2 + dz**2); dist[do > 2.0] = np.inf
nd = dist.min(axis=1)
d0 = d0.assign(halo_dist=nd)

print("Isolated (>=50 px from any indexed spot) unindexed spots vs SNR cut")
print(f"  {'cut':>5s} {'n':>6s} {'NImgs==1':>9s} {'med radius um':>14s} {'med SNR':>8s}")
for cut in (5, 6, 8, 10, 15, 20, 30, 50):
    s = d0[(d0["own_snr"] > cut) & (d0["halo_dist"] >= 50)]
    if len(s) == 0:
        print(f"  {cut:5d} {0:6d}"); continue
    print(f"  {cut:5d} {len(s):6d} {(s['NImgs']<=1).mean():8.0%} "
          f"{s['GrainRadius'].median():14.1f} {s['own_snr'].median():8.1f}")

print("\nIndexed spots for comparison, same cuts (a REAL population):")
for cut in (5, 10, 50, 100, 1000):
    s = I[I["own_snr"] > cut] if "own_snr" in I else None
    print(f"  {cut:5d} {len(m[idx & (snr>cut)]):6d}")

s44 = d0[(d0["own_snr"] > 5) & (d0["halo_dist"] >= 50)]
print(f"\nThe 44: SNR distribution")
h, e = np.histogram(s44["own_snr"], bins=[5, 6, 7, 8, 10, 15, 1e9])
for c, lo, hi in zip(h, e[:-1], e[1:]):
    print(f"  SNR {lo:>5.0f}-{hi if hi<1e8 else np.inf:<6.0f} {c:4d}  " + "#" * c)
print(f"\n  NImgs==1: {(s44['NImgs']<=1).sum()}/{len(s44)}")
print(f"  implied radius IQR {s44['GrainRadius'].quantile(.25):.1f}-"
      f"{s44['GrainRadius'].quantile(.75):.1f} um "
      f"(indexed IQR {I['GrainRadius'].quantile(.25):.0f}-"
      f"{I['GrainRadius'].quantile(.75):.0f} um)")
