"""How many 'credible' unindexed spots are just haloes of bright indexed spots?

The corrected figure showed several SNR>5 unindexed markers sitting on the
shoulder of a large saturated Bragg peak. If the peak finder is splitting the
tail of a strong reflection into extra 'spots', those are over-segmentation
artifacts of a grain we ALREADY have -- not evidence of new grains.

Test: for each credible unindexed spot, find the nearest INDEXED spot that is
close in omega (so both are on essentially the same frame). Proximity in
detector pixels then means it sits inside that spot's halo.
"""
from pathlib import Path
import numpy as np, pandas as pd

R = Path("/gdata/dm/1ID/2026/pokharel_jul26/analysis/au3_cubes_ff_000008")
m = pd.read_csv(R / "spot_noise_audit/spot_audit_snr.csv")
idx = m["indexed"].to_numpy().astype(bool)
ring = m["RingNumber"].to_numpy().astype(int)
snr = m["own_snr"].to_numpy()

I = m[idx]
iy, iz, io = (I["YRawPx"].to_numpy(), I["ZRawPx"].to_numpy(), I["Omega"].to_numpy())


def wrap(x):
    return (x + 180.0) % 360.0 - 180.0


def report(name, sel):
    d = m[sel]
    if len(d) == 0:
        print(f"  {name:34s} n=0"); return
    dy = d["YRawPx"].to_numpy()[:, None] - iy[None, :]
    dz = d["ZRawPx"].to_numpy()[:, None] - iz[None, :]
    do = np.abs(wrap(d["Omega"].to_numpy()[:, None] - io[None, :]))
    dist = np.sqrt(dy ** 2 + dz ** 2)
    dist[do > 2.0] = np.inf          # only count spots on ~the same frame
    nd = dist.min(axis=1)
    for cut in (20, 50):
        print(f"  {name:34s} n={len(d):5d}  within {cut:3d}px of an indexed "
              f"spot (|dOmega|<2 deg): {int((nd<cut).sum()):5d} "
              f"({(nd<cut).mean():5.1%})")


print("Halo / over-segmentation check")
report("credible unindexed (SNR>5)", ~idx & (ring > 0) & (snr > 5))
report("  of those, NImgs>=2", ~idx & (ring > 0) & (snr > 5) & (m["NImgs"] >= 2))
report("  of those, NImgs==1", ~idx & (ring > 0) & (snr > 5) & (m["NImgs"] <= 1))
report("non-credible unindexed (SNR<=5)", ~idx & (ring > 0) & (snr <= 5))

sel = ~idx & (ring > 0) & (snr > 5)
d = m[sel]
dy = d["YRawPx"].to_numpy()[:, None] - iy[None, :]
dz = d["ZRawPx"].to_numpy()[:, None] - iz[None, :]
do = np.abs(wrap(d["Omega"].to_numpy()[:, None] - io[None, :]))
dist = np.sqrt(dy ** 2 + dz ** 2); dist[do > 2.0] = np.inf
nd = dist.min(axis=1)
far = d[nd >= 50]
print(f"\nCredible unindexed spots NOT explainable as a halo (>=50 px away): "
      f"{len(far)} of {len(d)}")
print(f"  their implied GrainRadius: median {far['GrainRadius'].median():.1f} um "
      f"(p25 {far['GrainRadius'].quantile(.25):.1f}, "
      f"p75 {far['GrainRadius'].quantile(.75):.1f})")
print(f"  their SNR: median {far['own_snr'].median():.0f}")
print(f"\n=> completeness over spots that are credible AND not haloes: "
      f"{185/(185+len(far)):.1%}")
