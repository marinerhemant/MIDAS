"""Can single-frame detections be REAL signal? (They can.)

I recommended NImgs>=2 as a spot-list filter on the strength of NImgs tracking
cleanliness. But omega width is set by mosaicity, beam divergence and energy
bandwidth -- NOT by grain size -- so a small or undeformed grain can satisfy
Bragg within one 0.25 deg frame. And this dataset has exactly TWO large mosaic
grains spanning 12+ frames, which is the worst possible sample from which to
derive a general omega-multiplicity rule.

So: are there single-frame spots that are demonstrably real?
"""
from pathlib import Path
import numpy as np, pandas as pd

m = pd.read_csv(Path("/gdata/dm/1ID/2026/bt_1id_jul26/analysis/"
                     "au3_cubes_ff_000008/spot_noise_audit/spot_audit_snr.csv"))
snr = m["own_snr"].to_numpy(); ring = m["RingNumber"].to_numpy().astype(int)
idx = m["indexed"].to_numpy().astype(bool); nim = m["NImgs"].to_numpy()
base = np.isfinite(snr) & (ring > 0)

print("1. INDEXED (grain-explaining, certainly real) spots that are single-frame")
s = base & idx & (nim <= 1)
print(f"   n = {int(s.sum())} of {int((base & idx).sum())} indexed "
      f"({s.sum()/max((base&idx).sum(),1):.1%})")
if s.sum():
    print(f"   their SNR   : {', '.join(f'{v:.0f}' for v in np.sort(snr[s])[::-1])}")
    print(f"   their IMax  : median {np.median(m.loc[s,'IMax']):.0f}")
    print("   -> these are REAL spots the filter would delete.")

print("\n2. Credible (SNR>5) spots that are single-frame")
c = base & (snr > 5)
print(f"   {int((c & (nim<=1)).sum())} of {int(c.sum())} credible spots are "
      f"single-frame ({(c & (nim<=1)).sum()/max(c.sum(),1):.1%})")
print(f"   their median SNR {np.median(snr[c & (nim<=1)]):.1f}, "
      f"max {np.max(snr[c & (nim<=1)]):.0f}")

print("\n3. What NImgs>=2 actually costs, by SNR band")
for lo, hi, lbl in ((5, 10, "SNR 5-10"), (10, 100, "SNR 10-100"),
                    (100, 1e12, "SNR > 100")):
    sel = base & (snr >= lo) & (snr < hi)
    lost = sel & (nim <= 1)
    if sel.sum():
        print(f"   {lbl:12s} n={int(sel.sum()):5d}  lost to NImgs>=2: "
              f"{int(lost.sum()):5d} ({lost.sum()/sel.sum():5.1%})")

print("\n4. Is SNR the better discriminator? (does it subsume NImgs?)")
for lbl, sel in (("NImgs>=2 only          ", base & (nim >= 2)),
                 ("SNR>5 only             ", base & (snr > 5)),
                 ("SNR>5 AND NImgs>=2     ", base & (snr > 5) & (nim >= 2))):
    n_idx = int((sel & idx).sum())
    print(f"   {lbl} kept {int(sel.sum()):5d}   indexed kept "
          f"{n_idx}/{int((base&idx).sum())}")
