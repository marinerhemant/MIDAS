"""Can the spot list be cleaned using columns ALREADY recorded?

Mechanism test found the dirty population is dominated by single-frame
detections (NImgs==1: 18.5% clean) and by diverged fits (ReturnCode != -1).
Both are recorded per spot. So evaluate post-detection filters against the two
things that matter: how much noise they remove, and how many INDEXED (real,
grain-explaining) spots they cost.
"""
from pathlib import Path
import numpy as np, pandas as pd

m = pd.read_csv(Path("/gdata/dm/1ID/2026/pokharel_jul26/analysis/"
                     "au3_cubes_ff_000008/spot_noise_audit/spot_audit_snr.csv"))
snr = m["own_snr"].to_numpy(); ring = m["RingNumber"].to_numpy().astype(int)
idx = m["indexed"].to_numpy().astype(bool)
nim = m["NImgs"].to_numpy(); rc = m["ReturnCode"].to_numpy()
rmse = m["FitRMSE"].to_numpy()
base = np.isfinite(snr) & (ring > 0)

n_idx0 = int((base & idx).sum())
print(f"baseline: {int(base.sum())} ring-assigned spots, "
      f"{int((base & (snr>5)).sum())} clean (SNR>5), {n_idx0} indexed\n")
print(f"  {'filter':44s} {'kept':>6s} {'clean%':>7s} {'indexed kept':>13s}")
filters = [
    ("(none)", np.ones(len(m), bool)),
    ("NImgs >= 2", nim >= 2),
    ("NImgs >= 3", nim >= 3),
    ("ReturnCode == -1", rc == -1),
    ("NImgs >= 2 AND ReturnCode == -1", (nim >= 2) & (rc == -1)),
    ("NImgs >= 2 AND FitRMSE < 2000", (nim >= 2) & (rmse < 2000)),
    ("NImgs >= 2 AND RC==-1 AND FitRMSE<2000",
     (nim >= 2) & (rc == -1) & (rmse < 2000)),
]
for name, f in filters:
    s = base & f
    if s.sum() == 0:
        print(f"  {name:44s} {0:6d}"); continue
    ki = int((s & idx).sum())
    print(f"  {name:44s} {int(s.sum()):6d} {(snr[s]>5).mean():6.1%} "
          f"{ki:6d}/{n_idx0} ({ki/max(n_idx0,1):5.1%})")
