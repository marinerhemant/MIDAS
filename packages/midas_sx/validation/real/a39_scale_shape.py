"""A39: shape of the fitted smooth scale k(frame) per run (reference-free).  python -I a39_scale_shape.py <out dir> <runs>"""
import os, sys
_sx = os.environ.get("MIDAS_SX_PATH", "/Volumes/T5 EVO/travel/work/opt/MIDAS/packages/midas_sx")
sys.path[:0] = [_sx] + [os.path.join(os.path.dirname(_sx), p) for p in ("midas_hkls", "midas_defect", "midas_stress") if os.path.isdir(os.path.join(os.path.dirname(_sx), p))]
import numpy as np, pandas as pd
from midas_hkls import SpaceGroup
from midas_sx.merge import scale_smooth
out, runs = sys.argv[1], sys.argv[2].split(",")
sg = SpaceGroup.from_hm("P 21 21 21")
O = pd.concat([pd.read_pickle(os.path.join(out, r, "obs.pkl")).assign(run=r) for r in runs], ignore_index=True)
S, _ = scale_smooth(O, sg, n_knots=8)
S["bin"] = pd.cut(S.frame, np.linspace(0, 850, 9))
for r in runs:
    t = S[S.run == r].groupby("bin", observed=True).k_smooth.mean()
    print(r, "k(frame) at frame-bin centres 53..797:", np.round(t.to_numpy() / t.mean(), 3).tolist())
