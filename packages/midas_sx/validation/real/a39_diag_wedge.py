"""A39 diagnostic: how does the blind indexing score of the winning subset convention (identity|omega-, swing cols, tz -1, off -1) change with the omega range of the spots used?
Strongest 600 blobs within frames [0, n) for n = 100 ... 850; also the strongest 600 of each 100-frame window."""
import json, os, sys
sys.path[:0] = [os.environ.get("MIDAS_SX_PATH", "/Volumes/T5 EVO/travel/work/opt/MIDAS/packages/midas_sx"), os.path.dirname(os.path.abspath(__file__))]
_root = os.path.dirname(os.environ.get("MIDAS_SX_PATH", "/Volumes/T5 EVO/travel/work/opt/MIDAS/packages/midas_sx"))
sys.path[:0] = [os.path.join(_root, p) for p in ("midas_hkls", "midas_defect", "midas_stress") if os.path.isdir(os.path.join(_root, p))]
import numpy as np, pandas as pd, torch
from midas_hkls.ab_initio import index_ab_initio
from midas_defect.geometry import pixel_to_qlab, qlab_to_qsample
from midas_sx.conventions import Convention, transform_rc
from a39_geom import swing_geometry
out = sys.argv[1]
sp = pd.read_csv(os.path.join(out, "spots.csv"))
H = json.load(open(os.path.join(out, "info.json")))["header"]; nr, nc = H["shape"]
conv = Convention(0, -1)
def run(d):
    r, c = transform_rc(d.row.to_numpy(), d.col.to_numpy(), nr, nc, conv.dihedral)
    g, scan = swing_geometry(H, conv, "cols", -1, -1)
    q = pixel_to_qlab(r, c, g, device="cpu", dtype=torch.float64)
    om = torch.as_tensor(np.radians(scan.omega_of(d.frame.to_numpy())), dtype=torch.float64)
    res = index_ab_initio(qlab_to_qsample(q, om, 0.0).numpy(), two_pi=True, tol=0.15, min_reflections=30)
    return (int(res.n_indexed), len(d), np.round(res.cell[:3], 3) if res.success else None)
print("cumulative from frame 0:")
for n in (100, 200, 300, 450, 600, 850):
    d = sp[sp.frame < n].sort_values("integrated", ascending=False).head(600)
    print(n, run(d), flush=True)
print("100-frame windows:")
for a in range(0, 850, 100):
    d = sp[(sp.frame >= a) & (sp.frame < a + 100)].sort_values("integrated", ascending=False).head(600)
    print(a, run(d), flush=True)
