"""PREREGISTER A39 tier A null (with Amendment 1): strongest 600 blobs of the FIRST 100 frames of a run; the whole geometry selection (16 conventions x 9 swing hypotheses, blind ab-initio, tol 0.15)
is repeated on the real spots and on 8 omega-scrambled draws (frame labels permuted among the spots). Registered requirement: best scrambled fraction < 0.3 x the real one.

    python -I a39_null.py <step1 outdir holding spots.csv and info.json> [first_frames=100] [draws=8] [workers=8]
"""
import json, os, sys
sys.path[:0] = [os.environ.get("MIDAS_SX_PATH", "/Volumes/T5 EVO/travel/work/opt/MIDAS/packages/midas_sx"), os.path.dirname(os.path.abspath(__file__))]
_root = os.path.dirname(os.environ.get("MIDAS_SX_PATH", "/Volumes/T5 EVO/travel/work/opt/MIDAS/packages/midas_sx"))
sys.path[:0] = [os.path.join(_root, p) for p in ("midas_hkls", "midas_defect", "midas_stress") if os.path.isdir(os.path.join(_root, p))]
import numpy as np, pandas as pd, torch
from multiprocessing import Pool
from midas_hkls.ab_initio import index_ab_initio
from midas_defect.geometry import pixel_to_qlab, qlab_to_qsample
from midas_sx.conventions import candidate_conventions, transform_rc
from a39_geom import swing_geometry

out = sys.argv[1]; NF = int(sys.argv[2]) if len(sys.argv) > 2 else 100; DRAWS = int(sys.argv[3]) if len(sys.argv) > 3 else 8; W = int(sys.argv[4]) if len(sys.argv) > 4 else 8
sp = pd.read_csv(os.path.join(out, "spots.csv")); sp = sp[sp.frame < NF].sort_values("integrated", ascending=False).head(600).reset_index(drop=True)
H = json.load(open(os.path.join(out, "info.json")))["header"]; nr, nc = H["shape"]
HYP = [("cols", 0, 0)] + [(ax, t, o) for ax in ("cols", "rows") for t in (1, -1) for o in (1, -1)]
JOBS = [(c, h) for c in candidate_conventions() for h in HYP]


def score(job_frames):
    (conv, (ax, tzs, offs)), frames = job_frames
    torch.set_num_threads(1)
    r, c = transform_rc(sp.row.to_numpy(), sp.col.to_numpy(), nr, nc, conv.dihedral)
    geom, scan = swing_geometry(H, conv, ax, tzs, offs)
    q = pixel_to_qlab(r, c, geom, device="cpu", dtype=torch.float64)
    om = torch.as_tensor(np.radians(scan.omega_of(frames)), dtype=torch.float64)
    res = index_ab_initio(qlab_to_qsample(q, om, 0.0).numpy(), two_pi=True, tol=0.15, min_reflections=30)
    return int(res.n_indexed) if res.success else 0


if __name__ == "__main__":
    rng = np.random.default_rng(39); fr = sp.frame.to_numpy()
    with Pool(W) as pool:
        real = pool.map(score, [(j, fr) for j in JOBS]); best = int(np.argmax(real))
        print(f"{len(sp)} blobs of the first {NF} frames; real best {max(real)}/{len(sp)} = {max(real) / len(sp):.3f} at {JOBS[best][0].name} {JOBS[best][1]}", flush=True)
        fr_s = []
        for d in range(DRAWS):
            sc = pool.map(score, [(j, rng.permutation(fr)) for j in JOBS]); fr_s.append(max(sc))
            print(f"draw {d}: best scrambled {max(sc)}/{len(sp)} = {max(sc) / len(sp):.3f}", flush=True)
    ratio = max(fr_s) / max(real)
    print(f"max scrambled / real = {ratio:.3f} -> null {'HOLDS' if ratio < 0.3 else 'FAILS'} (registered: < 0.3)")
