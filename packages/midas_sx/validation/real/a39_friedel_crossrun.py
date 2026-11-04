"""PREREG_friedel_geometry.md extension (not part of the registered verdict): Friedel pairs ACROSS runs, shared detector geometry, one unknown sample-frame rotation per extra run
(midas_sx.friedel_geometry.fit_multirun; relation and weights in its docstring). Arms: shared geometry; tx and wedge allowed to differ per run.

    python -I a39_friedel_crossrun.py <out dir (rNN/obs.pkl, reduction.json)> <step1 dir> <runs> [n_boot=100]
"""
import json, os, sys
_sx = os.environ.get("MIDAS_SX_PATH", "/Volumes/T5 EVO/travel/work/opt/MIDAS/packages/midas_sx")
sys.path[:0] = [_sx, os.path.dirname(os.path.abspath(__file__))] + [os.path.join(os.path.dirname(_sx), p) for p in ("midas_hkls", "midas_defect", "midas_stress") if os.path.isdir(os.path.join(os.path.dirname(_sx), p))]
import numpy as np, pandas as pd
from midas_sx.conventions import Convention
from midas_sx.friedel_geometry import fit_multirun
from a39_geom import swing_geometry

d, step1, runs = sys.argv[1], sys.argv[2], sys.argv[3].split(",")
n_boot = int(sys.argv[4]) if len(sys.argv) > 4 else 100
H = json.load(open(os.path.join(step1, "run02", "info.json")))["header"]
g0, _ = swing_geometry(H, Convention(0, -1), "cols", -1, -1)
obs = [pd.read_pickle(os.path.join(d, r, "obs.pkl")) for r in runs]
fits = {r: json.load(open(os.path.join(d, r, "reduction.json"))) for r in runs}
pd.set_option("display.width", 200)
out = {}
for arm, per in (("shared geometry", ()), ("tx and wedge per run", ("tx", "wedge"))):
    f = fit_multirun(obs, g0, n_boot=n_boot, seed=3, per_run=per)
    print(f"\n== {arm}: {f.n_groups} hkl groups kept ({f.n_groups_trimmed} trimmed), {f.n_cross_run_groups} with a pair spanning runs; rms {f.rms_final:.5f}, robust scale {f.scale:.5f}")
    print(f.table().round(4).to_string(index=False))
    for k in range(1, len(runs)):
        print(f"   P_{runs[k]}: rotation {np.degrees(np.linalg.norm(f.rotvecs[k - 1])):.3f} deg")
    if per:
        for j, n_ in enumerate(per):
            print(f"   {n_}: run {runs[0]} (reference) {f.values[f.names.index(n_)]:+.4f}; deviations " + ", ".join(f"{runs[k]} {f.dev[k - 1, j]:+.4f} +/- {f.sigma_dev[k - 1, j]:.4f}" for k in range(1, len(runs))))
    out[arm] = dict(table=f.table().to_dict("list"), rms=f.rms_final, n_groups=f.n_groups, dev=None if f.dev is None else f.dev.tolist(), sigma_dev=None if f.sigma_dev is None else f.sigma_dev.tolist())
print("\norientation-fit values per run (reduction.json): " + "; ".join(f"{r}: tx {v['tilts_deg'][0]:+.3f} ty {v['tilts_deg'][1]:+.3f} tz {v['tilts_deg'][2]:+.3f} wedge {v['wedge_deg']:+.3f} bcz {v['centre_rc'][0]:.2f} bcy {v['centre_rc'][1]:.2f} lsd {v['lsd_mm']:.3f}" for r, v in fits.items()))
json.dump(out, open(os.path.join(d, "friedel_crossrun_" + "_".join(runs) + ".json"), "w"), indent=1, default=float)
