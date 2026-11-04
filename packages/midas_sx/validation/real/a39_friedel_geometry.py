"""PREREG_friedel_geometry.md, real data: Friedel-pair estimates of tx, wedge (and the rest of the geometry) per run, from the in-run pairs of obs.pkl.

    python -I a39_friedel_geometry.py <out dir (holds rNN/obs.pkl, reduction.json)> <step1 dir (run02/info.json)> <runs, e.g. r02,r03,r04> [n_boot=200] [n_null=20]

Starts: (1) header geometry (independent of the orientation fit), (2) the orientation-fit geometry (basin check). Null: mates scrambled within the run's pair list (n_null draws).
"""
import json, os, sys
_sx = os.environ.get("MIDAS_SX_PATH", "/Volumes/T5 EVO/travel/work/opt/MIDAS/packages/midas_sx")
sys.path[:0] = [_sx, os.path.dirname(os.path.abspath(__file__))] + [os.path.join(os.path.dirname(_sx), p) for p in ("midas_hkls", "midas_defect", "midas_stress") if os.path.isdir(os.path.join(os.path.dirname(_sx), p))]
import numpy as np, pandas as pd
from midas_sx.conventions import Convention
from midas_sx.friedel_geometry import FRIEDEL_PARAMS, fit_pairs, labelled_pairs, with_params
from a39_geom import swing_geometry

d, step1, runs = sys.argv[1], sys.argv[2], sys.argv[3].split(",")
n_boot = int(sys.argv[4]) if len(sys.argv) > 4 else 200; n_null = int(sys.argv[5]) if len(sys.argv) > 5 else 20
H = json.load(open(os.path.join(step1, "run02", "info.json")))["header"]
g_head, scan = swing_geometry(H, Convention(0, -1), "cols", -1, -1)
pd.set_option("display.width", 220); pd.set_option("display.max_columns", 30)
res = {}
for r in runs:
    red = json.load(open(os.path.join(d, r, "reduction.json"))); obs = pd.read_pickle(os.path.join(d, r, "obs.pkl"))
    fitgeom = dict(tx=red["tilts_deg"][0], ty=red["tilts_deg"][1], tz=red["tilts_deg"][2], wedge=red["wedge_deg"], bcz=red["centre_rc"][0], bcy=red["centre_rc"][1], lsd=red["lsd_mm"])
    pairs = labelled_pairs(obs)
    print(f"\n===== {r}: {len(obs)} observations, {len(pairs)} in-run pairs, {pairs.group.nunique()} hkl groups")
    out = {}
    for start_name, g0 in (("header start", g_head), ("orientation-fit start", with_params(g_head, fitgeom))):
        fit = fit_pairs(pairs, g0, n_boot=n_boot if start_name == "header start" else 0, seed=1)
        t = fit.table(); t["orientation_fit"] = [fitgeom[n] for n in fit.names]
        t["diff"] = t["value"] - t["orientation_fit"]
        print(f"-- {start_name}: success {fit.success}, rms start {fit.rms_start:.4f} -> final {fit.rms_final:.5f}, robust scale {fit.scale:.5f}; scaled singular values {np.round(fit.singular_values, 4)}")
        print(t.round(4).to_string(index=False))
        print("   weakest direction (scaled params):", dict(zip(fit.names, np.round(fit.weakest_direction, 2))))
        out[start_name] = dict(table=t.to_dict("list"), rms=fit.rms_final, sv=fit.singular_values.tolist(), n_pairs=fit.n_pairs, n_groups=fit.n_groups)
    # null: scrambled mates
    rng = np.random.default_rng(7); null_tx, null_w, null_rms = [], [], []
    for _ in range(n_null):
        bad = pairs.copy()
        for c in ("row2", "col2", "om2"):
            bad[c] = rng.permutation(bad[c].to_numpy())
        f = fit_pairs(bad, g_head)
        v = dict(zip(f.names, f.values)); null_tx.append(v["tx"]); null_w.append(v["wedge"]); null_rms.append(f.rms_final)
    print(f"-- null (scrambled mates, {n_null} draws): tx median {np.median(null_tx):+.3f} [{np.min(null_tx):+.2f}, {np.max(null_tx):+.2f}], wedge median {np.median(null_w):+.3f} [{np.min(null_w):+.2f}, {np.max(null_w):+.2f}], rms median {np.median(null_rms):.4f} (true pairs {out['header start']['rms']:.5f})")
    out["null"] = dict(tx=null_tx, wedge=null_w, rms=null_rms)
    res[r] = out
json.dump(res, open(os.path.join(d, "friedel_geometry_" + "_".join(runs) + ".json"), "w"), indent=1, default=float)
