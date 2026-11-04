"""A39 diagnostic: why do only ~half of the strongest blobs index? Predict every reflection with the REFINED geometry and U of one run, match every blob to the nearest prediction in (frame, row, col),
and read (a) the matched fraction for the strongest N blobs against a shifted-prediction null, (b) the position residuals of matched blobs against detector position / omega, (c) what the unmatched strong blobs are
(2theta histogram, positions).

    python -I a39_diag_unindexed.py <run dir with r02/ (spots_blind.csv, U.npy, reduction.json)> <step1 dir with run02/info.json> <cif> <rNN>
"""
import json, os, sys
_sx = os.environ.get("MIDAS_SX_PATH", "/Volumes/T5 EVO/travel/work/opt/MIDAS/packages/midas_sx")
sys.path[:0] = [_sx, os.path.dirname(os.path.abspath(__file__))] + [os.path.join(os.path.dirname(_sx), p) for p in ("midas_hkls", "midas_defect", "midas_stress") if os.path.isdir(os.path.join(os.path.dirname(_sx), p))]
import numpy as np, pandas as pd, gemmi
from dataclasses import replace
from scipy.spatial import cKDTree
from midas_hkls import Lattice
from midas_sx.conventions import Convention
from midas_sx.geometry import make_geometry, spot_directions
from midas_sx.predict import enumerate_hkl, predicted_crossings
from a39_geom import swing_geometry

d, step1, cif, run = sys.argv[1:5]
H = json.load(open(os.path.join(step1, "info.json")))["header"]
red = json.load(open(os.path.join(d, run, "reduction.json")))
U = np.load(os.path.join(d, run, "U.npy")); sp = pd.read_csv(os.path.join(d, run, "spots_blind.csv"))
ss = gemmi.read_small_structure(cif); lat = Lattice(ss.cell.a, ss.cell.b, ss.cell.c, 90, 90, 90)
g0, scan = swing_geometry(H, Convention(0, -1), "cols", -1, -1)
tx, ty, tz = red["tilts_deg"]; rc = red["centre_rc"]
geom = replace(g0, tx_deg=tx, ty_deg=ty, tz_deg=tz, wedge_deg=red["wedge_deg"], lsd_um=red["lsd_mm"] * 1000.0, bcz_px=rc[0], bcy_px=rc[1])
pred = predicted_crossings(U, lat, enumerate_hkl(lat, 0.60), geom, scan, margin_px=3.0)
print(f"{len(sp)} blobs, {len(pred)} predicted crossings on the detector inside the scan; geometry tilts {tx:+.3f} {ty:+.3f} {tz:+.3f} wedge {red['wedge_deg']:+.3f} centre {rc} Lsd {red['lsd_mm']:.3f}")
SF = 4.0                                                    # 1 frame counts as SF px in the match metric


def match(blobs, P):
    tree = cKDTree(np.c_[P.frame.to_numpy() * SF, P.row.to_numpy(), P.col.to_numpy()])
    dist, j = tree.query(np.c_[blobs.frame.to_numpy() * SF, blobs.row.to_numpy(), blobs.col.to_numpy()])
    return dist, j


sp = sp.sort_values("integrated", ascending=False).reset_index(drop=True)
rng = np.random.default_rng(1)
print("\nmatched fraction within distance (px-equivalent, 1 frame = 4 px) for the strongest N blobs; null = predictions shifted by a random (frame, row, col) offset, mean of 20")
for N in (100, 300, 600, 2000, len(sp)):
    b = sp.head(N); dist, _ = match(b, pred)
    nulls = []
    for _ in range(20):
        Ps = pred.copy(); Ps["frame"] += rng.uniform(15, 60) * rng.choice([-1, 1]); Ps["row"] += rng.uniform(20, 120) * rng.choice([-1, 1]); Ps["col"] += rng.uniform(20, 120) * rng.choice([-1, 1])
        nulls.append([(match(b, Ps)[0] < t).mean() for t in (2, 4, 8)])
    nl = np.mean(nulls, axis=0)
    print(f"N={N:6d}: <2: {(dist < 2).mean():.3f} (null {nl[0]:.3f})  <4: {(dist < 4).mean():.3f} (null {nl[1]:.3f})  <8: {(dist < 8).mean():.3f} (null {nl[2]:.3f})")

dist, j = match(sp, pred); sp["dist"] = dist
P = pred.iloc[j].reset_index(drop=True)
sp["dr"] = sp.row - P.row.to_numpy(); sp["dc"] = sp.col - P.col.to_numpy(); sp["dfr"] = sp.frame - P.frame.to_numpy()
sp["h"], sp["k"], sp["l"] = P.h.to_numpy(), P.k.to_numpy(), P.l.to_numpy()
s_, tth, _ = spot_directions(sp.row.to_numpy(), sp.col.to_numpy(), geom); sp["tth"] = tth
top = sp.head(600); mm = top[top.dist < 4]; um = top[top.dist >= 4]
print(f"\nstrongest 600: matched (<4) {len(mm)}, unmatched {len(um)}")
print("matched residuals (blob - prediction): median / robust sd   dr %.2f/%.2f px   dc %.2f/%.2f px   dframe %.2f/%.2f" % tuple(
    v for x in (mm.dr, mm.dc, mm.dfr) for v in (x.median(), 1.4826 * (x - x.median()).abs().median())))
def trend(col, bins, v):
    b = pd.cut(mm[col], bins); print(f"-- matched {v} vs {col}"); print(mm.groupby(b, observed=True)[v].agg(n="size", med="median").round(2).T.to_string())
trend("row", np.linspace(0, 1679, 8), "dr"); trend("col", np.linspace(0, 1475, 8), "dc"); trend("row", np.linspace(0, 1679, 8), "dc"); trend("col", np.linspace(0, 1475, 8), "dr")
trend("frame", np.linspace(0, 850, 8), "dfr"); trend("frame", np.linspace(0, 850, 8), "dr"); trend("frame", np.linspace(0, 850, 8), "dc")
print("\nstrongest-600 matched fraction by frame range:"); b = pd.cut(top.frame, np.linspace(0, 850, 9)); print(top.groupby(b, observed=True).dist.agg(n="size", frac_matched=lambda x: (x < 4).mean()).round(3).T.to_string())
print("\nstrongest-600 matched fraction by detector column:"); b = pd.cut(top.col, np.linspace(0, 1475, 9)); print(top.groupby(b, observed=True).dist.agg(n="size", frac_matched=lambda x: (x < 4).mean()).round(3).T.to_string())
print("\nunmatched strongest-600: 2theta histogram (deg) vs matched"); bins = np.arange(0, 50, 2.5)
print(pd.DataFrame(dict(unmatched=np.histogram(um.tth, bins)[0], matched=np.histogram(mm.tth, bins)[0]), index=bins[:-1]).T.to_string())
print("\nunmatched strongest: finer 2theta histogram, 0.25 deg bins, top 12 bins"); h, e = np.histogram(um.tth, np.arange(0, 50, 0.25)); o = np.argsort(-h)[:12]; print([(round(e[i], 2), int(h[i])) for i in o])
print("distance of unmatched to the nearest prediction (px-equivalent):", np.round(np.percentile(um.dist, [10, 25, 50, 75, 90]), 1))
sp.head(2000).to_csv(os.path.join(d, run, "diag_unindexed_top2000.csv"), index=False)
