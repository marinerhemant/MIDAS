"""PREREGISTER A38 tiers B1 and B2: merge the per-run observation tables (inter-run scale from `scale_groups`), completeness, R_int, and F² against (i) the CIF-computed F² and
(ii) the bundled SAINT / SADABS reduction.

    python -I a38_merge.py <out dir with r01 r02 ...> <cif> <saint hkl> <runs, e.g. r01,r02,r03,r05,r06>
"""
import json, os, sys
sys.path[:0] = [os.environ.get("MIDAS_SX_PATH", "/Volumes/T5 EVO/travel/work/opt/MIDAS/packages/midas_sx")]
import numpy as np, pandas as pd, gemmi, torch
from scipy.optimize import least_squares
from midas_hkls import Atom, Crystal, Lattice, SpaceGroup
from midas_sx.geometry import spot_directions
from midas_sx.merge import merge_observations, r_int, scale_groups, write_hkl
from midas_sx.predict import enumerate_hkl
from midas_sx.refine import EIGHT_PI2, RefineModel

out, cif, saint, runs = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4].split(",")
sg = SpaceGroup.from_hm("P 21 m a")
ss = gemmi.read_small_structure(cif)
crystal = Crystal(Lattice(ss.cell.a, ss.cell.b, ss.cell.c, ss.cell.alpha, ss.cell.beta, ss.cell.gamma), sg,
                  [Atom(s.element.name, (s.fract.x, s.fract.y, s.fract.z), s.occ, EIGHT_PI2 * s.u_iso, label=s.label) for s in ss.sites], "thiourea130K")
obs = pd.concat([pd.read_pickle(os.path.join(out, r, "obs.pkl")).assign(run=r) for r in runs], ignore_index=True)
print(f"{len(obs)} observations from {runs}: per run {obs.groupby('run').size().to_dict()}", flush=True)
res = {}
for name, d in (("unscaled", obs), ("scaled", None)):
    if d is None:
        d, tab = scale_groups(obs, sg)
        print("run scales:", tab.round(4).to_string(index=False), flush=True)
        res["run_scales"] = tab.to_dict("list")
    m = merge_observations(d, sg)
    res[name] = dict(n_unique=int(len(m)), n_ge2=int((m.n >= 2).sum()), r_int=float(r_int(d, sg)),
                     r_int_strong=float(r_int(d[d.F2 / d.sigF2 > 10], sg)) if (d.F2 / d.sigF2 > 10).sum() > 5 else float("nan"))
    print(name, res[name], flush=True)
obs_s, _ = scale_groups(obs, sg)
merged = merge_observations(obs_s, sg)
merged.to_csv(os.path.join(out, "merged.csv"), index=False)
write_hkl(os.path.join(out, "merged.hkl"), merged)

# completeness to the inscribed-edge d_min of the swung detector (the edge midpoint nearest to the beam, in 2theta)
ex = json.load(open(os.path.join(out, "r01", "reduction.json")))
info = json.load(open(os.path.join(out, "run01", "info.json")))["header"]
alpha = info["extra"]["detector_2theta_deg"]
foot = (info["extra"]["centre_xy_px"][1], info["extra"]["centre_xy_px"][0])
D = info["distance_mm"]; shift = D * np.tan(np.radians(alpha)) / info["pixel_mm"]
from midas_sx.workflow import ScanSetup
setup = ScanSetup(info["wavelength_A"], D / np.cos(np.radians(alpha)), info["pixel_mm"], (foot[0], foot[1] + shift), 0.0, 1.0, tilts_deg=(0, 0, alpha))
geom = setup.geometry(tuple(info["shape"]), 100)
nr, nc = info["shape"]
edge = np.array([[0, nc / 2], [nr - 1, nc / 2], [nr / 2, 0], [nr / 2, nc - 1]], float)
_, tth_edge, _ = spot_directions(edge[:, 0], edge[:, 1], geom)
th = np.radians(tth_edge.min() / 2.0)
d_min = info["wavelength_A"] / (2.0 * np.sin(th))
hk = enumerate_hkl(crystal.lattice, d_min, sg)
uniq = {tuple(sg.asu_representative(int(a), int(b), int(c))) for a, b, c in hk}
obs_keys = {(int(r.h), int(r.k), int(r.l)): r.n for r in merged.itertuples()}
n2 = sum(1 for k in uniq if obs_keys.get(k, 0) >= 2)
n1 = sum(1 for k in uniq if obs_keys.get(k, 0) >= 1)
res["completeness"] = dict(d_min=float(d_min), tth_edge_min=float(tth_edge.min()), n_expected=len(uniq), n_obs_ge1=n1, n_obs_ge2=n2, frac_ge1=n1 / len(uniq), frac_ge2=n2 / len(uniq))
print("completeness:", res["completeness"], flush=True)

# B2 (i): CIF F2
m0 = RefineModel(crystal)
hkl = merged[["h", "k", "l"]].to_numpy(int)
Fc2 = m0.make_f2(hkl)(torch.tensor(m0.x0())).detach().numpy()
lat = crystal.lattice
s2 = np.array([1.0 / (2.0 * lat.d_spacing(int(h), int(k), int(l))) ** 2 for h, k, l in hkl])                 # (sin theta / lambda)^2
Fo2, sg2 = merged.F2.to_numpy(), merged.sigF2.to_numpy()


def fit_scale_B(Fo2, sig, Fc2, s2):
    f = lambda p: (Fo2 - p[0] * np.exp(-2.0 * p[1] * s2) * Fc2) / sig
    sol = least_squares(f, [Fo2.sum() / Fc2.sum(), 0.0])
    return sol.x


def r_f2(Fo2, Fc2cal):
    return float(np.abs(Fo2 - Fc2cal).sum() / np.abs(Fo2).sum())


strong = (Fo2 / sg2 > 10)
k, B = fit_scale_B(Fo2[strong], sg2[strong], Fc2[strong], s2[strong])
Fcal = k * np.exp(-2.0 * B * s2) * Fc2
res["vs_cif"] = dict(n_strong=int(strong.sum()), scale=float(k), B=float(B), R_F2_strong=r_f2(Fo2[strong], Fcal[strong]), R_F2_all=r_f2(Fo2, Fcal),
                     median_ratio_strong=float(np.median(Fo2[strong] / Fcal[strong])))
print("vs CIF:", res["vs_cif"], flush=True)
# B2 (ii): SAINT / SADABS
rows = []
for line in open(saint):
    if len(line) >= 28:
        try:
            rows.append((int(line[0:4]), int(line[4:8]), int(line[8:12]), float(line[12:20]), float(line[20:28])))
        except ValueError:
            pass
sa = pd.DataFrame(rows, columns=["h", "k", "l", "F2", "sigF2"])
sa = sa[(sa.h != 0) | (sa.k != 0) | (sa.l != 0)]
sam = merge_observations(sa, sg)
j = merged.merge(sam, on=["h", "k", "l"], suffixes=("", "_saint"))
st = (j.F2 / j.sigF2 > 10) & (j.F2_saint / j.sigF2_saint > 10)
ks = float((j.F2[st] * j.F2_saint[st]).sum() / (j.F2_saint[st] ** 2).sum())
res["vs_saint"] = dict(n_saint_rows=int(len(sa)), n_saint_unique=int(len(sam)), n_common=int(len(j)), n_common_strong=int(st.sum()), scale=ks,
                       R_F2_strong=float(np.abs(j.F2[st] - ks * j.F2_saint[st]).sum() / j.F2[st].sum()), median_ratio=float(np.median(j.F2[st] / (ks * j.F2_saint[st]))))
print("vs SAINT/SADABS:", res["vs_saint"], flush=True)
json.dump(res, open(os.path.join(out, "merge_reading.json"), "w"), indent=1, default=float)
