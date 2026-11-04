"""PREREGISTER A39 tiers B1 and B2: merge the per-run observation tables of L-alanine (scale_groups), completeness, R_int, and F² against COD 1574527 (anisotropic U for C N O, isotropic H).

    python -I a39_merge.py <out dir holding r02 r03 r04 ...> <cif> <runs, e.g. r02,r03,r04> <dihedral> <omega_sign> <swing axis> <tz> <off>
Primary B2 metrics (registered): R_F2 over ALL merged reflections with n >= 2, over the 100 reflections with the largest calculated F², and the fitted overall B; the I/sigma > 10 subset is reported
but is not a headline (it is selected with our own sigma)."""
import json, os, sys
sys.path[:0] = [os.environ.get("MIDAS_SX_PATH", "/Volumes/T5 EVO/travel/work/opt/MIDAS/packages/midas_sx"), os.path.dirname(os.path.abspath(__file__))]
_root = os.path.dirname(os.environ.get("MIDAS_SX_PATH", "/Volumes/T5 EVO/travel/work/opt/MIDAS/packages/midas_sx"))
sys.path[:0] = [os.path.join(_root, p) for p in ("midas_hkls", "midas_defect", "midas_stress") if os.path.isdir(os.path.join(_root, p))]
import numpy as np, pandas as pd, gemmi, torch
from scipy.optimize import least_squares
from midas_hkls import Atom, Crystal, Lattice, SpaceGroup
from midas_sx.conventions import Convention
from midas_sx.geometry import spot_directions
from midas_sx.merge import merge_observations, r_int, scale_groups, write_hkl
from midas_sx.predict import enumerate_hkl
from midas_sx.refine import EIGHT_PI2, RefineModel
from a39_geom import swing_geometry

out, cif, runs = sys.argv[1], sys.argv[2], sys.argv[3].split(",")
conv = Convention(int(sys.argv[4]), int(sys.argv[5])); axis, tzs, offs = sys.argv[6], int(sys.argv[7]), int(sys.argv[8])
sg = SpaceGroup.from_hm("P 21 21 21")
ss = gemmi.read_small_structure(cif)


def atom(s, aniso):
    kw = {}
    if aniso and s.aniso.nonzero():
        a = s.aniso
        kw["U_aniso"] = (a.u11, a.u22, a.u33, a.u12, a.u13, a.u23)
    return Atom(s.element.name, (s.fract.x, s.fract.y, s.fract.z), s.occ, EIGHT_PI2 * s.u_iso, label=s.label, **kw)


lat = Lattice(ss.cell.a, ss.cell.b, ss.cell.c, ss.cell.alpha, ss.cell.beta, ss.cell.gamma)
crystal_iso = Crystal(lat, sg, [atom(s, False) for s in ss.sites], "ala_iso")
crystal_ani = Crystal(lat, sg, [atom(s, True) for s in ss.sites], "ala_aniso")
obs = pd.concat([pd.read_pickle(os.path.join(out, r, "obs.pkl")).assign(run=r) for r in runs], ignore_index=True)
print(f"{len(obs)} observations from {runs}: per run {obs.groupby('run').size().to_dict()}", flush=True)
res = {}
for name, d in (("unscaled", obs), ("scaled", None)):
    if d is None:
        d, tab = scale_groups(obs, sg)
        print("run scales:", tab.round(4).to_string(index=False), flush=True); res["run_scales"] = tab.to_dict("list")
    m = merge_observations(d, sg)
    res[name] = dict(n_unique=int(len(m)), n_ge2=int((m.n >= 2).sum()), r_int=float(r_int(d, sg)))
    print(name, res[name], flush=True)
obs_s, _ = scale_groups(obs, sg)
merged = merge_observations(obs_s, sg)
merged.to_csv(os.path.join(out, "merged.csv"), index=False); write_hkl(os.path.join(out, "merged.hkl"), merged)

# completeness to the inscribed-edge d_min of the swung detector (final geometry class; header start values)
info = json.load(open(os.path.join(out, "r02", "reduction.json")))
H = json.load(open(os.path.join(out, "run02", "info.json")))["header"]
g, _ = swing_geometry(H, conv, axis, tzs, offs)
nr, nc = (H["shape"] if conv.dihedral < 4 else H["shape"][::-1])
edge = np.array([[0, nc / 2], [nr - 1, nc / 2], [nr / 2, 0], [nr / 2, nc - 1]], float)
_, tth_edge, _ = spot_directions(edge[:, 0], edge[:, 1], g)
d_min = H["wavelength_A"] / (2.0 * np.sin(np.radians(tth_edge.min() / 2.0)))
uniq = {tuple(sg.asu_representative(int(a), int(b), int(c))) for a, b, c in enumerate_hkl(lat, d_min, sg)}
cnt = {(int(r.h), int(r.k), int(r.l)): r.n for r in merged.itertuples()}
n1, n2 = sum(1 for k in uniq if cnt.get(k, 0) >= 1), sum(1 for k in uniq if cnt.get(k, 0) >= 2)
res["completeness"] = dict(d_min=float(d_min), tth_edge_min=float(tth_edge.min()), n_expected=len(uniq), n_obs_ge1=n1, n_obs_ge2=n2, frac_ge1=n1 / len(uniq), frac_ge2=n2 / len(uniq))
print("completeness:", res["completeness"], flush=True)

hkl = merged[["h", "k", "l"]].to_numpy(int)
Fo2, sig = merged.F2.to_numpy(), merged.sigF2.to_numpy()
s2 = np.array([1.0 / (2.0 * lat.d_spacing(int(h), int(k), int(l))) ** 2 for h, k, l in hkl])
n_ = merged.n.to_numpy()


def fit(Fc2, sel):
    f = lambda p: (Fo2[sel] - p[0] * np.exp(-2.0 * p[1] * s2[sel]) * Fc2[sel]) / sig[sel]
    p = least_squares(f, [Fo2[sel].sum() / Fc2[sel].sum(), 0.0]).x
    cal = p[0] * np.exp(-2.0 * p[1] * s2) * Fc2
    return p, cal


for label, cr in (("aniso", crystal_ani), ("iso", crystal_iso)):
    m0 = RefineModel(cr)
    Fc2 = m0.make_f2(hkl)(torch.tensor(m0.x0())).detach().numpy()
    allm = n_ >= 2
    p, cal = fit(Fc2, allm)
    top = np.zeros(len(hkl), bool); top[np.argsort(-Fc2)[:100]] = True
    R = lambda sel: float(np.abs(Fo2[sel] - cal[sel]).sum() / np.abs(Fo2[sel]).sum())
    strong = (Fo2 / sig > 10)
    res[f"vs_cod_{label}"] = dict(n_all_ge2=int(allm.sum()), scale=float(p[0]), B=float(p[1]), R_F2_all_ge2=R(allm), R_F2_top100_Fc=R(top & allm) if (top & allm).sum() > 20 else R(top),
                                  n_top100_ge2=int((top & allm).sum()), R_F2_all_incl_n1=R(np.ones(len(hkl), bool)), R_F2_strong_Isig10=R(strong), n_strong=int(strong.sum()))
    print(f"vs COD ({label}):", res[f"vs_cod_{label}"], flush=True)
json.dump(res, open(os.path.join(out, "merge_reading.json"), "w"), indent=1, default=float)
