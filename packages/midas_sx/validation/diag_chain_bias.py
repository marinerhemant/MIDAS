"""Where does the ensemble-mean offset of the C2 pulls come from? (dev)   python diag_chain_bias.py
Three refinements of the SAME merged-table construction, all noise-free, compared with truth and with C2 v2's mean error:
  CHAIN   : noise-free frames -> real front end -> v2 reduction -> merge -> refine
  ORACLE  : each reduced spot's intensity replaced by the simulator's exact expected counts (the closed-form L,P,T still divide it)
  So CHAIN - ORACLE = the integration/background/hkl chain; ORACLE - truth = closed-form corrections vs the simulator's Monte-Carlo Lorentz (+ MC floor)."""
import sys, json, glob
sys.path.insert(0, "."); sys.path.insert(0, "..")
from pathlib import Path
import numpy as np, pandas as pd
import simulate as sim
from midas_hkls import Atom, Crystal, Lattice, SpaceGroup, corrections as C
from midas_sx.geometry import Scan, make_geometry, spot_directions
from midas_sx.integrate import build_profile
from midas_sx.merge import merge_observations
from midas_sx.pipeline import find_spots
from midas_sx.reduce import DacModel, PolarizationModel, Switches, _pt, reduce_spots
from midas_sx.assign import qsample_of_hkl
from scipy.optimize import brentq
from midas_sx.refine import EIGHT_PI2, RefineModel, perturbed_crystal, refine

st = sim.structure_s2_bilayer()
CACHE = Path("/private/tmp/claude-503/-Volumes-T5-EVO-travel-work-opt-MIDAS/0372d3ae-3aef-4d1a-ba24-2ca52591f359/scratchpad/c2_cache")
cache = CACHE / f"{st.name}_n100000_o1.npz"
z = np.load(cache)
exp = sim.Experiment(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in json.load(open(str(cache) + ".exp.json")).items()})
truth_tbl = pd.read_pickle(str(cache) + ".truth.pkl")
signal, U = z["signal"], z["U"]
raw = (signal + exp.background_map()[None]).astype(np.float32)
scan = Scan(exp.omega_start, exp.omega_step, exp.n_frames)
geom = make_geometry(wavelength_A=exp.wavelength_A, dist_mm=exp.dist_mm, pixel_mm=exp.pixel_mm, centre_rc=exp.centre_rc, shape=(exp.det_n, exp.det_n), scan=scan)
sg = SpaceGroup.from_hm(st.spacegroup); lattice = Lattice(*st.cell)
truth = Crystal(lattice, sg, [Atom(s.element, s.fract, s.occ, EIGHT_PI2 * s.u_iso, label=s.label) for s in st.sites], st.name)
spots, mask, info = find_spots(raw, geom, threshold=40.5)
s_all, tth_all, _ = spot_directions(spots.row.to_numpy(), spots.col.to_numpy(), geom)
L_all = C.rotation_lorentz(s_all, min_abs=1e-3)
ok = (tth_all > 2.0) & np.isfinite(L_all)
model, _ = build_profile(raw, spots[["frame", "row", "col"]].to_numpy(float)[ok], L_all[ok], mask=mask)
dac = DacModel(axis0=tuple(exp.dac_axis0()), t_plus_mm=exp.t_plus_mm, t_minus_mm=exp.t_minus_mm, mu_per_cm=exp.mu_diamond_per_mm * 10.0, half_opening_deg=exp.half_opening_deg)
EVAL = "measured" if "measured" in sys.argv else "calculated"          # v2 behaviour | v3 package default (PREREGISTER A15)
obs, rej, counts = reduce_spots(spots, raw, geom, scan, lattice, U, model, dac=dac, pol=PolarizationModel(exp.pol_fraction), mask=mask, eval_at=EVAL)
print("noise-free chain: kept", counts["kept"], "reflections")

# match to the exact truth to build ORACLE intensities
oracle, oracle_l = [], []
for r in obs.itertuples():
    m = truth_tbl[(truth_tbl.h == r.h) & (truth_tbl.k == r.k) & (truth_tbl.l == r.l)]
    j = (m.omega - r.omega).abs().idxmin()
    near = abs(m.omega[j] - r.omega) < 1.0
    oracle.append(m.expected_counts[j] / (r.L * r.PT) if near else np.nan)
    # ORACLE_L: divide by the simulator's REALISED Monte-Carlo Lorentz factor (L_em, in units where closed form is L * 1) instead of the closed form
    oracle_l.append(m.expected_counts[j] / (m.lorentz_emergent[j] * r.PT) if near else np.nan)
# PREDICTED-CROSSING corrections: L, P, T evaluated at the calculated Ewald crossing of the assigned reflection (U, hkl, lattice),
# not at the measured centroid direction of an extended spot
kk = 1.0 / exp.wavelength_A
qs = qsample_of_hkl(obs[["h", "k", "l"]].to_numpy(int), lattice, U)
Lp, PTp, om_star = [], [], []
pol_m = PolarizationModel(exp.pol_fraction)
for i, r in enumerate(obs.itertuples()):
    q = qs[i]
    f = lambda om: np.linalg.norm(np.array([kk, 0, 0]) + C.rotate_about_axis(q, [0, 0, 1], om)) - kk
    grid = np.linspace(r.omega - 2.0, r.omega + 2.0, 401)
    v = np.array([f(g) for g in grid])
    idx = np.flatnonzero(v[:-1] * v[1:] < 0)
    if len(idx) == 0:
        Lp.append(np.nan); PTp.append(np.nan); om_star.append(np.nan); continue
    j = idx[np.argmin(np.abs(grid[idx] - r.omega))]
    om = brentq(f, grid[j], grid[j + 1], xtol=1e-12)
    s = (np.array([kk, 0, 0]) + C.rotate_about_axis(q, [0, 0, 1], om)) / kk
    Lp.append(float(C.rotation_lorentz(s[None, :], min_abs=0.0)[0]))
    PTp.append(float(_pt(s[None, :], np.array([om]), dac, pol_m, Switches())[0]))
    om_star.append(om)
obs["L_pred"], obs["PT_pred"], obs["om_star"] = Lp, PTp, om_star
obs["F2_pred"] = obs.I / (obs.L_pred * obs.PT_pred)
print("L_measured/L_predicted: median %.4f rms-1 %.4f max|.-1| %.3f ; PT ratio rms-1 %.4f ; omega* - omega_measured: rms %.4f deg" % (
    (obs.L / obs.L_pred).median(), np.sqrt(np.nanmean((obs.L / obs.L_pred - 1) ** 2)), np.nanmax(np.abs(obs.L / obs.L_pred - 1)),
    np.sqrt(np.nanmean((obs.PT / obs.PT_pred - 1) ** 2)), np.sqrt(np.nanmean((obs.om_star - obs.omega) ** 2))))
obs["F2_oracle"] = oracle
obs["F2_oracleL"] = oracle_l
good = obs.F2_oracle.notna()          # NOT also F2_pred: the crossing grid search below returns NaN when the start is exactly on the root (f == 0)


def errors_for(table, col, label):
    mm = merge_observations(table, sg, value=col, sigma="sigF2")
    hkl = mm[["h", "k", "l"]].to_numpy(int)
    start = perturbed_crystal(truth, np.random.default_rng(3), shift_A=0.10)
    m = RefineModel(start)
    r = refine(m, hkl, mm[col].to_numpy(), mm.sigF2.to_numpy())
    out = {"converged": r.converged, "GooF": r.goof, "n_unique": len(hkl)}
    for j, (nm, kind) in enumerate(zip(r.names, r.kinds)):
        if kind == "scale":
            continue
        label_ = nm.split(".")[0]
        ia = [a.label for a in truth.atoms].index(label_)
        if kind == "U":
            out[nm] = (r.x[j] - truth.atoms[ia].B_iso / EIGHT_PI2, r.sigma[j])
        else:
            b = m.basis_frac[j]
            d = r.absolute.iloc[ia][["x", "y", "z"]].to_numpy(float) - np.array(truth.atoms[ia].fract)
            out[nm] = (float(d @ b) / float(b @ b), r.sigma[j])
    return out

res_chain = errors_for(obs[good], "F2", "chain")
res_oracle = errors_for(obs[good].assign(F2=obs[good].F2_oracle), "F2", "oracle")
res_oracleL = errors_for(obs[good].assign(F2=obs[good].F2_oracleL), "F2", "oracleL")
res_pred = errors_for(obs[good & obs.F2_pred.notna()].assign(F2=obs.F2_pred), "F2", "pred") if EVAL == "measured" else res_chain
c2 = pd.concat([pd.read_csv(g) for g in sorted(glob.glob("results/c2v2_S2_bilayer_like_*.csv"))], ignore_index=True)
c2 = c2[(c2.start == "N3p") & c2.converged]
rows = []
for nm in [k for k in res_chain if k not in ("converged", "GooF", "n_unique")]:
    e = c2["e:" + nm]
    rows.append(dict(param=nm, c2v2_mean_error=e.mean(), c2v2_SE=e.std(ddof=1) / np.sqrt(len(e)), noisefree_CHAIN=res_chain[nm][0],
                     noisefree_ORACLE=res_oracle[nm][0], noisefree_ORACLE_MCLorentz=res_oracleL[nm][0], noisefree_PREDICTED_CROSSING=res_pred[nm][0], formal_sigma_one_seed=res_chain[nm][1]))
t = pd.DataFrame(rows)
t["CHAIN/sigma"] = t.noisefree_CHAIN.abs() / t.formal_sigma_one_seed
t["PRED/sigma"] = t.noisefree_PREDICTED_CROSSING.abs() / t.formal_sigma_one_seed
pd.set_option("display.width", 250)
lines = [f"noise-free refinements, N3p-like start (shift 0.10 A, seed 3): CHAIN converged={res_chain['converged']} GooF={res_chain['GooF']:.3f}; ORACLE converged={res_oracle['converged']} GooF={res_oracle['GooF']:.3f}; ORACLE_MCLorentz converged={res_oracleL['converged']} GooF={res_oracleL['GooF']:.3f}; PREDICTED_CROSSING converged={res_pred['converged']} GooF={res_pred['GooF']:.3f}; n_unique {res_chain['n_unique']}",
         "errors = estimate - truth (A for z, A^2 for U); compare noisefree_* with c2v2_mean_error (+- SE over 50 noisy seeds)", t.to_string(index=False, float_format=lambda v: f"{v:.3e}")]
det = [n for n in t.param if n not in ("O2.U", "O3.U")]
acc = t[t.param.isin(det)]
lines.append("")
lines.append("A15 ACCEPTANCE (v3 package default = corrections at the calculated crossing): every determined parameter |error| <= 0.1 x formal sigma of one seed")
lines.append("  worst |CHAIN|/sigma over the 8 determined parameters: %.3f (%s) -> %s" % (acc["CHAIN/sigma"].max(), acc.loc[acc["CHAIN/sigma"].idxmax(), "param"], "MET" if acc["CHAIN/sigma"].max() <= 0.1 else "NOT MET"))
lines.append("  per parameter: " + ", ".join(f"{r.param} {r._asdict()['_8'] if False else r[acc.columns.get_loc('CHAIN/sigma') + 1]:.3f}" for r in acc.itertuples()))
print("\n".join(lines))
open("results/diag_chain_bias.txt", "w").write("\n".join(lines) + "\n")
