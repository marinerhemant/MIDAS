"""C2 of PREREGISTER.md (A4, A8): refine the REDUCED merged F2 for many noise seeds and read the pulls (dev).

    python run_c2.py <S1|S2> make-cache                 # simulate once, save the noise-free signal
    python run_c2.py <S1|S2> <seed_lo> <seed_hi>        # chain + refine for seeds lo..hi-1, appends results/c2_<S>.csv

Each seed: Poisson frames from the cached signal -> real midas_defect front end -> fixed-support reduction -> merge ->
refine from TWO starts (N3 = 0.02 fractional as registered, N3p = 0.10 A physical). The merged table of every seed is saved so
the refinement can be repeated without the 90 s front end. Nothing here sees the true parameters except to score the pulls.
"""
import sys, time, json, resource
sys.path.insert(0, "."); sys.path.insert(0, "..")
from dataclasses import asdict
from pathlib import Path
import numpy as np
import pandas as pd

import simulate as sim
from midas_hkls import Atom, Crystal, Lattice, SpaceGroup, corrections as C
from midas_sx.geometry import Scan, make_geometry, spot_directions
from midas_sx.integrate import build_profile, choose_support_scale, fit_profile_model
from midas_sx.merge import merge_observations
from midas_sx.pipeline import find_spots
from midas_sx.reduce import DacModel, PolarizationModel, reduce_spots
from midas_sx.refine import EIGHT_PI2, RefineModel, perturbed_crystal, refine

which = sys.argv[1]
mode = sys.argv[2]
CACHE = Path("/private/tmp/claude-503/-Volumes-T5-EVO-travel-work-opt-MIDAS/0372d3ae-3aef-4d1a-ba24-2ca52591f359/scratchpad/c2_cache")
CACHE.mkdir(exist_ok=True)
N_INT, ORI, MC = 1e5, 1, 2
st = sim.structure_cu() if which == "S1" else sim.structure_s2_bilayer()
cache = CACHE / f"{st.name}_n{N_INT:g}_o{ORI}.npz"
EXP_KW = dict(dac_tilt_deg=(2.0, 1.0), t_plus_mm=2.5, t_minus_mm=3.0, bandwidth_rel=1e-3, wavelength_A=12.3984 / 30.0)


def truth_crystal(st):
    atoms = [Atom(s.element, s.fract, s.occ, EIGHT_PI2 * s.u_iso, label=s.label) for s in st.sites]
    return Crystal(Lattice(*st.cell), SpaceGroup.from_hm(st.spacegroup), atoms, st.name)


if mode == "make-cache":
    exp = sim.make_experiment(**EXP_KW)
    t0 = time.time()
    res = sim.simulate(st, exp, orientation_seed=ORI, mc_seed=MC, n_int_max=N_INT, target_hits=100_000, max_pairs=20_000_000)
    np.savez(cache, signal=res.signal, U=res.U, scale=res.scale)
    res.truth.to_pickle(str(cache) + ".truth.pkl")
    json.dump(asdict(exp), open(str(cache) + ".exp.json", "w"))
    print(f"cache written {cache} ({time.time()-t0:.0f} s)")
    sys.exit(0)

z = np.load(cache)
exp = sim.Experiment(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in json.load(open(str(cache) + ".exp.json")).items()})
res = sim.SimResult(exp=exp, structure=st, U=z["U"], signal=z["signal"], truth=pd.read_pickle(str(cache) + ".truth.pkl"), scale=float(z["scale"]))
scan = Scan(exp.omega_start, exp.omega_step, exp.n_frames)
geom = make_geometry(wavelength_A=exp.wavelength_A, dist_mm=exp.dist_mm, pixel_mm=exp.pixel_mm, centre_rc=exp.centre_rc,
                     shape=(exp.det_n, exp.det_n), scan=scan)
truth = truth_crystal(st)
sg = SpaceGroup.from_hm(st.spacegroup)
lattice = Lattice(*st.cell)
dac = DacModel(axis0=tuple(exp.dac_axis0()), t_plus_mm=exp.t_plus_mm, t_minus_mm=exp.t_minus_mm,
               mu_per_cm=exp.mu_diamond_per_mm * 10.0, half_opening_deg=exp.half_opening_deg)
pol = PolarizationModel(fraction=exp.pol_fraction)
m_truth = RefineModel(truth)
outdir = Path("results"); outdir.mkdir(exist_ok=True)
(outdir / "c2_merged").mkdir(exist_ok=True)
lo_s, hi_s = int(sys.argv[2]), int(sys.argv[3])
PROF = sys.argv[4] if len(sys.argv) > 4 else "v1"          # v1 = Gaussian k-sigma support, v2 = measured extents (PREREGISTER A11)
PFX = {"v1": "c2", "v2": "c2v2", "v3": "c2v3"}[PROF]
EVAL = "calculated" if PROF == "v3" else "measured"      # v1/v2 = corrections at the measured centroid (as they were run); v3 = at the calculated crossing (A15)
(outdir / f"{PFX}_merged").mkdir(exist_ok=True)
csv = outdir / f"{PFX}_{st.name}_{lo_s}.csv"      # one file per process: no concurrent appends


def errors(res_, model):
    """Per-parameter error (estimate - truth) in the parameter's own units (A for coordinates, A^2 for U)."""
    out = {}
    for j, (nm, kind) in enumerate(zip(res_.names, res_.kinds)):
        if kind == "scale":
            continue
        label = nm.split(".")[0]
        ia = [a.label for a in truth.atoms].index(label)
        if kind == "U":
            out[nm] = res_.x[j] - truth.atoms[ia].B_iso / EIGHT_PI2
        else:
            b = model.basis_frac[j]
            dfrac = res_.absolute.iloc[ia][["x", "y", "z"]].to_numpy(float) - np.array(truth.atoms[ia].fract)
            out[nm] = float(dfrac @ b) / float(b @ b)
    return out


for seed in range(lo_s, hi_s):
    t0 = time.time()
    raw = res.frames(seed=1000 + seed)
    spots, mask, info = find_spots(raw, geom)
    s_all, tth_all, _ = spot_directions(spots.row.to_numpy(), spots.col.to_numpy(), geom)
    L_all = C.rotation_lorentz(s_all, min_abs=1e-3)
    ok = (tth_all > 2.0) & np.isfinite(L_all)
    centres = spots[["frame", "row", "col"]].to_numpy(float)[ok]
    try:
        if PROF in ("v2", "v3"):
            model, _pinfo = build_profile(raw, centres, L_all[ok], mask=mask)
        else:
            model, _ = fit_profile_model(raw, centres, L_all[ok], mask=mask)
            model, sweep = choose_support_scale(raw, centres, L_all[ok], model, mask=mask)
    except RuntimeError as e:
        pd.DataFrame([dict(seed=seed, start="-", note=f"profile model failed: {e}")]).to_csv(csv, mode="a", header=not csv.exists(), index=False)
        continue
    obs, rej, counts = reduce_spots(spots, raw, geom, scan, lattice, res.U, model, dac=dac, pol=pol, mask=mask, eval_at=EVAL)
    merged = merge_observations(obs, sg)
    hkl = merged[["h", "k", "l"]].to_numpy(int)
    np.savez(outdir / f"{PFX}_merged" / f"{st.name}_seed{seed}.npz", hkl=hkl, F2=merged.F2.to_numpy(), sig=merged.sigF2.to_numpy(),
             n=merged.n.to_numpy())
    del raw
    rows = []
    for start_name, kw in (("N3", dict(shift_frac=0.02)), ("N3p", dict(shift_A=0.10))):
        rng = np.random.default_rng(10_000 * seed + (1 if start_name == "N3" else 2))
        model_r = RefineModel(perturbed_crystal(truth, rng, **kw))
        r = refine(model_r, hkl, merged.F2.to_numpy(), merged.sigF2.to_numpy())
        err = errors(r, model_r)
        row = dict(seed=seed, start=start_name, converged=r.converged, poor_fit=r.poor_fit, reason=r.reason, goof=r.goof, chi2=r.chi2,
                   wR2=r.wR2, R1=r.R1, n_unique=len(hkl), n_obs=len(obs), n_par=r.n_par, nfev=r.nfev, sv_ratio=r.sv_ratio,
                   n_blobs=len(spots), rejected=json.dumps({k: v for k, v in counts.items() if v and k not in ("input", "kept")}),
                   k_support=model.k, profile=PROF, margin=model.margin, wallclock=time.time() - t0)
        for j, nm in enumerate(r.names):
            row[f"x:{nm}"] = r.x[j]
            row[f"s:{nm}"] = r.sigma[j]
            if nm in err:
                row[f"e:{nm}"] = err[nm]
        rows.append(row)
    pd.DataFrame(rows).to_csv(csv, mode="a", header=not csv.exists(), index=False)
    ok_n = [r_["converged"] for r_ in rows]
    print(f"seed {seed}: {len(hkl)} unique, converged N3={ok_n[0]} N3p={ok_n[1]}, GooF {rows[0]['goof']:.2f}/{rows[1]['goof']:.2f}, "
          f"{time.time()-t0:.0f} s, RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20:.0f} MB", flush=True)
