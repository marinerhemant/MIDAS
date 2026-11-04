"""C2 of PREREGISTER.md (A4, A8): refine the REDUCED merged F2 for many noise seeds and read the pulls (dev).

    python run_c2.py <S1|S2> make-cache                 # simulate once, save the noise-free signal
    python run_c2.py <S1|S2> <seed_lo> <seed_hi>        # chain + refine for seeds lo..hi-1, appends results/c2_<S>.csv

Each seed: Poisson frames from the cached signal -> real midas_defect front end -> fixed-support reduction -> merge ->
refine from TWO starts (N3 = 0.02 fractional as registered, N3p = 0.10 A physical). The merged table of every seed is saved so
the refinement can be repeated without the 90 s front end. Nothing here sees the true parameters except to score the pulls.
"""
import sys, time, json, resource, os
sys.path.insert(0, "."); sys.path.insert(0, "..")
from dataclasses import asdict, replace
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
from midas_sx.orient import index_and_refine
from midas_sx.assign import assign_hkl, q_sample_of_spots
from scipy.spatial.transform import Rotation as Rot
from midas_defect.seed_index import refine_U_from_centroids

which = sys.argv[1]
mode = sys.argv[2]
CACHE = Path("/private/tmp/claude-503/-Volumes-T5-EVO-travel-work-opt-MIDAS/0372d3ae-3aef-4d1a-ba24-2ca52591f359/scratchpad/c2_cache")
CACHE.mkdir(exist_ok=True)
N_INT, ORI, MC = 1e5, 1, 2
DOM2 = os.environ.get("DOM2", "")                  # PREREGISTER A23: second domain, orientation seed 3, intensity 0.3/0.7 of domain 1 (same absolute scale x 3/7)
if DOM2:
    ORI, MC = 3, 4
ORI_ENV = os.environ.get("ORI", "")                 # PREREGISTER A30: another crystal orientation (random_orientation seed); names get an _o<N> suffix so nothing collides with orientation 1
if ORI_ENV and not DOM2:
    ORI = int(ORI_ENV)
SFX = f"_o{ORI}" if (ORI_ENV and not DOM2) else ""
st = {"S1": sim.structure_cu, "S2": sim.structure_s2_bilayer, "S2A": sim.structure_s2_bilayer_aniso, "S2O": sim.structure_s2o, "S2OA": sim.structure_s2oa}[which]()          # S2A = anisotropic cations (PREREGISTER A22)
VAR = os.environ.get("EXT", "")
TWIN = os.environ.get("TWIN", "")                   # PREREGISTER A24: merohedral twin fraction in percent (S2O only): TWIN=30 -> m = 0.30
BG = os.environ.get("BG", "mean")                    # PREREGISTER A36: background estimator of the integration (mean | quadratic); quadratic outputs get a _bgq suffix
JOINT = os.environ.get("JOINT", "")                # PREREGISTER A33: JOINT=<percent twin> (S2OA only): aniso cations + twin + per-domain extinction (t_b = 5 um)
if JOINT:
    VAR = "J" + JOINT
if TWIN:
    VAR = "tw" + TWIN                  # PREREGISTER A21: T1_5, T1_10 (perfect blocks, um) or T2 (log-normal, median 7 um, sigma_ln 0.5); unset = kinematic (T0)
EXT_VARIANTS = {"T1_5": (5.0, 0.0), "T1_10": (10.0, 0.0), "T2": (7.0, 0.5)}
cache = CACHE / f"{st.name}_n{N_INT:g}_o{ORI}{'_' + VAR if VAR else ''}.npz"
cache1 = CACHE / f"{st.name}_n{N_INT:g}_o1.npz"
EXP_KW = dict(dac_tilt_deg=(2.0, 1.0), t_plus_mm=2.5, t_minus_mm=3.0, bandwidth_rel=1e-3, wavelength_A=12.3984 / 30.0)


def truth_crystal(st):
    atoms = [Atom(s.element, s.fract, s.occ, EIGHT_PI2 * s.u_iso, label=s.label) for s in st.sites]
    return Crystal(Lattice(*st.cell), SpaceGroup.from_hm(st.spacegroup), atoms, st.name)


if mode == "make-cache":
    exp = sim.make_experiment(**EXP_KW)
    t0 = time.time()
    f2_factor = None
    f2_override = sim.twin_f2(st, int(TWIN) / 100.0) if TWIN else None
    if JOINT:
        import extinction_truth as et
        f2_override = sim.joint_f2(st, int(JOINT) / 100.0, et.make_factor(st, exp.wavelength_A, 5.0, 0.0))
    if VAR and not TWIN and not JOINT:
        import extinction_truth as et
        f2_factor = et.make_factor(st, exp.wavelength_A, *EXT_VARIANTS[VAR])
    res = sim.simulate(st, exp, orientation_seed=ORI, mc_seed=MC, n_int_max=N_INT, target_hits=100_000, max_pairs=20_000_000, f2_factor=f2_factor,
                       scale_override=(float(np.load(cache1)['scale']) * 3.0 / 7.0 if DOM2 else None), f2_override=f2_override)
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
PFX = {"v1": "c2", "v2": "c2v2", "v3": "c2v3", "v4": "c2v4", "v5": "c2v5"}[PROF] + (f"_{VAR}" if VAR else "") + SFX + ("" if BG == "mean" else "_bgq") + os.environ.get("TAG", "")
EVAL = "calculated" if PROF in ("v3", "v4", "v5") else "measured"      # v5 = v3 reduction (exact U, exact geometry) but spots FOUND with the v4 wrong-start geometry (A20 control)      # v4 = v3 + orientation/geometry refinement from a wrong start (A19)
mis_deg = lambda A, B: float(np.degrees(np.linalg.norm(Rot.from_matrix(A @ B.T).as_rotvec())))      # v1/v2 = corrections at the measured centroid (as they were run); v3 = at the calculated crossing (A15)
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
    odiag = {}
    if PROF in ("v4", "v5"):
        # A19: a WRONG starting orientation and geometry, then index_and_refine from the measured spots
        rng_o = np.random.default_rng(777 + seed)
        ax = rng_o.normal(size=3); ax /= np.linalg.norm(ax)
        U_start = Rot.from_rotvec(np.radians(0.3) * ax).as_matrix() @ res.U
        phi = rng_o.uniform(0, 2 * np.pi)
        g_start = replace(geom, bcz_px=geom.bcz_px + 0.5 * np.cos(phi), bcy_px=geom.bcy_px + 0.5 * np.sin(phi),
                          lsd_um=geom.lsd_um * (1 + 0.01 * rng_o.choice([-1.0, 1.0])))
        geom_f = g_start
    else:
        geom_f = geom
    spots, mask, info = find_spots(raw, geom_f)
    if PROF == "v4":
        t_o = time.time()
        ores = index_and_refine(spots, lattice, U_start, g_start, scan)
        U_use, geom_use = ores.U, ores.geom
        odiag = dict(u_err_deg=mis_deg(ores.U, res.U), u_start_err_deg=mis_deg(U_start, res.U),
                     bc_row_err=ores.geom.bcz_px - geom.bcz_px, bc_col_err=ores.geom.bcy_px - geom.bcy_px,
                     dlsd_rel_err=ores.geom.lsd_um / geom.lsd_um - 1.0, o_converged=ores.converged, o_reason=ores.reason,
                     o_rms_px=ores.rms_px, o_rms_frame=ores.rms_frame, o_n_used=ores.n_used, o_n_spots=ores.n_spots,
                     o_sig_x=ores.sigma_rot_deg[0], o_sig_y=ores.sigma_rot_deg[1], o_sig_z=ores.sigma_rot_deg[2],
                     o_start_bc_row=g_start.bcz_px - geom.bcz_px, o_start_bc_col=g_start.bcy_px - geom.bcy_px,
                     o_start_dlsd=g_start.lsd_um / geom.lsd_um - 1.0, o_seconds=time.time() - t_o)
        # comparator: the existing direction-only tool, from the same wrong start
        try:
            s0, tth0, _ = spot_directions(spots.row.to_numpy(), spots.col.to_numpy(), g_start)
            om0 = scan.omega_of(spots.frame.to_numpy())
            hkl0, _, resid0 = assign_hkl(s0, om0, lattice, U_start, g_start.wavelength_A)
            use0 = (tth0 > 2.0) & (resid0 < 0.25) & np.any(hkl0 != 0, axis=1)
            q0 = q_sample_of_spots(s0[use0], om0[use0], g_start.wavelength_A) * 2 * np.pi
            Uex = refine_U_from_centroids(U_start, [(tuple(map(int, h_)), qq) for h_, qq in zip(hkl0[use0], q0)], a=st.cell[0], c=st.cell[2],
                                          n_steps=2000, lr=5e-3, device="cpu", dtype="float64")[0]
            odiag["u_err_existing_deg"] = mis_deg(Uex, res.U)
        except Exception as e:                                                # noqa: BLE001
            odiag["u_err_existing_deg"] = np.nan
    else:
        U_use, geom_use = res.U, geom                 # v1-v3 and the v5 control: exact U and exact geometry in the reduction
    s_all, tth_all, _ = spot_directions(spots.row.to_numpy(), spots.col.to_numpy(), geom_use)
    L_all = C.rotation_lorentz(s_all, min_abs=1e-3)
    ok = (tth_all > 2.0) & np.isfinite(L_all)
    centres = spots[["frame", "row", "col"]].to_numpy(float)[ok]
    try:
        if PROF in ("v2", "v3", "v4", "v5"):
            model, _pinfo = build_profile(raw, centres, L_all[ok], mask=mask, background=BG)
        else:
            model, _ = fit_profile_model(raw, centres, L_all[ok], mask=mask)
            model, sweep = choose_support_scale(raw, centres, L_all[ok], model, mask=mask)
    except RuntimeError as e:
        pd.DataFrame([dict(seed=seed, start="-", note=f"profile model failed: {e}")]).to_csv(csv, mode="a", header=not csv.exists(), index=False)
        continue
    obs, rej, counts = reduce_spots(spots, raw, geom_use, scan, lattice, U_use, model, dac=dac, pol=pol, mask=mask, eval_at=EVAL)
    merged = merge_observations(obs, sg)
    hkl = merged[["h", "k", "l"]].to_numpy(int)
    np.savez(outdir / f"{PFX}_merged" / f"{st.name}_seed{seed}.npz", hkl=hkl, F2=merged.F2.to_numpy(), sig=merged.sigF2.to_numpy(),
             n=merged.n.to_numpy())
    if PROF == "v4":
        # NULL arm (A19 i): reduce with the wrong start, unrefined
        obs_n, _, counts_n = reduce_spots(spots, raw, g_start, scan, lattice, U_start, model, dac=dac, pol=pol, mask=mask, eval_at=EVAL)
        merged_n = merge_observations(obs_n, sg)
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
                   k_support=model.k, profile=PROF, margin=model.margin, wallclock=time.time() - t0, **odiag)
        for j, nm in enumerate(r.names):
            row[f"x:{nm}"] = r.x[j]
            row[f"s:{nm}"] = r.sigma[j]
            if nm in err:
                row[f"e:{nm}"] = err[nm]
        rows.append(row)
    if PROF == "v4":
        rng = np.random.default_rng(10_000 * seed + 2)                      # the same start as N3p
        model_n = RefineModel(perturbed_crystal(truth, rng, shift_A=0.10))
        hkl_n = merged_n[["h", "k", "l"]].to_numpy(int)
        r = refine(model_n, hkl_n, merged_n.F2.to_numpy(), merged_n.sigF2.to_numpy())
        err = errors(r, model_n)
        row = dict(seed=seed, start="null_N3p", converged=r.converged, poor_fit=r.poor_fit, reason=r.reason, goof=r.goof, chi2=r.chi2, wR2=r.wR2,
                   R1=r.R1, n_unique=len(hkl_n), n_obs=len(obs_n), n_par=r.n_par, nfev=r.nfev, sv_ratio=r.sv_ratio, n_blobs=len(spots),
                   rejected=json.dumps({k: v for k, v in counts_n.items() if v and k not in ("input", "kept")}), k_support=model.k, profile=PROF,
                   margin=model.margin, wallclock=time.time() - t0, **odiag)
        for j, nm in enumerate(r.names):
            row[f"x:{nm}"] = r.x[j]; row[f"s:{nm}"] = r.sigma[j]
            if nm in err:
                row[f"e:{nm}"] = err[nm]
        rows.append(row)
    pd.DataFrame(rows).to_csv(csv, mode="a", header=not csv.exists(), index=False)
    ok_n = [r_["converged"] for r_ in rows]
    print(f"seed {seed}: {len(hkl)} unique, converged N3={ok_n[0]} N3p={ok_n[1]}, GooF {rows[0]['goof']:.2f}/{rows[1]['goof']:.2f}, " + (f"U err {odiag['u_err_deg']:.4f} deg (existing tool {odiag.get('u_err_existing_deg', float('nan')):.4f}), " if odiag else "") +
          f"{time.time()-t0:.0f} s, RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20:.0f} MB", flush=True)
