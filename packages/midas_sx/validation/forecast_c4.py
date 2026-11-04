"""C4 (PREREGISTER A5, A9): Fisher forecast of refinement uncertainty vs coverage and counts (dev).

    python forecast_c4.py calibrate <c_min_lo> <c_min_hi>      # scan the detectability cut at the nominal geometry
    python forecast_c4.py grid <c_min>                         # the full table

For every geometry: which reflections cross the scan inside the opening cone, their expected integrated counts (the simulator's own
normalisation: strongest complete reflection = n_int_max), the Poisson sigma of the fixed-support box the reduction uses, the
reduction's own rejection rules, internal merge, then the refiner's covariance (the refiner called on exact data at the true
parameters, so the model and the covariance code are the real ones). Averaged over random orientations.
"""
import sys, time, json
sys.path.insert(0, "."); sys.path.insert(0, "..")
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

import simulate as sim
from midas_hkls import Atom, Crystal, Lattice, SpaceGroup, corrections as C
from midas_sx.integrate import ProfileModel
from midas_sx.reduce import DacModel, PolarizationModel, Switches, _pt, _profile_variation
from midas_sx.refine import EIGHT_PI2, RefineModel, refine

st = sim.structure_s2_bilayer()
sg = SpaceGroup.from_hm(st.spacegroup)
lat = Lattice(*st.cell)
truth = Crystal(lat, sg, [Atom(s.element, s.fract, s.occ, EIGHT_PI2 * s.u_iso, label=s.label) for s in st.sites], st.name)
B = sim._reciprocal_basis(st)
LAM = 12.3984 / 30.0
# the profile model fitted by the reduction on the S2 runs (c1_S2 result file: sigma_px 0.704, sigma_f 0.179 + 0.064 L, k = 2)
PROFILE_V1 = ProfileModel(sigma_px=0.704, sf0=0.179, sf1=0.064, k=2.0, shell_px=3, shell_frames=2)
# v2 support measured on the same signal (diag_integration_bias.py S2 noisy v2): half-extent 1.0 + 0.2065 L frames, 2.92 px, margin 0 (PREREGISTER A13)
PROFILE_V2 = ProfileModel(sigma_px=2.92 / 3.3, sf0=1.0 / 3.0, sf1=0.2065 / 3.0, k=3.0, shell_px=3, shell_frames=2,
                          ext_f0=1.0, ext_f1=0.2065, ext_p=2.92, margin=0)
PROFILE = PROFILE_V2 if "--v1profile" not in sys.argv else PROFILE_V1
STEP = 0.25
TOL_X, TOL_U = 0.02, 0.00127
NOMINAL = dict(W=40.0, alpha=35.0, n_int=1e5)


def base_experiment(W_max=60.0):
    return sim.make_experiment(dac_tilt_deg=(2.0, 1.0), t_plus_mm=2.5, t_minus_mm=3.0, bandwidth_rel=1e-3, wavelength_A=LAM,
                               omega_start=-W_max, n_frames=int(round(2 * W_max / STEP)))


EXP60 = base_experiment(60.0)
D_MIN = EXP60.wavelength_A / (2.0 * np.sin(np.radians(EXP60.tth_max_deg() / 2.0)))
HKL_ALL = sim.enumerate_hkl(st, D_MIN)
F2_ALL = sim.f2_truth(st, HKL_ALL)
_keep = F2_ALL > 1e-6 * F2_ALL.max()
HKL_ALL, F2_ALL = HKL_ALL[_keep], F2_ALL[_keep]
ASU = {tuple(h): sg.asu_representative(*map(int, h)) for h in map(tuple, HKL_ALL)}
_cache = {}


def crossings(orient_seed):
    """All Ewald crossings inside +-60 deg for one orientation: hkl index, omega*, s_nom, q_sample, pixel, F2."""
    if orient_seed in _cache:
        return _cache[orient_seed]
    U = sim.random_orientation(orient_seed)
    rows = []
    k = EXP60.k
    for i, h in enumerate(HKL_ALL):
        q0 = U @ (B @ h)
        for om, slope in sim._crossings(q0, EXP60):
            if slope < 1e-4:
                continue
            s = (k * sim.S0 + Rotation.from_rotvec(np.radians(om) * sim.AXIS).apply(q0)) / k
            if s[0] <= 1e-6:
                continue
            col = EXP60.centre_rc[1] - EXP60.dist_mm * s[1] / (s[0] * EXP60.pixel_mm)
            row = EXP60.centre_rc[0] + EXP60.dist_mm * s[2] / (s[0] * EXP60.pixel_mm)
            if not (abs(row - EXP60.centre_rc[0]) < EXP60.det_n / 2 - 3 and abs(col - EXP60.centre_rc[1]) < EXP60.det_n / 2 - 3):
                continue
            rows.append((i, om, *s, *q0, row, col))
    arr = np.array(rows)
    _cache[orient_seed] = (U, arr)
    return _cache[orient_seed]


def geometry_table(orient_seed, W, alpha):
    """Per-crossing corrected quantities for scan half-range W and opening half-angle alpha (deg)."""
    U, arr = crossings(orient_seed)
    idx = arr[:, 0].astype(int)
    om = arr[:, 1]
    s = arr[:, 2:5]
    q0 = arr[:, 5:8]
    row, col = arr[:, 8], arr[:, 9]
    L = C.rotation_lorentz(s, min_abs=1e-3)
    dac = DacModel(axis0=tuple(EXP60.dac_axis0()), t_plus_mm=EXP60.t_plus_mm, t_minus_mm=EXP60.t_minus_mm,
                   mu_per_cm=EXP60.mu_diamond_per_mm * 10.0, half_opening_deg=alpha)
    pol = PolarizationModel(fraction=EXP60.pol_fraction)
    PT = _pt(s, om, dac, pol, Switches())
    tth = np.degrees(np.arccos(np.clip(s[:, 0], -1, 1)))
    ok = np.isfinite(L) & (PT > 0) & (tth > 2.0) & (np.abs(om) <= W)
    hf = np.array([PROFILE.half_widths(l)[0] if np.isfinite(l) else 0 for l in L])
    hp = PROFILE.half_widths(1.0)[1]
    ok &= (np.abs(om) + hf * STEP + 0.5 * STEP) <= W                           # support inside the scan (complete)
    var = np.full(len(om), np.nan)
    for j in np.flatnonzero(ok):
        var[j] = _profile_variation(q0[j], float(om[j]), hf[j] * STEP, LAM, dac, pol, Switches())
    ok &= np.isfinite(var) & (var <= 0.02)
    return dict(U=U, idx=idx, om=om, s=s, row=row, col=col, L=L, PT=PT, hf=hf, hp=hp, ok=ok, tth=tth)


@np.errstate(divide='ignore', invalid='ignore')
def merged_data(g, n_int, c_min):
    """Merged noise-free F2_obs and sigma for one geometry table at counts n_int and detectability cut c_min."""
    ok = g["ok"]
    F2 = F2_ALL[g["idx"]]
    drive = F2 * g["L"] * g["PT"]
    norm = np.nanmax(np.where(ok, drive, np.nan))
    counts = n_int * drive / norm
    found = ok & (counts >= c_min)
    if not found.any():
        return None
    bg_map = EXP60.bg0 + EXP60.bg1 * np.exp(-g["tth"] / 8.0)
    nb = (2 * g["hf"] + 1) * (2 * g["hp"] + 1) ** 2
    shell = (2 * (g["hf"] + 2) + 1) * (2 * (g["hp"] + 3) + 1) ** 2 - nb
    var_I = counts + nb * bg_map + nb ** 2 * bg_map / np.maximum(shell, 1)
    corr = g["L"] * g["PT"]
    f2_obs = counts / corr
    sig = np.sqrt(var_I) / corr
    df = pd.DataFrame(dict(h=HKL_ALL[g["idx"], 0], k=HKL_ALL[g["idx"], 1], l=HKL_ALL[g["idx"], 2], f2=f2_obs, w=1 / sig ** 2))[found]
    rep = [ASU[(int(a), int(b), int(c))] for a, b, c in zip(df.h, df.k, df.l)]
    df[["uh", "uk", "ul"]] = np.array(rep)
    out = df.groupby(["uh", "uk", "ul"]).agg(f2=("f2", "mean"), w=("w", "sum"), n=("f2", "size")).reset_index()
    return out[["uh", "uk", "ul"]].to_numpy(int), out.f2.to_numpy(), 1 / np.sqrt(out.w.to_numpy()), int(found.sum())


def forecast_sigma(hkl, f2, sig):
    """The refiner's formal sigma at the TRUE parameters on noise-free data (so no optimisation is involved)."""
    m = RefineModel(truth)
    return refine(m, hkl, f2, sig, x_start=m.x0())


def run_point(orient_seed, W, alpha, n_int, c_min):
    g = geometry_table(orient_seed, W, alpha)
    md = merged_data(g, n_int, c_min)
    if md is None:
        return None
    hkl, f2, sig, n_obs = md
    res = forecast_sigma(hkl, f2, sig)
    row = dict(orient=orient_seed, W=W, alpha=alpha, n_int=n_int, c_min=c_min, n_obs=n_obs, n_unique=len(hkl),
               full_rank=res.full_rank, n_par=res.n_par)
    for nm, kind, sg_ in zip(res.names, res.kinds, res.sigma):
        row[f"s:{nm}"] = sg_
    return row


if __name__ == "__main__":
    mode = sys.argv[1]
    orients = list(range(1, 9))
    if mode == "calibrate":
        lo, hi = float(sys.argv[2]), float(sys.argv[3])
        t0 = time.time()
        print("c_min   median n_unique over orientations 1..8 (nominal geometry W=40, alpha=35, n_int=1e5)   [orientation 1 alone]")
        for cm in np.geomspace(lo, hi, 9):
            nu = []
            for o in orients:
                r = run_point(o, NOMINAL["W"], NOMINAL["alpha"], NOMINAL["n_int"], cm)
                nu.append(r["n_unique"] if r else 0)
            print(f"{cm:8.0f}  {np.median(nu):5.1f}   [{nu[0]}]   ({time.time()-t0:.0f} s)", flush=True)
    elif mode == "calibrate1":
        target = float(sys.argv[2])
        rows = []
        for cm in np.arange(600, 2600, 25):
            r = run_point(1, NOMINAL["W"], NOMINAL["alpha"], NOMINAL["n_int"], cm)
            rows.append((cm, r["n_unique"] if r else 0))
        arr = np.array(rows)
        best = np.abs(arr[:, 1] - target).min()
        hit = arr[np.abs(arr[:, 1] - target) == best, 0]
        print("orientation 1, nominal geometry: c_min -> n_unique")
        print(" ".join(f"{int(c)}:{int(n)}" for c, n in arr[::4]))
        print(f"target n_unique {target:g}: closest {best:g} away, achieved for c_min in [{hit.min():.0f}, {hit.max():.0f}] -> using the midpoint {hit.mean():.0f}")
    elif mode == "grid":
        cm = float(sys.argv[2])
        rows = []
        t0 = time.time()
        points = [(NOMINAL["W"], NOMINAL["alpha"], NOMINAL["n_int"])] + \
                 [(W, a, n) for n in (1e3, 1e4, 1e5) for a in (25.0, 30.0, 35.0, 40.0) for W in (30.0, 45.0, 60.0)]
        for (W, a, n) in points:
            for o in orients:
                r = run_point(o, W, a, n, cm)
                if r:
                    rows.append(r)
            print(f"W={W:.0f} alpha={a:.0f} n_int={n:g}  done ({time.time()-t0:.0f} s)", flush=True)
        pd.DataFrame(rows).to_csv(f"results/c4_forecast_cmin{cm:.0f}.csv", index=False)
