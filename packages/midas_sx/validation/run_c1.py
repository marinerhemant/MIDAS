"""C1 / C5 of PREREGISTER.md: F2 recovery through the REAL front end, with ablations (dev).

    python run_c1.py <S1|S2> [n_int_max=1e5] [noise_seed=11] [bandwidth_rel=1e-3] [energy_keV=30] [anvil_mm=2.5/3.0] [orient_seed=1]

Everything the reduction receives is what a user would have: raw frames, nominal geometry, the cell, the
space group, the orientation, and the DAC model. It never sees F2_true, B or coordinates.
Writes results/c1_<structure>_<counts>_<seed>.txt and prints the same.
"""
import sys, time, resource, json
sys.path.insert(0, "."); sys.path.insert(0, "..")      # simulate.py here; midas_sx one level up (not pip-installed)
from pathlib import Path
import numpy as np
import pandas as pd

import simulate as sim
from midas_hkls import Lattice, SpaceGroup, corrections as C
from midas_sx.geometry import Scan, make_geometry, spot_directions
from midas_sx.integrate import build_profile, choose_support_scale, fit_profile_model
from midas_sx.merge import merge_observations, r_int
from midas_sx.pipeline import find_spots
from midas_sx.reduce import DacModel, PolarizationModel, Switches, reduce_spots

which = sys.argv[1] if len(sys.argv) > 1 else "S1"
n_int = float(sys.argv[2]) if len(sys.argv) > 2 else 1e5
seed = int(sys.argv[3]) if len(sys.argv) > 3 else 11
bw = float(sys.argv[4]) if len(sys.argv) > 4 else 1e-3
energy = float(sys.argv[5]) if len(sys.argv) > 5 else 30.0
anvil = float(sys.argv[6]) if len(sys.argv) > 6 and sys.argv[6] != '' else None
oseed = int(sys.argv[7]) if len(sys.argv) > 7 else 1
PROF = sys.argv[8] if len(sys.argv) > 8 else "v1"          # v1 Gaussian k-sigma support | v2 measured extents (PREREGISTER A11)
pd.set_option("display.width", 250)
out_lines = []


def say(*a):
    line = " ".join(str(x) for x in a)
    print(line, flush=True)
    out_lines.append(line)


def rss():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20


st = sim.structure_cu() if which == "S1" else sim.structure_s2_bilayer()
t_plus, t_minus = (2.5, 3.0) if anvil is None else (anvil, anvil)
exp = sim.make_experiment(dac_tilt_deg=(2.0, 1.0), t_plus_mm=t_plus, t_minus_mm=t_minus, bandwidth_rel=bw,
                          wavelength_A=12.3984 / energy)
t0 = time.time()
res = sim.simulate(st, exp, orientation_seed=oseed, n_int_max=n_int, target_hits=100_000, max_pairs=20_000_000)
raw = res.frames(seed=seed)
say(f"# C1 ({PROF} support) {st.name}  n_int_max={n_int:g}  noise_seed={seed}  orient_seed={oseed}  E={energy:g} keV  mu_diamond={exp.mu_diamond_per_mm*10:.3f} /cm  bandwidth_rel={bw:g}  mosaic={exp.mosaic_fwhm_deg} deg  "
    f"DAC tilt={exp.dac_tilt_deg} t+={exp.t_plus_mm} t-={exp.t_minus_mm} mm opening={exp.half_opening_deg} deg  (run {time.strftime('%F')})")
say(f"simulated {len(res.truth)} true spots ({int(res.truth.complete.sum())} complete) in {time.time()-t0:.0f} s; RSS {rss():.0f} MB")

scan = Scan(exp.omega_start, exp.omega_step, exp.n_frames)
geom = make_geometry(wavelength_A=exp.wavelength_A, dist_mm=exp.dist_mm, pixel_mm=exp.pixel_mm,
                     centre_rc=exp.centre_rc, shape=(exp.det_n, exp.det_n), scan=scan)
lattice = Lattice(*st.cell)
sg = SpaceGroup.from_hm(st.spacegroup)
t0 = time.time()
spots, mask, info = find_spots(raw, geom)
say(f"front end: {len(spots)} blobs in {time.time()-t0:.0f} s (noise {info['noise']:.2f}, threshold {info['threshold']:.1f}); RSS {rss():.0f} MB")

# profile model from the strong spots, NOT from anything the simulator knows
s_all, tth_all, _ = spot_directions(spots.row.to_numpy(), spots.col.to_numpy(), geom)
L_all = C.rotation_lorentz(s_all, min_abs=1e-3)
ok = (tth_all > 2.0) & np.isfinite(L_all)
centres = spots[["frame", "row", "col"]].to_numpy(float)[ok]
if PROF == "v2":
    model, pinfo2 = build_profile(raw, centres, L_all[ok], mask=mask)
    say(f"profile v2 (measured extents from {pinfo2['extents']['n_strong']} strong spots, L up to {pinfo2['extents']['L_max']:.0f}): half-extent {model.ext_f0:.2f}+{model.ext_f1:.3f}*L frames, "
        f"{model.ext_p:.2f} px, margin {model.margin} (plateau converged={pinfo2['margin']['converged']})")
    say("margin sweep (m, median I(m)/I(mmax), p10, n): " + "; ".join(f"{m}:{a:.4f}/{b:.4f}/{n}" for m, a, b, n in pinfo2["margin"]["table"]))
else:
    model, pinfo = fit_profile_model(raw, centres, L_all[ok], mask=mask)
    model, sweep = choose_support_scale(raw, centres, L_all[ok], model, mask=mask)
    say(f"profile model v1 (from {pinfo['n_strong']} strong spots): sigma_px={model.sigma_px:.3f} sigma_f={model.sf0:.3f}+{model.sf1:.3f}*L  "
        f"k={model.k} (converged={sweep['converged']})")
    say("support sweep (k, median I(k)/I(kmax), p10, n): " + "; ".join(f"{k:g}:{m:.4f}/{p:.4f}/{n}" for k, m, p, n in sweep["table"]))

dac = DacModel(axis0=tuple(exp.dac_axis0()), t_plus_mm=exp.t_plus_mm, t_minus_mm=exp.t_minus_mm,
               mu_per_cm=exp.mu_diamond_per_mm * 10.0, half_opening_deg=exp.half_opening_deg)
pol = PolarizationModel(fraction=exp.pol_fraction)


def truth_f2(hkls):
    reps = [sg.asu_representative(int(h), int(k), int(l)) for h, k, l in hkls]
    return sim.f2_truth(st, np.array(reps))


def metrics(obs, col="F2", scol="sigF2"):
    m = merge_observations(obs, sg, value=col, sigma=scol) if len(obs) else pd.DataFrame()
    if m.empty:
        return dict(n_obs=len(obs), n_unique=0, R_F2=np.nan, slope=np.nan, dB=np.nan, r_int=np.nan, n_sel=0)
    ft = truth_f2(list(zip(m.h, m.k, m.l)))
    sel = (m[col] / m[scol] > 10) & (m[col] > 0) & (ft > 0)
    if sel.sum() < 3:
        return dict(n_obs=len(obs), n_unique=len(m), R_F2=np.nan, slope=np.nan, dB=np.nan, r_int=np.nan, n_sel=int(sel.sum()))
    x, y = m.loc[sel, col].to_numpy(), ft[sel.to_numpy()]
    k = (x * y).sum() / (x * x).sum()
    R = np.abs(k * x - y).sum() / y.sum()
    d = np.array([lattice.d_spacing(int(h), int(kk), int(l)) for h, kk, l in zip(m.h[sel], m.k[sel], m.l[sel])])
    s2 = (1.0 / (2.0 * d)) ** 2
    if sel.sum() >= 4:
        coef, cov = np.polyfit(s2, np.log(k * x / y), 1, cov=True)
        slope, slope_se = float(coef[0]), float(np.sqrt(cov[0, 0]))
    else:
        slope, slope_se = float(np.polyfit(s2, np.log(k * x / y), 1)[0]), np.nan
    return dict(n_obs=len(obs), n_unique=len(m), R_F2=float(R), slope=slope, slope_se=slope_se, dB=float(-slope / 2.0),
                r_int=float(r_int(obs, sg, value=col)), n_sel=int(sel.sum()))


arms = {
    "FULL (all corrections)": Switches(True, True, True),
    "N1 (no corrections)": Switches(False, False, False),
    "ablate L": Switches(False, True, True),
    "ablate P": Switches(True, False, True),
    "ablate T": Switches(True, True, False),
}
rows, store = [], {}
full_keys = None
for name, sw in arms.items():
    obs, rej, counts = reduce_spots(spots, raw, geom, scan, lattice, res.U, model, dac=dac, pol=pol,
                                    switches=sw, mask=mask)
    # Every arm is scored on the reflections FULL kept. Switching a correction off also switches off its
    # rejection test (factor_variation uses P*T), so an unrestricted ablation would be scored on a
    # DIFFERENT reflection set and the comparison would move for a reason unrelated to the correction.
    # (Found by the verify artifact lens: one seed's 'ablate T DETECTED' was this, not absorption.)
    key = list(zip(obs["frame"].round(3), obs["row"].round(3), obs["col"].round(3))) if len(obs) else []
    if full_keys is None:
        full_keys = set(key)
    else:
        n_before = len(obs)
        obs = obs[[k in full_keys for k in key]].reset_index(drop=True)
        counts["dropped_to_match_FULL"] = n_before - len(obs)
    store[name] = (obs, counts)
    mt = metrics(obs)
    rows.append(dict(arm=name, **mt))
    say(f"[{name}] kept {counts['kept']}/{counts['input']}" + (f" (scored on FULL's set: {counts['dropped_to_match_FULL']} extra dropped)" if counts.get('dropped_to_match_FULL') else "") + "; rejected: " +
        ", ".join(f"{r}={counts[r]}" for r in counts if r not in ('input', 'kept', 'dropped_to_match_FULL') and counts[r]))

# C5: the blob-label sum through the SAME corrections on the SAME spot set as FULL
obs_full = store["FULL (all corrections)"][0].copy()
corr = (obs_full.L * obs_full.PT).to_numpy()
blob = obs_full.assign(F2=obs_full.blob_sum / corr, sigF2=obs_full.sigF2)            # sigma unchanged: only the intensity differs
mb = metrics(blob)
rows.append(dict(arm="C5: blob label sum, same spots/corrections", **mb))

# wrong omega sign: positive control, the reduction must visibly fail
bad_scan = Scan(-exp.omega_end + exp.omega_step * 0, -exp.omega_step, exp.n_frames)   # mirror: omega -> -omega, frame order kept
obs_bad, rej_bad, c_bad = reduce_spots(spots, raw, geom, bad_scan, lattice, res.U, model, dac=dac, pol=pol, mask=mask)
rows.append(dict(arm="control: wrong omega sign", **metrics(obs_bad)))
say(f"[control: wrong omega sign] kept {c_bad['kept']}/{c_bad['input']}; rejected: " +
    ", ".join(f"{r}={c_bad[r]}" for r in c_bad if r not in ('input', 'kept') and c_bad[r]))

tab = pd.DataFrame(rows)
say("")
say(tab.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

# completeness vs the simulator's own truth: how many true, complete spots made it into FULL?
full = store["FULL (all corrections)"][0]
say("")
say(f"true spots {len(res.truth)}; FULL observations {len(full)}; unique reflections merged {len(merge_observations(full, sg))}")

# ---- preregistered reading
f = tab.set_index("arm")
R_full, sl_full = f.loc["FULL (all corrections)", "R_F2"], f.loc["FULL (all corrections)", "slope"]
R_n1 = f.loc["N1 (no corrections)", "R_F2"]
say("")
say("PREREGISTERED READING (C1): R_F2 <= 5 %, |slope| <= 0.2 A^2, FULL beats N1 by >= 2x in R_F2")
se_full = f.loc["FULL (all corrections)", "slope_se"]
say(f"  R_F2(FULL)={R_full:.4f}  slope={sl_full:+.4f} +- {se_full:.4f} (regression SE; replicate scatter of this design is ~0.04, see DESIGN.md)  "
    f"R_F2(N1)={R_n1:.4f}  ratio N1/FULL={R_n1 / R_full if R_full else np.nan:.2f}")
c1 = (R_full <= 0.05) and (abs(sl_full) <= 0.2) and (R_n1 >= 2 * R_full)
say(f"  C1 numeric criteria met: {c1}   (still PROVISIONAL: synthetic, one noise seed, not /verify'd)")
say("  sensitivity (each ablation must raise R_F2 by >= 0.02 absolute or move |slope| by >= 0.2):")
for a in ("ablate L", "ablate P", "ablate T", "control: wrong omega sign"):
    r_, s_ = f.loc[a, "R_F2"], f.loc[a, "slope"]
    if a.startswith("control"):
        n_ok = int(f.loc[a, "n_obs"])
        say(f"    {a:30s} observations surviving hkl assignment = {n_ok} of {len(spots)} blobs -> "
            f"{'DETECTED (reduction fails at assignment)' if n_ok <= 0.2 * f.loc['FULL (all corrections)', 'n_obs'] else 'NOT DETECTED'}")
        continue
    seen = (np.isfinite(r_) and r_ - R_full >= 0.02) or (np.isfinite(s_) and abs(s_ - sl_full) >= 0.2)
    say(f"    {a:30s} R_F2={r_:.4f} slope={s_:.4f} -> {'DETECTED' if seen else 'NOT DETECTED (that correction is UNTESTED at this geometry)'}")
say(f"  C5: blob-sum R_F2={f.loc['C5: blob label sum, same spots/corrections','R_F2']:.4f} slope={f.loc['C5: blob label sum, same spots/corrections','slope']:.4f}"
    f"  vs fixed-support R_F2={R_full:.4f} slope={sl_full:.4f}")
say(f"peak RSS {rss():.0f} MB")

outdir = Path("results"); outdir.mkdir(exist_ok=True)
(outdir / f"c1{'' if PROF == 'v1' else 'v2'}_{st.name}_{n_int:g}_n{seed}_o{oseed}_E{energy:g}_t{(t_plus if anvil is None else anvil):g}.txt").write_text("\n".join(out_lines) + "\n")
