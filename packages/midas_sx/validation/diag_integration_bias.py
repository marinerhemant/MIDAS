"""Where does a bias in the reduced F2 come from? Reduce NOISE-FREE frames and compare each integrated intensity with the
simulator's exact expected counts (dev).   python diag_integration_bias.py S2"""
import sys, json
sys.path.insert(0, "."); sys.path.insert(0, "..")
from pathlib import Path
import numpy as np, pandas as pd
import simulate as sim
from midas_hkls import Lattice, SpaceGroup, corrections as C
from midas_sx.geometry import Scan, make_geometry, spot_directions
from midas_sx.integrate import build_profile, choose_support_scale, fit_profile_model
from midas_sx.pipeline import find_spots
from midas_sx.reduce import DacModel, PolarizationModel, reduce_spots

which = sys.argv[1]
st = sim.structure_cu() if which == "S1" else sim.structure_s2_bilayer()
CACHE = Path("/private/tmp/claude-503/-Volumes-T5-EVO-travel-work-opt-MIDAS/0372d3ae-3aef-4d1a-ba24-2ca52591f359/scratchpad/c2_cache")
cache = CACHE / f"{st.name}_n100000_o1.npz"
z = np.load(cache)
exp = sim.Experiment(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in json.load(open(str(cache) + ".exp.json")).items()})
truth = pd.read_pickle(str(cache) + ".truth.pkl")
signal = z["signal"]; U = z["U"]
bg = exp.background_map()
NOISY = "noisy" in sys.argv
if NOISY:
    rng = np.random.default_rng(2024)
    raw = np.empty(signal.shape, dtype=np.float32)
    for i in range(signal.shape[0]):
        raw[i] = rng.poisson(signal[i] + bg).astype(np.float32)
else:
    raw = (signal + bg[None]).astype(np.float32)                   # NOISE-FREE frames
scan = Scan(exp.omega_start, exp.omega_step, exp.n_frames)
geom = make_geometry(wavelength_A=exp.wavelength_A, dist_mm=exp.dist_mm, pixel_mm=exp.pixel_mm, centre_rc=exp.centre_rc,
                     shape=(exp.det_n, exp.det_n), scan=scan)
spots, mask, info = find_spots(raw, geom, threshold=40.5)           # the threshold of the noisy runs (7 sigma of their noise)
print("blobs", len(spots), info["blob_counts"])
s_all, tth_all, _ = spot_directions(spots.row.to_numpy(), spots.col.to_numpy(), geom)
L_all = C.rotation_lorentz(s_all, min_abs=1e-3)
ok = (tth_all > 2.0) & np.isfinite(L_all)
cen = spots[["frame", "row", "col"]].to_numpy(float)[ok]
V2 = "v2" in sys.argv
if V2:
    model, pinfo = build_profile(raw, cen, L_all[ok], mask=mask)
    print("profile v2:", model, "\n  extents from", pinfo["extents"]["n_strong"], "strong spots, L up to", round(pinfo["extents"]["L_max"], 1),
          "; margin table (m, median, p10, n):", [(m, round(a, 4), round(b, 4), n) for m, a, b, n in pinfo["margin"]["table"]], "chosen", pinfo["margin"]["chosen"])
else:
    model, _ = fit_profile_model(raw, cen, L_all[ok], mask=mask)
    model, sweep = choose_support_scale(raw, cen, L_all[ok], model, mask=mask)
    print("profile v1", model, "converged", sweep["converged"])
lattice = Lattice(*st.cell)
dac = DacModel(axis0=tuple(exp.dac_axis0()), t_plus_mm=exp.t_plus_mm, t_minus_mm=exp.t_minus_mm,
               mu_per_cm=exp.mu_diamond_per_mm * 10.0, half_opening_deg=exp.half_opening_deg)
obs, rej, counts = reduce_spots(spots, raw, geom, scan, lattice, U, model, dac=dac, pol=PolarizationModel(exp.pol_fraction), mask=mask)
print("kept", counts["kept"], "of", counts["input"])
# match to the exact truth by (h,k,l) and omega
t = truth.copy()
rows = []
for r in obs.itertuples():
    m = t[(t.h == r.h) & (t.k == r.k) & (t.l == r.l)]
    if m.empty: continue
    j = (m.omega - r.omega).abs().idxmin()
    if abs(m.omega[j] - r.omega) > 1.0: continue
    rows.append(dict(h=r.h, k=r.k, l=r.l, I=r.I, exp_counts=m.expected_counts[j], tth=r.tth, L=r.L, sigI=r.sigI, bg=r.bg, n_box=r.n_box))
d = pd.DataFrame(rows)
d["ratio"] = d.I / d.exp_counts
d["lc"] = np.log10(d.exp_counts)
print(f"\nmatched {len(d)} spots; I_integrated / I_exact (noise-free frames)")
print("  all:      median %.4f  mean %.4f  sd %.4f  min %.4f  max %.4f" % (d.ratio.median(), d.ratio.mean(), d.ratio.std(), d.ratio.min(), d.ratio.max()))
for lo, hi in ((0, 3e3), (3e3, 1e4), (1e4, 3e4), (3e4, 1e9)):
    g = d[(d.exp_counts >= lo) & (d.exp_counts < hi)]
    if len(g): print(f"  expected counts {lo:>7.0f}-{hi:<9.0f}: n={len(g):3d}  median ratio {g.ratio.median():.4f}  mean {g.ratio.mean():.4f}")
s2 = (np.sin(np.radians(d.tth / 2)) / exp.wavelength_A) ** 2
coef = np.polyfit(s2, np.log(d.ratio), 1)
print("  trend ln(ratio) vs (sin theta/lambda)^2: slope %+.4f /A^-2  (a B error would show as -2*dB; slope 0 = no resolution-dependent bias)" % coef[0])
print("  trend ln(ratio) vs log10(expected counts): slope %+.4f" % np.polyfit(d.lc, np.log(d.ratio), 1)[0])
d["eta"] = np.nan
import itertools
w = obs.reset_index(drop=True)
m = d.merge(w[["h","k","l","omega","eta","row","col","n_box"]].rename(columns={"n_box":"nb"}), on=["h","k","l"], how="left", suffixes=("","_o")) if False else d
bad = d.sort_values("ratio").head(12)
pd.set_option("display.width", 220)
print("\nworst 12 spots by I/I_exact (noise-free):")
tmp = obs.copy(); tmp["_k"] = list(zip(tmp.h, tmp.k, tmp.l, tmp.omega.round(3)))
print(bad[["h","k","l","ratio","exp_counts","tth","L","n_box"]].round(4).to_string(index=False))
print("  profile hf,hp used for these L:", [model.half_widths(l) for l in bad.L.values[:6]])
print("  trend ln(ratio) vs L: slope %+.5f" % np.polyfit(d.L, np.log(d.ratio), 1)[0])

if NOISY:
    d["z"] = (d.I - d.exp_counts) / d.sigI
    print(f"\nNOISY frames: z = (I - I_exact)/sigI over {len(d)} spots: mean {d.z.mean():+.3f}  sd {d.z.std():.3f}  (1.000 if sigma is honest)")
    for lo, hi in ((0, 3e3), (3e3, 1e4), (1e4, 3e4), (3e4, 1e9)):
        g = d[(d.exp_counts >= lo) & (d.exp_counts < hi)]
        if len(g) > 2: print(f"  expected counts {lo:>7.0f}-{hi:<9.0f}: n={len(g):3d}  z mean {g.z.mean():+.3f}  z sd {g.z.std():.3f}  median sigI/I {np.median(g.sigI/g.I):.3f}")
    print("  relative error sd / quoted relative sigma, by expected counts is the same ratio as z sd above")

# ---- A11 acceptance
print("\nA11 ACCEPTANCE (" + ("v2" if V2 else "v1") + ", " + ("noisy" if NOISY else "noise-free") + "):")
if not NOISY:
    print(f"  min ratio {d.ratio.min():.4f} (need >= 0.98): {'MET' if d.ratio.min() >= 0.98 else 'NOT MET'};  |median-1| {abs(d.ratio.median()-1):.4f} (need <= 0.005): {'MET' if abs(d.ratio.median()-1) <= 0.005 else 'NOT MET'}")
else:
    ok_all = True
    for lo, hi in ((0, 3e3), (3e3, 1e4), (1e4, 3e4), (3e4, 1e9)):
        g = d[(d.exp_counts >= lo) & (d.exp_counts < hi)]
        if len(g) >= 10:
            met = g.z.std() <= 1.3; ok_all &= met
            print(f"  class {lo:.0f}-{hi:.0f}: n={len(g)} z sd {g.z.std():.3f} (need <= 1.3): {'MET' if met else 'NOT MET'}")
        else:
            print(f"  class {lo:.0f}-{hi:.0f}: n={len(g)} (< 10, not scored)")
    print("  overall z mean %+.3f sd %.3f; all scored classes met: %s" % (d.z.mean(), d.z.std(), ok_all))
