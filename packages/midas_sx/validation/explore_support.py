"""Why does the support sweep show a ~8 % loss at the 10th percentile? Per-spot I(k)/I(6) vs L (dev)."""
import sys; sys.path.insert(0, "."); sys.path.insert(0, "..")
import numpy as np, pandas as pd
import simulate as sim
from midas_hkls import corrections as C
from midas_sx.geometry import Scan, make_geometry, spot_directions
from midas_sx.integrate import fit_profile_model, integrate_box
from midas_sx.pipeline import find_spots
pd.set_option("display.width", 250)
exp = sim.make_experiment(dac_tilt_deg=(2.0, 1.0), t_plus_mm=2.5, t_minus_mm=3.0)
res = sim.simulate(sim.structure_cu(), exp, n_int_max=1e5, target_hits=100_000, max_pairs=20_000_000)
raw = res.frames(seed=11)
scan = Scan(exp.omega_start, exp.omega_step, exp.n_frames)
geom = make_geometry(wavelength_A=exp.wavelength_A, dist_mm=exp.dist_mm, pixel_mm=exp.pixel_mm, centre_rc=exp.centre_rc, shape=(exp.det_n, exp.det_n), scan=scan)
spots, mask, info = find_spots(raw, geom)
s, tth, _ = spot_directions(spots.row.to_numpy(), spots.col.to_numpy(), geom)
L = C.rotation_lorentz(s, min_abs=1e-3); ok = (tth > 2) & np.isfinite(L)
cen = spots[["frame", "row", "col"]].to_numpy(float)[ok]; Lk = L[ok]
model, pinfo = fit_profile_model(raw, cen, Lk, mask=mask)
rows = []
for (cf, cr, cc), Li in zip(cen, Lk):
    ref = integrate_box(raw, (cf, cr, cc), model.with_k(6).half_widths(Li), mask=mask, shell=(2, 3))
    if not ref.ok or ref.clipped_frames or ref.clipped_detector: continue
    rec = dict(L=Li, peak=ref.peak, I6=ref.intensity)
    for k in (2, 3, 4, 5):
        hw = model.with_k(k).half_widths(Li); r = integrate_box(raw, (cf, cr, cc), hw, mask=mask, shell=(2, 3))
        rec[f"k{k}_hf,hp"] = f"{hw[0]},{hw[1]}"; rec[f"k{k}"] = r.intensity / ref.intensity
    hw6 = model.with_k(6).half_widths(Li); rec["k6_hf,hp"] = f"{hw6[0]},{hw6[1]}"
    rows.append(rec)
d = pd.DataFrame(rows).sort_values("L")
print(model)
print(d.round(3).to_string(index=False))
