"""PREREGISTER A38 null: the omega-scrambled spots of run 01 (frame index randomly permuted), blind ab-initio indexing with the winning geometry; the registered requirement is a
fraction indexed below 0.3 of the real one. Also the same with the wrong geometry class (no swing) as a second reference."""
import json, os, sys
sys.path[:0] = ["/Volumes/T5 EVO/travel/work/opt/MIDAS/packages/midas_sx"]
import numpy as np, pandas as pd, torch
from midas_hkls.ab_initio import index_ab_initio
from midas_defect.geometry import pixel_to_qlab, qlab_to_qsample
from midas_sx.workflow import ScanSetup
O = "/Volumes/T5 EVO/open_data/thiourea_zenodo_18910433/out/"
H = json.load(open(O + "run01/info.json"))["header"]; ex = H["extra"]
alpha = ex["detector_2theta_deg"]; foot = (ex["centre_xy_px"][1], ex["centre_xy_px"][0]); shift = H["distance_mm"] * np.tan(np.radians(alpha)) / H["pixel_mm"]
setup = ScanSetup(H["wavelength_A"], H["distance_mm"] / np.cos(np.radians(alpha)), H["pixel_mm"], (foot[0], foot[1] + shift), H["angle_start"], H["angle_step"], tilts_deg=(0, 0, alpha))
geom, scan = setup.geometry(tuple(H["shape"]), H["n_frames"]), setup.scan(H["n_frames"])
sp = pd.read_csv(O + "run01/spots.csv").sort_values("integrated", ascending=False).head(700).reset_index(drop=True)
def frac(frames):
    q = pixel_to_qlab(sp.row.to_numpy(), sp.col.to_numpy(), geom, device="cpu", dtype=torch.float64)
    om = torch.as_tensor(np.radians(scan.omega_of(frames)), dtype=torch.float64)
    res = index_ab_initio(qlab_to_qsample(q, om, 0.0).numpy(), two_pi=True, tol=0.15, min_reflections=30)
    return (res.n_indexed / len(sp) if res.success else 0.0), res
real, res = frac(sp.frame.to_numpy()); print(f"real: {real:.3f} (cell {np.round(res.cell[:3], 3)})")
rng = np.random.default_rng(38)
sc = [frac(rng.permutation(sp.frame.to_numpy()))[0] for _ in range(8)]
print("scrambled:", np.round(sc, 3), "max", max(sc), "ratio to real", round(max(sc) / real, 3), "-> null", "HOLDS" if max(sc) < 0.3 * real else "FAILS")
