"""A38 diagnostic: what profile (support) does build_profile measure on thiourea run 01, and what do the box and its background do to the weak reflections?"""
import os, sys
sys.path[:0] = ["/Volumes/T5 EVO/travel/work/opt/MIDAS/packages/midas_sx"]
import numpy as np, pandas as pd
from midas_hkls import corrections as C
from midas_sx.geometry import spot_directions
from midas_sx.integrate import build_profile
from midas_sx.io_frames import read_sfrm_mask, read_sfrm_scan
from midas_sx.workflow import ScanSetup
from midas_sx.reduce import DacModel, PolarizationModel
D="/Volumes/T5 EVO/open_data/thiourea_zenodo_18910433/x/2_T130K_18MPC001"
st, h = read_sfrm_scan(D, prefix="mo_18MPC001_rampT160K_01_"); dead = read_sfrm_mask(D + "/mo_18MPC001_rampT160K_xa_01_0001.sfrm")
raw = st.astype(np.float32); raw[:, dead] = 0
alpha = h.extra["detector_2theta_deg"]; foot = (h.extra["centre_xy_px"][1], h.extra["centre_xy_px"][0]); shift = h.distance_mm*np.tan(np.radians(alpha))/h.pixel_mm
setup = ScanSetup(h.wavelength_A, h.distance_mm/np.cos(np.radians(alpha)), h.pixel_mm, (foot[0], foot[1]+shift), h.angle_start, h.angle_step, tilts_deg=(0,0,alpha))
geom = setup.geometry(h.shape, h.n_frames)
sp = pd.read_csv("/Volumes/T5 EVO/open_data/thiourea_zenodo_18910433/out/run01/spots.csv")
s, tth, _ = spot_directions(sp.row.to_numpy(), sp.col.to_numpy(), geom)
L = C.rotation_lorentz(s, min_abs=1e-3); ok = (tth > 2) & np.isfinite(L)
model, info = build_profile(raw, sp[["frame","row","col"]].to_numpy(float)[ok], L[ok], mask=np.load("/Volumes/T5 EVO/open_data/thiourea_zenodo_18910433/out/run01/mask.npy"))
print(model)
e = info["extents"]; print({k: (v if np.isscalar(v) else np.asarray(v)[:8]) for k, v in e.items()}); print("margin table", info["margin"])
print("half widths at L=2,4,8:", [model.half_widths(l) for l in (2,4,8)])
