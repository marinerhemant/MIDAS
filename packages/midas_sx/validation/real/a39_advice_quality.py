"""Run the advice QUALITY checks on the real L-alanine reductions (stand-ins built from the files a39_run.py wrote).
    python -I a39_advice_quality.py <out dir with rNN/ (obs.pkl, spots_blind.csv, reduction.json, rejected.pkl)> <step1 dir with run02/info.json> <cif> <runs, e.g. r02,r03,r04> [merged csv]
Acquisition facts from the CBF header: count cutoff 791927, exposure 0.1964 s; no I0 recorded, no calibrant frame in the Zenodo set."""
import json, os, sys
_sx = os.environ.get("MIDAS_SX_PATH", "/Volumes/T5 EVO/travel/work/opt/MIDAS/packages/midas_sx")
sys.path[:0] = [_sx, os.path.dirname(os.path.abspath(__file__))] + [os.path.join(os.path.dirname(_sx), p) for p in ("midas_hkls", "midas_defect", "midas_stress") if os.path.isdir(os.path.join(os.path.dirname(_sx), p))]
from dataclasses import replace
from types import SimpleNamespace
import numpy as np, pandas as pd, gemmi
from midas_hkls import Atom, Crystal, Lattice, SpaceGroup
from midas_sx.advice import Acquisition, assess
from midas_sx.conventions import Convention
from midas_sx.refine import EIGHT_PI2
from a39_geom import swing_geometry

out, step1, cif, runs = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4].split(",")
ss = gemmi.read_small_structure(cif)
crystal = Crystal(Lattice(ss.cell.a, ss.cell.b, ss.cell.c, ss.cell.alpha, ss.cell.beta, ss.cell.gamma), SpaceGroup.from_hm("P 21 21 21"),
                  [Atom(s.element.name, (s.fract.x, s.fract.y, s.fract.z), s.occ, EIGHT_PI2 * s.u_iso, label=s.label) for s in ss.sites], "L-alanine")
H = json.load(open(os.path.join(step1, "run02", "info.json")))["header"]
g0, scan = swing_geometry(H, Convention(0, -1), "cols", -1, -1)
reds, header = {}, {}
for r in runs:
    j = json.load(open(os.path.join(out, r, "reduction.json")))
    tx, ty, tz = j["tilts_deg"]; rc = j["centre_rc"]
    g = replace(g0, tx_deg=tx, ty_deg=ty, tz_deg=tz, wedge_deg=j["wedge_deg"], lsd_um=j["lsd_mm"] * 1000.0, bcz_px=rc[0], bcy_px=rc[1], wavelength_A=j.get("wavelength_A", g0.wavelength_A))
    rejp = os.path.join(out, r, "rejected.pkl")
    reds[r] = SimpleNamespace(obs=pd.read_pickle(os.path.join(out, r, "obs.pkl")), spots=pd.read_csv(os.path.join(out, r, "spots_blind.csv")),
                              rejected=pd.read_pickle(rejp) if os.path.exists(rejp) else None, counts=j["counts"], geom=g, U=np.load(os.path.join(out, r, "U.npy")), profile=None,
                              orientation=SimpleNamespace(rms_px=j["rms_px"], rms_frame=j["rms_frame"], converged=j["converged"], lambda_cell_scale=j.get("lambda_cell_scale", 0.0),
                                                          sigma_lambda_cell_scale=j.get("sigma_lambda_cell_scale", float("nan"))))
    header[r] = g0
quality = ["systematic_floor", "intensity_drift", "dynamic_range", "omega_sampling", "geometry_consistency", "contamination"]
rep = assess(reds, crystal, checks=quality, acquisition=Acquisition(exposure_s=0.1964, count_cutoff=791927.0, has_i0=False, has_calibrant=False), header_geometry=header)
print(rep.text())
json.dump(rep.to_dict(), open(os.path.join(out, "advice_quality.json"), "w"), indent=1, default=str)
