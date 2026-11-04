"""Run the advice.coverage checks on the real L-alanine runs (stand-ins built from out_p0/rNN/{obs.pkl, U.npy, reduction.json}; refined geometry as a39_diag_unindexed.py builds it).
python -I a39_advice_coverage.py <out dir> <step1 dir> <cif> <runs, e.g. r02,r03,r04>"""
import json, os, sys, time
from types import SimpleNamespace
_sx = os.environ.get("MIDAS_SX_PATH", "/Volumes/T5 EVO/travel/work/opt/MIDAS/packages/midas_sx")
sys.path[:0] = [_sx, os.path.dirname(os.path.abspath(__file__))] + [os.path.join(os.path.dirname(_sx), p) for p in ("midas_hkls", "midas_defect", "midas_stress") if os.path.isdir(os.path.join(os.path.dirname(_sx), p))]
import numpy as np, pandas as pd, gemmi
from dataclasses import replace
from midas_hkls import Atom, Crystal, Lattice, SpaceGroup
from midas_sx.advice import Acquisition, assess
from midas_sx.conventions import Convention
from midas_sx.refine import EIGHT_PI2
from midas_sx.workflow import merge_runs
from a39_geom import swing_geometry
out, step1, cif, names = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4].split(",")
ss = gemmi.read_small_structure(cif)
crystal = Crystal(Lattice(ss.cell.a, ss.cell.b, ss.cell.c, ss.cell.alpha, ss.cell.beta, ss.cell.gamma), SpaceGroup.from_hm("P 21 21 21"),
                  [Atom(s.element.name, (s.fract.x, s.fract.y, s.fract.z), s.occ, EIGHT_PI2 * s.u_iso, label=s.label) for s in ss.sites], "L-alanine 100K")
H = json.load(open(os.path.join(step1, "run02", "info.json")))["header"]
g0, scan = swing_geometry(H, Convention(0, -1), "cols", -1, -1)
runs = {}
for r in names:
    red = json.load(open(os.path.join(out, r, "reduction.json"))); tx, ty, tz = red["tilts_deg"]; rc = red["centre_rc"]
    g = replace(g0, tx_deg=tx, ty_deg=ty, tz_deg=tz, wedge_deg=red["wedge_deg"], lsd_um=red["lsd_mm"] * 1000.0, bcz_px=rc[0], bcy_px=rc[1])
    runs[r] = SimpleNamespace(geom=g, U=np.load(os.path.join(out, r, "U.npy")), obs=pd.read_pickle(os.path.join(out, r, "obs.pkl")), spots=None, counts=red["counts"],
                              orientation=SimpleNamespace(rms_px=red["rms_px"]))
merged = merge_runs({k: v.obs for k, v in runs.items()}, crystal.space_group)
t0 = time.time()
rep = assess(runs, crystal, merged_runs=merged, acquisition=Acquisition(), checks=["scan_range", "completeness_redundancy", "resolution_limit"])
print(f"[3 checks without next_setting: {time.time() - t0:.0f} s]")
t1 = time.time()
rep2 = assess(runs, crystal, merged_runs=merged, checks=["next_setting"], setting_step_deg=15.0)
print(f"[next_setting: {time.time() - t1:.0f} s]")
print(rep.text()); print(); print(rep2.text())
json.dump(dict(coverage=rep.to_dict(), next_setting=rep2.to_dict()), open(os.path.join(out, "advice_coverage.json"), "w"), indent=1, default=str)
# single run r02: lower completeness, so next_setting is active; time it at the real detector size
from midas_sx.advice import coverage as C, AssessContext
one = {names[0]: runs[names[0]]}
m1 = merge_runs({names[0]: runs[names[0]].obs}, crystal.space_group)
t2 = time.time(); rep3 = assess(one, crystal, merged_runs=m1, checks=["completeness_redundancy", "next_setting"], setting_step_deg=15.0)
print(f"\n[single run {names[0]}: completeness + next_setting: {time.time() - t2:.0f} s]"); print(rep3.text())
# the numbers behind checks that raised nothing (the report only holds findings)
ctx = AssessContext(runs, crystal, merged, Acquisition(), {})
ex = C._expected(ctx); cv = C._coverage_numbers(ctx)
n = cv["n_obs"]
print("\nall three runs, numbers behind the silent completeness check:", dict(corner_d_min=round(ex["d_corner"], 3), effective_d_min=round(ex["d_min"], 3), expected_unique=len(n), partial_ring_unique_dropped=ex["n_dropped_partial_rings"],
      reachable=int(cv["reach"].sum()), n1=round(float((n >= 1).mean()), 3), n2=round(float((n >= 2).mean()), 3), strong_n2=round(float((n[cv["strong"]] >= 2).mean()), 3)))
