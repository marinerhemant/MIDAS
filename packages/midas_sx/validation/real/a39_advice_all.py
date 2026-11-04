"""PREREG_advice.md, real data: run the WHOLE advice module on the L-alanine reductions and check the registered expectations (each within 5 % relative).
python -I a39_advice_all.py <out dir (rNN/ with obs.pkl, spots_blind.csv, rejected.pkl, reduction.json, U.npy)> <step1 dir with run02/info.json> <cif> <runs, e.g. r02,r03,r04>
Acquisition from the CBF header: count cutoff 791927, exposure 0.1964 s; no I0, no calibrant frame in the Zenodo set."""
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
runs, header = {}, {}
for r in names:
    j = json.load(open(os.path.join(out, r, "reduction.json"))); tx, ty, tz = j["tilts_deg"]; rc = j["centre_rc"]
    g = replace(g0, tx_deg=tx, ty_deg=ty, tz_deg=tz, wedge_deg=j["wedge_deg"], lsd_um=j["lsd_mm"] * 1000.0, bcz_px=rc[0], bcy_px=rc[1])
    rejp = os.path.join(out, r, "rejected.pkl")
    runs[r] = SimpleNamespace(obs=pd.read_pickle(os.path.join(out, r, "obs.pkl")), spots=pd.read_csv(os.path.join(out, r, "spots_blind.csv")),
                              rejected=pd.read_pickle(rejp) if os.path.exists(rejp) else None, counts=j["counts"], geom=g, U=np.load(os.path.join(out, r, "U.npy")), profile=None,
                              orientation=SimpleNamespace(rms_px=j["rms_px"], rms_frame=j["rms_frame"], converged=j["converged"], lambda_cell_scale=j.get("lambda_cell_scale", 0.0),
                                                          sigma_lambda_cell_scale=j.get("sigma_lambda_cell_scale", float("nan"))))
    header[r] = g0
mr = merge_runs({k: v.obs for k, v in runs.items()}, crystal.space_group)
acq = Acquisition(exposure_s=0.1964, count_cutoff=791927.0, has_i0=False, has_calibrant=False)
t0 = time.time()
rep = assess(runs, crystal, merged_runs=mr, acquisition=acq, header_geometry=header, setting_step_deg=15.0)
print(f"[all checks: {time.time() - t0:.0f} s]"); print(rep.text())
os.makedirs(os.path.join(out, "advice"), exist_ok=True)
open(os.path.join(out, "advice", "advice_all.txt"), "w").write(rep.text()); json.dump(rep.to_dict(), open(os.path.join(out, "advice", "advice_all.json"), "w"), indent=1, default=str)

F = {f.id: f for f in rep.findings}
exp = []
def chk(label, got, want, tol=0.05, absolute=False):
    if got is None:
        exp.append((label, got, want, "NOT EXPOSED")); return
    ok = abs(got - want) <= (tol if absolute else tol * abs(want))
    exp.append((label, got, want, "PASS" if ok else "FAIL"))
d = F.get("intensity_drift")
if d is not None:
    lr = d.measured["log_scale_range_per_run"]
    for r, w in zip(names, (0.37, 0.47, 0.58)):
        chk(f"smooth-scale log range {r}", lr.get(r), w)
    chk("systematic floor with run scales only", F["systematic_floor"].measured.get("floor_with_run_scales_only") if "systematic_floor" in F else None, 0.142)
else:
    exp.append(("intensity_drift finding", None, "present", "FAIL"))
for r, w in zip(names, (0.413, 0.310, 0.291)):
    chk(f"refined wedge {r} (deg, from geom)", runs[r].geom.wedge_deg, w)
sr = F.get("scan_range")
if sr is not None:
    per = sr.measured["per_run"]
    for r, w in zip(names, (38, 57, 47)):
        chk(f"in-run Friedel pairs observed {r}", per[r]["pairs_observed"], w)
    cr = sr.measured.get("cross_run", {})
    chk("cross-run mates (signed hkl)", cr.get("signed_hkl_with_mate_in_another_run"), 1802, tol=0.0, absolute=True)
    chk("signed hkl observed", cr.get("signed_hkl_observed"), 3120, tol=0.0, absolute=True)
else:
    exp.append(("scan_range finding", None, "present", "FAIL"))
pk = runs[names[0]].spots["peak_counts"]
print("\nblobs with peak_counts > 1e5 (hand definition), r02:", int((pk > 1e5).sum()), "(hand: 12)")
dr = F.get("dynamic_range")
bl = None if dr is None else dr.measured.get("blobs_above_1e5_per_run")
chk("blobs with peak_counts > 1e5 in r02 (exposed by the module)", None if bl is None else bl.get(names[0]), 12, tol=0.0, absolute=True)
exp.append(("no blocking finding", len(rep.by_severity("blocking")), 0, "PASS" if not rep.by_severity("blocking") else "FAIL"))
print("\nREGISTERED REAL-DATA EXPECTATIONS (within 5 % relative unless stated)")
for e in exp:
    print("  %-62s got %-12s want %-8s %s" % (e[0], e[1] if not isinstance(e[1], float) else round(e[1], 4), e[2], e[3]))
json.dump([dict(label=a, got=b, want=c, result=d_) for a, b, c, d_ in exp], open(os.path.join(out, "advice", "expectations.json"), "w"), indent=1, default=str)
