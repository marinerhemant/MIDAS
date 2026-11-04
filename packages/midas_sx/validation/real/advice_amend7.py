"""Amendment 7 of PREREG_advice.md: do dynamic_range and omega_sampling DISCRIMINATE D2/D3 from clean frames?
Runs on the cached high-count reductions of advice_validate.py (copland). Usage: MIDAS_SX_PATH=<code dir> python -I advice_amend7.py <out.json>"""
import json
import os
import pickle
import sys
import warnings
from dataclasses import replace

import numpy as np

os.environ["ADVICE_HI"] = "1"
os.environ["ADVICE_DMIN"] = "0.66"
sys.path.insert(0, os.path.join(os.environ["MIDAS_SX_PATH"], "validation", "real"))
import advice_validate as av


def load(prefix):
    return {f"r{s}": pickle.load(open(os.path.join(av.WORK, "red", f"{prefix}_{s}.pkl"), "rb")) for s in av.SEEDS}


def assess(reds, acq):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return av.assess(reds, av.CRYSTAL, acquisition=acq, header_geometry={k: v.header_geometry for k, v in reds.items()}, checks=["dynamic_range", "omega_sampling"], d_min=0.66)


def summary(rep):
    out = []
    for f in rep.findings:
        keep = ("fraction_clipped", "observations_clipped", "blobs_clipped_per_run", "fraction", "single_frame_fraction", "median_rms_width_frames", "max_peak_counts", "cutoff", "cutoff_is_a_guess")
        out.append(dict(id=f.id, severity=f.severity, title=f.title, **{k: f.measured[k] for k in keep if k in f.measured}))
    return dict(findings=out, numbers=rep.numbers.get("dynamic_range"))


clean, d2, d3 = load("H_C"), load("H_D2"), load("H_D3")
cut_d2 = 226888.0                                                    # the cutoff D2 was clipped at (4 x median clean peak)
pk_clean_max = max(float(np.max(r.spots["peak_counts"])) for r in clean.values())
res = {}
res["D2"] = summary(assess(d2, replace(av.ACQ, count_cutoff=cut_d2)))
res["D2c_clean_cutoff_1.5xmax"] = summary(assess(clean, replace(av.ACQ, count_cutoff=1.5 * pk_clean_max)))
res["D2n_clean_no_cutoff"] = summary(assess(clean, av.ACQ))
res["D2_no_cutoff"] = summary(assess(d2, av.ACQ))
res["D3"] = summary(assess(d3, av.ACQ))
res["C_omega"] = summary(assess(clean, av.ACQ))
res["clean_max_peak"] = pk_clean_max
json.dump(res, open(sys.argv[1], "w"), indent=1, default=str)
for k, v in res.items():
    print(k, json.dumps(v, default=str)[:900])
