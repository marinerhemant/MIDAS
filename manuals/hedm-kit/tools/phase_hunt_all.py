#!/usr/bin/env python3
"""Run phase_hunt.py on the middle data file of every scan folder listed in a CSV, then print one summary table.

Usage:  phase_hunt_all.py <samples.csv> <out_dir> [--workers 4] [--ndraw 5000] [--primary SG:a,b,c,al,be,ga] [--candidates c.json]
        phase_hunt_all.py --summary <out_dir> ["candidate name"]   (re-print the table from existing *_hunt.json; the focus candidate is the first one in the candidates file unless named)

samples.csv columns:  name,folder,energy_keV,lsd_mm      (folder = absolute path of ONE scan folder holding *.vrx.h5 layers;
                      energy and Lsd are FACTS you must supply per folder: Varex 20-ID files do not record the energy)
Each run reads one 24 GB sweep (about 2 minutes) and needs ~10 GB RAM: keep --workers small (the zipper-like memory traps apply).
The summary lists, per folder, the number of ring-level and spot-level lines the primary phase does not explain and, for every
candidate cell, matches / lines in the window / chance mean / p. A match to a candidate is a hypothesis about identity (a line list
fits any phase with enough lines), and "no lines" is NOT a negative unless phase_hunt.py --plant recovered its planted rings.
"""
import csv
import glob
import json
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).parent


def middle_file(folder):
    fs = sorted(f for f in Path(folder).glob("*.vrx.h5") if not re.search(r"dark_(before|after)", f.name))
    return fs[len(fs) // 2] if fs else None


def summary(out, focus=None):
    """One row per folder. `ring`/`spot` = number of ring-level / spot-level lines the primary phase does NOT explain.
    For the best candidate (smallest chance probability) and for the focus candidate (default: the first one in the candidates file):
    'k of n obs' = k of the n observed ring lines lie within 0.4 % in d of a line of that cell; 'w in window' = lines the cell has in the
    observed d window; 'chance' = mean k for n random radii; p = fraction of random draws with >= k (printed '<1/N' when none reached it)."""
    print("ring/spot = lines NOT explained by the primary phase (ring level / spot level). 'k of n obs, w in window, chance, p' are explained in the script header.")
    print(f"{'folder':46s} ring spot | best candidate -> | focus candidate")
    def fmt(v):
        n = v["n_obs_lines"]; p = v["p_chance_ge"]; nd = v.get("ndraw", 3000)
        return f"{v['matched']} of {n} obs, {v['lines_in_window']} in window, chance {v['chance_matched_mean']:.2f}, p " + (f"<{1 / nd:.0e}" if p == 0 else f"{p:.3g}")
    for f in sorted(glob.glob(str(Path(out) / "*_hunt.json"))):
        r = json.load(open(f))
        nm = Path(f).name[:-10]
        c = r.get("candidates", {}).get("ring", {})
        best = min(c.items(), key=lambda kv: (kv[1]["p_chance_ge"], -kv[1]["matched"]), default=None)
        fk = focus or (next(iter(c)) if c else None)
        fv = c.get(fk)
        b = f"{best[0]}: {fmt(best[1])}" if best else "no ring lines"
        ft = f"{fk}: {fmt(fv)}" if fv else "-"
        print(f"{nm:46s} {r.get('n_L_ring', 0):4d} {r.get('n_L_spot', 0):4d} | {b} | {ft}")
    print("NOTE: 'no lines' or a weak match is a NEGATIVE only if phase_hunt.py --plant recovered its planted rings at this sensitivity (see the README).")


def main():
    a = sys.argv[1:]
    if not a or a[0] in ("-h", "--help"):
        sys.exit(__doc__)
    if a[0] == "--summary":
        return summary(a[1], a[2] if len(a) > 2 else None)
    csvf, out = a[0], Path(a[1])
    opt = {a[i]: a[i + 1] for i in range(2, len(a) - 1, 2)}
    out.mkdir(parents=True, exist_ok=True)
    jobs = list(csv.DictReader(open(csvf)))

    def run(j):
        f = middle_file(j["folder"])
        if f is None:
            return j["name"], "no data file"
        if (out / f"{j['name']}_hunt.json").exists():
            return j["name"], "exists"
        cmd = [sys.executable, "-B", "-I", str(HERE / "phase_hunt.py"), str(f), str(out / j["name"]), j["energy_keV"], j["lsd_mm"], "--ndraw", opt.get("--ndraw", "5000")]
        for k in ("--primary", "--candidates", "--px"):
            if k in opt:
                cmd += [k, opt[k]]
        r = subprocess.run(cmd, capture_output=True, text=True)
        (out / f"{j['name']}.log").write_text(r.stdout + r.stderr)
        return j["name"], r.returncode

    with ThreadPoolExecutor(int(opt.get("--workers", 4))) as ex:
        for name, rc in ex.map(run, jobs):
            print(name, rc, flush=True)
    summary(out)


if __name__ == "__main__":
    main()
