#!/usr/bin/env python3
"""Phase-0 FF parity gate: prove midas-ff-pipeline == midas-pipeline (--scan-mode ff)
on the same synthetic FF layer, byte/grain-level on Grains.csv.

Usage:  python ff_parity_gate.py <params_template.txt> [--backend c-omp|python] [--workdir DIR]
Exits 0 if PARITY, 1 if mismatch. Run inside the isolated migration env.
"""
import argparse, subprocess, sys, math
from pathlib import Path
import numpy as np

def run(cmd, log):
    print("  $", " ".join(str(c) for c in cmd), flush=True)
    with open(log, "w") as f:
        r = subprocess.run([str(c) for c in cmd], stdout=f, stderr=subprocess.STDOUT)
    return r.returncode

def load_grains(p):
    g = np.genfromtxt(p, comments="%")
    if g.ndim == 1: g = g.reshape(1, -1)
    return dict(OM=g[:, 1:10], pos=g[:, 10:13], strain=g[:, 13:19], rad=g[:, 19], n=len(g))

def misorient_min(om1, om2):
    # cubic-agnostic raw misorientation (deg) — small if same grain
    R = om1.reshape(3,3) @ om2.reshape(3,3).T
    c = (np.trace(R) - 1) / 2
    return math.degrees(math.acos(max(-1, min(1, c))))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("params_template")
    ap.add_argument("--backend", default="c-omp")
    ap.add_argument("--workdir", default="/tmp/ff_parity")
    ap.add_argument("--n-grains", type=int, default=50)
    ap.add_argument("--n-cpus", type=int, default=8)
    a = ap.parse_args()
    work = Path(a.workdir); work.mkdir(parents=True, exist_ok=True)

    # 1. synthetic FF layer (shared generator)
    from midas_ff_pipeline.testing import generate_synthetic_dataset
    synth = work / "synth"; synth.mkdir(exist_ok=True)
    print("[1/4] generating synthetic FF dataset ...", flush=True)
    zip_path = generate_synthetic_dataset(out_dir=synth, params_template=Path(a.params_template),
                                          n_grains=a.n_grains, seed=42, n_cpus=a.n_cpus, device="cpu")
    print("   zarr:", zip_path, flush=True)
    P = a.params_template

    # 2. ff-pipeline
    ff = work / "ff_out"; (ff / "LayerNr_1").mkdir(parents=True, exist_ok=True)
    print("[2/4] midas-ff-pipeline ...", flush=True)
    rc1 = run(["midas-ff-pipeline", "run", "--params", P, "--result", str(ff),
               "--zarr", str(zip_path), "--layers", "1-1", "--device", "cpu",
               "--n-cpus", str(a.n_cpus), "--indexer-backend", a.backend,
               "--pg-mode", "spot_aware"], work / "ff.log")

    # 3. pipeline --scan-mode ff
    pp = work / "pipe_out"; (pp / "LayerNr_1").mkdir(parents=True, exist_ok=True)
    print("[3/4] midas-pipeline --scan-mode ff ...", flush=True)
    rc2 = run(["midas-pipeline", "run", "--params", P, "--result", str(pp),
               "--zarr", str(zip_path), "--scan-mode", "ff", "--layers", "1-1",
               "--device", "cpu", "--indexer-backend", a.backend,
               "--pg-mode", "spot_aware"], work / "pipe.log")
    print(f"   exit codes: ff={rc1} pipe={rc2}", flush=True)

    # 4. compare
    print("[4/4] comparing Grains.csv ...", flush=True)
    fg = ff / "LayerNr_1" / "Grains.csv"; pg = pp / "LayerNr_1" / "Grains.csv"
    if not fg.exists() or not pg.exists():
        print(f"FAIL: missing Grains.csv (ff={fg.exists()} pipe={pg.exists()}). See logs in {work}")
        return 1
    A, B = load_grains(fg), load_grains(pg)
    print(f"   n_grains: ff={A['n']}  pipe={B['n']}")
    if A["n"] != B["n"]:
        print("FAIL: grain count differs"); return 1
    # match each ff grain to nearest pipe grain by orientation
    max_mis = 0.0; max_dpos = 0.0; max_dstrain = 0.0
    for i in range(A["n"]):
        j = min(range(B["n"]), key=lambda k: misorient_min(A["OM"][i], B["OM"][k]))
        max_mis = max(max_mis, misorient_min(A["OM"][i], B["OM"][j]))
        max_dpos = max(max_dpos, float(np.abs(A["pos"][i]-B["pos"][j]).max()))
        max_dstrain = max(max_dstrain, float(np.abs(A["strain"][i]-B["strain"][j]).max()))
    print(f"   max misorientation {max_mis:.4f}°  max Δpos {max_dpos:.3f} µm  max Δstrain {max_dstrain:.2e}")
    ok = (max_mis < 0.05) and (max_dpos < 1.0) and (max_dstrain < 1e-4)
    print("PARITY ✓" if ok else "MISMATCH ✗ (see tolerances)")
    return 0 if ok else 1

if __name__ == "__main__":
    sys.exit(main())
