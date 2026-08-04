"""Cross-implementation parity of the FF refiner on the real 2 Au grains.

Six implementations of the same refinement, all starting from the SAME seeds,
the same matched spots and the same geometry, so the refiner is the only
variable:

  c-orig      FitPosOrStrainsOMP        (original C, FF_HEDM/src)
  c-omp       midas_fitgrain            (unified C shipped in midas-fit-grain)
  py-f64-cpu  midas_fit_grain (torch)   float64, CPU
  py-f64-gpu  midas_fit_grain (torch)   float64, CUDA
  py-f32-cpu  midas_fit_grain (torch)   float32, CPU
  py-f32-gpu  midas_fit_grain (torch)   float32, CUDA

Comparison is SEED-AWARE: each implementation is scored by how far it moved
from ITS OWN seed, and pairwise against the others. Scoring every backend
against one arbitrary reference hides the case where the reference is the
outlier -- which is exactly what happened here before, when a float32 refiner
that silently returned its input looked "close to C" only because the C answer
had not moved much either.

Reads ``Results/OrientPosFit.bin`` through
``midas_fit_grain.io_binary.read_orient_pos_fit`` rather than reimplementing
the layout -- it is **27** doubles per row with orientation at cols 1:10,
position at 11:14 and lattice at 15:21, not the compact packing one might
assume. Guessing it wrong silently yields a non-integer row count and looks
like "no output".
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

R = Path("/gdata/dm/1ID/2026/bt_1id_jul26/analysis/au3_cubes_ff_000008")
SRC = R / "c_ff_fmt"
OUT = R / "refiner_crosscheck"
ENVB = Path("/home/beams12/S1IDUSER/opt/envs/midas/bin")
PY = ENVB / "python"
C_ORIG = Path.home() / "opt/ffbuild/bin/FitPosOrStrainsOMP"
N_SEEDS = 189
NPROC = 16
ROW = 27          # doubles per row (io_binary.ORIENT_POS_FIT_NCOLS)
C_ORIENT, C_POS, C_LAT = slice(1, 10), slice(11, 14), slice(15, 21)


def _stage(tag: str) -> Path:
    """A private run directory seeded identically to every other."""
    d = OUT / tag
    if d.exists():
        shutil.rmtree(d)
    (d / "Output").mkdir(parents=True)
    (d / "Results").mkdir(parents=True)
    for f in (SRC / "Output").glob("IndexBest*.bin"):
        shutil.copy(f, d / "Output" / f.name)
    for name in ("SpotsToIndex.csv", "ExtraInfo.bin", "InputAll.csv",
                 "InputAllExtraInfoFittingAll.csv", "hkls.csv", "IDsHash.csv",
                 "Spots.bin", "Data.bin"):
        src = SRC / name
        if src.exists():
            shutil.copy(src, d / name)
    txt = (SRC / "paramstest.txt").read_text().splitlines()
    out = []
    for ln in txt:
        if ln.startswith("OutputFolder "):
            ln = f"OutputFolder {d}/Output"
        elif ln.startswith("ResultFolder "):
            ln = f"ResultFolder {d}/Results"
        out.append(ln)
    (d / "paramstest.txt").write_text("\n".join(out) + "\n")
    return d


def _read_fit(d: Path):
    p = d / "Results" / "OrientPosFit.bin"
    if not p.exists() or p.stat().st_size == 0:
        return None
    sys.path.insert(0, str(ENVB.parent / "lib/python3.11/site-packages"))
    from midas_fit_grain.io_binary import read_orient_pos_fit

    return np.asarray(read_orient_pos_fit(p))


def _seeds():
    """Seed orientations/positions the refiner starts from (IndexBest)."""
    for name in ("IndexBest.bin", "IndexBest_all.bin"):
        p = SRC / "Output" / name
        if p.exists() and p.stat().st_size:
            a = np.fromfile(p, dtype=np.float64)
            if a.size % ROW == 0:
                return a.reshape(-1, ROW)
    return None


def run_c_orig(tag="c-orig"):
    d = _stage(tag)
    t0 = time.time()
    r = subprocess.run([str(C_ORIG), str(d / "paramstest.txt"), "0", "1",
                        str(N_SEEDS), str(NPROC)],
                       cwd=str(d), capture_output=True)
    (d / "run.log").write_bytes(r.stdout + b"\n---stderr---\n" + r.stderr)
    return tag, d, time.time() - t0, r.returncode


def run_c_omp(tag="c-omp"):
    sys.path.insert(0, str(ENVB.parent / "lib/python3.11/site-packages"))
    from midas_fit_grain import backend_c

    d = _stage(tag)
    t0 = time.time()
    r = backend_c.run_refiner(d / "paramstest.txt", block_nr=0, n_blocks=1,
                              n_work=N_SEEDS, num_procs=NPROC, cwd=d)
    (d / "run.log").write_bytes(r.stdout + b"\n---stderr---\n" + r.stderr)
    return tag, d, time.time() - t0, r.returncode


def run_py(tag, device, dtype):
    d = _stage(tag)
    env = os.environ.copy()
    env.update(PATH=f"{ENVB}:{env['PATH']}",
               PYTHONPATH=str(Path.home() / "opt/midas_overlay"),
               KMP_DUPLICATE_LIB_OK="TRUE", CUDA_DEVICE_ORDER="PCI_BUS_ID")
    t0 = time.time()
    r = subprocess.run(
        [str(PY), "-m", "midas_fit_grain", str(d / "paramstest.txt"),
         "0", "1", str(N_SEEDS), str(NPROC),
         "--solver", "lbfgs", "--loss", "full3d",
         "--device", device, "--dtype", dtype, "--mode", "all_at_once"],
        cwd=str(d), env=env, capture_output=True)
    (d / "run.log").write_bytes(r.stdout + b"\n---stderr---\n" + r.stderr)
    return tag, d, time.time() - t0, r.returncode


def main():
    OUT.mkdir(exist_ok=True)
    runs = [run_c_orig(), run_c_omp(),
            run_py("py-f64-cpu", "cpu", "float64"),
            run_py("py-f32-cpu", "cpu", "float32"),
            run_py("py-f64-gpu", "cuda", "float64"),
            run_py("py-f32-gpu", "cuda", "float32")]

    seeds = _seeds()
    res = {}
    print(f"{'implementation':14s} {'exit':>5s} {'secs':>7s} {'rows':>6s} "
          f"{'moved from own seed (um)':>26s}")
    for tag, d, secs, rc in runs:
        a = _read_fit(d)
        res[tag] = a
        if a is None:
            print(f"{tag:14s} {rc:5d} {secs:7.1f} {'-':>6s}   NO OUTPUT")
            continue
        moved = "-"
        if seeds is not None and seeds.shape[0] >= a.shape[0]:
            dv = np.linalg.norm(a[:, C_POS] - seeds[:a.shape[0], C_POS], axis=1)
            ok = np.isfinite(dv)
            moved = f"median {np.median(dv[ok]):8.2f}  max {dv[ok].max():8.2f}"
        print(f"{tag:14s} {rc:5d} {secs:7.1f} {a.shape[0]:6d}   {moved}")

    tags = [t for t in res if res[t] is not None]
    if len(tags) < 2:
        print("\nnot enough successful runs to compare")
        return

    counts = {t: res[t].shape[0] for t in tags}
    n = min(counts.values())
    if len(set(counts.values())) > 1:
        print(f"\nNOTE: row counts differ between implementations: {counts}. "
              f"Comparing the first {n} rows only -- a differing count is "
              f"itself a discrepancy, not just a bookkeeping detail.")
    print(f"\nPairwise agreement over the first {n} refined seeds")
    print(f"  {'pair':28s} {'|dpos| um':>22s} {'|dOrient| deg':>16s} "
          f"{'|da| A':>12s}")
    for i, ta in enumerate(tags):
        for tb in tags[i + 1:]:
            A, B = res[ta][:n], res[tb][:n]
            dp = np.linalg.norm(A[:, C_POS] - B[:, C_POS], axis=1)
            oa = A[:, C_ORIENT].reshape(-1, 3, 3)
            ob = B[:, C_ORIENT].reshape(-1, 3, 3)
            rel = np.einsum("nij,nkj->nik", oa, ob)
            tr = np.clip((np.trace(rel, axis1=1, axis2=2) - 1.0) / 2.0, -1, 1)
            dang = np.degrees(np.arccos(tr))
            da = np.abs(A[:, C_LAT.start] - B[:, C_LAT.start])
            f = np.isfinite(dp) & np.isfinite(dang)
            print(f"  {ta+' vs '+tb:28s} "
                  f"med {np.median(dp[f]):7.2f} max {np.max(dp[f]):7.2f} "
                  f"med {np.median(dang[f]):6.3f} "
                  f"med {np.median(da[f]):8.2e}")

    np.save(OUT / "fits.npy",
            {t: res[t] for t in tags}, allow_pickle=True)
    print(f"\nwrote {OUT/'fits.npy'}")


if __name__ == "__main__":
    main()
