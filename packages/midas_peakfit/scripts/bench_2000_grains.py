#!/usr/bin/env python
"""Benchmark driver: 2000-grain Au dataset, forward sim, then peakfit C / Py-CPU / Py-GPU.

Compares fitted peak positions against:
  (a) the C-tool's output as oracle
  (b) the simulator's ground truth from SpotMatrixGen.csv

Usage (on alleppey, after sourcing midas_env):
  python bench_2000_grains.py [--n-grains 2000] [--workdir /scratch/s1iduser/peakfit_bench]
                              [--skip-sim] [--skip-c] [--skip-cpu] [--skip-gpu]
                              [--n-cpus 64] [--gpu-id 0]

Each stage is checkpointed: if --skip-sim is passed (or sim outputs already exist),
the run resumes from the next stage. Useful for re-running comparisons without
re-doing the slow simulation.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import zarr  # noqa: F401  (loaded by the enrichment helper)

MIDAS_HOME = Path(os.environ.get("MIDAS_HOME", "/home/beams/S1IDUSER/opt/MIDAS"))
sys.path.insert(0, str(MIDAS_HOME / "utils"))
sys.path.insert(0, str(MIDAS_HOME / "tests"))


# ─── Stage 1: generate grains + Parameters.txt with absolute paths ─────────
def write_parameters_with_n_grains(
    src_param: Path, work_dir: Path, n_grains: int, seed: int
) -> Path:
    """Generate GrainsSim.csv with N random grains; emit Parameters.txt with abs paths."""
    grains_csv = work_dir / "GrainsSim.csv"
    new_param = work_dir / "Parameters.txt"

    # Use existing generate_grains.py
    from generate_grains import generate_grains_csv

    # Parse the source param file to extract sample dimensions + lattice
    rsample, hbeam, beam_thickness, sg = 2000, 2000, 200, 225
    lat = [4.08, 4.08, 4.08, 90, 90, 90]
    src_lines = src_param.read_text().splitlines()
    for line in src_lines:
        toks = line.split("#", 1)[0].split()
        if not toks:
            continue
        if toks[0] == "Rsample" and len(toks) > 1:
            rsample = float(toks[1])
        elif toks[0] == "Hbeam" and len(toks) > 1:
            hbeam = float(toks[1])
        elif toks[0] == "BeamThickness" and len(toks) > 1:
            beam_thickness = float(toks[1])
        elif toks[0] == "SpaceGroup" and len(toks) > 1:
            sg = int(toks[1])
        elif toks[0] == "LatticeConstant" and len(toks) >= 7:
            lat = [float(toks[i]) for i in range(1, 7)]

    print(f"[gen] Generating {n_grains} grains → {grains_csv}")
    print(
        f"  Rsample={rsample}µm Hbeam={hbeam}µm BeamThickness={beam_thickness}µm "
        f"SG={sg} LatticeConstant={lat}"
    )
    generate_grains_csv(
        grains_csv, n_grains, lat, rsample, hbeam, beam_thickness,
        space_group=sg, seed=seed,
    )

    # Rewrite Parameters.txt with absolute paths
    out_stem = "Au_FF_2k"
    with open(new_param, "w") as f:
        for line in src_lines:
            toks = line.split()
            if not toks:
                f.write(line + "\n")
                continue
            if toks[0] == "InFileName":
                f.write(f"InFileName {grains_csv}\n")
            elif toks[0] == "OutFileName":
                f.write(f"OutFileName {work_dir / out_stem}\n")
            else:
                f.write(line + "\n")

    return new_param


# ─── Stage 2: forward simulation ───────────────────────────────────────────
def run_forward_sim(param_file: Path, work_dir: Path, n_cpus: int) -> Path:
    """Run ForwardSimulationCompressed; return path to the final renamed Zarr."""
    bin_path = MIDAS_HOME / "FF_HEDM" / "bin" / "ForwardSimulationCompressed"
    cmd = [str(bin_path), str(param_file), str(n_cpus)]
    print(f"[sim] Command: {' '.join(cmd)}")
    t0 = time.time()
    rc = subprocess.run(cmd, cwd=str(work_dir)).returncode
    if rc != 0:
        raise RuntimeError(f"ForwardSimulationCompressed failed (rc={rc})")
    print(f"[sim] Done in {time.time() - t0:.1f}s")

    # Output is {OutFileName}_scanNr_0.zip → rename to {OutFileName}.analysis.MIDAS.zip
    out_stem = "Au_FF_2k"
    raw_zip = work_dir / f"{out_stem}_scanNr_0.zip"
    final_zip = work_dir / f"{out_stem}.analysis.MIDAS.zip"
    if not raw_zip.exists():
        raise RuntimeError(f"Forward-sim output {raw_zip} missing")
    shutil.move(str(raw_zip), str(final_zip))
    return final_zip


# ─── Stage 3: enrich Zarr with analysis params ─────────────────────────────
def enrich_zarr(zip_path: Path, params_dict: dict) -> None:
    """Inject analysis_parameters / scan_parameters into the Zarr archive."""
    import zarr as _zarr

    with _zarr.ZipStore(str(zip_path), mode="a") as store:
        try:
            zRoot = _zarr.group(store=store)
        except _zarr.errors.GroupNotFoundError:
            zRoot = _zarr.group(store=store, overwrite=True)
        if "analysis" not in zRoot:
            zRoot.create_group("analysis/process/analysis_parameters")
        if "measurement" not in zRoot:
            zRoot.create_group("measurement/process/scan_parameters")
        sp_ana = zRoot.require_group("analysis/process/analysis_parameters")
        sp_pro = zRoot.require_group("measurement/process/scan_parameters")

        data_dtype = str(zRoot["exchange/data"].dtype)
        sp_pro.create_dataset(
            "datatype", data=np.bytes_(data_dtype.encode("UTF-8")), overwrite=True
        )

        from midas_zipper.ff_zip import write_analysis_parameters

        z_groups = {"sp_pro_analysis": sp_ana, "sp_pro_meas": sp_pro}
        write_analysis_parameters(z_groups, params_dict)


def parse_param_file(filepath: Path) -> dict:
    from midas_zipper.ff_zip import parse_parameter_file

    return parse_parameter_file(str(filepath))


# ─── Stage 4: run peakfit (each variant in its own subdir) ─────────────────
def run_c_peakfit(zip_path: Path, work_dir: Path, n_cpus: int) -> Path:
    """Run PeaksFittingOMPZarrRefactor; return Path to AllPeaks_PS.bin."""
    out_dir = work_dir / "out_c"
    out_dir.mkdir(exist_ok=True)
    (out_dir / "Temp").mkdir(exist_ok=True)
    # The C tool reads hkls.csv from {ResultFolder}/hkls.csv, not work_dir
    src_hkl = work_dir / "hkls.csv"
    if src_hkl.exists():
        shutil.copy2(src_hkl, out_dir / "hkls.csv")
    bin_path = MIDAS_HOME / "FF_HEDM" / "bin" / "PeaksFittingOMPZarrRefactor"
    log = out_dir / "stdout.log"
    err = out_dir / "stderr.log"
    cmd = [str(bin_path), str(zip_path), "0", "1", str(n_cpus), str(out_dir)]
    print(f"[c]  Command: {' '.join(cmd)}")
    t0 = time.time()
    with open(log, "w") as flog, open(err, "w") as ferr:
        rc = subprocess.run(cmd, stdout=flog, stderr=ferr).returncode
    elapsed = time.time() - t0
    print(f"[c]  rc={rc}, elapsed={elapsed:.1f}s")
    print(f"[c]  log: {log}")
    return out_dir / "Temp" / "AllPeaks_PS.bin"


def run_py_peakfit(
    zip_path: Path, work_dir: Path, n_cpus: int, *, device: str, dtype: str = "float64",
    gpu_id: int = 0, n_gpus: int = 1, interleave_blocks: bool = False,
) -> Path:
    """Run peakfit_torch; return Path to AllPeaks_PS.bin.

    If ``n_gpus > 1`` and ``device == "cuda"``, launches ``n_gpus`` peakfit_torch
    processes in parallel (one per GPU, each on its own block of frames),
    then merges their outputs into a single ``AllPeaks_PS.bin``.
    """
    suffix = f"py_{device}_{dtype}"
    if n_gpus > 1 and device == "cuda":
        suffix = f"{suffix}_x{n_gpus}gpu"
    out_dir = work_dir / f"out_{suffix}"
    out_dir.mkdir(exist_ok=True)
    (out_dir / "Temp").mkdir(exist_ok=True)
    src_hkl = work_dir / "hkls.csv"
    if src_hkl.exists():
        shutil.copy2(src_hkl, out_dir / "hkls.csv")

    if n_gpus == 1 or device != "cuda":
        # Single-GPU / CPU path
        log = out_dir / "stdout.log"
        err = out_dir / "stderr.log"
        env = os.environ.copy()
        if device == "cuda":
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
        cmd = [
            "peakfit_torch",
            str(zip_path), "0", "1", str(n_cpus),
            str(out_dir), "--device", device, "--dtype", dtype,
        ]
        print(f"[py-{device}] Command: {' '.join(cmd)}")
        t0 = time.time()
        with open(log, "w") as flog, open(err, "w") as ferr:
            rc = subprocess.run(cmd, stdout=flog, stderr=ferr, env=env).returncode
        elapsed = time.time() - t0
        print(f"[py-{device}] rc={rc}, elapsed={elapsed:.1f}s")
        return out_dir / "Temp" / "AllPeaks_PS.bin"

    # Multi-GPU: spawn ``n_gpus`` parallel peakfit_torch instances, each on
    # its own (block, GPU) pair. Frame range is split via the (blockNr, nBlocks)
    # CLI args. Outputs go into per-block subfolders, then we merge.
    print(f"[py-cuda x{n_gpus}gpu] Spawning {n_gpus} parallel block runs…")
    block_dirs: list[Path] = []
    procs: list[tuple[subprocess.Popen, Path, Path, int]] = []
    cpus_per_gpu = max(1, n_cpus // n_gpus)
    t0 = time.time()
    for block_nr in range(n_gpus):
        block_dir = out_dir / f"block_{block_nr}"
        block_dir.mkdir(exist_ok=True)
        (block_dir / "Temp").mkdir(exist_ok=True)
        if src_hkl.exists():
            shutil.copy2(src_hkl, block_dir / "hkls.csv")
        block_log = block_dir / "stdout.log"
        block_err = block_dir / "stderr.log"
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(block_nr)
        env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
        cmd = [
            "peakfit_torch",
            str(zip_path), str(block_nr), str(n_gpus), str(cpus_per_gpu),
            str(block_dir), "--device", "cuda", "--dtype", dtype,
        ]
        if interleave_blocks:
            cmd.append("--interleave-blocks")
        flog = open(block_log, "w")
        ferr = open(block_err, "w")
        proc = subprocess.Popen(cmd, stdout=flog, stderr=ferr, env=env)
        procs.append((proc, block_log, block_err, block_nr))
        block_dirs.append(block_dir)
        print(f"  [block {block_nr} → GPU {block_nr}] pid={proc.pid}")

    # Wait for all
    for proc, _, _, block_nr in procs:
        rc = proc.wait()
        if rc != 0:
            print(f"  [block {block_nr}] rc={rc}")
    elapsed = time.time() - t0
    print(f"[py-cuda x{n_gpus}gpu] all {n_gpus} blocks done, elapsed={elapsed:.1f}s")

    # Merge block outputs into out_dir/Temp/AllPeaks_PS.bin
    print(f"[py-cuda x{n_gpus}gpu] Merging outputs…")
    merge_t0 = time.time()
    sys.path.insert(0, str(MIDAS_HOME / "packages" / "midas_peakfit"))
    from midas_peakfit.compat.merge_blocks import merge_block_outputs
    from midas_peakfit.compat.reference_decoder import read_ps

    # Need n_total_frames + NrPixels from any block's output
    sample_ps = read_ps(block_dirs[0] / "Temp" / "AllPeaks_PS.bin")
    # Read NrPixels from PX header
    with open(block_dirs[0] / "Temp" / "AllPeaks_PX.bin", "rb") as f:
        import numpy as _np
        _ = _np.frombuffer(f.read(4), dtype=_np.int32)[0]
        nr_pixels = int(_np.frombuffer(f.read(4), dtype=_np.int32)[0])

    merge_block_outputs(
        block_dirs,
        out_folder=out_dir,
        n_total_frames=int(sample_ps.n_frames),
        nr_pixels=nr_pixels,
    )
    print(f"[py-cuda x{n_gpus}gpu] merge done in {time.time() - merge_t0:.1f}s")
    return out_dir / "Temp" / "AllPeaks_PS.bin"


# ─── Stage 5: ground truth from SpotMatrixGen.csv ──────────────────────────
def load_ground_truth(work_dir: Path) -> np.ndarray:
    """Read SpotMatrixGen.csv → array of (Omega, YCen, ZCen, Eta, RingNr).

    Columns in the file (0-indexed):
      0 GrainID, 1 SpotID, 2 Omega, 3 DetectorHor (=YCen),
      4 DetectorVert (=ZCen), 5 OmeRaw, 6 Eta, 7 RingNr, ...
    """
    fn = work_dir / "SpotMatrixGen.csv"
    if not fn.exists():
        raise FileNotFoundError(fn)
    rows = []
    with open(fn) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("%"):
                continue
            toks = line.split()
            try:
                ome = float(toks[2])
                y = float(toks[3])
                z = float(toks[4])
                eta = float(toks[6])
                ring = int(float(toks[7]))
                rows.append((ome, y, z, eta, ring))
            except (IndexError, ValueError):
                continue
    return np.array(
        rows,
        dtype=[("ome", "f8"), ("y", "f8"), ("z", "f8"), ("eta", "f8"), ("ring", "i4")],
    )


# ─── Stage 6: comparison ───────────────────────────────────────────────────
def load_ps_peaks(ps_path: Path, params_dict: dict) -> np.ndarray:
    """Read AllPeaks_PS.bin → array of (Omega, YCen, ZCen, Eta, R).

    Frame omega is computed from OmegaStart + frame_index * OmegaStep
    (matching the C tool's per-frame omega assignment).
    """
    sys.path.insert(0, str(MIDAS_HOME / "packages" / "midas_peakfit"))
    from midas_peakfit.compat.reference_decoder import read_ps

    ps = read_ps(ps_path)
    out = []
    omega_start = float(params_dict.get("OmegaStart", 0))
    omega_step = float(params_dict.get("OmegaStep", 1))

    for f_idx in range(ps.n_frames):
        ome = omega_start + omega_step * f_idx
        rows = ps.rows_per_frame[f_idx]
        for r in rows:
            # cols: 2 Omega (per-frame), 3 YCen, 4 ZCen, 7 Eta, 6 R
            out.append((ome, r[3], r[4], r[7], r[6]))
    return np.array(
        out,
        dtype=[("ome", "f8"), ("y", "f8"), ("z", "f8"), ("eta", "f8"), ("r", "f8")],
    )


def match_peaks(
    a: np.ndarray, b: np.ndarray, *, ome_tol: float = 0.5, yz_tol: float = 5.0
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized nearest-neighbor match using cKDTree per omega bucket.

    Two peaks match if ``|Δomega| ≤ ome_tol`` AND ``||(Δy, Δz)|| ≤ yz_tol``.
    Within each omega bucket we do a single batched ``cKDTree.query`` and
    reject by yz_tol; greedy unique assignment across buckets resolves
    multi-claim conflicts in ascending-distance order.

    Drops 231k×231k matching from minutes (Python-loop) to ~1-2s.
    """
    if a.size == 0 or b.size == 0:
        return np.zeros(0, dtype=a.dtype), np.zeros(0, dtype=b.dtype)

    from scipy.spatial import cKDTree

    a_bin = np.floor(a["ome"] / ome_tol).astype(np.int64)
    b_bin = np.floor(b["ome"] / ome_tol).astype(np.int64)

    a_groups: dict[int, np.ndarray] = {
        int(k): np.where(a_bin == k)[0] for k in np.unique(a_bin)
    }
    b_groups: dict[int, np.ndarray] = {
        int(k): np.where(b_bin == k)[0] for k in np.unique(b_bin)
    }

    cand_a, cand_b, cand_d = [], [], []
    for k, a_idx in a_groups.items():
        b_idx_parts = [b_groups.get(k + s) for s in (-1, 0, 1)]
        b_idx_parts = [arr for arr in b_idx_parts if arr is not None and arr.size]
        if not b_idx_parts:
            continue
        b_idx = np.concatenate(b_idx_parts)
        a_yz = np.column_stack([a["y"][a_idx], a["z"][a_idx]])
        b_yz = np.column_stack([b["y"][b_idx], b["z"][b_idx]])
        tree = cKDTree(b_yz)
        dists, nn = tree.query(a_yz, k=1, distance_upper_bound=yz_tol)
        good = dists < yz_tol
        if not good.any():
            continue
        a_local = a_idx[good]
        b_local = b_idx[nn[good]]
        ome_diff = np.abs(a["ome"][a_local] - b["ome"][b_local])
        ok = ome_diff <= ome_tol
        cand_a.append(a_local[ok])
        cand_b.append(b_local[ok])
        cand_d.append(dists[good][ok])

    if not cand_a:
        return np.zeros(0, dtype=a.dtype), np.zeros(0, dtype=b.dtype)

    cand_a = np.concatenate(cand_a)
    cand_b = np.concatenate(cand_b)
    cand_d = np.concatenate(cand_d)

    # Greedy unique assignment in ascending distance order.
    order = np.argsort(cand_d, kind="stable")
    cand_a = cand_a[order]
    cand_b = cand_b[order]
    used_a = np.zeros(a.size, dtype=bool)
    used_b = np.zeros(b.size, dtype=bool)
    out_a = []
    out_b = []
    for ai, bi in zip(cand_a.tolist(), cand_b.tolist()):
        if used_a[ai] or used_b[bi]:
            continue
        used_a[ai] = True
        used_b[bi] = True
        out_a.append(ai)
        out_b.append(bi)
    matched_a = np.asarray(out_a, dtype=np.int64)
    matched_b = np.asarray(out_b, dtype=np.int64)
    return a[matched_a], b[matched_b]


def compute_stats(label: str, a: np.ndarray, b: np.ndarray) -> dict:
    """Per-axis position errors (a is reference, b is candidate)."""
    if a.size == 0:
        return {"label": label, "n": 0}
    dy = b["y"] - a["y"]
    dz = b["z"] - a["z"]
    deta = b["eta"] - a["eta"]
    # Wrap eta diff to [-180, 180]
    deta = ((deta + 180.0) % 360.0) - 180.0
    dist = np.sqrt(dy * dy + dz * dz)

    def pcts(v):
        return {
            "mean": float(np.mean(v)),
            "median": float(np.median(v)),
            "p95": float(np.percentile(v, 95)),
            "p99": float(np.percentile(v, 99)),
            "max": float(np.max(v)),
            "std": float(np.std(v)),
        }

    return {
        "label": label,
        "n": int(a.size),
        "dist_px": pcts(dist),
        "dy_px": pcts(np.abs(dy)),
        "dz_px": pcts(np.abs(dz)),
        "deta_deg": pcts(np.abs(deta)),
    }


def fmt_stats(s: dict) -> str:
    if s["n"] == 0:
        return f"{s['label']:25s}  (no matches)"
    lines = [
        f"== {s['label']} ==",
        f"  matched = {s['n']:>9d}",
    ]
    for axis, key in [
        ("|YZ| dist (px)", "dist_px"),
        ("|dY|     (px)", "dy_px"),
        ("|dZ|     (px)", "dz_px"),
        ("|dEta|  (deg)", "deta_deg"),
    ]:
        d = s[key]
        lines.append(
            f"  {axis:14s}  mean={d['mean']:.4f}  med={d['median']:.4f}  "
            f"p95={d['p95']:.4f}  p99={d['p99']:.4f}  max={d['max']:.4f}"
        )
    return "\n".join(lines)


def make_cdf_plot(stats_lists: dict, work_dir: Path) -> Path:
    """CDF of |YZ| position errors for each comparison."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    for label, dists in stats_lists.items():
        if len(dists) == 0:
            continue
        sorted_d = np.sort(dists)
        cdf = np.arange(1, sorted_d.size + 1) / sorted_d.size
        ax.plot(sorted_d, cdf, label=f"{label} (n={sorted_d.size})")
    ax.set_xlabel("|YZ| position error (pixels)")
    ax.set_ylabel("CDF")
    ax.set_xscale("log")
    ax.set_xlim(1e-3, 10)
    ax.grid(alpha=0.3)
    ax.legend()
    ax.set_title("Peak-fit position recovery — CDF")
    fig.tight_layout()
    out_path = work_dir / "comparison_cdf.png"
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


# ─── Driver ────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-grains", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workdir", default="/scratch/s1iduser/peakfit_bench")
    ap.add_argument("--n-cpus", type=int, default=64)
    ap.add_argument("--gpu-id", type=int, default=0)
    ap.add_argument("--n-gpus", type=int, default=1,
                    help="Number of GPUs for the GPU run (each handles 1 block, run in parallel)")
    ap.add_argument("--skip-sim", action="store_true")
    ap.add_argument("--skip-c", action="store_true")
    ap.add_argument("--skip-cpu", action="store_true")
    ap.add_argument("--skip-gpu", action="store_true")
    ap.add_argument("--dtype", choices=["float64", "float32"], default="float64")
    ap.add_argument("--interleave-blocks", action="store_true",
                    help="Multi-GPU: stripe frames across blocks for better load balance")
    args = ap.parse_args()

    work_dir = Path(args.workdir)
    work_dir.mkdir(parents=True, exist_ok=True)

    src_params = MIDAS_HOME / "FF_HEDM" / "Example" / "Parameters.txt"
    final_zip = work_dir / "Au_FF_2k.analysis.MIDAS.zip"

    # Stage 1+2: gen + sim
    if args.skip_sim and final_zip.exists():
        print(f"[sim] Skipping (using existing {final_zip})")
        param_file = work_dir / "Parameters.txt"
    else:
        # Need hkls.csv next to Parameters.txt for ring radii
        # Copy from Example
        for fn in ("hkls.csv",):
            src = MIDAS_HOME / "FF_HEDM" / "Example" / fn
            if src.exists():
                shutil.copy2(src, work_dir / fn)
        param_file = write_parameters_with_n_grains(
            src_params, work_dir, args.n_grains, args.seed
        )
        final_zip = run_forward_sim(param_file, work_dir, args.n_cpus)
        params_dict = parse_param_file(param_file)
        enrich_zarr(final_zip, params_dict)

    # Always re-parse for omega lookup
    params_dict = parse_param_file(work_dir / "Parameters.txt")

    # Stage 4: run peakfit variants
    ps_paths: dict[str, Path] = {}
    timings: dict[str, float] = {}

    if not args.skip_c:
        t0 = time.time()
        ps_paths["C"] = run_c_peakfit(final_zip, work_dir, args.n_cpus)
        timings["C"] = time.time() - t0
    else:
        ps_paths["C"] = work_dir / "out_c" / "Temp" / "AllPeaks_PS.bin"

    if not args.skip_gpu:
        gpu_label = f"Py-GPU" if args.n_gpus == 1 else f"Py-{args.n_gpus}GPU"
        t0 = time.time()
        ps_paths[gpu_label] = run_py_peakfit(
            final_zip, work_dir, args.n_cpus,
            device="cuda", dtype=args.dtype, gpu_id=args.gpu_id,
            n_gpus=args.n_gpus,
            interleave_blocks=args.interleave_blocks,
        )
        timings[gpu_label] = time.time() - t0
    else:
        gpu_label = f"Py-GPU" if args.n_gpus == 1 else f"Py-{args.n_gpus}GPU"
        suf = f"py_cuda_{args.dtype}" if args.n_gpus == 1 else f"py_cuda_{args.dtype}_x{args.n_gpus}gpu"
        ps_paths[gpu_label] = work_dir / f"out_{suf}" / "Temp" / "AllPeaks_PS.bin"

    if not args.skip_cpu:
        t0 = time.time()
        ps_paths["Py-CPU"] = run_py_peakfit(
            final_zip, work_dir, args.n_cpus, device="cpu", dtype=args.dtype,
        )
        timings["Py-CPU"] = time.time() - t0
    else:
        ps_paths["Py-CPU"] = work_dir / f"out_py_cpu_{args.dtype}" / "Temp" / "AllPeaks_PS.bin"

    # Stage 5: ground truth
    print("\n[gt]  Loading SpotMatrixGen.csv …")
    gt = load_ground_truth(work_dir)
    print(f"[gt]  {gt.size} ground-truth spots")

    # Stage 6: load each PS.bin and compare
    fits = {}
    for label, path in ps_paths.items():
        if not path.exists():
            print(f"[WARN] {label}: {path} not found, skipping")
            continue
        peaks = load_ps_peaks(path, params_dict)
        print(f"[load] {label}: {peaks.size} fitted peaks")
        fits[label] = peaks

    print("\n" + "=" * 70)
    print("  TIMINGS")
    print("=" * 70)
    for label, t in timings.items():
        print(f"  {label:8s}  {t:8.1f} s")

    print("\n" + "=" * 70)
    print("  POSITION RECOVERY: candidate vs Ground Truth (SpotMatrixGen.csv)")
    print("=" * 70)
    cdf_lists: dict[str, np.ndarray] = {}
    for label, peaks in fits.items():
        gt_m, p_m = match_peaks(gt, peaks, ome_tol=0.5, yz_tol=5.0)
        s = compute_stats(f"GT vs {label}", gt_m, p_m)
        print(fmt_stats(s))
        if "dist_px" in s:
            dy = p_m["y"] - gt_m["y"]
            dz = p_m["z"] - gt_m["z"]
            cdf_lists[f"GT vs {label}"] = np.sqrt(dy * dy + dz * dz)

    if "C" in fits:
        print("\n" + "=" * 70)
        print("  PARITY: Python vs C (C is oracle)")
        print("=" * 70)
        for label in ["Py-GPU", "Py-CPU"]:
            if label not in fits:
                continue
            c_m, p_m = match_peaks(fits["C"], fits[label], ome_tol=0.5, yz_tol=5.0)
            s = compute_stats(f"C vs {label}", c_m, p_m)
            print(fmt_stats(s))
            if c_m.size > 0 and p_m.size > 0:
                dy = p_m["y"] - c_m["y"]
                dz = p_m["z"] - c_m["z"]
                cdf_lists[f"C vs {label}"] = np.sqrt(dy * dy + dz * dz)

    # CDF plot
    if cdf_lists:
        out_png = make_cdf_plot(cdf_lists, work_dir)
        print(f"\n[plot] CDF saved to {out_png}")

    # Persist a markdown report
    report = work_dir / "BENCHMARK_REPORT.md"
    with open(report, "w") as f:
        f.write(f"# Benchmark report — {args.n_grains} grains\n\n")
        f.write("## Timings\n\n")
        for label, t in timings.items():
            f.write(f"- `{label}`: {t:.1f} s\n")
        f.write("\n## Stats (matched peaks)\n\n```\n")
        for label, peaks in fits.items():
            gt_m, p_m = match_peaks(gt, peaks)
            f.write(fmt_stats(compute_stats(f"GT vs {label}", gt_m, p_m)) + "\n\n")
        if "C" in fits:
            for label in ["Py-GPU", "Py-CPU"]:
                if label not in fits:
                    continue
                c_m, p_m = match_peaks(fits["C"], fits[label])
                f.write(fmt_stats(compute_stats(f"C vs {label}", c_m, p_m)) + "\n\n")
        f.write("```\n")
    print(f"[report] {report}")


if __name__ == "__main__":
    main()
