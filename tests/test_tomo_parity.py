#!/usr/bin/env python3
"""
MIDAS Tomography GPU Parity & Benchmark Test

Validates GPU reconstruction against CPU reconstruction and benchmarks
performance at production scale.

Usage:
    # Quick parity check (256px, 2 slices)
    python test_tomo_parity.py

    # Large-scale benchmark: 1000 slices, 2048px, 96 CPUs vs GPU
    python test_tomo_parity.py --size 2048 --n-slices 1000 --n-cpus 96 --benchmark

    # FFTW-bridge mode (byte-exact parity)
    python test_tomo_parity.py --fftw-bridge

    # Custom configuration
    python test_tomo_parity.py --size 512 --n-slices 100 --n-cpus 48 --benchmark
"""

import argparse
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Locate MIDAS
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
MIDAS_HOME = SCRIPT_DIR.parent

sys.path.insert(0, str(MIDAS_HOME / "TOMO"))
from midas_tomo_python import run_tomo_from_sinos


# ---------------------------------------------------------------------------
# Phantom & Sinogram generation
# ---------------------------------------------------------------------------
def shepp_logan_phantom(size: int = 256) -> np.ndarray:
    """Generate a Shepp-Logan phantom.  Returns float32 in [0, 1]."""
    ellipses = [
        ( 1.0,   0.6900, 0.9200,  0.0000,  0.0000,    0),
        (-0.8,   0.6624, 0.8740,  0.0000, -0.0184,    0),
        (-0.2,   0.1100, 0.3100,  0.2200,  0.0000,  -18),
        (-0.2,   0.1600, 0.4100, -0.2200,  0.0000,   18),
        ( 0.1,   0.2100, 0.2500,  0.0000,  0.3500,    0),
        ( 0.1,   0.0460, 0.0460,  0.0000,  0.1000,    0),
        ( 0.1,   0.0460, 0.0460,  0.0000, -0.1000,    0),
        ( 0.1,   0.0460, 0.0230, -0.0800, -0.6050,    0),
        ( 0.1,   0.0230, 0.0230,  0.0000, -0.6060,    0),
        ( 0.1,   0.0230, 0.0460,  0.0600, -0.6050,    0),
    ]
    img = np.zeros((size, size), dtype=np.float64)
    x = np.linspace(-1, 1, size)
    xg, yg = np.meshgrid(x, x)
    for intensity, a, b, x0, y0, theta_deg in ellipses:
        cos_t = np.cos(np.radians(theta_deg))
        sin_t = np.sin(np.radians(theta_deg))
        xr = cos_t * (xg - x0) + sin_t * (yg - y0)
        yr = -sin_t * (xg - x0) + cos_t * (yg - y0)
        mask = (xr / a) ** 2 + (yr / b) ** 2 <= 1.0
        img[mask] += intensity
    img = np.clip(img, 0, None)
    if img.max() > 0:
        img /= img.max()
    return img.astype(np.float32)


def radon_transform(image: np.ndarray, thetas_deg: np.ndarray) -> np.ndarray:
    """Radon transform via scikit-image or scipy fallback."""
    try:
        from skimage.transform import radon
        sino = radon(image, theta=thetas_deg, circle=True)
        return sino.T.astype(np.float32)
    except ImportError:
        from scipy.ndimage import rotate
        N = image.shape[0]
        sinogram = np.zeros((len(thetas_deg), N), dtype=np.float32)
        for i, theta in enumerate(thetas_deg):
            rotated = rotate(image, theta, reshape=False, order=1)
            sinogram[i, :] = np.sum(rotated, axis=0)
        return sinogram


# ---------------------------------------------------------------------------
# Parity comparison
# ---------------------------------------------------------------------------
def compare_recons(cpu_recon, gpu_recon, label=""):
    """Compare CPU and GPU reconstructions, return (pass, stats_dict)."""
    if cpu_recon.shape != gpu_recon.shape:
        print(f"  [FAIL ❌]  {label}")
        print(f"    Shape mismatch: CPU {cpu_recon.shape} vs GPU {gpu_recon.shape}")
        return False, {}

    cpu_flat = cpu_recon.ravel().astype(np.float64)
    gpu_flat = gpu_recon.ravel().astype(np.float64)

    diff = np.abs(cpu_flat - gpu_flat)
    max_diff = float(diff.max())
    mean_diff = float(diff.mean())
    rms_diff = float(np.sqrt(np.mean(diff ** 2)))

    # Correlation
    a = cpu_flat - cpu_flat.mean()
    b = gpu_flat - gpu_flat.mean()
    denom = np.sqrt(np.dot(a, a) * np.dot(b, b))
    corr = float(np.dot(a, b) / denom) if denom > 0 else 0.0

    # Float-exact count
    n_exact = int(np.sum(cpu_recon.ravel() == gpu_recon.ravel()))
    n_total = cpu_recon.size
    exact_pct = 100.0 * n_exact / n_total

    stats = {
        'max_diff': max_diff,
        'mean_diff': mean_diff,
        'rms_diff': rms_diff,
        'correlation': corr,
        'exact_pixel_pct': exact_pct,
    }

    # Thresholds
    ok_corr = corr > 0.999
    ok_diff = max_diff < 1e-3

    passed = ok_corr and ok_diff
    status = "PASS ✅" if passed else "FAIL ❌"

    print(f"\n  [{status}]  {label}")
    print(f"    Correlation:     {corr:.8f}  {'PASS' if ok_corr else 'FAIL'}")
    print(f"    Max diff:        {max_diff:.6e}  {'PASS' if ok_diff else 'FAIL'}")
    print(f"    Mean diff:       {mean_diff:.6e}")
    print(f"    RMS diff:        {rms_diff:.6e}")
    print(f"    Exact pixels:    {n_exact}/{n_total} ({exact_pct:.1f}%)")

    return passed, stats


# ---------------------------------------------------------------------------
# Main test
# ---------------------------------------------------------------------------
def run_parity_test(phantom_size=256, n_thetas=1800, n_slices=2, n_cpus=1,
                    test_fftw_bridge=False, benchmark=False):
    # Header
    data_size_gb = (n_slices * n_thetas * phantom_size * 4) / (1024**3)
    print("=" * 70)
    print("  MIDAS Tomography GPU Parity & Benchmark Test")
    print("=" * 70)
    print(f"  Phantom size:  {phantom_size} x {phantom_size}")
    print(f"  Projections:   {n_thetas}")
    print(f"  Slices:        {n_slices}")
    print(f"  CPUs:          {n_cpus}")
    print(f"  Data size:     {data_size_gb:.2f} GB (sinograms)")
    print()

    # Generate phantom = single slice sinogram
    print("[1/5] Generating Shepp-Logan phantom...")
    phantom = shepp_logan_phantom(phantom_size)
    print(f"  Shape: {phantom.shape}")

    print("[2/5] Computing Radon transform for one slice...")
    thetas = np.linspace(0, 179.9, n_thetas, dtype=np.float32)
    t0 = time.time()
    sino_1slice = radon_transform(phantom, thetas)
    print(f"  Sinogram: {sino_1slice.shape}, computed in {time.time()-t0:.1f}s")

    # Replicate into n_slices
    print(f"[3/5] Replicating to {n_slices} slices...")
    t0 = time.time()
    sinos = np.tile(sino_1slice[np.newaxis, :, :], (n_slices, 1, 1))
    print(f"  Sinograms array: {sinos.shape}, "
          f"{sinos.nbytes / (1024**3):.2f} GB, "
          f"replicated in {time.time()-t0:.1f}s")

    results = []

    # ─── CPU reconstruction ───
    print(f"\n[4/5] CPU reconstruction ({n_cpus} threads)...")
    cpu_dir = tempfile.mkdtemp(prefix='tomo_parity_cpu_')
    t_cpu_start = time.time()
    recon_cpu = run_tomo_from_sinos(sinos, cpu_dir, thetas, numCPUs=n_cpus,
                                   useGPU=False)
    t_cpu = time.time() - t_cpu_start
    print(f"  CPU total time:  {t_cpu:.3f}s")
    print(f"  CPU throughput:  {n_slices/t_cpu:.1f} slices/s")
    print(f"  Recon shape:     {recon_cpu.shape}")

    # ─── GPU reconstruction (cuFFT) ───
    print(f"\n[5/5] GPU reconstruction (cuFFT)...")
    gpu_dir = tempfile.mkdtemp(prefix='tomo_parity_gpu_')
    t_gpu_start = time.time()
    recon_gpu = run_tomo_from_sinos(sinos, gpu_dir, thetas, numCPUs=1,
                                   useGPU=True)
    t_gpu = time.time() - t_gpu_start
    print(f"  GPU total time:  {t_gpu:.3f}s")
    print(f"  GPU throughput:  {n_slices/t_gpu:.1f} slices/s")
    print(f"  Recon shape:     {recon_gpu.shape}")

    # Compare first slice for parity
    passed, stats = compare_recons(recon_cpu[0, 0], recon_gpu[0, 0],
                                   "Slice 0: CPU vs GPU (cuFFT)")
    stats['cpu_time'] = t_cpu
    stats['gpu_time'] = t_gpu
    stats['speedup'] = t_cpu / t_gpu if t_gpu > 0 else 0
    results.append(('cuFFT (slice 0)', passed, stats))

    # Also compare a middle and last slice if we have multiple
    if n_slices > 2:
        mid = n_slices // 2
        p2, s2 = compare_recons(recon_cpu[0, mid], recon_gpu[0, mid],
                                f"Slice {mid}: CPU vs GPU (cuFFT)")
        results.append((f'cuFFT (slice {mid})', p2, s2))

        p3, s3 = compare_recons(recon_cpu[0, -1], recon_gpu[0, -1],
                                f"Slice {n_slices-1}: CPU vs GPU (cuFFT)")
        results.append((f'cuFFT (slice {n_slices-1})', p3, s3))

    # ─── FFTW-bridge mode ───
    if test_fftw_bridge:
        print(f"\n[5b] GPU reconstruction (FFTW-bridge)...")
        bridge_dir = tempfile.mkdtemp(prefix='tomo_parity_bridge_')
        t_bridge_start = time.time()
        recon_bridge = run_tomo_from_sinos(sinos, bridge_dir, thetas,
                                          numCPUs=1, useGPU=True,
                                          fftwBridge=True)
        t_bridge = time.time() - t_bridge_start
        print(f"  Bridge time: {t_bridge:.3f}s")

        pb, sb = compare_recons(recon_cpu[0, 0], recon_bridge[0, 0],
                                "Slice 0: CPU vs GPU (FFTW-bridge)")
        sb['bridge_time'] = t_bridge
        results.append(('FFTW-bridge', pb, sb))

    # ─── Repeated benchmark ───
    if benchmark:
        n_runs = 3
        print(f"\n[BENCH] Repeated benchmark ({n_runs} runs)...")
        cpu_times = []
        gpu_times = []
        for i in range(n_runs):
            print(f"  Run {i+1}/{n_runs}...", end=" ", flush=True)
            d = tempfile.mkdtemp(prefix=f'tomo_bench_cpu_{i}_')
            t0 = time.time()
            run_tomo_from_sinos(sinos, d, thetas, numCPUs=n_cpus,
                                useGPU=False)
            ct = time.time() - t0
            cpu_times.append(ct)

            d = tempfile.mkdtemp(prefix=f'tomo_bench_gpu_{i}_')
            t0 = time.time()
            run_tomo_from_sinos(sinos, d, thetas, numCPUs=1, useGPU=True)
            gt = time.time() - t0
            gpu_times.append(gt)
            print(f"CPU={ct:.2f}s, GPU={gt:.2f}s, "
                  f"speedup={ct/gt:.2f}x")

        avg_cpu = np.mean(cpu_times)
        avg_gpu = np.mean(gpu_times)
        speedup = avg_cpu / avg_gpu if avg_gpu > 0 else 0

        print(f"\n  ┌──────────────────────────────────────────────────┐")
        print(f"  │  Benchmark: {n_slices} slices × {phantom_size}px × "
              f"{n_thetas} angles{' ' * max(0, 10 - len(str(n_slices)))}│")
        print(f"  ├──────────────────────────────────────────────────┤")
        print(f"  │  {'':8s}  {'CPU ('+str(n_cpus)+'t)':>12s}  "
              f"{'GPU':>10s}  {'Speedup':>8s}  │")
        print(f"  │  {'Mean':8s}  {avg_cpu:12.2f}s  {avg_gpu:10.2f}s  "
              f"{speedup:7.2f}x  │")
        print(f"  │  {'Min':8s}  {min(cpu_times):12.2f}s  "
              f"{min(gpu_times):10.2f}s  "
              f"{max(cpu_times)/min(gpu_times):7.2f}x  │")
        print(f"  │  {'Thruput':8s}  {n_slices/avg_cpu:10.1f}/s  "
              f"{n_slices/avg_gpu:8.1f}/s  {'':>8s}  │")
        print(f"  └──────────────────────────────────────────────────┘")

    # ─── Summary ───
    print("\n" + "=" * 70)
    all_passed = all(r[1] for r in results)
    n_pass = sum(1 for r in results if r[1])
    n_total = len(results)

    for label, passed, stats in results:
        status = "✅" if passed else "❌"
        extra = ""
        if 'speedup' in stats:
            extra = (f"  [CPU {stats['cpu_time']:.2f}s vs "
                     f"GPU {stats['gpu_time']:.2f}s = "
                     f"{stats['speedup']:.2f}x]")
        print(f"  {status} {label}: corr={stats['correlation']:.8f}, "
              f"max_diff={stats['max_diff']:.2e}{extra}")

    print()
    if all_passed:
        print(f"  ✅ ALL PARITY CHECKS PASSED ({n_pass}/{n_total})")
    else:
        print(f"  ❌ PARITY CHECKS FAILED ({n_pass}/{n_total} passed)")
    print("=" * 70)

    return 0 if all_passed else 1


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="MIDAS Tomography GPU Parity & Benchmark Test")
    parser.add_argument("--size", type=int, default=256,
                        help="Phantom size in pixels (default: 256)")
    parser.add_argument("--n-thetas", type=int, default=1800,
                        help="Number of projection angles (default: 1800)")
    parser.add_argument("--n-slices", type=int, default=2,
                        help="Number of slices (default: 2)")
    parser.add_argument("--n-cpus", type=int, default=1,
                        help="Number of CPU threads (default: 1)")
    parser.add_argument("--fftw-bridge", action="store_true",
                        help="Also test FFTW-bridge mode")
    parser.add_argument("--benchmark", action="store_true",
                        help="Run repeated benchmark (3 runs)")
    args = parser.parse_args()

    sys.exit(run_parity_test(
        phantom_size=args.size,
        n_thetas=args.n_thetas,
        n_slices=args.n_slices,
        n_cpus=args.n_cpus,
        test_fftw_bridge=args.fftw_bridge,
        benchmark=args.benchmark,
    ))
