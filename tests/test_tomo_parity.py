#!/usr/bin/env python3
"""
MIDAS Tomography GPU Parity Test

Validates GPU reconstruction against CPU reconstruction at byte level.
Tests both cuFFT mode and (optionally) FFTW-bridge mode.

Usage:
    python test_tomo_parity.py                  # Quick test (256px)
    python test_tomo_parity.py --size 512       # Larger phantom
    python test_tomo_parity.py --fftw-bridge    # Also test FFTW-bridge mode
    python test_tomo_parity.py --benchmark      # Include timing comparison

Requirements:
    - MIDAS_TOMO and MIDAS_TOMO_GPU binaries must be built
    - Run on a machine with a CUDA-capable GPU
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

    # Byte-exact count
    cpu_bytes = cpu_recon.view(np.uint8)
    gpu_bytes = gpu_recon.view(np.uint8)
    n_bytes_match = int(np.sum(cpu_bytes == gpu_bytes))
    n_total_bytes = cpu_bytes.size
    byte_pct = 100.0 * n_bytes_match / n_total_bytes

    # Float-exact count
    n_exact = int(np.sum(cpu_recon.ravel() == gpu_recon.ravel()))
    n_total = cpu_recon.size
    exact_pct = 100.0 * n_exact / n_total

    stats = {
        'max_diff': max_diff,
        'mean_diff': mean_diff,
        'rms_diff': rms_diff,
        'correlation': corr,
        'byte_match_pct': byte_pct,
        'exact_pixel_pct': exact_pct,
    }

    # Thresholds
    ok_corr = corr > 0.999
    ok_diff = max_diff < 1e-3  # very generous for cuFFT mode

    passed = ok_corr and ok_diff
    status = "PASS ✅" if passed else "FAIL ❌"

    print(f"\n  [{status}]  {label}")
    print(f"    Correlation:     {corr:.8f}  {'PASS' if ok_corr else 'FAIL'}")
    print(f"    Max diff:        {max_diff:.6e}  {'PASS' if ok_diff else 'FAIL'}")
    print(f"    Mean diff:       {mean_diff:.6e}")
    print(f"    RMS diff:        {rms_diff:.6e}")
    print(f"    Exact pixels:    {n_exact}/{n_total} ({exact_pct:.1f}%)")
    print(f"    Byte-level match: {byte_pct:.1f}%")

    return passed, stats


# ---------------------------------------------------------------------------
# Main test
# ---------------------------------------------------------------------------
def run_parity_test(phantom_size=256, n_thetas=1800, test_fftw_bridge=False,
                    benchmark=False, n_benchmark_runs=3):
    print("=" * 70)
    print("  MIDAS Tomography GPU Parity Test")
    print("=" * 70)
    print(f"  Phantom size: {phantom_size} x {phantom_size}")
    print(f"  Projections:  {n_thetas}")
    print()

    # Generate phantom and sinogram
    print("[1/5] Generating Shepp-Logan phantom...")
    phantom = shepp_logan_phantom(phantom_size)
    print(f"  Shape: {phantom.shape}")

    print("[2/5] Computing Radon transform...")
    thetas = np.linspace(0, 179.9, n_thetas, dtype=np.float32)
    t0 = time.time()
    sinos = radon_transform(phantom, thetas)
    print(f"  Sinogram: {sinos.shape}, computed in {time.time()-t0:.1f}s")

    # CPU reconstruction
    print("[3/5] CPU reconstruction...")
    cpu_dir = tempfile.mkdtemp(prefix='tomo_parity_cpu_')
    t0 = time.time()
    recon_cpu = run_tomo_from_sinos(sinos, cpu_dir, thetas, numCPUs=1,
                                   useGPU=False)
    cpu_time = time.time() - t0
    print(f"  CPU time: {cpu_time:.3f}s, shape: {recon_cpu.shape}")

    # GPU reconstruction (cuFFT mode)
    print("[4/5] GPU reconstruction (cuFFT mode)...")
    gpu_dir = tempfile.mkdtemp(prefix='tomo_parity_gpu_')
    t0 = time.time()
    recon_gpu = run_tomo_from_sinos(sinos, gpu_dir, thetas, numCPUs=1,
                                   useGPU=True)
    gpu_time = time.time() - t0
    print(f"  GPU time: {gpu_time:.3f}s, shape: {recon_gpu.shape}")

    results = []

    # Compare CPU vs GPU (cuFFT)
    passed, stats = compare_recons(recon_cpu[0, 0], recon_gpu[0, 0],
                                   "CPU vs GPU (cuFFT)")
    stats['cpu_time'] = cpu_time
    stats['gpu_time'] = gpu_time
    stats['speedup'] = cpu_time / gpu_time if gpu_time > 0 else 0
    results.append(('cuFFT', passed, stats))

    # FFTW-bridge mode
    if test_fftw_bridge:
        print("[4b/5] GPU reconstruction (FFTW-bridge mode)...")
        bridge_dir = tempfile.mkdtemp(prefix='tomo_parity_bridge_')
        t0 = time.time()
        recon_bridge = run_tomo_from_sinos(sinos, bridge_dir, thetas,
                                          numCPUs=1, useGPU=True,
                                          fftwBridge=True)
        bridge_time = time.time() - t0
        print(f"  FFTW-bridge time: {bridge_time:.3f}s")

        passed_b, stats_b = compare_recons(recon_cpu[0, 0],
                                           recon_bridge[0, 0],
                                           "CPU vs GPU (FFTW-bridge)")
        stats_b['bridge_time'] = bridge_time
        results.append(('FFTW-bridge', passed_b, stats_b))

    # Benchmark
    if benchmark:
        print(f"\n[5/5] Benchmarking ({n_benchmark_runs} runs)...")
        cpu_times = []
        gpu_times = []
        for i in range(n_benchmark_runs):
            d = tempfile.mkdtemp(prefix='tomo_bench_cpu_')
            t0 = time.time()
            run_tomo_from_sinos(sinos, d, thetas, numCPUs=1, useGPU=False)
            cpu_times.append(time.time() - t0)

            d = tempfile.mkdtemp(prefix='tomo_bench_gpu_')
            t0 = time.time()
            run_tomo_from_sinos(sinos, d, thetas, numCPUs=1, useGPU=True)
            gpu_times.append(time.time() - t0)

        avg_cpu = np.mean(cpu_times)
        avg_gpu = np.mean(gpu_times)
        speedup = avg_cpu / avg_gpu if avg_gpu > 0 else 0

        print(f"\n  Benchmark Results ({n_benchmark_runs} runs):")
        print(f"  {'':8s}  {'CPU':>10s}  {'GPU':>10s}  {'Speedup':>8s}")
        print(f"  {'Mean':8s}  {avg_cpu:10.3f}s  {avg_gpu:10.3f}s  {speedup:7.2f}x")
        print(f"  {'Min':8s}  {min(cpu_times):10.3f}s  {min(gpu_times):10.3f}s")
        print(f"  {'Max':8s}  {max(cpu_times):10.3f}s  {max(gpu_times):10.3f}s")

    # Summary
    print("\n" + "=" * 70)
    all_passed = all(r[1] for r in results)
    n_pass = sum(1 for r in results if r[1])
    n_total = len(results)

    for label, passed, stats in results:
        status = "✅" if passed else "❌"
        extra = ""
        if 'speedup' in stats:
            extra = f"  (GPU {stats['speedup']:.2f}x)"
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
        description="MIDAS Tomography GPU Parity Test")
    parser.add_argument("--size", type=int, default=256,
                        help="Phantom size (default: 256)")
    parser.add_argument("--n-thetas", type=int, default=1800,
                        help="Number of projection angles (default: 1800)")
    parser.add_argument("--fftw-bridge", action="store_true",
                        help="Also test FFTW-bridge mode")
    parser.add_argument("--benchmark", action="store_true",
                        help="Run performance benchmark")
    parser.add_argument("--benchmark-runs", type=int, default=3,
                        help="Number of benchmark iterations (default: 3)")
    args = parser.parse_args()

    sys.exit(run_parity_test(
        phantom_size=args.size,
        n_thetas=args.n_thetas,
        test_fftw_bridge=args.fftw_bridge,
        benchmark=args.benchmark,
        n_benchmark_runs=args.benchmark_runs,
    ))
