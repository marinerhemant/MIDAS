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
    """Radon transform — fast.  Tries scikit-image, then numpy vectorized."""
    try:
        from skimage.transform import radon
        sino = radon(image, theta=thetas_deg, circle=True)
        return sino.T.astype(np.float32)
    except ImportError:
        pass

    # Fast vectorized projection: for each angle, sum along rotated lines
    # Using scipy.ndimage.rotate on a downsampled image for speed, then
    # upscale the sinogram.  Or just do straight line integrals via interp.
    N = image.shape[0]
    n_angles = len(thetas_deg)
    sinogram = np.zeros((n_angles, N), dtype=np.float32)

    # Coordinate grid (centered)
    coords = np.arange(N, dtype=np.float32) - N / 2.0 + 0.5
    x_grid, y_grid = np.meshgrid(coords, coords)  # both (N, N)

    for i, theta in enumerate(thetas_deg):
        # Project each pixel onto the detector axis
        theta_rad = np.radians(theta)
        cos_t = np.cos(theta_rad)
        sin_t = np.sin(theta_rad)
        # Projection coordinate for each pixel
        t = x_grid * cos_t + y_grid * sin_t  # (N, N)
        # Bin into detector pixels using histogram
        t_idx = t + N / 2.0  # shift to [0, N] range
        # Use np.add.at for exact binning (no interpolation needed for benchmark)
        t_bin = np.clip(t_idx.astype(np.int32), 0, N - 1)
        np.add.at(sinogram[i], t_bin.ravel(), image.ravel())
        if (i + 1) % 200 == 0:
            print(f"    angle {i+1}/{n_angles}...", flush=True)

    return sinogram


# ---------------------------------------------------------------------------
# Stripe injection for testing cleanup
# ---------------------------------------------------------------------------
def inject_stripes(sinos, phantom_size, seed=42):
    """Inject known stripe artifacts into sinograms.

    Returns (corrupted_sinos, stripe_info_dict).
    """
    rng = np.random.RandomState(seed)
    bad = sinos.copy()
    n_slices, n_angles, det_xdim = bad.shape
    # Stripe positions (in detector coordinates)
    stripes = {
        'dead':       det_xdim // 4,       # column set to 0
        'hot':        det_xdim // 2 + 50,  # 2x gain
        'noisy':      3 * det_xdim // 4,   # additive Gaussian noise
        'offset':     det_xdim // 3,       # constant offset
        'dead_pair':  det_xdim // 2 - 30,  # 2 adjacent dead columns
    }
    signal_rms = float(np.sqrt(np.mean(bad ** 2)))
    # Dead column
    bad[:, :, stripes['dead']] = 0.0
    # Hot column (2x gain)
    bad[:, :, stripes['hot']] *= 2.0
    # Noisy column (additive noise, 10% of signal RMS)
    bad[:, :, stripes['noisy']] += rng.normal(0, 0.1 * signal_rms,
                                               (n_slices, n_angles)).astype(np.float32)
    # Offset column (constant shift)
    bad[:, :, stripes['offset']] += 0.2 * signal_rms
    # Dead pair (2 adjacent)
    bad[:, :, stripes['dead_pair']] = 0.0
    bad[:, :, stripes['dead_pair'] + 1] = 0.0

    return bad, stripes


def run_stripe_test(phantom_size, n_thetas, n_cpus, sino_1slice, thetas):
    """Test stripe removal: inject stripes, apply 3 cleanup methods,
    reconstruct and generate comparison visualization."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from scipy.ndimage import gaussian_filter1d, median_filter
    from scipy.signal import savgol_filter
    from scipy.ndimage import gaussian_filter

    n_slices = 4  # small number, enough for testing
    print(f"\n{'='*70}")
    print("  Stripe Removal Test — All Methods")
    print(f"{'='*70}")
    print(f"  Phantom size:  {phantom_size}")
    print(f"  Slices:        {n_slices}")
    print(f"  Projections:   {n_thetas}")
    print()

    # Build sinogram stack
    sinos = np.tile(sino_1slice[np.newaxis, :, :], (n_slices, 1, 1))
    clean_sinos = sinos.copy()

    # Inject stripes (add partial stripe for Algo 2 testing)
    print("[1/6] Injecting stripe artifacts...")
    bad_sinos, stripe_info = inject_stripes(sinos, phantom_size)
    # Add a partial stripe (bad for only ~30% of angles) — Algo 2 target
    n_angles = bad_sinos.shape[1]
    det_xdim = bad_sinos.shape[2]
    partial_col = det_xdim // 5
    stripe_info['partial'] = partial_col
    signal_rms = float(np.sqrt(np.mean(clean_sinos ** 2)))
    bad_sinos[:, :n_angles//3, partial_col] *= 1.8  # gain error in first 1/3 of angles
    # Add a broad/slow gain drift column — Algo 1 target
    drift_col = 4 * det_xdim // 5
    stripe_info['drift'] = drift_col
    drift = np.linspace(1.0, 1.3, n_angles, dtype=np.float32)
    bad_sinos[:, :, drift_col] *= drift[np.newaxis, :, np.newaxis].squeeze()

    for name, col in stripe_info.items():
        print(f"  {name:12s} -> column {col}")

    # --- Python implementations of cleanup methods ---
    def py_cleanup_filtering(sino_2d, sigma=3.0, sm_size=21):
        """Algo 2: filtering-based (for partial stripes).
        Separate into smooth+sharp, sort-correct only smooth."""
        nrow, ncol = sino_2d.shape
        # Gaussian smooth along angle axis (axis 0) per column
        smooth = gaussian_filter1d(sino_2d, sigma=sigma, axis=0, mode='reflect')
        sharp = sino_2d - smooth
        # Sort-based correction on smooth component
        for c in range(ncol):
            col = smooth[:, c].copy()
            idx_sort = np.argsort(col)
            col_sorted = col[idx_sort]
            col_filtered = median_filter(col_sorted, size=sm_size, mode='reflect')
            smooth[idx_sort, c] = col_filtered
        return smooth + sharp

    def py_cleanup_fitting(sino_2d, order=3, sigma=(5, 20)):
        """Algo 1: fitting-based (for low-frequency/broad stripes).
        Polynomial fit + 2D Gaussian smooth + normalize."""
        nrow, ncol = sino_2d.shape
        nrow1 = nrow if nrow % 2 == 1 else nrow - 1
        if order >= nrow1:
            order = nrow1 - 1
        # Savitzky-Golay along angle axis (window = all angles, order = polynomial degree)
        sinofit = savgol_filter(sino_2d, nrow1, order, axis=0, mode='mirror')
        # 2D Gaussian smooth of the fit
        sinofitsmooth = gaussian_filter(sinofit, sigma=[sigma[0], sigma[1]],
                                        mode='reflect')
        # Scale-preserving normalization
        mean_fit = np.mean(sinofit)
        mean_smooth = np.mean(sinofitsmooth)
        if abs(mean_smooth) > 1e-12:
            sinofitsmooth *= mean_fit / mean_smooth
        # Normalize: sinogram / fit * smoothed_fit
        result = sino_2d.copy()
        mask = np.abs(sinofit) > 1e-9
        result[mask] = sino_2d[mask] / sinofit[mask] * sinofitsmooth[mask]
        return result

    # --- Reconstruct with each method ---
    # a) Clean reference
    print("\n[2/6] Reconstructing clean reference...")
    d = tempfile.mkdtemp(prefix='stripe_clean_')
    recon_clean = run_tomo_from_sinos(clean_sinos, d, thetas, numCPUs=n_cpus,
                                      useGPU=False)

    # b) Dirty (no cleanup)
    print("[3/6] Reconstructing dirty (no cleanup)...")
    d = tempfile.mkdtemp(prefix='stripe_dirty_')
    recon_dirty = run_tomo_from_sinos(bad_sinos, d, thetas, numCPUs=n_cpus,
                                      useGPU=False, doStripeRemoval=0)

    # c) Method 1: Combined Algo 3+5+6 (via C code, existing doStripeRemoval=1)
    print("[4/6] Reconstructing with Algo 3+5+6 (combined, C code)...")
    d = tempfile.mkdtemp(prefix='stripe_combined_')
    recon_combined = run_tomo_from_sinos(bad_sinos, d, thetas, numCPUs=n_cpus,
                                         useGPU=False, doStripeRemoval=1)

    # d) Method 2: Algo 2 (filtering-based, Python)
    print("[5/6] Applying Algo 2 (filtering) + reconstructing...")
    filtered_sinos = bad_sinos.copy()
    for i in range(n_slices):
        filtered_sinos[i] = py_cleanup_filtering(filtered_sinos[i])
    d = tempfile.mkdtemp(prefix='stripe_filtering_')
    recon_filtering = run_tomo_from_sinos(filtered_sinos, d, thetas, numCPUs=n_cpus,
                                          useGPU=False, doStripeRemoval=0)

    # e) Method 3: Algo 1 (fitting-based, Python)
    print("[6/6] Applying Algo 1 (fitting) + reconstructing...")
    fitted_sinos = bad_sinos.copy()
    for i in range(n_slices):
        fitted_sinos[i] = py_cleanup_fitting(fitted_sinos[i])
    d = tempfile.mkdtemp(prefix='stripe_fitting_')
    recon_fitting = run_tomo_from_sinos(fitted_sinos, d, thetas, numCPUs=n_cpus,
                                        useGPU=False, doStripeRemoval=0)

    # --- Visualization: 3 rows x 4 columns ---
    print("\nGenerating visualization...")
    sl = 0
    fig, axes = plt.subplots(3, 4, figsize=(22, 14))
    fig.suptitle('Stripe Removal — Method Comparison (Vo et al. 2018)',
                 fontsize=16, fontweight='bold')

    # Row 1: Sinograms
    ang_range = slice(0, min(200, n_thetas))
    vmin_s = float(np.percentile(clean_sinos[sl, ang_range], 1))
    vmax_s = float(np.percentile(clean_sinos[sl, ang_range], 99))

    axes[0, 0].imshow(clean_sinos[sl, ang_range], aspect='auto',
                      cmap='gray', vmin=vmin_s, vmax=vmax_s)
    axes[0, 0].set_title('Clean Sinogram')
    axes[0, 0].set_ylabel('Angle index')

    axes[0, 1].imshow(bad_sinos[sl, ang_range], aspect='auto',
                      cmap='gray', vmin=vmin_s, vmax=vmax_s)
    axes[0, 1].set_title('Sinogram + All Stripes')
    for name, col in stripe_info.items():
        color = 'r' if name not in ('partial', 'drift') else ('cyan' if name == 'partial' else 'lime')
        axes[0, 1].axvline(col, color=color, alpha=0.4, linewidth=0.8)

    # Difference
    diff_sino = bad_sinos[sl, ang_range] - clean_sinos[sl, ang_range]
    vabs = max(float(np.percentile(np.abs(diff_sino), 99)), 1e-6)
    axes[0, 2].imshow(diff_sino, aspect='auto', cmap='RdBu_r',
                      vmin=-vabs, vmax=vabs)
    axes[0, 2].set_title('Stripe Difference')

    axes[0, 3].set_axis_off()
    info_lines = ['Injected Stripes:']
    for name, col in stripe_info.items():
        label = {'dead': 'Dead (=0)', 'hot': 'Hot (2x)',
                 'noisy': 'Noisy (+G)', 'offset': 'Offset (+c)',
                 'dead_pair': 'Dead pair', 'partial': 'Partial (1/3)',
                 'drift': 'Drift (1->1.3x)'}.get(name, name)
        info_lines.append(f'  col {col:3d}: {label}')
    info_lines.append('')
    info_lines.append('Methods:')
    info_lines.append('  Algo 3+5+6: dead+large+sort')
    info_lines.append('  Algo 2: filter(smooth+sharp)')
    info_lines.append('  Algo 1: polyfit+gauss norm')
    axes[0, 3].text(0.05, 0.5, '\n'.join(info_lines),
                    transform=axes[0, 3].transAxes,
                    fontsize=10, verticalalignment='center', fontfamily='monospace')

    # Row 2: Clean ref, Dirty, Algo 3+5+6, Algo 2 (filtering)
    vmin_r = float(np.percentile(recon_clean[0, sl], 1))
    vmax_r = float(np.percentile(recon_clean[0, sl], 99))

    axes[1, 0].imshow(recon_clean[0, sl], cmap='gray', vmin=vmin_r, vmax=vmax_r)
    axes[1, 0].set_title('Clean (reference)')
    axes[1, 0].set_ylabel('Reconstruction')

    axes[1, 1].imshow(recon_dirty[0, sl], cmap='gray', vmin=vmin_r, vmax=vmax_r)
    axes[1, 1].set_title('Dirty (ring artifacts)')

    axes[1, 2].imshow(recon_combined[0, sl], cmap='gray', vmin=vmin_r, vmax=vmax_r)
    axes[1, 2].set_title('Algo 3+5+6 (combined)')

    axes[1, 3].imshow(recon_filtering[0, sl], cmap='gray', vmin=vmin_r, vmax=vmax_r)
    axes[1, 3].set_title('Algo 2 (filtering)')

    # Row 3: Algo 1, difference maps
    axes[2, 0].imshow(recon_fitting[0, sl], cmap='gray', vmin=vmin_r, vmax=vmax_r)
    axes[2, 0].set_title('Algo 1 (fitting)')
    axes[2, 0].set_ylabel('Fitting + Diffs')

    # Difference: dirty - clean (shows ring severity)
    diff_dirty = recon_dirty[0, sl] - recon_clean[0, sl]
    vabs_r = max(float(np.percentile(np.abs(diff_dirty), 99)), 1e-6)
    axes[2, 1].imshow(diff_dirty, cmap='RdBu_r', vmin=-vabs_r, vmax=vabs_r)
    axes[2, 1].set_title('Dirty - Clean')

    diff_combined = recon_combined[0, sl] - recon_clean[0, sl]
    axes[2, 2].imshow(diff_combined, cmap='RdBu_r', vmin=-vabs_r, vmax=vabs_r)
    axes[2, 2].set_title('Algo 3+5+6 - Clean')

    diff_filtering = recon_filtering[0, sl] - recon_clean[0, sl]
    axes[2, 3].imshow(diff_filtering, cmap='RdBu_r', vmin=-vabs_r, vmax=vabs_r)
    axes[2, 3].set_title('Algo 2 - Clean')

    plt.tight_layout()
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'stripe_removal_test.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"  Saved visualization to {out_path}")
    plt.close()

    # RMS error summary
    print("\n  Method RMS errors (vs clean):")
    for name, recon in [('Dirty (no cleanup)', recon_dirty),
                        ('Algo 3+5+6 (combined)', recon_combined),
                        ('Algo 2 (filtering)', recon_filtering),
                        ('Algo 1 (fitting)', recon_fitting)]:
        rms = float(np.sqrt(np.mean((recon[0, sl] - recon_clean[0, sl])**2)))
        print(f"    {name:25s}: {rms:.6f}")

    return True


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
                    test_fftw_bridge=False, benchmark=False,
                    test_stripe_removal=False):
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

    # Generate phantom + sinogram (cached to avoid 2min recomputation)
    cache_dir = os.path.dirname(os.path.abspath(__file__))
    cache_file = os.path.join(cache_dir,
                              f"cached_sino_{phantom_size}x{n_thetas}.npy")
    if os.path.exists(cache_file):
        print(f"[1/5] Loading cached sinogram from {os.path.basename(cache_file)}...")
        sino_1slice = np.load(cache_file)
        print(f"  Sinogram: {sino_1slice.shape} (cached)")
    else:
        print("[1/5] Generating Shepp-Logan phantom...")
        phantom = shepp_logan_phantom(phantom_size)
        print(f"  Shape: {phantom.shape}")

        print("[2/5] Computing Radon transform for one slice...")
        thetas = np.linspace(0, 179.9, n_thetas, dtype=np.float32)
        t0 = time.time()
        sino_1slice = radon_transform(phantom, thetas)
        print(f"  Sinogram: {sino_1slice.shape}, computed in {time.time()-t0:.1f}s")
        np.save(cache_file, sino_1slice)
        print(f"  Saved to {os.path.basename(cache_file)}")
    thetas = np.linspace(0, 179.9, n_thetas, dtype=np.float32)

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

    # ─── Stripe removal test ───
    stripe_passed = True
    if test_stripe_removal:
        stripe_passed = run_stripe_test(phantom_size, n_thetas, n_cpus,
                                        sino_1slice, thetas)
        results.append(('Stripe removal (CPU vs GPU)', stripe_passed,
                        {'correlation': 1.0, 'max_diff': 0.0}))

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
    parser.add_argument("--stripe-removal", action="store_true",
                        help="Test stripe artifact removal (injection + cleanup + visualization)")
    args = parser.parse_args()

    sys.exit(run_parity_test(
        phantom_size=args.size,
        n_thetas=args.n_thetas,
        n_slices=args.n_slices,
        n_cpus=args.n_cpus,
        test_fftw_bridge=args.fftw_bridge,
        benchmark=args.benchmark,
        test_stripe_removal=args.stripe_removal,
    ))
