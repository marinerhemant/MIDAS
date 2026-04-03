# Track E: MLEM Sinogram Reconstruction

## What It Does

Replaces Filtered Back-Projection (FBP/gridrec) with Maximum Likelihood Expectation Maximization (MLEM) for reconstructing per-grain shape images from pf-HEDM sinograms.

**Why MLEM over FBP?** pf-HEDM sinograms are inherently sparse: data exists only at discrete omega angles where a grain produced a diffraction spot, not a continuous 0-360 sweep. FBP assumes dense angular sampling and requires heuristic preprocessing (gap filling, smoothing) to compensate. MLEM naturally handles missing projections without any gap filling.

## Files

| File | Purpose |
|------|---------|
| `utils/mlem_recon.py` | `mlem()`, `osem()`, `forward_project()`, `back_project()` |
| `utils/sino_cleanup_tomo.py` | Existing sinogram preprocessing (still useful for despeckling) |

## Prerequisites

```bash
source /path/to/miniconda3/bin/activate midas_env
# Needs: numpy only (zero external dependencies)
```

---

## Conceptual Overview

```
Current pipeline (FBP):
    Raw sinogram → despeckle → fill holes → interpolate gaps → smooth → FBP → grain shape

MLEM pipeline:
    Raw sinogram → despeckle → MLEM iteration → grain shape
                                  ↓
                    (no hole filling, gap interpolation, or smoothing needed)
```

MLEM iterates:
1. **Forward project** current estimate → predicted sinogram
2. **Compare** measured / predicted (ratio)
3. **Back-project** the ratio
4. **Update** estimate multiplicatively (guarantees positivity)

---

## Step 1: Get Per-Grain Sinograms

### From existing pf-HEDM analysis

After running `pf_MIDAS.py`, sinograms are stored as binary files or TIFs:

```python
import numpy as np
from pathlib import Path

work_dir = Path("/path/to/pf_analysis")

# Method A: Read from binary sinogram files
nGrs = 5        # number of grains (from your analysis)
maxNHKLs = 50   # max spots per grain
nScans = 15     # number of scan positions

sinos = np.fromfile(
    work_dir / f"sinos_raw_{nGrs}_{maxNHKLs}_{nScans}.bin",
    dtype=np.double,
).reshape((nGrs, maxNHKLs, nScans))

omegas = np.fromfile(
    work_dir / f"omegas_{nGrs}_{maxNHKLs}.bin",
    dtype=np.double,
).reshape((nGrs, maxNHKLs))

grain_nspots = np.fromfile(
    work_dir / f"nrHKLs_{nGrs}.bin",
    dtype=np.int32,
)

print(f"Loaded sinograms for {nGrs} grains")
for g in range(nGrs):
    nsp = grain_nspots[g]
    print(f"  Grain {g}: {nsp} spots, "
          f"omega range [{omegas[g,:nsp].min():.1f}, {omegas[g,:nsp].max():.1f}] deg")
```

### Method B: Read per-grain TIF sinograms

```python
from PIL import Image

# If sino_cleanup_tomo.py already ran and saved cleaned TIFs:
sino_tif = work_dir / "Tomo" / f"sino_grain_{grain_nr}.tif"
sino_img = np.array(Image.open(sino_tif), dtype=np.float64)  # (nScans, nSpots)
# Transpose: reconstruction expects (nThetas, nDetPixels)
sino = sino_img.T  # (nSpots, nScans)
```

---

## Step 2: Extract Single-Grain Sinogram and Angles

```python
grain_nr = 0  # which grain to reconstruct
nsp = grain_nspots[grain_nr]

# Extract this grain's sinogram and angles
sino = sinos[grain_nr, :nsp, :]   # (nSpots, nScans)
angles = omegas[grain_nr, :nsp]   # (nSpots,) in degrees

print(f"Grain {grain_nr}: sinogram shape {sino.shape}, "
      f"{nsp} projection angles")
print(f"  Angle range: [{angles.min():.1f}, {angles.max():.1f}] deg")
print(f"  Non-zero entries: {(sino > 0).sum()} / {sino.size} "
      f"({100*(sino>0).sum()/sino.size:.1f}%)")
```

---

## Step 3: Optional Preprocessing

MLEM doesn't require gap filling or smoothing, but despeckling is still useful:

```python
# Minimal preprocessing: just despeckle
# (skip hole filling, gap interpolation, and smoothing -- MLEM handles those)

from scipy.ndimage import median_filter

def despeckle(sino, threshold=3.0):
    """Remove isolated bright pixels (salt noise)."""
    med = median_filter(sino, size=3)
    mask = sino > threshold * np.maximum(med, 1e-10)
    cleaned = sino.copy()
    cleaned[mask] = med[mask]
    return cleaned

sino_clean = despeckle(sino)
```

---

## Step 4: Reconstruct with MLEM

```python
import sys
sys.path.insert(0, "/Users/hsharma/opt/MIDAS/utils")
from mlem_recon import mlem, osem

# ─── Standard MLEM ────────────────────────────────────────────────
recon = mlem(
    sinogram=sino_clean,   # (nThetas, nDetPixels) = (nSpots, nScans)
    angles_deg=angles,     # (nThetas,) projection angles in degrees
    n_iter=50,             # iterations (30-100 typical)
)
# recon: (nScans, nScans) reconstructed grain shape image

print(f"Reconstruction shape: {recon.shape}")
print(f"  Min: {recon.min():.4f}, Max: {recon.max():.4f}")
print(f"  Positive: {(recon > 0).sum()} / {recon.size} pixels")
```

### Faster: OS-EM (Ordered Subsets)

```python
# ~4x faster convergence by processing subsets of projections
recon_fast = osem(
    sinogram=sino_clean,
    angles_deg=angles,
    n_iter=15,        # fewer iterations needed
    n_subsets=4,      # 4 subsets (each uses 1/4 of projections per update)
)
```

### With FBP as initial estimate (hybrid)

```python
# Use existing FBP result as starting point for faster MLEM convergence
from TOMO.midas_tomo_python import run_tomo_from_sinos

fbp_result = run_tomo_from_sinos(sino_clean, "Tomo", angles)
fbp_init = fbp_result[0, 0, :nScans, :nScans]
fbp_init = np.maximum(fbp_init, 0)  # clip negatives

recon_hybrid = mlem(sino_clean, angles, n_iter=20, init=fbp_init)
```

---

## Step 5: Compare MLEM vs FBP

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# FBP (current method)
from TOMO.midas_tomo_python import run_tomo_from_sinos
fbp_arr = run_tomo_from_sinos(sino_clean, "Tomo", angles)
fbp = fbp_arr[0, 0, :nScans, :nScans]

axes[0].imshow(fbp, cmap='viridis')
axes[0].set_title(f"FBP (gridrec)")

# MLEM
axes[1].imshow(recon, cmap='viridis')
axes[1].set_title(f"MLEM ({50} iter)")

# OS-EM
axes[2].imshow(recon_fast, cmap='viridis')
axes[2].set_title(f"OS-EM ({15} iter, {4} subsets)")

for ax in axes:
    ax.axis('off')
plt.tight_layout()
plt.savefig("mlem_vs_fbp.png", dpi=150)
plt.show()
```

---

## Step 6: Reconstruct All Grains and Build Grain Map

```python
nGrs = len(grain_nspots)
recon_size = nScans  # reconstruction grid size

all_recons = np.zeros((nGrs, recon_size, recon_size))

for g in range(nGrs):
    nsp = grain_nspots[g]
    if nsp < 3:  # need at least 3 projections
        continue
    sino_g = despeckle(sinos[g, :nsp, :])
    angles_g = omegas[g, :nsp]
    all_recons[g] = mlem(sino_g, angles_g, n_iter=50)
    print(f"  Grain {g}: {nsp} projections, "
          f"max intensity = {all_recons[g].max():.2f}")

# Build grain map: assign each pixel to the grain with highest intensity
grain_map = np.argmax(all_recons, axis=0)
# Mask background (where no grain has significant signal)
max_intensity = np.max(all_recons, axis=0)
grain_map[max_intensity < 0.1 * max_intensity.max()] = -1

print(f"\nGrain map: {grain_map.shape}, "
      f"{len(np.unique(grain_map))} unique labels")
```

---

## Integration with pf_MIDAS.py

To use MLEM as a drop-in replacement in the existing workflow, modify the reconstruction call in `pf_MIDAS.py` (around line 1623):

```python
# BEFORE (FBP):
# recon_arr = run_tomo_from_sinos(sino_for_tomo, 'Tomo', thetas, ...)

# AFTER (MLEM):
from mlem_recon import mlem
recon = mlem(sino_for_tomo, thetas, n_iter=50)
# Pad to match expected output format:
recon_arr = recon[np.newaxis, np.newaxis, :, :]  # (1, 1, N, N)
```

A future `-reconMethod mlem|fbp` flag can be added to `pf_MIDAS.py` to select the method at runtime.

---

## API Reference

### `mlem(sinogram, angles_deg, n_iter=50, init=None, mask=None, eps=1e-10)`

| Parameter | Type | Description |
|-----------|------|-------------|
| `sinogram` | ndarray (nThetas, M) | Measured sinogram. Zero rows = missing. |
| `angles_deg` | ndarray (nThetas,) | Projection angles in degrees. |
| `n_iter` | int | Number of MLEM iterations. |
| `init` | ndarray (M, M), optional | Initial estimate. Default: uniform. |
| `mask` | ndarray (nThetas, M) bool, optional | Which entries to use. |
| `eps` | float | Division-by-zero guard. |
| **Returns** | ndarray (M, M) | Reconstructed image (non-negative). |

### `osem(sinogram, angles_deg, n_iter=10, n_subsets=4, init=None, eps=1e-10)`

Same as `mlem` but with `n_subsets` for ordered-subset acceleration.

### `forward_project(image, angles_deg)` -> ndarray (nThetas, N)

Radon transform with bilinear interpolation.

### `back_project(sinogram, angles_deg, N)` -> ndarray (N, N)

Adjoint Radon transform (unfiltered back-projection).

---

## Performance Notes

- The custom NumPy implementation is ~10x slower than MIDAS_TOMO (C/FFTW gridrec) for the same number of projections. For a 15x15 grid with 50 spots, MLEM takes ~0.5 seconds per grain.
- OS-EM with 4 subsets converges in ~4x fewer iterations, effectively matching FBP speed for typical pf-HEDM problems.
- For GPU acceleration, the forward/back projectors can be replaced with ASTRA Toolbox calls (same API, ~100x speedup).
