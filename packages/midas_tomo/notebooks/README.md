# MIDAS TOMO tutorial notebooks

Six interactive notebooks that walk through the full TOMO workflow,
from raw inputs to publication-quality reconstructions. Every
notebook generates a small synthetic dataset on first run so it works
anywhere — replace the synthetic data block with a path to your own
data to use the same workflow on real scans.

| # | File | Topic |
|---|---|---|
| 1 | `01_quickstart_hdf5.ipynb`   | APS HDF5 → reconstruction in 10 steps |
| 2 | `02_from_tiff_stack.ipynb`   | Per-frame TIFFs → reconstruction |
| 3 | `03_cleanup_tuning.ipynb`    | Vo stripe-removal parameter sweep |
| 4 | `04_shift_search.ipynb`      | Finding the rotation centre |
| 5 | `05_from_sinograms.ipynb`    | Pre-formed sinograms (e.g. PF-HEDM output) |
| 6 | `06_troubleshooting.ipynb`   | Symptom → fix recipes |

## Setup

```bash
# Activate the MIDAS env
source ~/miniconda3/bin/activate midas_env

# Build MIDAS_TOMO (one-time, after pulling new code)
cd ~/opt/MIDAS/build && cmake --build . --target MIDAS_TOMO

# Launch
cd ~/opt/MIDAS/TOMO/notebooks
jupyter lab
```

## Shared utilities

- `_phantom.py` — synthetic-data generator used by every notebook
  (phantom volume, forward projector, HDF5 / TIFF / sinogram
  writers, controlled ring-source injection). Edit it once to
  change phantom geometry or noise levels across all six notebooks.

## Recommended reading order

1. **01** first — establishes the basic vocabulary (dark, white,
   sinogram, shift, recon cube).
2. **02** if your data is TIFFs, **05** if it is sinograms.
3. **03** for almost every real dataset — rings are nearly
   universal.
4. **04** if you don't trust the stored shift.
5. **06** any time something looks wrong.
