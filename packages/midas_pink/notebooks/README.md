# midas-pink notebooks

Runnable, CPU-only, synthetic notebooks for `midas-pink`.

| Notebook | What it shows |
|---|---|
| [`01_spectrum_aware_recovery.ipynb`](01_spectrum_aware_recovery.ipynb) | Build a `ParameterisedSpectrum` (narrow pink Gaussian), construct a per-energy mono `build_pink_bank`, plant a grain and splat its observed ROIs, then recover orientation + lattice with `recover_grain_state` from a perturbed seed. Shows the monochromatic→pink extension is just the spectrum width. |

## Running

```bash
source /Users/hsharma/miniconda3/bin/activate midas_env
export KMP_DUPLICATE_LIB_OK=TRUE
cd packages/midas_pink/notebooks
python _build.py
jupyter nbconvert --to notebook --execute --inplace 01_spectrum_aware_recovery.ipynb
```

The notebooks are generated from `_build.py` (cells as `(kind, source)`
tuples) — edit that file, not the `.ipynb` JSON, then rerun `python _build.py`.

Requires `midas-diffract` and `midas-hkls` (installed in `midas_env`).
No GPU, no network, no real data.

For joint S(E) + grain recovery (`recover_joint`), two-stage recovery,
and the paper's bandwidth/noise/calibrant protocols, see
`dev/paper/scripts/`.
