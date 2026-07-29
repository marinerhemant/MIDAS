# midas-grain-odf notebooks

Runnable, CPU-only, synthetic notebooks for `midas-grain-odf`.

| Notebook | What it shows |
|---|---|
| [`01_odf_round_trip.ipynb`](01_odf_round_trip.ipynb) | Plant a known per-grain ODF (3 orientation particles in a tight ball), render ODF-weighted spot patches, recover it with a `ParticleODF` + `fit_grain_odf`, and report intensity-weighted recovery (fraction of ODF mass placed near the planted particles). Compares the `ParticleODF` / `BinghamMixtureODF` / `VoxelGridODF` parameterisations. |

## Running

```bash
source /Users/hsharma/miniconda3/bin/activate midas_env
export KMP_DUPLICATE_LIB_OK=TRUE
cd packages/midas_grain_odf/notebooks
python _build.py
jupyter nbconvert --to notebook --execute --inplace 01_odf_round_trip.ipynb
```

The notebook reuses the package's own test scaffolding
(`tests/conftest.py`) for the forward model and ground-truth orientation,
so it is a thin, faithful wrapper around `tests/test_synth_particle.py`.

The notebooks are generated from `_build.py` (cells as `(kind, source)`
tuples) — edit that file, not the `.ipynb` JSON, then rerun `python _build.py`.

Requires `midas-diffract` (installed in `midas_env`). No GPU, no network,
no real data.
