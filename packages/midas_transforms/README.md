# midas-transforms

Pure-Python / PyTorch FF-HEDM intermediate transforms — the four stages
between peak-fitting and indexing in the MIDAS workflow.

Drop-in replacement for these C binaries:

| C binary                          | Python entry point         |
|-----------------------------------|----------------------------|
| `MergeOverlappingPeaksAllZarr`    | `midas-merge-peaks`        |
| `CalcRadiusAllZarr`               | `midas-calc-radius`        |
| `FitSetupZarr`                    | `midas-fit-setup`          |
| `SaveBinData`                     | `midas-bin-data`           |

Plus an end-to-end `midas-transforms pipeline <zarr>` that runs all four on
GPU with no CSV / binary disk round-trips between stages.

## Why

- **Speed.** Vectorised PyTorch kernels target ≥ 1× C on CPU and 5–50× on
  GPU at production scale. Binning and merge are the workflow's longest
  tails; both are now broadcast + scatter operations.
- **Bit-matching with C.** Every deterministic stage targets byte-exact
  output at float64. The geometry refine (`DoFit==1`) targets
  physics-meaningful tolerance via `midas-calibrate.refine_geometry`
  (LM with ADAM fallback — no NLopt, no Nelder-Mead).
- **Differentiable.** All geometry parameters that flow through detector
  projection (`Lsd`, `BC_y`, `BC_z`, `tx/ty/tz`, `p0..p14`, `wedge`,
  `dLsd`, `dP2`, residual-correction map values) carry full autograd
  through `apply_tilt_distortion` — useful for joint calibration with
  downstream grain mapping.
- **CPU/GPU portable.** Single `device=` switch (cpu / cuda / mps). No
  separate `.cu` codebase. No C extensions, no `cibuildwheel`.

## Install

```bash
pip install -e packages/midas_transforms
```

(Until uploaded to PyPI; once released, `pip install midas-transforms`.)

## Notebooks

Worked-example Jupyter notebooks live in `notebooks/`. They are **not shipped with `pip install`** — get them by cloning the [MIDAS repository](https://github.com/marinerhemant/MIDAS/tree/master/packages/midas_transforms/notebooks).

## Quick start

### Per-stage CLI (drop-in for the C binaries)

```bash
midas-merge-peaks scan.zip
midas-calc-radius scan.zip
midas-fit-setup scan.zip
midas-bin-data --result-folder .
```

### End-to-end Pipeline (intermediates stay on GPU)

```bash
midas-transforms pipeline scan.zip --device cuda --out-dir /scratch/run42
```

### Library

```python
from midas_transforms import Pipeline

pipe = Pipeline.from_zarr("scan.zip", device="cuda")
result = pipe.run()
result.merge        # (N, 17) on GPU
result.radius       # (N, 24) on GPU
result.fit_setup    # InputAll/Extra tensors on GPU
result.bins         # Spots/ExtraInfo/Data/nData tensors on GPU
pipe.dump("/scratch/run42")
```

Or use stages individually:

```python
from midas_transforms import merge_overlapping_peaks, calc_radius, fit_setup, bin_data

merge_overlapping_peaks(zarr_path="scan.zip", device="cuda")
calc_radius(result_folder=".", zarr_params=..., device="cuda")
fit_setup(result_folder=".", zarr_params=..., device="cuda")
bin_data(result_folder=".", device="cuda")
```

## Wiring into `ff_MIDAS.py`

`ff_MIDAS.py` accepts a `--useTorchTransforms 1` flag (mirrors the
existing `--useTorchIndexer` / `--peakFitGPU` flags). When set, the
workflow's `merge_overlaps`, `calc_radius`, `data_transform`, and
`binning` stages dispatch to the Python entry points instead of the C
binaries. Outputs are byte-compatible (or numerically equivalent for
the `DoFit` geometry refine).

## Limits

- **Scanning workflows** (`SaveBinDataScanning`, `MergeMultipleScans`) are
  out of scope here; they belong with the broader scanning pipeline.
- **ResidualCorrectionMap** filename is parsed from Zarr but the per-pixel
  ΔR map isn't yet sampled in `apply_tilt_distortion` (the bilinear
  sampler exists; the `np.fromfile(...).reshape(NrPixelsZ, NrPixelsY)`
  load is a TODO in `fit_setup/core.py`).

## See also

- `dev/implementation_plan.md` — full design, scope, and risk register.
- `midas-peakfit` — upstream peak-fitting (writes the consolidated HDF5).
- `midas-index` — downstream indexer (consumes `Spots.bin`/`Data.bin`).
- `midas-calibrate` — provides `geometry_torch.pixel_to_REta_torch` and
  `refine_geometry` (LM solver) used by `fit_setup`.
- `midas-hkls` — upstream `hkls.csv` generation.
