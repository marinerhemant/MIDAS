# midas-nf-pipeline

**Pure-Python NF-HEDM pipeline orchestrator** — drop-in replacement for the
legacy `NF_HEDM/workflows/nf_MIDAS.py` and
`nf_MIDAS_Multiple_Resolutions.py` shell drivers, with no calls to C
binaries (every stage is a function call into a sibling MIDAS package).

## Stages

| Stage | Python module |
| --- | --- |
| Denoise (opt-in) | `MIDAS-NF-preProc` |
| HKL list | `midas_hkls.write_nf_hkls_csv` |
| Seed orientations | `midas_nf_preprocess.seed_orientations` |
| Hex grid | `midas_nf_preprocess.hex_grid` |
| Tomo / mask filter | `midas_nf_preprocess.tomo_filter` |
| Diffraction-spot simulation | `midas_nf_preprocess.diffr_spots` |
| Image processing (median + peaks) | `midas_nf_preprocess.process_images` |
| Orientation fitting | `midas_nf_fitorientation` |
| `ParseMic` | `midas_nf_pipeline.parse_mic` (this package) |
| `Mic2GrainsList` | `midas_nf_pipeline.mic2grains` (this package) |
| Consolidated H5 | `midas_nf_pipeline.consolidate` (this package) |

All crystal-symmetry / quaternion / misorientation math goes through
`midas_stress.orientation` — the canonical Python port of
`NF_HEDM/src/GetMisorientation.h`.

## Install

```bash
pip install midas-nf-pipeline                   # base install
pip install midas-nf-pipeline[denoise]          # also pulls MIDAS-NF-preProc
```

Local dev install:

```bash
cd packages/midas_nf_pipeline
pip install -e '.[dev]'
```

## Single-resolution = `NumLoops=0` special case of multi-resolution

There is exactly one workflow in this package — `run_layer_pipeline`. It
runs the multi-resolution loop:

- Loop 0 (initial): unseeded fit at `StartingGridSize`
- Loops 1..N (refinement): seeded → bad-voxel filter → unseeded → binary merge

When the param file does **not** include `GridRefactor`, `NumLoops` is
treated as 0 and only loop 0 runs — that's the "single-resolution" case.
No code branch, no separate workflow.

Multi-layer is always supported via `--start-layer` / `--end-layer`.

## CLI

```bash
midas-nf-pipeline run        params.txt [options]   # full pipeline
midas-nf-pipeline parse-mic  params.txt              # just ParseMic
midas-nf-pipeline mic2grains params.txt mic out      # just Mic2GrainsList
midas-nf-pipeline consolidate mic_text                # rebuild H5
midas-nf-pipeline refine-params params.txt [--multi-point] [--row-nr N]
```

`midas-nf-pipeline run` flags:

| Flag | Default | Meaning |
| --- | --- | --- |
| `--n-cpus` | 1 | CPU count |
| `--device` | auto | cpu / cuda |
| `--ff-seed-orientations` | off | seeds from FF `Grains.csv` instead of cache |
| `--no-image-processing` | off | skip `ProcessImagesCombined` (assumes `SpotsInfo.bin` exists) |
| `--start-layer / --end-layer` | 1 / 1 | layer range |
| `--result-folder` | param's `OutputDirectory` | per-layer outputs go to `<rf>/LayerNr_<n>/` |
| `--min-confidence` | 0.6 | `MinConfidence` for `Mic2GrainsList` post-step |
| `--resume <h5>` | — | auto-detect last completed stage |
| `--restart-from <stage>` | — | force restart at named stage |

## Python API

```python
from argparse import Namespace
from midas_nf_pipeline.workflows import run_layer_pipeline, run_multi_layer

args = Namespace(
    paramFN='params.txt', nCPUs=8, device='auto',
    ffSeedOrientations=False, doImageProcessing=1,
    startLayerNr=1, endLayerNr=1, resultFolder='/scratch/au',
    minConfidence=0.6, resume='', restartFrom='',
)
h5 = run_layer_pipeline(args)             # one layer
h5_list = run_multi_layer(args)           # multiple layers
```

Per-stage helpers in `midas_nf_pipeline.stages` if you want to invoke
them individually (debugging, swapping in custom code, etc.):

```python
from midas_nf_pipeline import stages
stages.run_get_hkls(p, paramFN)
stages.run_hex_grid(p, paramFN)
stages.run_image_processing(p, paramFN)
stages.run_fitting(p, paramFN, n_cpus=8, device='cuda')
stages.run_parse_mic(p)
stages.run_consolidate(p, param_text)
```

## Notebooks

The `notebooks/` directory holds hands-on walkthroughs. They are **not shipped
with `pip install`** — get them by cloning the [MIDAS repository](https://github.com/marinerhemant/MIDAS/tree/master/packages/midas_nf_pipeline/notebooks):

| Notebook | Topic |
| --- | --- |
| `00_quickstart_au.ipynb` | end-to-end on the bundled Au example |
| `01_single_resolution.ipynb` | stage-by-stage walkthrough |
| `02_multi_resolution.ipynb` | grid refinement + bad-voxel filter |
| `03_refine_parameters.ipynb` | calibration / parameter refinement |
| `04_resume_restart.ipynb` | `PipelineH5`, resume/restart |
| `05_multi_layer_batch.ipynb` | multi-layer 3D scans |

## Tests

```bash
# Fast unit tests
python -m pytest tests/ -q

# Slow Au end-to-end integration test (~5 min)
MIDAS_RUN_INTEGRATION=1 python -m pytest tests/integration/ -q
```

Byte-parity tests vs the C reference outputs:

| Module | Test file | Reference |
| --- | --- | --- |
| `parse_mic` | `tests/test_parse_mic.py` | `Au_txt_Reconstructed.mic*` from `NF_HEDM/Example/sim/` |
| `mic2grains` | `tests/test_mic2grains.py` | live C `Mic2GrainsList` invoked at test time |

## Migration from `nf_MIDAS.py` / `nf_MIDAS_Multiple_Resolutions.py`

Old | New
--- | ---
`python NF_HEDM/workflows/nf_MIDAS.py -paramFN p.txt -nCPUs 8` | `midas-nf-pipeline run p.txt --n-cpus 8`
`python NF_HEDM/workflows/nf_MIDAS_Multiple_Resolutions.py -paramFN p.txt -startLayerNr 1 -endLayerNr 5` | `midas-nf-pipeline run p.txt --start-layer 1 --end-layer 5`
`-machineName umich -nNodes 4` | (dropped — Parsl removed; runs single-process)
`-resume <h5>` / `-restartFrom <stage>` | `--resume <h5>` / `--restart-from <stage>`
`-gpuFit 1` | `--device cuda`
`-doImageProcessing 0` | `--no-image-processing`

## What's *not* here

- **Parsl multi-node configs**: dropped. The new fit-orientation kernel
  is fast enough on a single H100 (~5 s on the full Au grid) that
  multi-node makes little sense. If you need it, wrap
  `run_layer_pipeline` in your own scheduler glue.
- **`MMapImageInfo` legacy fallback**: dropped. The new
  `ProcessImagesCombined` path writes `SpotsInfo.bin` directly.
- **`FitOrientationGPU` C binary**: dropped — superseded by the Triton
  fast path in `midas_nf_fitorientation`.
- **Per-machine config modules** (`uMichConfig`, `purdueConfig`, …): dropped.

## Versioning

Each release bumps `pyproject.toml` `version` and the matching
`midas_nf_pipeline.__version__`. Use `release.sh`:

```bash
./release.sh 0.1.1                 # prepare local artifacts only
./release.sh 0.1.1 --dry-run       # build but don't commit/tag
./release.sh 0.1.1 --publish       # tag, push, GitHub release, PyPI
```
