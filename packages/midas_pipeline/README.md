# midas-pipeline

End-to-end MIDAS HEDM orchestrator. **FF is the single-scan degeneracy of PF.** One package, one CLI, two scan modes.

## Status

**0.1.0 — end-to-end PF and FF paths live.** The scanning indexer matches the C `IndexerScanningOMP` reference on its 1-voxel C-parity gate (seed identity, solution counts, voxel-center positions exact; orientation matrices within mrad-scale, the refiner closes the gap downstream). Real-data validation: Wenxi CP-Ti consolidation_pf reproduces the legacy `pf_MIDAS.py` grain count (770 == 770, all common). Park22 P5c parity gate now runs in ~6.7s vs the original 790s after the scanning-indexer position-grid fix.

Stages call in-process Python kernels via `midas-index` / `midas-fit-grain` / `midas-transforms` / `midas-stress`. FF mode shells out to `python -m midas_index` and `python -m midas_fit_grain` (same kernels, subprocess for the FF parity-preserving pattern). No CUDA C; GPU is torch-only.

## Install

```bash
pip install -e packages/midas_pipeline
```

## Notebooks

Worked-example Jupyter notebooks live in `notebooks/`. They are **not shipped with `pip install`** — get them by cloning the [MIDAS repository](https://github.com/marinerhemant/MIDAS/tree/master/packages/midas_pipeline/notebooks).

## CLI

```bash
midas-pipeline run --scan-mode {ff,pf,auto} --params Parameters.txt --result rundir/
midas-pipeline status rundir/
midas-pipeline resume rundir/ --from <stage>
midas-pipeline reprocess rundir/
midas-pipeline inspect rundir/LayerNr_1/
midas-pipeline simulate --out simdir/ --params Parameters.txt
midas-pipeline seed --params ... --output UniqueOrientations.csv
```

When `--scan-mode` is omitted (default `auto`), the CLI sniffs the parameter file: `nScans > 1` or presence of `BeamSize` / scanning keys → `pf`, otherwise `ff`. For PF mode, `--n-scans`, `--scan-step`, `--beam-size`, and `--scan-pos-tol` default to values in the params file (CLI flags override).

### Indexer backend

```bash
midas-pipeline run --indexer-backend {python,c-omp} ...
```

`python` (default) — in-process numba/torch indexer. Portable (CPU/CUDA/MPS), differentiable, slower on large PF datasets.

`c-omp` — bundled unified C binary (`midas_indexer`) from `midas-index`. Requires midas-index installed with a working OpenMP toolchain (macOS: `brew install libomp`). ~290× faster than the Python path on real PF datasets (per the Wenxi CP-Ti benchmark in `packages/midas_index/dev/`). Output is bit-identical to the Python path on the PF parity gate.

## Coexistence with `midas-ff-pipeline`

The legacy `midas-ff-pipeline` console-script is preserved as an independent FF orchestrator (its own kernels, its own CLI). It is **not** deprecated by `midas-pipeline run --scan-mode ff` — both paths invoke the same `midas-index` / `midas-fit-grain` kernels under the hood, and both stay green on the FF parity gate. Pick whichever is more convenient for your workflow.

## Architecture

- **One orchestrator** with a mode-dependent `STAGE_ORDER`.
- **Shared kernel packages** (`midas-index`, `midas-fit-grain`, `midas-transforms`, etc.) extended in place; FF behavior preserved by parity gates.
- **PF-only modules** live inside `midas_pipeline` (`find_grains/`, `sinogen`, `recon/`, `fuse`, `potts`, `em_refine`, `seeding/`).
- **Differentiability + multi-device** mandatory on every new compute path (CPU / CUDA / MPS via torch).

## Constraints

- No CUDA C; GPU support is torch-only.
- No deletions of legacy code in this effort.
- `midas-process-grains` is FF-only; PF consolidation is fresh pure-Python.
- `utils/calcMiso.py` is not imported by this package; all orientation math comes from `midas-stress`.
- `TOMO/midas_tomo_python.py` is imported in place, not relocated.
