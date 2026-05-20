# midas-pipeline notebooks

Runnable Jupyter notebooks for the unified MIDAS HEDM orchestrator
(FF + PF, `--scan-mode {ff,pf,auto}`).

All notebooks run on **CPU** with **synthetic** data — no external
datasets, no GPU. Source of truth is `_build.py`; the `.ipynb` files are
derived artefacts. Set `$MIDAS_HOME` if the repo is not at `~/opt/MIDAS`.

## Rebuild

```bash
source /Users/hsharma/miniconda3/bin/activate midas_env
cd packages/midas_pipeline/notebooks
python _build.py
KMP_DUPLICATE_LIB_OK=TRUE jupyter nbconvert --to notebook --execute --inplace *.ipynb \
    --ExecutePreprocessor.timeout=300
```

(`KMP_DUPLICATE_LIB_OK=TRUE` works around a duplicate-libomp issue on
macOS; the notebooks also set it internally.)

## Index

| Notebook | What it covers | Status |
| --- | --- | --- |
| `01_synthetic_ff_walkthrough.ipynb` | Forward-simulate a synthetic single-detector FF dataset, drive `midas-pipeline run --scan-mode ff` **through binning + indexing**, inspect per-stage `status` + artefacts, confirm `Data.bin` is populated and the indexer recovers grain solutions, and show `ScanGeometry.ff()` as the single-scan degeneracy of `pf_uniform`. | runs clean (through indexing) |
| `02_indexer_backends.ipynb` | Indexer backend selector: `--indexer-backend python` vs `c-omp`. Runs both over the identical binned `Spots.bin` / `Data.bin` and shows they read the same observed spots (the C binary is exercised when it is built; otherwise the cell documents its absence). | runs clean |
| `03_vmap_soft_attribution.ipynb` | `--vmap-run` + `--soft-attribution`: computes the per-spot relative-volume V-map signal (`calc_radius_v` → `Radius_V.csv`) on FF, renders the V-map diagnostic image, and explains why the *joint* `refine_vmap` + spatial soft-attribution need PF multi-scan data. | runs clean |
| `04_ff_is_pf_degeneracy.ipynb` | FF as the single-scan degeneracy of PF, made literal in the bytes: the FF run writes the PF-shaped 10-column `Spots.bin` (col 9 = ScanNr, all 0), an `int64` `Data.bin`, and a one-line `positions.csv`. | runs clean |

## Notes on PF-only features

The *joint* V-map refinement (`refine_vmap_joint`) and the spatial
soft-attribution profile operate over a **multi-scan voxel grid**, which
is a PF concept — with a single FF beam position there is one voxel and
nothing to attribute across. Notebook 03 demonstrates the FF-observable
part of the V-map (per-spot relative volume) and documents this. A full
spatially-resolved PF demonstration additionally needs a multi-scan
synthetic generator; `midas_ff_pipeline.testing` ships only single-scan
FF generators today (`midas-pipeline simulate` is a P1 scaffold), so the
PF V-map / soft-attribution demo is deferred until that lands.
