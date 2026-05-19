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
jupyter nbconvert --to notebook --execute --inplace *.ipynb \
    --ExecutePreprocessor.timeout=300
```

## Index

| Notebook | What it covers | Status |
| --- | --- | --- |
| `01_synthetic_ff_walkthrough.ipynb` | Forward-simulate a synthetic single-detector FF dataset, drive `midas-pipeline run --scan-mode ff` through its upstream stages (ingest → hkl → peakfit → merge → radius → transforms → binning), inspect per-stage `status` + artefacts, and show `ScanGeometry.ff()` as the single-scan degeneracy of `pf_uniform`. | runs clean (upstream stages) |

## Deferred notebooks

The following were planned but are blocked on the same root cause and
are **not** shipped yet:

| Notebook | Topic | Blocker |
| --- | --- | --- |
| 02 | indexer backend selector (`python` vs `c-omp`) | indexing stage |
| 03 | V-map joint refinement + soft beam attribution | indexing stage |
| 04 | FF as PF degeneracy — full end-to-end run | indexing stage |

**Root cause (0.2.0).** The FF binning stage writes a zero-byte
`Data.bin` (the per-bin spot-index table). Both indexer backends then
fail to mmap it:

```
ValueError: cannot mmap an empty file        # --indexer-backend python
mmap ./Data.bin failed: Invalid argument     # --indexer-backend c-omp
```

This is independent of grain count and of backend. Until binning
populates `Data.bin`, the indexing → refinement → consolidation tail
(and a grain table) is not reproducible from a notebook.

A full **PF** end-to-end notebook additionally needs a multi-scan
synthetic generator: `midas-pipeline simulate` is a P1 scaffold in 0.2.0
(prints a notice and exits non-zero), and the sibling
`midas_ff_pipeline.testing.generate_synthetic_dataset` produces only a
single-scan FF dataset.
