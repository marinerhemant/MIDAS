# midas-index

Pure-Python/PyTorch FF-HEDM indexer. Drop-in replacement for `IndexerOMP` /
`IndexerGPU` from MIDAS, with seamless CPU / CUDA / MPS device switching.

**Status:** v0.4.x — production. Bit-identical to C `IndexerOMP` on the
500/500 seed FF parity gate, plus the new scanning indexer that matches
C `IndexerScanningOMP` on the 1-voxel PF parity gate. Auto dense ↔
jagged `compare_spots` strategy picker (see
`midas_index.compute.matching.pick_compare_strategy`) keeps GPU runs
inside the available memory budget without OOM. Detailed design doc
lives in `dev/implementation_plan.md` (gitignored).

## Install

```bash
pip install midas-index
```

For local development:

```bash
cd packages/midas_index
pip install -e .[dev]
```

## Notebooks

Worked-example Jupyter notebooks live in `notebooks/`. They are **not shipped with `pip install`** — get them by cloning the [MIDAS repository](https://github.com/marinerhemant/MIDAS/tree/master/packages/midas_index/notebooks).

## Quick start

```bash
# CLI — drop-in for IndexerOMP / IndexerGPU
midas-index paramstest.txt 0 1 1000 8

# Pin device / dtype via env vars (auto-detect: CUDA -> MPS -> CPU)
MIDAS_INDEX_DEVICE=cuda MIDAS_INDEX_DTYPE=float32 \
    midas-index paramstest.txt 0 1 1000 8
```

Library API:

```python
from midas_index import Indexer

result = Indexer.from_param_file("paramstest.txt", device="cuda").run(
    block_nr=0, n_blocks=1, n_spots_to_index=1000,
)
```

## C backend (`midas_indexer`)

A bundled C binary, `midas_indexer`, ships alongside the Python+numba
path. It supersedes the legacy MIDAS `IndexerOMP` (FF) and
`IndexerScanningOMP` (PF) binaries with one unified algorithm: PF
subsumes FF as the `nScans=1` specialization, Friedel-pair plane normals
and 3D per-seed position search activate when `nScans=1`, scan-position
filter activates when `nScans>1`. Output is always consolidated
(`IndexBest_all.bin` + `IndexKey_all.bin` + `IndexBest_IDs_all.bin`) +
a Phase 8 sidecar `IndexBest_weights_all.bin` carrying per-match
soft-attribution weights.

Build: scikit-build-core compiles `c_src/IndexerUnified.c` at
`pip install` time and installs the binary at
`<site-packages>/midas_index/bin/midas_indexer`. Requires OpenMP. On
macOS install libomp first (`brew install libomp`); on Linux gcc/clang
with libgomp suffices. If OpenMP is missing the install still
succeeds — only the Python path is available, and
`backend_c.available()` returns `False`.

### Use from the library

```python
from midas_index import Indexer

ind = Indexer.from_param_file("paramstest.txt")
ind.run_scanning(
    scan_positions=positions, out_path="Output/IndexBest_all.bin",
    backend="c-omp",         # "python" (default) | "c-omp"
    num_procs=8,
)
```

Indexer.run() takes the same `backend=` kwarg. `paramstest_path` is
required when `backend="c-omp"` unless the Indexer was constructed via
`from_param_file` (which captures the path).

### Use from `midas-pipeline`

```bash
midas-pipeline run --indexer-backend c-omp ...
```

### Soft beam attribution

Set in `paramstest.txt`:

```
SoftAttrMode      gaussian   # none | top_hat | gaussian
SoftAttrFwhm      2.5        # FWHM in µm
SoftAttrTruncate  6.0        # gaussian tail cut (µm); 0 = unbounded
SoftAttrFalloff   1.0        # top-hat edge ramp (µm); 0 = strict
```

Mode `none` (default) preserves the legacy `ScanPosTol` hard window
bit-identically. Modes `top_hat` and `gaussian` widen the candidate
window and emit per-match weights into the
`IndexBest_weights_all.bin` sidecar (1.0 weights for mode `none` so
downstream code can rely on the file's presence).

## Drive from `ff_MIDAS.py`

Pass `-useTorchIndexer 1` to switch the indexing stage from C `IndexerOMP` /
`IndexerGPU` to this package:

```bash
python ff_MIDAS.py -paramFN paramstest.txt … -useTorchIndexer 1
```

## Architecture

`midas-index` is a thin orchestration layer. Heavy lifting is delegated to:

- [`midas-diffract`](../midas_diffract/) — forward simulation (HKL -> theoretical spots).
- [`midas-stress`](../midas_stress/) — orientation conversions, symmetry, fundamental zone.

This package itself owns: seed enumeration, orientation / position grid layout,
binned matching, scoring, I/O, and the CLI / library API.

## Benchmark

A bundled benchmark drives the full per-seed pipeline end-to-end:

```bash
python -m midas_index.benchmarks.bench_seed --n-grains 5 --n-iter 3
```

## License

BSD-3-Clause. Part of [MIDAS](https://github.com/marinerhemant/MIDAS).
