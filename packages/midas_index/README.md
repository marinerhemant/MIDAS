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
