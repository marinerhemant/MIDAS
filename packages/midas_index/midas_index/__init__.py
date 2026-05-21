"""midas-index: Pure-Python/PyTorch FF-HEDM indexer.

Drop-in replacement for the C binaries `IndexerOMP` and `IndexerGPU` from
MIDAS, with seamless CPU/CUDA/MPS device switching.

See `dev/implementation_plan.md` (gitignored) for design and roadmap.

Quick start
-----------
    # CLI
    midas-index paramstest.txt 0 1 1000 8

    # Library
    from midas_index import Indexer, IndexerParams
    result = Indexer.from_param_file("paramstest.txt", device="cuda").run(
        block_nr=0, n_blocks=1, n_spots_to_index=1000,
    )
"""

__version__ = "0.7.2"

from .params import IndexerParams
from .result import IndexerResult, SeedResult
from .indexer import Indexer

__all__ = [
    "Indexer",
    "IndexerParams",
    "IndexerResult",
    "SeedResult",
    "__version__",
]
