"""Where the PF grain list lives.

Two files share the name ``UniqueOrientations.csv``:

- ``<layer>/UniqueOrientations.csv`` is the SEED list, written by the
  ``seeding`` stage for the indexer's seeded path (568 rows on ma5608).
- ``<layer>/Output/UniqueOrientations.csv`` is the GRAIN list, written by
  ``find_grains`` (204 rows on ma5608). Its rows are the grains the
  sinograms, ``UniqueIndexSingleKey.bin`` and the per-grain recons are
  indexed by.

Every consumer of grain index ``g`` must read the grain list. Reading the
layer-level file pairs grain ``g`` with seed row ``g`` in a seeded run and
finds no file at all in an unseeded one.
"""
from __future__ import annotations

from pathlib import Path
from typing import Union

GRAIN_LIST_NAME = "UniqueOrientations.csv"


def grain_list_path(layer_dir: Union[str, Path]) -> Path:
    """``<layer_dir>/Output/UniqueOrientations.csv`` (find_grains' output)."""
    return Path(layer_dir) / "Output" / GRAIN_LIST_NAME


def require_grain_list(layer_dir: Union[str, Path]) -> Path:
    p = grain_list_path(layer_dir)
    if not p.is_file():
        raise FileNotFoundError(
            f"{p} not found. The grain list is written by find_grains; "
            f"the layer-level {GRAIN_LIST_NAME} is the seed list and is not "
            f"a substitute. Run find_grains first.")
    return p
