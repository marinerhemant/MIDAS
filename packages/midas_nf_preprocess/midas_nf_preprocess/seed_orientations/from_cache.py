"""Load (and, on request, build) seed-orientation libraries on disk.

The legacy C workflow keeps pre-extracted CSVs and binary lookup tables under
``NF_HEDM/seedOrientations/`` for all 12 Laue-group buckets. Those files are
generated artefacts and are NOT tracked in git (``.gitignore``), so a clean
checkout or a pip install does not have them. This module gives a
torch-tensor view over whichever cache is present, and can build a missing
one deterministically with the package's own from-scratch generator
(:func:`build_seed_cache`).

Two file formats are supported:

  - ``seed_<lookup_type>.csv`` -- one CSV per bucket, comma-separated quaternions
    ``w, x, y, z``. The fastest path: a single ``np.loadtxt`` (or csv parse) and
    we are done. This is also what :func:`build_seed_cache` writes.
  - ``orientations_master.bin`` + ``lookup_<lookup_type>.bin`` -- the binary
    layout used by ``GenerateSeedLookupTables.c``. Useful when the CSV has not
    yet been extracted; we use the ``np.fromfile`` path identical to
    ``utils/extract_seed_orientations.ensure_seed_orientations``.

Cache directory resolution (first match wins):

  1. ``seed_dir=`` / ``--seed-dir``;
  2. ``$MIDAS_NF_SEED_DIR``;
  3. otherwise BOTH ``NF_HEDM/seedOrientations/`` in the source tree
     (:data:`DEFAULT_SEED_DIR`) and the per-user cache
     (:data:`USER_SEED_DIR`, ``$XDG_CACHE_HOME/midas/nf_seed_orientations``)
     are searched, in that order.

When no cache file is found we raise :class:`SeedCacheNotFound` naming the
missing file, every directory searched, and the exact command that builds it.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch

from ..device import resolve_device, resolve_dtype
from .crystal import LOOKUP_TYPES, REPRESENTATIVE_SG, space_group_to_lookup_type


# Location of the legacy C seed cache, relative to the MIDAS repo. Present only
# in a source checkout where GenerateSeedLookupTables has been run.
DEFAULT_SEED_DIR = Path(__file__).resolve().parents[3] / ".." / "NF_HEDM" / "seedOrientations"

# Per-user cache, written by build_seed_cache when no directory is given.
USER_SEED_DIR = (
    Path(os.environ.get("XDG_CACHE_HOME") or (Path.home() / ".cache"))
    / "midas" / "nf_seed_orientations"
)


#: ``generate_uniform_seeds`` resolutions that reproduce the legacy cache's
#: seed DENSITY per lookup type (matched by prefix). Same table as
#: ``midas_nf_pipeline.stages._SCRATCH_RESOLUTION_DEG``: cubic 2.8 deg ->
#: 251,545 seeds against the cache's 243,129; hexagonal 2.25 deg -> 486,946
#: against 486,755. The sampler's nominal 1.5 deg would give ~2.8x as many.
CACHE_EQUIVALENT_RESOLUTION_DEG = {"cubic": 2.8, "hexagonal": 2.25}
CACHE_EQUIVALENT_RESOLUTION_DEFAULT = 2.5


def cache_equivalent_resolution(lookup_type: str) -> float:
    """From-scratch resolution matching the legacy cache density."""
    for prefix, res in CACHE_EQUIVALENT_RESOLUTION_DEG.items():
        if lookup_type.startswith(prefix):
            return res
    return CACHE_EQUIVALENT_RESOLUTION_DEFAULT


class SeedCacheNotFound(FileNotFoundError):
    """No cached seed file found for the requested lookup type."""


def _candidate_seed_dirs(seed_dir: Optional[Union[str, Path]]) -> list[Path]:
    if seed_dir is not None:
        return [Path(seed_dir)]
    env = os.environ.get("MIDAS_NF_SEED_DIR")
    if env:
        return [Path(env)]
    return [DEFAULT_SEED_DIR, USER_SEED_DIR]


def _build_dir(seed_dir: Optional[Union[str, Path]]) -> Path:
    """Where :func:`build_seed_cache` writes when not told otherwise."""
    if seed_dir is not None:
        return Path(seed_dir)
    env = os.environ.get("MIDAS_NF_SEED_DIR")
    if env:
        return Path(env)
    return USER_SEED_DIR


def _has_cache(d: Path, lookup_type: str) -> bool:
    return (d / f"seed_{lookup_type}.csv").is_file() or (
        (d / "orientations_master.bin").is_file()
        and (d / f"lookup_{lookup_type}.bin").is_file()
    )


def _not_found(lookup_type: str, searched: list[Path],
               seed_dir: Optional[Union[str, Path]]) -> SeedCacheNotFound:
    sg = REPRESENTATIVE_SG[lookup_type]
    dir_flag = f" --seed-dir {seed_dir}" if seed_dir is not None else ""
    return SeedCacheNotFound(
        f"Seed cache file seed_{lookup_type}.csv (or orientations_master.bin + "
        f"lookup_{lookup_type}.bin) not found. Searched: "
        + ", ".join(str(d) for d in searched)
        + ". These are generated files, not tracked in git. Build it with:\n"
        f"    midas-nf-preprocess seed-orientations --method cache "
        f"--space-group {sg}{dir_flag} --build-cache --output seeds.csv\n"
        f"(or in Python: build_seed_cache('{lookup_type}'"
        + (f", seed_dir='{seed_dir}'" if seed_dir is not None else "")
        + ")), or point MIDAS_NF_SEED_DIR at an existing cache."
    )


def _load_csv(path: Path) -> np.ndarray:
    """Load a comma-separated quaternion CSV (w,x,y,z per line).

    ``np.loadtxt`` collapses a single-row file to a 1D ``(4,)`` array; promote
    back to ``(1, 4)`` so the caller can rely on the 2D shape.
    """
    arr = np.loadtxt(path, delimiter=",", dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr


def _load_from_master_lookup(seed_dir: Path, lookup_type: str) -> np.ndarray:
    """Reconstitute quaternions from orientations_master.bin + lookup_*.bin.

    Mirrors ``utils/extract_seed_orientations.ensure_seed_orientations`` L73-L75.
    """
    master_path = seed_dir / "orientations_master.bin"
    lookup_path = seed_dir / f"lookup_{lookup_type}.bin"
    master = np.fromfile(master_path, dtype=np.float64).reshape(-1, 4)
    indices = np.fromfile(lookup_path, dtype=np.int32)
    return master[indices]


def build_seed_cache(
    lookup_type: str,
    *,
    seed_dir: Optional[Union[str, Path]] = None,
    resolution_deg: Optional[float] = None,
    seed: int = 42,
    overwrite: bool = False,
) -> Path:
    """Write ``seed_<lookup_type>.csv`` with the from-scratch generator.

    Deterministic: the same ``(lookup_type, resolution_deg, seed)`` always
    produces the same file (Shoemake sampling with a fixed CPU RNG seed, FZ
    reduction with the bucket's representative space group, deduplication at
    ``resolution_deg / 2``). ``resolution_deg`` defaults to
    :func:`cache_equivalent_resolution`, which matches the legacy cache's
    seed count. The file is NOT identical to the legacy
    ``GenerateSeedLookupTables`` cache (different sampler, same density), so
    the orientation grid differs point-by-point from runs seeded with it.

    Writes into ``seed_dir``, else ``$MIDAS_NF_SEED_DIR``, else
    :data:`USER_SEED_DIR`. Returns the path of the CSV. An existing file is
    left alone unless ``overwrite``.
    """
    from .from_scratch import generate_uniform_seeds
    from .io import write_seeds_csv

    if lookup_type not in LOOKUP_TYPES:
        raise ValueError(
            f"Unknown lookup_type {lookup_type!r}; expected one of {LOOKUP_TYPES}"
        )
    out = _build_dir(seed_dir) / f"seed_{lookup_type}.csv"
    if out.exists() and not overwrite:
        return out
    if resolution_deg is None:
        resolution_deg = cache_equivalent_resolution(lookup_type)
    quats = generate_uniform_seeds(
        REPRESENTATIVE_SG[lookup_type],
        resolution_deg=resolution_deg,
        seed=seed,
        device="cpu",
        dtype="fp64",
    )
    tmp = out.with_suffix(".csv.tmp")
    write_seeds_csv(quats, tmp)
    os.replace(tmp, out)
    return out


def load_seeds_for_lookup_type(
    lookup_type: str,
    *,
    seed_dir: Optional[Union[str, Path]] = None,
    device: Optional[Union[str, torch.device]] = None,
    dtype: Optional[Union[str, torch.dtype]] = None,
    build_if_missing: bool = False,
) -> torch.Tensor:
    """Load a cached seed-orientation file by MIDAS lookup-type name.

    Parameters
    ----------
    lookup_type : one of the 12 names in ``LOOKUP_TYPES``
        (``cubic_high``, ``hexagonal_high``, ...).
    seed_dir : override the cache directory. See the module docstring for
        the default search order.
    build_if_missing : when no cache file is found, build it with
        :func:`build_seed_cache` (cache-equivalent density, seed 42) instead
        of raising.

    Returns
    -------
    Tensor of shape ``(N, 4)`` -- quaternions ``(w, x, y, z)``.
    """
    searched = _candidate_seed_dirs(seed_dir)
    found = next((d for d in searched if _has_cache(d, lookup_type)), None)
    if found is None:
        if not build_if_missing:
            raise _not_found(lookup_type, searched, seed_dir)
        csv_path = build_seed_cache(lookup_type, seed_dir=seed_dir)
        found = csv_path.parent

    csv_path = found / f"seed_{lookup_type}.csv"
    if csv_path.exists():
        arr = _load_csv(csv_path)
    else:
        arr = _load_from_master_lookup(found, lookup_type)

    if arr.ndim != 2 or arr.shape[1] != 4:
        raise ValueError(
            f"{csv_path}: expected (N, 4) quaternions, got shape {arr.shape}"
        )

    device = resolve_device(device)
    dtype = resolve_dtype(device, dtype)
    return torch.from_numpy(arr).to(device=device, dtype=dtype)


def load_seeds_for_space_group(
    space_group: int,
    *,
    seed_dir: Optional[Union[str, Path]] = None,
    device: Optional[Union[str, torch.device]] = None,
    dtype: Optional[Union[str, torch.dtype]] = None,
    build_if_missing: bool = False,
) -> torch.Tensor:
    """Convenience wrapper: SG -> lookup type -> cached seeds.

    See :func:`load_seeds_for_lookup_type` for parameter docs.
    """
    lookup_type = space_group_to_lookup_type(space_group)
    return load_seeds_for_lookup_type(
        lookup_type, seed_dir=seed_dir, device=device, dtype=dtype,
        build_if_missing=build_if_missing,
    )
