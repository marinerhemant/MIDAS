"""High-level dispatcher for the three seed-orientation generation paths."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import torch

from .crystal import crystal_system_to_space_group, space_group_to_lookup_type
from .from_cache import load_seeds_for_space_group
from .from_grains import grains_to_quaternions, read_grains_orientations
from .from_scratch import generate_uniform_seeds


def generate_seeds(
    *,
    method: str,
    space_group: Optional[int] = None,
    crystal_system: Optional[str] = None,
    resolution_deg: float = 1.5,
    n_master: Optional[int] = None,
    seed: int = 42,
    deduplicate: bool = True,
    seed_dir: Optional[Union[str, Path]] = None,
    grains_file: Optional[Union[str, Path]] = None,
    device: Optional[Union[str, torch.device]] = None,
    dtype: Optional[Union[str, torch.dtype]] = None,
    build_cache: bool = False,
) -> torch.Tensor:
    """One-stop API to produce NF seed quaternions.

    Pick the path with ``method``:

      - ``"cache"``       -- look up a cached file in ``seed_dir`` for the
                             requested space group. Default 1.5deg spacing.
                             With ``build_cache=True`` a missing cache file is
                             built deterministically first
                             (:func:`~.from_cache.build_seed_cache`).
      - ``"from_scratch"`` -- generate uniform random quats at ``resolution_deg``
                              and reduce to the FZ.
      - ``"from_grains"``  -- parse an FF Grains.csv at ``grains_file``.

    ``space_group`` may be given directly or derived from ``crystal_system``
    (one of ``triclinic``, ``monoclinic``, ``orthorhombic``, ``tetragonal``,
    ``trigonal``, ``hexagonal``, ``cubic``). For ``method="from_grains"`` SG
    is not required.

    Returns
    -------
    Tensor of shape ``(N, 4)``.
    """
    method = method.lower()
    if method not in ("cache", "from_scratch", "from_grains"):
        raise ValueError(
            f"method must be 'cache', 'from_scratch', or 'from_grains'; "
            f"got {method!r}"
        )

    sg = _resolve_sg(space_group, crystal_system, required=method != "from_grains")

    if method == "cache":
        return load_seeds_for_space_group(
            sg, seed_dir=seed_dir, device=device, dtype=dtype,
            build_if_missing=build_cache,
        )
    if method == "from_scratch":
        return generate_uniform_seeds(
            sg,
            resolution_deg=resolution_deg,
            n_master=n_master,
            seed=seed,
            deduplicate=deduplicate,
            device=device,
            dtype=dtype,
        )
    # from_grains
    if grains_file is None:
        raise ValueError("method='from_grains' requires grains_file=...")
    grains = read_grains_orientations(grains_file, device=device, dtype=dtype)
    return grains_to_quaternions(grains).to(
        # grains_to_quaternions stacks per-grain quats, already on device.
    )


def _resolve_sg(
    space_group: Optional[int],
    crystal_system: Optional[str],
    *,
    required: bool,
) -> int:
    if space_group is not None and crystal_system is not None:
        raise ValueError(
            "Pass either space_group or crystal_system, not both."
        )
    if space_group is not None:
        return int(space_group)
    if crystal_system is not None:
        return crystal_system_to_space_group(crystal_system)
    if required:
        raise ValueError(
            "Either space_group or crystal_system is required."
        )
    return 0  # placeholder; from_grains ignores
