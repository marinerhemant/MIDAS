"""seed_orientations: NF-HEDM seed-orientation generation.

Three production paths, all returning a torch tensor of quaternions
``[N, 4]`` in the ``(w, x, y, z)`` convention used by ``diffr_spots``:

  1. ``from_cache``   -- load pre-computed lookup files in the
                         ``NF_HEDM/seedOrientations/`` tree (the existing C
                         workflow's default; ~1.5 deg average spacing).
  2. ``from_scratch`` -- generate uniform random quaternions (Shoemake) at a
                         user-specified resolution, then reduce to the
                         fundamental zone of the requested space group.
  3. ``from_grains``  -- parse an FF-HEDM ``Grains.csv`` file and emit the
                         per-grain orientation as quaternions (port of
                         ``GenSeedOrientationsFF2NFHEDM.c``).

Reduction to the fundamental zone is delegated to
``midas_stress.orientation.fundamental_zone`` (torch-vectorised over all 12
MIDAS Laue groups; no orix dependency).

Public API:

  - ``generate_seeds(method=..., ...)`` -- one-stop dispatcher.
  - ``from_cache.load_seeds_for_space_group(...)``
  - ``from_scratch.generate_uniform_seeds(...)``
  - ``from_grains.read_grains_orientations(...)``
  - ``write_seeds_csv(...)``, ``read_seeds_csv(...)``
  - ``crystal_system_to_space_group(...)``, ``space_group_to_lookup_type(...)``
"""

from .crystal import (
    crystal_system_to_space_group,
    space_group_to_lookup_type,
    space_group_to_sym_order,
    LOOKUP_TYPES,
    REPRESENTATIVE_SG,
)
from .from_cache import (
    DEFAULT_SEED_DIR,
    USER_SEED_DIR,
    SeedCacheNotFound,
    build_seed_cache,
    load_seeds_for_space_group,
    load_seeds_for_lookup_type,
)
from .from_scratch import (
    generate_uniform_seeds,
    n_master_for_resolution,
    shoemake_uniform_quaternions,
    deduplicate_quaternions,
)
from .from_grains import (
    GrainOrientation,
    read_grains_orientations,
)
from .io import (
    read_seeds_csv,
    write_seeds_csv,
    write_seeds_with_lattice_csv,
)
from .dispatch import generate_seeds

__all__ = [
    # Crystal-system mapping
    "crystal_system_to_space_group",
    "space_group_to_lookup_type",
    "space_group_to_sym_order",
    "LOOKUP_TYPES",
    "REPRESENTATIVE_SG",
    # Cache path
    "DEFAULT_SEED_DIR",
    "USER_SEED_DIR",
    "SeedCacheNotFound",
    "build_seed_cache",
    "load_seeds_for_space_group",
    "load_seeds_for_lookup_type",
    # From-scratch path
    "generate_uniform_seeds",
    "n_master_for_resolution",
    "shoemake_uniform_quaternions",
    "deduplicate_quaternions",
    # From-grains path
    "GrainOrientation",
    "read_grains_orientations",
    # I/O
    "read_seeds_csv",
    "write_seeds_csv",
    "write_seeds_with_lattice_csv",
    # Dispatcher
    "generate_seeds",
]
