"""IO submodule.

Binary readers (mmap'd) for the per-seed records produced by the upstream
indexing + refinement stages, plus CSV writers for the canonical MIDAS
grain-output files.
"""

from .binary import (
    BinaryInputs,
    TailPaddedBinary,
    materialize,
    read_index_best,
    read_index_best_full,
    read_fit_best,
    read_fit_best_final,
    read_orient_pos_fit,
    read_key,
    read_process_key,
    read_all,
)
from .csv import (
    write_grains_csv,
    write_spot_matrix_csv,
    write_grain_ids_key_csv,
)
from .hkls import HklTable, load_hkl_table
from .ids_hash import IDsHash, load_ids_hash
from .spot_diag import (
    PF_SPOT_MATRIX_COLS,
    SPOT_DIAG_COLS,
    SpotDiag,
    load_spot_diag,
    write_pf_spot_matrix,
)

__all__ = [
    "BinaryInputs",
    "TailPaddedBinary",
    "materialize",
    "read_index_best",
    "read_index_best_full",
    "read_fit_best",
    "read_fit_best_final",
    "read_orient_pos_fit",
    "read_key",
    "read_process_key",
    "read_all",
    "write_grains_csv",
    "write_spot_matrix_csv",
    "write_grain_ids_key_csv",
    "HklTable",
    "load_hkl_table",
    "IDsHash",
    "load_ids_hash",
    "SpotDiag",
    "load_spot_diag",
    "SPOT_DIAG_COLS",
    "PF_SPOT_MATRIX_COLS",
    "write_pf_spot_matrix",
]
