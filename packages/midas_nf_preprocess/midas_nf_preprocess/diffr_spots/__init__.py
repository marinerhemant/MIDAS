"""diffr_spots: per-orientation diffraction-spot prediction (port of MakeDiffrSpots).

Differentiable PyTorch port of ``NF_HEDM/src/MakeDiffrSpots.c``. For each seed
orientation and each HKL, solve the Bragg condition:

  - ``omega(s)`` (1 or 2 solutions): rotation angles where the reciprocal-space
    G-vector intersects the Ewald sphere
  - ``eta``: azimuthal angle on the detector
  - ``(yl, zl)``: lab-frame detector position at the primary distance

then filter by ``OmegaRange``, ``BoxSize``, and ``ExcludePoleAngle``. Output is
either a structured tensor bundle or the same three binary files the C
executable produces:

  - ``DiffractionSpots.bin`` : ``[TotalSpots, 3]`` float64 ``(yl, zl, omega)``
  - ``Key.bin``              : ``[N_orient, 2]`` int32 ``(count_i, offset_i)``
  - ``OrientMat.bin``        : ``[N_orient, 3, 3]`` float64 row-major

Differentiability:

  - ``omega``, ``eta``, ``yl``, ``zl`` are differentiable in the input
    quaternions / Euler angles / orientation matrices and in the lattice
    parameters that produced ``hkls``.
  - The hard ``OmegaRange`` / ``BoxSize`` / ``ExcludePoleAngle`` filters return
    a boolean ``valid`` mask (no gradient), matching the convention used by
    ``midas_diffract.SpotDescriptors``.

The geometry primitives (``calc_omega``, ``calc_eta``, ``calc_spot_position``)
match the C implementations in ``MakeDiffrSpots.c`` L99-L204 line-by-line and
extend them to batched, vectorized PyTorch operations.
"""

from .params import DiffrSpotsParams
from .orientations import (
    quat_to_orient_matrix,
    orient_matrix_to_quat,
    euler_to_orient_matrix,
)
from .geometry import (
    bragg_omega_eta,
    calc_eta_deg,
    calc_spot_position,
    rotate_around_z,
)
from .pipeline import (
    DiffrSpotsPipeline,
    resolve_hkls_csv,
    DiffrSpotsResult,
    predict_spots,
)
from .binary_io import (
    write_diffr_spots_bin,
    write_key_bin,
    write_orient_mat_bin,
    read_diffr_spots_bin,
    read_key_bin,
    read_orient_mat_bin,
    write_all,
)
from .hkls import HKLEntry, read_hkls_csv, read_seed_orientations

__all__ = [
    "DiffrSpotsParams",
    "quat_to_orient_matrix",
    "orient_matrix_to_quat",
    "euler_to_orient_matrix",
    "bragg_omega_eta",
    "calc_eta_deg",
    "calc_spot_position",
    "rotate_around_z",
    "DiffrSpotsPipeline",
    "resolve_hkls_csv",
    "DiffrSpotsResult",
    "predict_spots",
    "write_diffr_spots_bin",
    "write_key_bin",
    "write_orient_mat_bin",
    "read_diffr_spots_bin",
    "read_key_bin",
    "read_orient_mat_bin",
    "write_all",
    "HKLEntry",
    "read_hkls_csv",
    "read_seed_orientations",
]
