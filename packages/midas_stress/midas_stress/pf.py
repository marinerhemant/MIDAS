"""Per-voxel stress for pf-HEDM (scanning 3DXRD) maps.

Reads the voxel table a MIDAS pf run writes (``Recons/microstrFull.csv``,
or the ``microstr`` dataset of ``Recons/microstructure.hdf``) and turns the
per-voxel strain into per-voxel stress with :func:`midas_stress.hooke_stress`.
Nothing here re-derives Hooke's law or the Mandel rotation; it is column
bookkeeping, unit handling and frame handling around the existing functions.

Column layout of ``microstrFull.csv`` (43 columns, 0-based)
-----------------------------------------------------------
Header: ``midas_pipeline/stages/consolidation_pf.py:82-87`` (``MICROSTR_HEADER``),
identical to legacy ``pf_MIDAS.py:2472-2473``::

    0 SpotID | 1-9 O11..O33 | 10 SpotID | 11-13 x y z | 14 SpotID |
    15-20 a b c alpha beta gamma | 21 SpotID | 22-24 PosErr OmeErr InternalAngle |
    25 Radius | 26 Completeness | 27-35 E11..E33 | 36-38 Eul1..Eul3 |
    39-42 Quat1..Quat4 (fundamental-zone quaternion, written by consolidation)

Strain frame and units (read from the refiner source, not assumed)
------------------------------------------------------------------
Columns 27-35 are copied verbatim from the c-omp refiner's per-voxel
``FitBest_*.csv`` (``consolidation_pf.py:158-176`` reads the first 43 tokens).
The refiner writes them at ``midas_fit_grain/c_src/FitUnified.c:2669-2671`` as
``StrainTensorSample[i][j] * 1000000``, i.e. **microstrain**. That tensor is
fit in ``StrainTensorKenesei`` (``FitUnified.c:1185-1273``) from the strain-gauge
equation ``eps_ij g_i g_j = (d_obs - d_0)/d_0``, where ``g`` is the observed
scattering vector rotated back by omega (``SpotToGv``, ``FitUnified.c:459-479``;
filled at ``FitUnified.c:538-541`` and copied to ``SpotsComp[4..6]`` at
``FitUnified.c:616``). So E11..E33 is the **d-spacing ("Kenesei") strain in
the frame in which the orientation matrix O is expressed** (``v_lab = O @ v_crystal``):
the MIDAS lab frame at omega = 0 when Wedge = 0, and the rotation-stage frame when
Wedge != 0 (see ``midas_diffract`` ``forward.py``). It is not the crystal frame. ``d_0`` comes from the run's ``LatticeConstant``, so the hydrostatic part
is only as good as that reference cell.

Caveat: the pure-Python refiner (``midas_fit_grain/scan_driver.py:215,243``)
writes ZEROS in 27-35. A pf run refined with it has no strain; this module
raises if every voxel's strain is exactly zero rather than return zero stress.
"""

from __future__ import annotations

from typing import Union

import numpy as np

from .hooke import hooke_stress
from .materials import get_stiffness
from .tensor import hydrostatic, von_mises

#: 0-based column indices in microstrFull.csv (see module docstring).
COL_OM = slice(1, 10)
COL_POS = slice(11, 14)
COL_LATTICE = slice(15, 21)
COL_COMPLETENESS = 26
COL_STRAIN = slice(27, 36)
COL_EULER = slice(36, 39)
COL_QUAT = slice(39, 43)
N_COLS_MICROSTR = 43

#: microstrFull.csv stores strain in microstrain (FitUnified.c:2671).
MICROSTRAIN = 1e-6


def read_microstr_full(path: str) -> np.ndarray:
    """Load ``microstrFull.csv`` (comma-delimited, ``#`` header) or the
    ``microstr`` dataset of ``microstructure.hdf`` as an ``(N, 43)`` array."""
    if str(path).endswith((".hdf", ".h5", ".hdf5")):
        import h5py
        with h5py.File(path, "r") as f:
            data = np.asarray(f["microstr"][()], dtype=np.float64)
    else:
        data = np.loadtxt(path, delimiter=",", comments="#", ndmin=2)
    if data.shape[1] < N_COLS_MICROSTR:
        raise ValueError(
            f"{path}: {data.shape[1]} columns, expected {N_COLS_MICROSTR} "
            "(microstrFull.csv layout)")
    return data


def voxel_stress(
    microstr: Union[str, np.ndarray],
    stiffness: Union[str, np.ndarray],
    strain_scale: float = MICROSTRAIN,
    allow_zero_strain: bool = False,
) -> dict:
    """Per-voxel lab-frame stress from a MIDAS pf voxel table.

    Parameters
    ----------
    microstr : str or ndarray (N, >=43)
        Path to ``microstrFull.csv`` / ``microstructure.hdf``, or the array.
    stiffness : str or ndarray (6, 6)
        A :data:`midas_stress.STIFFNESS_LIBRARY` name (e.g. ``"Zr"``) or a
        crystal-frame Mandel stiffness in GPa.
    strain_scale : float
        Factor taking columns 27-35 to dimensionless strain. Default 1e-6
        (the file is in microstrain; see module docstring).
    allow_zero_strain : bool
        If False (default), raise when every strain entry is exactly 0,
        the signature of the Python refiner which does not fit strain.

    Returns
    -------
    dict with
        ``xyz`` (N, 3) voxel position, um;
        ``orientation`` (N, 3, 3) crystal -> lab;
        ``completeness`` (N,);
        ``strain_lab`` (N, 3, 3) dimensionless, in the frame of ``orientation`` (lab at omega=0 for Wedge=0);
        ``stress_lab`` (N, 3, 3) MPa, same frame;
        ``stress_voigt`` (N, 6) MPa, plain (non-Mandel) order
        ``[xx, yy, zz, xy, xz, yz]``;
        ``hydrostatic`` (N,) MPa, tr(sigma)/3;
        ``von_mises`` (N,) MPa.
    """
    data = read_microstr_full(microstr) if isinstance(microstr, str) \
        else np.asarray(microstr, dtype=np.float64)
    if data.ndim != 2 or data.shape[1] < N_COLS_MICROSTR:
        raise ValueError(f"expected (N, >={N_COLS_MICROSTR}) array, got {data.shape}")
    C = get_stiffness(stiffness) if isinstance(stiffness, str) \
        else np.asarray(stiffness, dtype=np.float64)

    om = data[:, COL_OM].reshape(-1, 3, 3)
    eps = data[:, COL_STRAIN].reshape(-1, 3, 3) * strain_scale
    if not allow_zero_strain and data.shape[0] > 0 and not np.any(data[:, COL_STRAIN]):
        raise ValueError(
            "every voxel has zero strain (cols 27-35): this table was likely "
            "refined by the Python refiner, which does not fit strain. Pass "
            "allow_zero_strain=True to override.")
    eps = 0.5 * (eps + np.swapaxes(eps, -1, -2))   # file is symmetric to 1e-6 print precision

    sig = hooke_stress(eps, C, orient=om, frame="lab") * 1e3   # GPa -> MPa
    voigt = np.stack([sig[:, 0, 0], sig[:, 1, 1], sig[:, 2, 2],
                      sig[:, 0, 1], sig[:, 0, 2], sig[:, 1, 2]], axis=-1)
    return {
        "xyz": data[:, COL_POS].copy(),
        "orientation": om,
        "completeness": data[:, COL_COMPLETENESS].copy(),
        "strain_lab": eps,
        "stress_lab": sig,
        "stress_voigt": voigt,
        "hydrostatic": hydrostatic(sig),
        "von_mises": von_mises(sig),
    }
