"""I/O for MIDAS grain data: Grains.csv and consolidated HDF5 files.

The CSV parser is header-driven so it works with all MIDAS flavours:
the historical ``Grains.csv`` layout, the newer ``GrainsSim.csv`` /
``Grains_allatonce.csv`` layout (with ``DiffPos``, ``DiffOme``, ...
between positions and strains), and any other file as long as the
column header line begins with ``%GrainID``.

The returned dict exposes a single ``strain`` key (the *d-spacing
strain*, historically called the Kenesei form in MIDAS). It is
fit from the per-reflection strain-gauge equation
``eps_ij g_i g_j = (d_obs - d_0) / d_0`` using a least-squares
solve over the six symmetric components, so it is tied directly
to the raw diffraction observables.

The alternative *lattice-parameter strain* (historically called
the Fable-Beaudoin form) is derived from the fitted lattice
parameters via the deformation gradient ``F = A @ A0^-1`` and
``eps = 0.5 (F + F^T) - I``. It is returned under
``strain_lattice`` when present. At typical HEDM strain magnitudes
the two forms are numerically equivalent to second order in
strain; the d-spacing form has better noise properties and is
the recommended default.
"""

import os

import numpy as np


# Standard column name -> output key mapping for the 3x3 tensor blocks.
_TENSOR_BLOCKS = {
    "orientations": ["O11", "O12", "O13",
                     "O21", "O22", "O23",
                     "O31", "O32", "O33"],
    # Primary strain: d-spacing (strain-gauge) form, fit directly
    # from per-reflection (d_obs - d_0)/d_0. Historically named
    # "Kenesei" in the MIDAS CSV columns (eKen*).
    "strain":         ["eKen11", "eKen12", "eKen13",
                       "eKen21", "eKen22", "eKen23",
                       "eKen31", "eKen32", "eKen33"],
    # Alternate: lattice-parameter strain, eps = 0.5(F + F^T) - I
    # with F = A @ A0^-1 built from fitted lattice parameters.
    # Historically named "Fable-Beaudoin" in the MIDAS columns (eFab*).
    "strain_lattice": ["eFab11", "eFab12", "eFab13",
                       "eFab21", "eFab22", "eFab23",
                       "eFab31", "eFab32", "eFab33"],
}

_VECTOR_BLOCKS = {
    "positions":      ["X", "Y", "Z"],
    "lattice_params": ["a", "b", "c", "alpha", "beta", "gamma"],
    "euler_angles":   ["Eul0", "Eul1", "Eul2"],
}

# Scalar columns: output_key -> list of accepted header names (first match wins).
_SCALAR_COLUMNS = {
    "grain_ids":   ["GrainID"],
    "radii":       ["GrainRadius", "Radius"],
    "confidences": ["Confidence"],
    "phase":       ["PhaseNr"],
    "rms_error":   ["RMSErrorStrain"],
}


def _find_header_line(filepath: str) -> tuple:
    """Locate the ``%GrainID ...`` header line and the number of lines to skip.

    Returns
    -------
    columns : list of str
    skip_header : int
    """
    skip = 0
    with open(filepath, 'r') as f:
        for line in f:
            stripped = line.lstrip()
            if stripped.startswith("%GrainID") or stripped.startswith("% GrainID"):
                header = stripped.lstrip("%").strip()
                columns = header.split()
                return columns, skip + 1
            skip += 1
    raise ValueError(
        f"Could not locate '%GrainID ...' column header line in {filepath}"
    )


def _col_index(columns, name):
    """Find column index by exact name, return None if absent."""
    try:
        return columns.index(name)
    except ValueError:
        return None


def read_grains_csv(filepath: str) -> dict:
    """Read grain data from a MIDAS ``Grains.csv``-style file.

    The parser is header-driven: it locates the line starting with
    ``%GrainID`` and maps columns by name, so it works with all MIDAS
    output flavours (``Grains.csv``, ``GrainsSim.csv``,
    ``Grains_allatonce.csv``, etc.).

    Parameters
    ----------
    filepath : str

    Returns
    -------
    dict with keys (present only if the file contains them):
        'raw'           : ndarray (N, ncols) — full numeric data
        'columns'       : list of column names parsed from the header
        'grain_ids'     : ndarray (N,)
        'orientations'  : ndarray (N, 3, 3)
        'positions'     : ndarray (N, 3)  [X, Y, Z] (micrometers)
        'lattice_params': ndarray (N, 6)  [a, b, c, alpha, beta, gamma]
        'strain'         : ndarray (N, 3, 3) — d-spacing
                           (strain-gauge) form, recommended default.
                           Historically the Kenesei form in MIDAS.
        'strain_lattice' : ndarray (N, 3, 3) — lattice-parameter
                           form, eps = 0.5(F+F^T) - I with
                           F = A @ A0^-1. Historically the
                           Fable-Beaudoin form in MIDAS.
        'confidences'   : ndarray (N,)
        'radii'         : ndarray (N,)
        'phase'         : ndarray (N,)
        'rms_error'     : ndarray (N,)
        'euler_angles'  : ndarray (N, 3) — radians
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Grains file not found: {filepath}")

    columns, skip = _find_header_line(filepath)
    data = np.genfromtxt(filepath, skip_header=skip)
    if data.ndim == 1:
        data = data.reshape(1, -1)

    N = data.shape[0]
    result = {'raw': data, 'columns': columns}

    # 3x3 tensor blocks
    for out_key, col_names in _TENSOR_BLOCKS.items():
        idxs = [_col_index(columns, c) for c in col_names]
        if all(i is not None for i in idxs):
            result[out_key] = data[:, idxs].reshape(N, 3, 3)

    # Vector blocks
    for out_key, col_names in _VECTOR_BLOCKS.items():
        idxs = [_col_index(columns, c) for c in col_names]
        if all(i is not None for i in idxs):
            result[out_key] = data[:, idxs]

    # Scalar columns
    for out_key, aliases in _SCALAR_COLUMNS.items():
        for name in aliases:
            idx = _col_index(columns, name)
            if idx is not None:
                col = data[:, idx]
                if out_key == "grain_ids" or out_key == "phase":
                    result[out_key] = col.astype(int)
                else:
                    result[out_key] = col
                break

    return result


def read_grains_h5(filepath: str) -> dict:
    """Read grain data from MIDAS consolidated HDF5 output.

    Parameters
    ----------
    filepath : str
        Path to consolidated_Output.h5 or similar.

    Returns
    -------
    dict with keys:
        'orientations'  : ndarray (N, 3, 3)
        'euler_angles'  : ndarray (N, 3)
        'positions'     : ndarray (N, 3)
        'lattice_params': ndarray (N, 6)
        'strain'         : ndarray (N, 3, 3) — d-spacing (strain-gauge)
                           form, recommended default
        'strain_lattice' : ndarray (N, 3, 3) — lattice-parameter
                           alternate
        'radii'         : ndarray (N,)
        'confidences'   : ndarray (N,)
        'grain_ids'     : list of str
    """
    import h5py

    grains = {
        'orientations': [], 'euler_angles': [], 'positions': [],
        'lattice_params': [], 'strain': [], 'strain_lattice': [],
        'radii': [], 'confidences': [], 'grain_ids': [],
    }

    with h5py.File(filepath, 'r') as f:
        grp = f['grains']
        for gid in sorted(grp.keys()):
            g = grp[gid]
            grains['grain_ids'].append(gid)
            grains['orientations'].append(g['orientation'][()])
            grains['euler_angles'].append(g['euler_angles'][()])
            grains['positions'].append(g['position'][()])
            grains['lattice_params'].append(g['lattice_params_fit'][()])
            # Primary: d-spacing (strain-gauge) form, stored on
            # disk under the legacy "strain_kenesei" dataset name.
            grains['strain'].append(g['strain_kenesei'][()])
            # Alternate: lattice-parameter form, stored on disk
            # under the legacy "strain_fable" dataset name.
            grains['strain_lattice'].append(g['strain_fable'][()])
            grains['radii'].append(float(g['radius'][()]))
            grains['confidences'].append(float(g['confidence'][()]))

    for key in grains:
        if key != 'grain_ids':
            grains[key] = np.array(grains[key])

    return grains


def read_grains(filepath: str) -> dict:
    """Read grains from CSV or HDF5 (auto-detected by extension).

    Parameters
    ----------
    filepath : str

    Returns
    -------
    dict — see ``read_grains_csv`` or ``read_grains_h5`` for keys.
    """
    if filepath.endswith('.h5') or filepath.endswith('.hdf5'):
        return read_grains_h5(filepath)
    return read_grains_csv(filepath)


def example_data_path(filename: str = "GrainsSim.csv") -> str:
    """Return the absolute path to a file shipped under ``examples/data/``.

    Useful inside notebooks / tests so the default example works on any
    install (editable or wheel):

        >>> import midas_stress as ms
        >>> g = ms.read_grains(ms.example_data_path())

    Parameters
    ----------
    filename : str
        Name of the bundled example file. Default: ``"GrainsSim.csv"``.

    Returns
    -------
    str — absolute filesystem path.

    Raises
    ------
    FileNotFoundError if the file is not shipped with the package.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    # Package layout: packages/midas_stress/midas_stress/io.py
    #                 packages/midas_stress/examples/data/<file>
    candidates = [
        os.path.join(here, "..", "examples", "data", filename),
        # Fallback for wheel installs that bundle data inside the package
        os.path.join(here, "examples", "data", filename),
        os.path.join(here, "_data", filename),
    ]
    for p in candidates:
        p = os.path.abspath(p)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        f"Example data file '{filename}' not found; searched "
        f"{[os.path.abspath(c) for c in candidates]}"
    )
