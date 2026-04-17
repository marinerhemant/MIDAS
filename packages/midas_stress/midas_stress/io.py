"""I/O for MIDAS grain data: Grains.csv and consolidated HDF5 files."""

import os

import numpy as np

GRAINS_HEADER_LINES = 9

GRAINS_COLS = [
    'GrainID',
    'O11', 'O12', 'O13', 'O21', 'O22', 'O23', 'O31', 'O32', 'O33',
    'X', 'Y', 'Z',
    'a', 'b', 'c', 'alpha', 'beta', 'gamma',
    'eFab11', 'eFab12', 'eFab13', 'eFab21', 'eFab22', 'eFab23',
    'eFab31', 'eFab32', 'eFab33',
    'eKen11', 'eKen12', 'eKen13', 'eKen21', 'eKen22', 'eKen23',
    'eKen31', 'eKen32', 'eKen33',
    'RMSErrorStrain', 'Confidence', 'Reserved1', 'Reserved2',
    'PhaseNr', 'Radius', 'Eul0', 'Eul1', 'Eul2', 'Reserved3', 'Reserved4',
]


def read_grains_csv(filepath: str) -> dict:
    """Read grain data from MIDAS Grains.csv.

    Parameters
    ----------
    filepath : str
        Path to Grains.csv.

    Returns
    -------
    dict with keys:
        'raw': ndarray (N, ncols) — full data array
        'grain_ids': ndarray (N,)
        'orientations': ndarray (N, 3, 3)
        'positions': ndarray (N, 3) — [X, Y, Z] in micrometers
        'lattice_params': ndarray (N, 6) — [a, b, c, alpha, beta, gamma]
        'strain_fable': ndarray (N, 3, 3)
        'strain_kenesei': ndarray (N, 3, 3)
        'confidences': ndarray (N,)
        'radii': ndarray (N,)
        'euler_angles': ndarray (N, 3) — in radians
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Grains file not found: {filepath}")

    data = np.genfromtxt(filepath, skip_header=GRAINS_HEADER_LINES)
    if data.ndim == 1:
        data = data.reshape(1, -1)

    N = data.shape[0]
    result = {'raw': data}
    result['grain_ids'] = data[:, 0].astype(int)
    result['orientations'] = data[:, 1:10].reshape(N, 3, 3)
    result['positions'] = data[:, 10:13]
    result['lattice_params'] = data[:, 13:19]
    result['strain_fable'] = data[:, 19:28].reshape(N, 3, 3)
    result['strain_kenesei'] = data[:, 28:37].reshape(N, 3, 3)

    if data.shape[1] > 38:
        result['confidences'] = data[:, 38]
    if data.shape[1] > 42:
        result['radii'] = data[:, 42]
    if data.shape[1] > 45:
        result['euler_angles'] = data[:, 43:46]

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
        'orientations': ndarray (N, 3, 3)
        'euler_angles': ndarray (N, 3)
        'positions': ndarray (N, 3)
        'lattice_params': ndarray (N, 6)
        'strain_fable': ndarray (N, 3, 3)
        'strain_kenesei': ndarray (N, 3, 3)
        'radii': ndarray (N,)
        'confidences': ndarray (N,)
        'grain_ids': list of str
    """
    import h5py

    grains = {
        'orientations': [], 'euler_angles': [], 'positions': [],
        'lattice_params': [], 'strain_fable': [], 'strain_kenesei': [],
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
            grains['strain_fable'].append(g['strain_fable'][()])
            grains['strain_kenesei'].append(g['strain_kenesei'][()])
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
