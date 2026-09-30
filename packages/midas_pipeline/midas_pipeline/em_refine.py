"""EM Spot-Ownership integration for pf-HEDM pipeline.

**Copy** (not move) of ``FF_HEDM/workflows/em_pf_integration.py``. The
original file stays untouched per the project's "no deletions" rule;
the new package re-exports it under ``midas_pipeline.em_refine``.

Bridges between pf-HEDM data formats (``Spots.bin``,
``Output/UniqueOrientations.csv``, ``paramstest.txt``) and the EM model
(``fwd_sim/em_spot_ownership.py``). Generates weighted sinograms from
EM ownership probabilities, replacing the hard-assignment sinograms
from ``findSingleSolutionPFRefactored``.

Usage from the stage shell::

    from midas_pipeline.em_refine import run_em_spot_ownership
    run_em_spot_ownership(topdir, n_scans=…, …)

The function imports ``EMSpotOwnership`` lazily from ``fwd_sim/`` so
this module imports cleanly without requiring ``fwd_sim`` to be
installed.
"""

import glob
import logging
import math
import os
import sys
from pathlib import Path

import numpy as np

from ._grain_list import require_grain_list

# Add fwd_sim to path for forward model and EM imports. We anchor on
# the *MIDAS repo root* (four levels up from this file) the same way
# the legacy module anchors on its file. Path layout is
# packages/midas_pipeline/midas_pipeline/em_refine.py.
_HERE = Path(__file__).resolve()
MIDAS_HOME = _HERE.parents[3]
_FWD_SIM = MIDAS_HOME / "fwd_sim"
if _FWD_SIM.is_dir() and str(_FWD_SIM) not in sys.path:
    sys.path.insert(0, str(_FWD_SIM))

logger = logging.getLogger("midas_pipeline.em_refine")

DEG2RAD = math.pi / 180.0
RAD2DEG = 180.0 / math.pi


def parse_params_for_em(param_file):
    """Parse paramstest.txt for geometry parameters needed by EM.

    Returns a dict with: Lsd, px, y_BC, z_BC, omega_start, omega_step,
    n_frames, n_pixels_y, n_pixels_z, min_eta, wavelength, tol_ome,
    tol_eta, sgnum.
    """
    params = {
        'Lsd': 1_000_000.0, 'px': 200.0,
        'y_BC': 1024.0, 'z_BC': 1024.0,
        'omega_start': 0.0, 'omega_step': 0.25,
        'n_pixels_y': 2048, 'n_pixels_z': 2048,
        'min_eta': 6.0, 'wavelength': 0.172979,
        'tol_ome': 0.5, 'tol_eta': 5.0,
        'sgnum': 225,
        'start_nr': 1, 'end_nr': 1440,
    }
    with open(param_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            key, val = parts[0], parts[1].rstrip(';')
            if key == 'Lsd':
                params['Lsd'] = float(val)
            elif key in ('LsdFit',):
                params['Lsd'] = float(val)
            elif key == 'px':
                params['px'] = float(val)
            elif key in ('BC', 'BeamCenter'):
                params['y_BC'] = float(val)
                if len(parts) > 2:
                    params['z_BC'] = float(parts[2].rstrip(';'))
            elif key == 'ty':
                params['y_BC'] = float(val)
            elif key == 'tz':
                params['z_BC'] = float(val)
            elif key in ('OmegaFirstFile', 'OmegaStart'):
                params['omega_start'] = float(val)
            elif key == 'OmegaStep':
                params['omega_step'] = float(val)
            elif key == 'NrFilesPerSweep':
                params['nr_files_per_sweep'] = int(val)
            elif key == 'NrPixels':
                params['n_pixels_y'] = int(val)
                params['n_pixels_z'] = int(val)
            elif key in ('NrPixelsY', 'numPxY'):
                params['n_pixels_y'] = int(val)
            elif key in ('NrPixelsZ', 'numPxZ'):
                params['n_pixels_z'] = int(val)
            elif key == 'MinEta':
                params['min_eta'] = float(val)
            elif key == 'Wavelength':
                params['wavelength'] = float(val)
            elif key == 'TolOme':
                params['tol_ome'] = float(val)
            elif key == 'TolEta':
                params['tol_eta'] = float(val)
            elif key == 'SpaceGroup':
                params['sgnum'] = int(val)
            elif key == 'StartNr':
                params['start_nr'] = int(val)
            elif key == 'EndNr':
                params['end_nr'] = int(val)
            elif key == 'OmegaRange':
                params['omega_range_min'] = float(val)
                if len(parts) > 2:
                    params['omega_range_max'] = float(parts[2].rstrip(';'))
    if 'omega_range_min' in params and 'omega_range_max' in params:
        ome_min = params['omega_range_min']
        ome_max = params['omega_range_max']
        step = abs(params['omega_step'])
        if step > 0:
            params['n_frames'] = int(round(abs(ome_max - ome_min) / step))
            params['omega_start'] = ome_min
            params['omega_step'] = step
    elif 'end_nr' in params and 'start_nr' in params:
        params['n_frames'] = params['end_nr'] - params['start_nr'] + 1
    return params


def load_spots(topdir):
    """Load Spots.bin and return observed spot data.

    Spots.bin format: 10 doubles per spot
    [yCen, zCen, omega, intensity, spotID, ringNum, eta, theta, dspacing, scanNum]
    """
    spots_file = os.path.join(topdir, 'Spots.bin')
    raw = np.fromfile(spots_file, dtype=np.double)
    n_spots = len(raw) // 10
    data = raw.reshape((n_spots, 10))
    two_theta = data[:, 7] * DEG2RAD
    eta = data[:, 6] * DEG2RAD
    omega = data[:, 2] * DEG2RAD
    intensity = data[:, 3]
    ring_num = data[:, 5].astype(int)
    scan_num = data[:, 9].astype(int)
    spot_id = data[:, 4].astype(int)
    import torch
    obs_spots = torch.tensor(
        np.column_stack([two_theta, eta, omega]),
        dtype=torch.float64,
    )
    return (obs_spots,
            torch.tensor(ring_num, dtype=torch.long),
            torch.tensor(scan_num, dtype=torch.long),
            torch.tensor(intensity, dtype=torch.float64),
            torch.tensor(spot_id, dtype=torch.long))


def load_grain_orientations(topdir, refined=False):
    """Load Output/UniqueOrientations.csv → orientation matrices + grain IDs.

    Format: ``grainID rowNr nSpots startRowNr listStartPos OM1..OM9``
    (14 cols, space-delimited).
    """
    # The unrefined list is find_grains' grain list (Output/); the refined one
    # is written next to the layer by update_unique_orientations_from_refinement.
    fn = (os.path.join(topdir, 'UniqueOrientations_refined.csv') if refined
          else str(require_grain_list(topdir)))
    data = np.genfromtxt(fn, delimiter=' ')
    if data.ndim == 1:
        data = data.reshape(1, -1)
    n_grains = data.shape[0]
    orient_matrices = data[:, 5:14].reshape(n_grains, 3, 3)
    grain_ids = data[:, 0].astype(int)
    return orient_matrices, grain_ids


def orient_mat_to_euler(om):
    """3x3 orientation matrix → Euler angles (ZXZ, radians).

    The C code stores OMs in a convention that is the transpose of
    what HEDMForwardModel.euler2mat produces. We transpose first so
    that ``euler2mat(orient_mat_to_euler(OM)) == OM``.
    """
    omt = om.T
    if abs(omt[2, 2]) < 1.0 - 1e-10:
        Phi = math.acos(np.clip(omt[2, 2], -1.0, 1.0))
        phi1 = math.atan2(omt[2, 0], -omt[2, 1])
        phi2 = math.atan2(omt[0, 2], omt[1, 2])
    else:
        Phi = 0.0 if omt[2, 2] > 0 else math.pi
        phi1 = math.atan2(omt[0, 1], omt[0, 0])
        phi2 = 0.0
    return np.array([phi1, Phi, phi2])


def load_hkls(topdir):
    """Load hkls.csv for the forward model."""
    data = np.loadtxt(os.path.join(topdir, 'hkls.csv'), skiprows=1)
    import torch
    hkls_cart = torch.tensor(data[:, 5:8], dtype=torch.float64)
    thetas = torch.tensor(data[:, 8] * DEG2RAD, dtype=torch.float64)
    ring_indices = torch.tensor(data[:, 4].astype(int), dtype=torch.long)
    return hkls_cart, thetas, ring_indices


def load_scan_to_spatial(topdir, n_scans):
    """positions.csv → file-order to spatial-order mapping."""
    positions = np.loadtxt(os.path.join(topdir, 'positions.csv'))
    sort_idx = np.argsort(positions)
    scan_to_spatial = np.empty(n_scans, dtype=int)
    for i, si in enumerate(sort_idx):
        scan_to_spatial[si] = i
    return scan_to_spatial


def build_forward_model(topdir, params):
    """Build HEDMForwardModel from a parameter file and hkls.csv."""
    from hedm_forward import HEDMForwardModel, HEDMGeometry

    hkls_cart, thetas, ring_indices = load_hkls(topdir)
    geometry = HEDMGeometry(
        Lsd=params['Lsd'],
        y_BC=params['y_BC'],
        z_BC=params['z_BC'],
        px=params['px'],
        omega_start=params['omega_start'],
        omega_step=params['omega_step'],
        n_frames=params['n_frames'],
        n_pixels_y=params['n_pixels_y'],
        n_pixels_z=params['n_pixels_z'],
        min_eta=params['min_eta'],
        wavelength=params['wavelength'],
    )
    model = HEDMForwardModel(hkls=hkls_cart, thetas=thetas, geometry=geometry)
    model.ring_indices = ring_indices
    return model, ring_indices


def reweight_sinograms_with_ownership(topdir, ownership, spot_ids, n_grains,
                                        max_n_hkls, n_scans):
    """Re-weight the C code's sinograms using EM ownership probabilities.

    Only re-weights sinogram cells where a spot is shared between
    multiple grains. Spots exclusively owned by one grain are left
    untouched.
    """
    sino_fn = os.path.join(topdir, f'sinos_{n_grains}_{max_n_hkls}_{n_scans}.bin')
    sinos = np.fromfile(sino_fn, dtype=np.float64).reshape(
        n_grains, max_n_hkls, n_scans
    )
    mapping_fn = os.path.join(topdir, f'spotMapping_{n_grains}_{max_n_hkls}_{n_scans}.bin')
    spot_mapping = np.fromfile(mapping_fn, dtype=np.int32).reshape(
        n_grains, max_n_hkls, n_scans
    )
    own_np = ownership.detach().cpu().numpy()
    sid_np = spot_ids.cpu().numpy()
    sid_to_row = {int(sid): i for i, sid in enumerate(sid_np)}

    sid_grain_count = {}
    for g in range(n_grains):
        for h in range(max_n_hkls):
            for s in range(n_scans):
                sid = spot_mapping[g, h, s]
                if sid <= 0:
                    continue
                sid_grain_count.setdefault(sid, set()).add(g)
    shared_sids = {sid for sid, grains in sid_grain_count.items()
                   if len(grains) > 1}
    logger.info(
        "SpotMapping: %d unique spots, %d shared between multiple grains",
        len(sid_grain_count), len(shared_sids),
    )
    weighted = sinos.copy()
    n_reweighted = 0
    n_skipped_unshared = 0
    for g in range(n_grains):
        for h in range(max_n_hkls):
            for s in range(n_scans):
                sid = spot_mapping[g, h, s]
                if sid <= 0:
                    continue
                if sid not in shared_sids:
                    n_skipped_unshared += 1
                    continue
                row = sid_to_row.get(sid)
                if row is None:
                    continue
                w = own_np[row, g]
                weighted[g, h, s] = sinos[g, h, s] * w
                n_reweighted += 1
    logger.info(
        "Re-weighted %d shared-spot sinogram cells, left %d unshared cells untouched",
        n_reweighted, n_skipped_unshared,
    )
    return weighted


def save_sinograms(topdir, sinograms, omegas, nr_hkls):
    """Save sinograms in C-compatible binary format for drop-in replacement."""
    n_grains, max_n_hkls, n_scans = sinograms.shape
    base_fn = f"sinos_{n_grains}_{max_n_hkls}_{n_scans}.bin"
    sinograms.astype(np.float64).tofile(os.path.join(topdir, base_fn))
    raw_fn = f"sinos_raw_{n_grains}_{max_n_hkls}_{n_scans}.bin"
    sinograms.astype(np.float64).tofile(os.path.join(topdir, raw_fn))
    omegas_fn = f"omegas_{n_grains}_{max_n_hkls}.bin"
    omegas.astype(np.float64).tofile(os.path.join(topdir, omegas_fn))
    hkls_fn = f"nrHKLs_{n_grains}.bin"
    nr_hkls.astype(np.int32).tofile(os.path.join(topdir, hkls_fn))
    logger.info(
        "Saved EM sinograms: %s (%d grains, %d HKLs, %d scans)",
        base_fn, n_grains, max_n_hkls, n_scans,
    )


def update_unique_orientations_from_refinement(topdir, n_scans):
    """Re-derive unique grain orientations from pre-tomo refinement results."""
    orig_data = np.genfromtxt(
        require_grain_list(topdir), delimiter=' '
    )
    if orig_data.ndim == 1:
        orig_data = orig_data.reshape(1, -1)
    n_grains = orig_data.shape[0]
    key_data = np.fromfile(
        os.path.join(topdir, 'Output/UniqueIndexSingleKey.bin'),
        dtype=np.uintp, count=n_scans * n_scans * 5
    ).reshape((-1, 5))
    _ = key_data  # for parity with original; not used further
    result_files = glob.glob(os.path.join(topdir, 'Results', 'FitBest_*.csv'))
    if not result_files:
        logger.warning("No refinement results found. Using original orientations.")
        return
    refined = {}
    for rf in result_files:
        try:
            basename = os.path.basename(rf)
            parts = basename.replace('.csv', '').split('_')
            vox_nr = int(parts[1])
            with open(rf, 'r') as f:
                f.readline()
                line2 = f.readline().strip()
                if not line2:
                    continue
                vals = [float(x) for x in line2.split()]
                if len(vals) < 27:
                    continue
                om = np.array(vals[1:10]).reshape(3, 3)
                completeness = vals[26] if len(vals) > 26 else 0.0
                refined[vox_nr] = (om, completeness)
        except (ValueError, IndexError):
            continue
    logger.info("Loaded %d refined voxel orientations", len(refined))
    updated = orig_data.copy()
    n_updated = 0
    for g in range(n_grains):
        best_vox = int(orig_data[g, 1])
        if best_vox in refined:
            om, _comp = refined[best_vox]
            updated[g, 5:14] = om.flatten()
            n_updated += 1
    out_fn = os.path.join(topdir, 'UniqueOrientations_refined.csv')
    np.savetxt(out_fn, updated, fmt='%.10f', delimiter=' ')
    logger.info(
        "Updated %d/%d grain orientations from refinement → %s",
        n_updated, n_grains, out_fn,
    )


def run_em_spot_ownership(topdir, n_scans,
                           n_iter=20, sigma_init=0.1, sigma_min=0.005,
                           sigma_decay=0.85, tol_ome_override=None,
                           tol_eta_override=None,
                           n_opt_steps=5, lr=0.005,
                           refine_orientations=True,
                           use_refined_orientations=True):
    """Run EM spot-ownership and generate weighted sinograms.

    Main entry point. Imports ``EMSpotOwnership`` and ``HEDMForwardModel``
    lazily from ``fwd_sim/`` so a fwd_sim-less environment can still
    import this module.
    """
    import torch
    from em_spot_ownership import EMSpotOwnership

    logger.info("=== EM Spot-Ownership: Starting ===")

    param_file = os.path.join(topdir, 'paramstest.txt')
    params = parse_params_for_em(param_file)
    logger.info(
        "Geometry: Lsd=%s, px=%s, omega_step=%s, tol_ome=%s, tol_eta=%s",
        params['Lsd'], params['px'], params['omega_step'],
        params['tol_ome'], params['tol_eta'],
    )

    obs_spots, obs_rings, scan_numbers, intensities, spot_ids = load_spots(topdir)
    logger.info("Loaded %d observed spots", obs_spots.shape[0])

    refined = use_refined_orientations and os.path.exists(
        os.path.join(topdir, 'UniqueOrientations_refined.csv'))
    orient_matrices, grain_ids = load_grain_orientations(topdir, refined=refined)
    n_grains = orient_matrices.shape[0]
    logger.info("Loaded %d grain orientations (refined=%s)", n_grains, refined)

    orient_tensors = torch.tensor(orient_matrices, dtype=torch.float64)
    positions = torch.zeros(n_grains, 3, dtype=torch.float64)

    model, ring_indices = build_forward_model(topdir, params)
    logger.info(
        "Forward model: %d HKLs, %d unique rings",
        model.hkls.shape[0], ring_indices.unique().shape[0],
    )

    tol_ome_deg = tol_ome_override if tol_ome_override is not None else params['tol_ome']
    tol_eta_deg = tol_eta_override if tol_eta_override is not None else params['tol_eta']
    tol_ome_rad = tol_ome_deg * DEG2RAD
    tol_eta_rad = tol_eta_deg * DEG2RAD
    logger.info("EM tolerances: tol_omega=%s deg, tol_eta=%s deg", tol_ome_deg, tol_eta_deg)

    grain_scan_mask_tensor = None
    sino_fns = glob.glob(os.path.join(topdir, "sinos_*.bin"))
    base_sinos = [f for f in sino_fns
                  if os.path.basename(f).count('_') == 3
                  and os.path.basename(f).split('_')[1].isdigit()]
    if base_sinos and n_scans > 1:
        base_fn_tmp = os.path.basename(base_sinos[0])
        parts_tmp = base_fn_tmp.replace('.bin', '').split('_')
        n_grs_tmp = int(parts_tmp[1])
        max_hkls_tmp = int(parts_tmp[2])
        n_scans_tmp = int(parts_tmp[3])
        sinos_tmp = np.fromfile(base_sinos[0], dtype=np.float64).reshape(
            n_grs_tmp, max_hkls_tmp, n_scans_tmp)
        grain_scan_mask_np = np.any(sinos_tmp > 0, axis=1)
        grain_scan_mask_tensor = torch.tensor(grain_scan_mask_np, dtype=torch.bool)
        logger.info(
            "Grain-scan mask: %s, active cells: %d/%d",
            grain_scan_mask_np.shape,
            grain_scan_mask_np.sum(),
            grain_scan_mask_np.size,
        )
        for g in range(n_grains):
            active_scans = np.where(grain_scan_mask_np[g])[0].tolist()
            logger.info("  Grain %d: present at scans %s", g, active_scans)
    else:
        logger.info("Single-scan or no sinograms found — skipping scan filter")

    em = EMSpotOwnership(
        forward_model=model,
        sigma_init=sigma_init,
        sigma_min=sigma_min,
        sigma_decay=sigma_decay,
        tol_omega=tol_ome_rad,
        tol_eta=tol_eta_rad,
    )

    result = em.fit_from_orient(
        obs_spots=obs_spots,
        orient_matrices=orient_tensors,
        positions=positions,
        obs_rings=obs_rings,
        obs_scan_nrs=scan_numbers,
        grain_scan_mask=grain_scan_mask_tensor,
        n_iter=n_iter,
        verbose=True,
    )

    logger.info(
        "EM converged. Final ownership: %d/%d spots assigned",
        (result.ownership.max(dim=1).values > 0.1).sum().item(),
        obs_spots.shape[0],
    )

    sino_fns = glob.glob(os.path.join(topdir, "sinos_*.bin"))
    base_sinos = [f for f in sino_fns
                  if os.path.basename(f).count('_') == 3
                  and os.path.basename(f).split('_')[1].isdigit()]
    if not base_sinos:
        logger.error("No base sinogram file found from findSingleSolutionPFRefactored")
        return

    base_fn = os.path.basename(base_sinos[0])
    parts = base_fn.replace('.bin', '').split('_')
    n_grs_sino = int(parts[1])
    max_n_hkls = int(parts[2])
    n_scans_sino = int(parts[3])

    weighted = reweight_sinograms_with_ownership(
        topdir, result.ownership, spot_ids,
        n_grs_sino, max_n_hkls, n_scans_sino,
    )

    weighted.astype(np.float64).tofile(os.path.join(topdir, base_fn))
    raw_fn = f"sinos_raw_{n_grs_sino}_{max_n_hkls}_{n_scans_sino}.bin"
    weighted.astype(np.float64).tofile(os.path.join(topdir, raw_fn))
    logger.info("Saved EM-reweighted sinograms to %s and %s", base_fn, raw_fn)

    nr_hkls_arr = np.fromfile(
        os.path.join(topdir, f"nrHKLs_{n_grs_sino}.bin"), dtype=np.int32
    )
    for g in range(n_grains):
        n_sp = nr_hkls_arr[g] if g < len(nr_hkls_arr) else max_n_hkls
        filled_orig = np.sum(
            np.fromfile(os.path.join(topdir, base_fn), dtype=np.float64
                        ).reshape(n_grs_sino, max_n_hkls, n_scans_sino)[g, :n_sp, :] > 0
        )
        filled_w = np.sum(weighted[g, :n_sp, :] > 0)
        total = n_sp * n_scans
        logger.info(
            "  Grain %d: %d HKLs, filled=%d/%d (reweighted), original=%d/%d",
            g, n_sp, filled_w, total, filled_orig, total,
        )

    logger.info("=== EM Spot-Ownership: Complete ===")
