"""Shared loaders that turn MIDAS Phase-2 grain outputs (Grains.csv +
SpotMatrix.csv) into the per-grain ``ObservedSpots`` / ``MatchResult`` objects
the HEDM residual consumes.

Extracted from ``runners/run_real_phase3_joint.py`` so the joint runner *and*
the lightweight grain-geometry refiner (:mod:`grain_refine`) share one
definition (no dual tree). The functions here are pure I/O + the
forward-then-associate step that initialises the spot matching.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from midas_fit_grain.matching import MatchResult, associate, ring_slot_lookup
from midas_fit_grain.observations import ObservedSpots
# The canonical, name-resolving readers for the two ProcessGrains artefacts.
# Both files have been widened repeatedly (Grains 19 -> 21 -> 47 -> 53,
# SpotMatrix 12 -> 28) and are written under two header tokens (%ID and
# %GrainID). Every positional reader in the tree froze one snapshot of that and
# drifted silently; see the module docstring of midas_process_grains.io.read.
from midas_process_grains.io import read_grains_csv as _read_grains_csv
from midas_process_grains.io import read_spot_matrix as _read_spot_matrix


def euler_zxz_from_om(R: np.ndarray) -> np.ndarray:
    """Inverse of ``midas_diffract.HEDMForwardModel.euler2mat`` (ZXZ).

    ``R`` has shape (3, 3) (or 9,); returns (phi1, Phi, phi2) in radians.
    """
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    if abs(R[2, 2]) < 1.0 - 1e-9:
        phi1 = np.arctan2(R[0, 2], -R[1, 2])
        Phi = np.arccos(np.clip(R[2, 2], -1.0, 1.0))
        phi2 = np.arctan2(R[2, 0], R[2, 1])
    else:  # Gimbal lock at Phi = 0 or π
        Phi = 0.0 if R[2, 2] > 0 else np.pi
        phi1 = np.arctan2(R[1, 0], R[0, 0])
        phi2 = 0.0
    return np.array([phi1, Phi, phi2])


def load_grains_csv(path: Path) -> dict:
    """Read a ``Grains.csv`` of any width (19 / 21 / 23 / 47 / 53 columns) and
    either header token (``%ID`` or ``%GrainID``).

    Every column is resolved BY NAME through
    :func:`midas_process_grains.io.read_grains_csv`. This used to be a
    positional parser that hard-coded the 21-column legacy layout behind a
    ``len(cols) < 21`` guard — which passes on a 53-column file, so it never
    raised. On anything wider than 21 columns it returned column 19
    (``DiffPos``) as ``radius`` and column 20 (``DiffOme``) as ``confidence``.
    Measured on a 47-column Ti-7Al set (``bt_20id_jul26b``, 208 grains):
    median radius 286.9 µm instead of 25.2 µm, median confidence 0.117
    instead of 0.580.

    The confidence error was not cosmetic. :mod:`grain_refine` SELECTS grains
    with ``np.argsort(-grains["confidence"])``, so it was ranking on DiffOme
    descending — i.e. deliberately keeping the *worst-fitting* grains. On that
    same file the top-10 selection shares only 1 grain with the correct one.
    A separate real-data run reported the resulting refined ``tx`` moving by
    ~10 % and ``Wedge`` changing sign.

    Returned keys are unchanged so callers do not move; three are added:

    ``strains``
        (n, 6) Voigt strain, DIMENSIONLESS, ordered
        ``[e11, e22, e33, e23, e13, e12]``. Taken from the Kenesei
        (lab/sample-frame) 3x3 block and divided by 1e6 on the 47/53-column
        files, which store it in microstrain; taken from the legacy
        ``E11..E23`` block verbatim on a genuine 21-column file (that
        generation never recorded its units). Cols 13-18 are the LATTICE on
        anything wider than 21 columns — reading them as strain is the bug
        this docstring exists to prevent.
    ``strain_ken_ue`` / ``strain_fab_ue``
        (n, 3, 3) microstrain, or ``None`` when the file predates them.
    ``lattice_per_grain``
        (n, 6) per-grain ``a b c alpha beta gamma``, or ``None`` on a legacy
        21-column file. Distinct from ``lattice``, which is the single
        reference lattice from the ``%`` preamble.
    """
    t = _read_grains_csv(path)
    n = t.n_grains

    # Voigt pack in the [11, 22, 33, 23, 13, 12] order the downstream
    # deviatoric-norm helpers document (NOT the legacy header's
    # E11 E22 E33 E12 E13 E23 order — that mismatch is itself a live bug).
    if t.strain_ken is not None:
        k = t.strain_ken
        strains = np.column_stack([
            k[:, 0, 0], k[:, 1, 1], k[:, 2, 2],
            k[:, 1, 2], k[:, 0, 2], k[:, 0, 1],
        ]) * 1e-6                      # microstrain -> dimensionless
    elif t.strain_voigt is not None:
        strains = np.asarray(t.strain_voigt, dtype=np.float64)
    else:
        strains = np.zeros((n, 6), dtype=np.float64)

    zeros = np.zeros(n, dtype=np.float64)
    return {
        "n_grains": n,
        "ids": np.asarray(t.ids, dtype=np.int64),
        # (n, 9) row-major, as before: euler_zxz_from_om reshapes to (3, 3).
        "orient_mat": np.asarray(t.orient_mat, dtype=np.float64).reshape(n, 9),
        "positions": (np.zeros((n, 3), dtype=np.float64) if t.positions is None
                      else np.asarray(t.positions, dtype=np.float64)),
        "strains": strains,
        "radius": (zeros if t.grain_radius is None
                   else np.asarray(t.grain_radius, dtype=np.float64)),
        "confidence": (zeros if t.confidence is None
                       else np.asarray(t.confidence, dtype=np.float64)),
        # Kept as a plain tuple / None: grain_refine does `grains["lattice"] or
        # tuple(v1.LatticeConstant)`, which an ndarray would turn into a
        # "truth value of an array is ambiguous" ValueError.
        "sg": t.space_group,
        "lattice": (tuple(t.lattice_parameter)
                    if t.lattice_parameter is not None else None),
        "strain_ken_ue": t.strain_ken,
        "strain_fab_ue": t.strain_fab,
        "lattice_per_grain": t.lattice,
    }


def load_spot_matrix(path: Path) -> dict:
    """Read a ``SpotMatrix.csv`` of either width (12 legacy / 28 expanded).

    Columns are resolved by name through
    :func:`midas_process_grains.io.read_spot_matrix`, which also drops the
    ``Matched == 0`` rows by default. Those rows are reflections the grain was
    PREDICTED to produce and that were never observed: they carry ``-1`` in
    ``SpotID``/``RingNr`` and NaN in every observed column, and they are ~3.3 %
    of rows on real data. Left in, they reached
    :func:`build_observations_and_matches` and raised ``KeyError: -1`` out of
    the ring-slot lookup; had they got past that, η would have been recomputed
    from NaN ``YLab``/``ZLab``.

    SpotMatrix column-6 "Eta" is a peak-fit diagnostic, NOT the η angle, so we
    recompute η in the model's convention ``atan2(-YLab, ZLab)`` from the
    lab-frame (YLab, ZLab) columns (which are in µm).
    """
    t = _read_spot_matrix(path, matched_only=True)
    eta_from_lab_deg = np.rad2deg(np.arctan2(-t.y_lab, t.z_lab))
    return {
        "grain_id": t.grain_id,
        "spot_id": t.spot_id,
        "omega": t.omega,              # deg
        "det_hor": t.detector_hor,     # px
        "det_vert": t.detector_vert,   # px
        "ome_raw": t.ome_raw,
        "eta": eta_from_lab_deg,       # deg, recomputed from (YLab, ZLab)
        "ring_nr": t.ring_nr,
        "y_lab": t.y_lab,              # µm
        "z_lab": t.z_lab,              # µm
        "theta": t.theta,              # deg (= 2θ/2)
        "strain_error": t.strain_error,
        # Provenance for the completeness deficit that was silently dropped.
        "n_rows_total": t.n_rows_total,
        "n_rows_unmatched": t.n_rows_unmatched,
    }


def grain_lattice_from_reference(grains: dict) -> np.ndarray:
    """Per-grain reference lattice (the header lattice, tiled). The geometry
    pass keeps strain frozen; per-grain strain is refined separately."""
    n = grains["n_grains"]
    lat0 = (np.array(grains["lattice"], dtype=np.float64)
            if grains["lattice"] is not None else np.zeros(6))
    return np.tile(lat0[None, :], (n, 1))


def load_ring_two_theta(hkls_csv: Path) -> Dict[int, float]:
    """Map ring-number → 2θ (deg) from a MIDAS hkls.csv (col 4 = ring,
    col 9 = 2θ)."""
    ring_two_theta: Dict[int, float] = {}
    with open(hkls_csv) as f:
        next(f)  # header
        for line in f:
            cols = line.split()
            if len(cols) < 11:
                continue
            rn = int(cols[4]); tt = float(cols[9])
            ring_two_theta.setdefault(rn, tt)
    return ring_two_theta


def load_phase2_grains_and_spots(layer_dir: Path):
    """Read Grains.csv + SpotMatrix.csv from a Phase-2 layer dir.

    Returns ``(grain_eulers (n,3) rad, positions (n,3) µm, lattices (n,6),
    spots_per_grain, grains_dict, spot_dict)``.
    """
    grains_csv = layer_dir / "Grains.csv"
    spot_csv = layer_dir / "SpotMatrix.csv"
    if not grains_csv.exists() or not spot_csv.exists():
        raise FileNotFoundError(
            f"Phase-2 outputs not found in {layer_dir}: need Grains.csv + "
            f"SpotMatrix.csv (process_grains stage)")
    g = load_grains_csv(grains_csv)
    s = load_spot_matrix(spot_csv)
    grain_eulers = np.zeros((g["n_grains"], 3))
    for i in range(g["n_grains"]):
        grain_eulers[i] = euler_zxz_from_om(g["orient_mat"][i])

    grain_id_to_idx = {int(gid): i for i, gid in enumerate(g["ids"])}
    spots_per_grain: List[Dict[str, np.ndarray]] = [{} for _ in range(g["n_grains"])]
    for k, gid in enumerate(s["grain_id"]):
        i = grain_id_to_idx.get(int(gid))
        if i is None:
            continue
        bag = spots_per_grain[i]
        for col in ("spot_id", "y_lab", "z_lab", "omega", "eta", "ring_nr",
                    "theta", "det_hor", "det_vert"):
            bag.setdefault(col, []).append(s[col][k])
    for i in range(g["n_grains"]):
        for col, vals in spots_per_grain[i].items():
            spots_per_grain[i][col] = np.array(vals)
    return (grain_eulers, g["positions"], grain_lattice_from_reference(g),
            spots_per_grain, g, s)


def _empty_observation() -> ObservedSpots:
    z_i = torch.zeros(0, dtype=torch.int64)
    z_f = torch.zeros(0, dtype=torch.float64)
    return ObservedSpots(
        spot_id=z_i, ring_nr=z_i, y_lab=z_f, z_lab=z_f, omega=z_f, eta=z_f,
        two_theta=z_f, grain_radius=z_f, fit_rmse=z_f, y_orig=z_f, z_orig=z_f,
        omega_ini=z_f, mask_touched=torch.zeros(0, dtype=torch.bool),
    )


def _empty_match() -> MatchResult:
    z_i = torch.zeros(0, dtype=torch.int64)
    z_f = torch.zeros(0, dtype=torch.float64)
    return MatchResult(k_idx=z_i, m_idx=z_i, mask=torch.zeros(0, dtype=torch.bool),
                       delta_omega=z_f, delta_eta=z_f)


def build_observations_and_matches(
    model,
    spots: List[Dict[str, np.ndarray]],
    grain_eulers_init: np.ndarray,
    grain_pos_init: np.ndarray,
    grain_lat_init: np.ndarray,
    grain_radius_um: np.ndarray,
    ring_two_theta_by_ring: Dict[int, float],
    *,
    omega_tol_deg: float = 2.0,
    eta_tol_deg: float = 3.0,
    ring_match_tol_deg: float = 0.05,
) -> Tuple[List[ObservedSpots], List[MatchResult]]:
    """Build per-grain ``ObservedSpots`` and an initial ``MatchResult`` by
    forward-modelling each grain at its init pose and associating predicted
    with observed spots.

    ``model`` is a ``midas_diffract.HEDMForwardModel``. The predicted ring slot
    for each model reflection is the nearest observed-ring 2θ within
    ``ring_match_tol_deg`` (else -1).
    """
    n = len(spots)
    obs_ring_nrs = sorted({int(r) for bag in spots for r in bag.get("ring_nr", [])})
    if len(obs_ring_nrs) == 0:
        raise RuntimeError("No observed rings in any grain's spot bag")
    ring_tt_arr = np.array([ring_two_theta_by_ring[r] for r in obs_ring_nrs],
                           dtype=np.float64)
    pred_tt_deg = np.rad2deg(2 * model.thetas.detach().cpu().numpy())
    diffs = np.abs(pred_tt_deg[:, None] - ring_tt_arr[None, :])
    nearest = diffs.argmin(axis=1)
    nearest_d = diffs[np.arange(diffs.shape[0]), nearest]
    pred_ring_slot = torch.from_numpy(
        np.where(nearest_d < ring_match_tol_deg, nearest, -1)).long()

    observations: List[ObservedSpots] = []
    matches: List[MatchResult] = []
    for g in range(n):
        bag = spots[g]
        if "spot_id" not in bag or len(bag["spot_id"]) == 0:
            observations.append(_empty_observation())
            matches.append(_empty_match())
            continue
        S = len(bag["spot_id"])
        ring_nr = torch.from_numpy(bag["ring_nr"]).long()
        omega_rad = torch.from_numpy(np.deg2rad(bag["omega"])).double()
        eta_rad = torch.from_numpy(np.deg2rad(bag["eta"])).double()
        theta_rad = torch.from_numpy(np.deg2rad(bag["theta"])).double()
        two_theta = 2.0 * theta_rad
        y_lab = torch.from_numpy(bag["y_lab"]).double()
        z_lab = torch.from_numpy(bag["z_lab"]).double()
        observations.append(ObservedSpots(
            spot_id=torch.from_numpy(bag["spot_id"]).long(),
            ring_nr=ring_nr, y_lab=y_lab, z_lab=z_lab,
            omega=omega_rad, eta=eta_rad, two_theta=two_theta,
            grain_radius=torch.full((S,), float(grain_radius_um[g]), dtype=torch.float64),
            fit_rmse=torch.zeros(S, dtype=torch.float64),
            y_orig=y_lab.clone(), z_orig=z_lab.clone(),
            omega_ini=omega_rad.clone(),
            mask_touched=torch.zeros(S, dtype=torch.bool),
        ))
        eu_g = torch.from_numpy(grain_eulers_init[g][None, None, :]).double()
        po_g = torch.from_numpy(grain_pos_init[g][None, None, :]).double()
        la_g = torch.from_numpy(grain_lat_init[g][None, :]).double()
        pred = model(eu_g, po_g, lattice_params=la_g)
        pred_omega = pred.omega.squeeze(0).squeeze(0).double()
        pred_eta = pred.eta.squeeze(0).squeeze(0).double()
        pred_valid = pred.valid.squeeze(0).squeeze(0).bool()
        obs_slot = ring_slot_lookup(obs_ring_nrs, ring_nr)
        matches.append(associate(
            obs_ring_nr=ring_nr, obs_omega=omega_rad, obs_eta=eta_rad,
            pred_ring_slot=pred_ring_slot, pred_omega=pred_omega,
            pred_eta=pred_eta, pred_valid=pred_valid, obs_ring_slot=obs_slot,
            omega_tolerance=math.radians(omega_tol_deg),
            eta_tolerance=math.radians(eta_tol_deg),
        ))
    return observations, matches


__all__ = [
    "euler_zxz_from_om",
    "load_grains_csv",
    "load_spot_matrix",
    "grain_lattice_from_reference",
    "load_ring_two_theta",
    "load_phase2_grains_and_spots",
    "build_observations_and_matches",
]
