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
    """Read Grains.csv (21 cols: ID + OM[9] + XYZ[3] + strain[6] + radius +
    confidence). Returns arrays plus the header SpaceGroup / Lattice."""
    ids, om, pos, strain, rad, conf = [], [], [], [], [], []
    sg, lattice = None, None
    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            if line.startswith("%"):
                if line.startswith("%\tSpaceGroup:"):
                    sg = int(line.split(":")[1].strip())
                elif line.startswith("%\tLattice"):
                    lattice = tuple(float(x) for x in line.split(":")[1].strip().split())
                continue
            cols = line.split("\t")
            if len(cols) < 21:
                continue
            ids.append(int(cols[0]))
            om.append([float(c) for c in cols[1:10]])
            pos.append([float(c) for c in cols[10:13]])
            strain.append([float(c) for c in cols[13:19]])
            rad.append(float(cols[19]))
            conf.append(float(cols[20]))
    n = len(ids)
    return {
        "n_grains": n,
        "ids": np.array(ids, dtype=np.int64),
        "orient_mat": np.array(om, dtype=np.float64),
        "positions": np.array(pos, dtype=np.float64),
        "strains": np.array(strain, dtype=np.float64),
        "radius": np.array(rad, dtype=np.float64),
        "confidence": np.array(conf, dtype=np.float64),
        "sg": sg, "lattice": lattice,
    }


def load_spot_matrix(path: Path) -> dict:
    """Read SpotMatrix.csv (12 cols: GrainID, SpotID, Omega, DetectorHor,
    DetectorVert, OmeRaw, Eta, RingNr, YLab, ZLab, Theta, StrainError).

    SpotMatrix column-6 "Eta" is a peak-fit diagnostic, NOT the η angle, so we
    recompute η in the model's convention ``atan2(-YLab, ZLab)`` from the
    lab-frame (YLab, ZLab) columns (which are in µm).
    """
    rows = []
    with open(path) as f:
        for line in f:
            if line.startswith("%") or not line.strip():
                continue
            cols = line.rstrip().split("\t")
            if len(cols) < 12:
                continue
            rows.append([float(x) for x in cols])
    arr = np.array(rows, dtype=np.float64)
    eta_from_lab_deg = np.rad2deg(np.arctan2(-arr[:, 8], arr[:, 9]))
    return {
        "grain_id": arr[:, 0].astype(np.int64),
        "spot_id": arr[:, 1].astype(np.int64),
        "omega": arr[:, 2],            # deg
        "det_hor": arr[:, 3],          # px
        "det_vert": arr[:, 4],         # px
        "ome_raw": arr[:, 5],
        "eta": eta_from_lab_deg,       # deg, recomputed from (YLab, ZLab)
        "ring_nr": arr[:, 7].astype(np.int64),
        "y_lab": arr[:, 8],            # µm
        "z_lab": arr[:, 9],            # µm
        "theta": arr[:, 10],           # deg (= 2θ/2)
        "strain_error": arr[:, 11],
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
