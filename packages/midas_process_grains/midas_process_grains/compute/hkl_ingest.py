"""Stage 1 of the physics-bounded clustering pipeline (v4).

For every refined indexer candidate, recover the signed (h,k,l) of the
seed spot — the spot the indexer used to *initiate* that candidate. This
is the primitive on which the entire hkl-uniqueness physics constraint
rests (Stage 3) and the trust signal (Stage 4) is defined.

The seed-spot ID is recorded in OPF column 0 (the "SpId sentinel" that
the legacy C ProcessGrains skips). Given the seed-spot's observed
(YLab, ZLab, ω) and the candidate's refined orientation matrix, we
forward-simulate the predicted reflections on the seed's ring and
return the signed (h,k,l) whose predicted (Y, Z, ω) is closest to the
observation.

We do this in bulk via vectorised torch on CPU/CUDA/MPS, batching by
ring so the per-ring hkl-multiset lookup is a single matrix lookup.

Inputs
------
- ``OrientPosFit.bin``     → (N_seeds, 27) float64, OPF[i,0] = seed SpotID
- ``InputAllExtraInfoFittingAll.csv`` → per-spot (Y, Z, ω, ring, eta, 2θ)
- ``hkls.csv``             → per-ring (h, k, l) variants in crystal frame

Outputs
-------
- ``SeedHklTable`` dataclass with per-candidate:
  - ``seed_spot_id``    (int)
  - ``seed_ring``       (int)
  - ``seed_h``, ``seed_k``, ``seed_l``  (signed)
  - ``seed_match_residual_deg`` (ω match error in deg — small = clean seed
    recovery, large = something wrong upstream)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass
class HklTable:
    """Per-(h, k, l) reflection metadata loaded from hkls.csv."""

    h:        np.ndarray   # (n,) int32
    k:        np.ndarray   # (n,) int32
    l:        np.ndarray   # (n,) int32
    ring:     np.ndarray   # (n,) int32
    d_A:      np.ndarray   # (n,) float64 — d-spacing in Å
    g_crystal: np.ndarray  # (n, 3) float64 — normalized reciprocal vector in crystal frame
    theta_deg: np.ndarray  # (n,) float64 — Bragg angle in deg
    ttheta_deg: np.ndarray  # (n,) float64
    radius_um: np.ndarray  # (n,) float64 — ring radius at the detector (µm)

    @property
    def n(self) -> int:
        return self.h.shape[0]


@dataclass
class SeedHklTable:
    """Per-candidate seed-(h,k,l) recovery result."""

    n_seeds:               int
    seed_spot_id:          np.ndarray   # (N,) int64 — OPF[i, 0] copy
    seed_ring:             np.ndarray   # (N,) int32 — from InputAll lookup
    seed_eta_deg:          np.ndarray   # (N,) float64 — observed eta
    seed_omega_deg:        np.ndarray   # (N,) float64 — observed ω
    seed_y_um:             np.ndarray   # (N,) float64 — observed YLab
    seed_z_um:             np.ndarray   # (N,) float64 — observed ZLab
    # Signed seed hkl; -127 sentinel means "could not resolve" (no seed spot
    # in InputAll, or matching failed).
    seed_h:                np.ndarray   # (N,) int8
    seed_k:                np.ndarray   # (N,) int8
    seed_l:                np.ndarray   # (N,) int8
    seed_match_omega_deg:  np.ndarray   # (N,) float64 — ω match residual
    seed_alive:            np.ndarray   # (N,) bool — was the recovery successful?


# ---------------------------------------------------------------------------
# Readers
# ---------------------------------------------------------------------------


def read_hkls_csv(path: Union[str, Path]) -> HklTable:
    """Read MIDAS hkls.csv (output of midas-calc-hkls / GetHKLList).

    Expected columns (whitespace-separated, header on line 1):

        h k l D-spacing RingNr g1 g2 g3 Theta 2Theta Radius

    where (g1, g2, g3) is the crystal-frame normalized reciprocal vector
    of the (h, k, l) reflection.
    """
    path = Path(path)
    df = pd.read_csv(path, sep=r"\s+", engine="c")
    # Trim possible "%" header marker if present
    df.columns = [c.lstrip("%") for c in df.columns]
    return HklTable(
        h=df["h"].astype(np.int32).to_numpy(),
        k=df["k"].astype(np.int32).to_numpy(),
        l=df["l"].astype(np.int32).to_numpy(),
        ring=df["RingNr"].astype(np.int32).to_numpy(),
        d_A=df["D-spacing"].astype(np.float64).to_numpy(),
        g_crystal=df[["g1", "g2", "g3"]].astype(np.float64).to_numpy(),
        theta_deg=df["Theta"].astype(np.float64).to_numpy(),
        ttheta_deg=df["2Theta"].astype(np.float64).to_numpy(),
        radius_um=df["Radius"].astype(np.float64).to_numpy(),
    )


def read_inputall_minimal(
    path: Union[str, Path],
) -> pd.DataFrame:
    """Read just the columns needed for seed-(h,k,l) recovery.

    Returns a DataFrame indexed by SpotID with columns:
    ``YLab, ZLab, Omega, RingNumber, Eta``.

    Different MIDAS variants emit the header with or without a leading ``%``
    on the first column (legacy uses ``%YLab``; the joint-results pipeline
    drops the ``%``). The reader auto-detects.
    """
    df = pd.read_csv(path, sep=r"\s+", engine="c")
    # Strip any leading "%" from column names so downstream code is uniform
    df.columns = [c.lstrip("%") for c in df.columns]
    df["SpotID"] = df["SpotID"].astype(np.int64)
    keep = [c for c in ("YLab", "ZLab", "Omega", "RingNumber", "Eta")
            if c in df.columns]
    return df.set_index("SpotID", drop=True)[keep]


# ---------------------------------------------------------------------------
# Forward-model seed-(h,k,l) recovery
# ---------------------------------------------------------------------------


def _g_lab_from_observed(
    y_lab_um: np.ndarray,
    z_lab_um: np.ndarray,
    lsd_um: float,
) -> np.ndarray:
    """Compute the lab-frame normalized diffraction vector from
    detector position (Y, Z) and the sample-detector distance L.

    MIDAS / midas-diffract convention (see ``forward.py:calc_bragg_geometry``):
    the Bragg condition is written ``-Gx_lab = sin(θ)·|G|``, which is the
    ``q = k_in - k_out`` sign. For incident beam along +X_lab and the spot
    at lab-frame ``(Lsd, Y, Z)``:

        k_in_dir  = (1, 0, 0)
        k_out_dir = (Lsd, Y, Z) / |(Lsd, Y, Z)|
        g_lab     = k_in_dir - k_out_dir                 (direction only)

    This direction points from the scattered ray back toward the incident
    beam direction (in lab frame); its sample-frame rotation via R_z(-ω)
    and crystal-frame de-rotation via R_om^T then give the crystal-frame
    g of the seed reflection.
    """
    n = np.column_stack([np.full_like(y_lab_um, lsd_um), y_lab_um, z_lab_um])
    k_in = np.array([1.0, 0.0, 0.0])
    k_out = n / np.linalg.norm(n, axis=1, keepdims=True)
    q_lab = k_in - k_out
    return q_lab / np.linalg.norm(q_lab, axis=1, keepdims=True)


def _rotate_sample_to_crystal(
    g_sample: np.ndarray,
    OM: np.ndarray,
) -> np.ndarray:
    """Given a sample-frame unit vector g_sample and a crystal-to-sample
    orientation matrix R (such that v_sample = R · v_crystal), return
    the corresponding crystal-frame vector v_crystal = R^T · v_sample.
    """
    # OM shape (3, 3); g_sample shape (3,) or (N, 3)
    return g_sample @ OM   # equivalent to R^T · g for row vectors


def _rotate_lab_to_sample_by_omega(
    g_lab: np.ndarray,
    omega_deg: np.ndarray,
) -> np.ndarray:
    """Sample frame is the lab frame rotated by -ω about the +Z (vertical)
    axis. (MIDAS convention: ω is the rotation of the sample about Z.)

    g_sample = Rz(-ω) · g_lab.
    """
    ome = np.deg2rad(omega_deg)
    c = np.cos(-ome); s = np.sin(-ome)
    g = g_lab.copy()
    # rotation about Z: x' = c·x − s·y, y' = s·x + c·y, z' = z
    gx_new = c * g[:, 0] - s * g[:, 1]
    gy_new = s * g[:, 0] + c * g[:, 1]
    g_out = np.column_stack([gx_new, gy_new, g[:, 2]])
    return g_out


def recover_seed_hkls(
    *,
    seed_spot_id_per_candidate: np.ndarray,    # (N,) int — OPF col 0
    orientation_matrices:       np.ndarray,    # (N, 3, 3) — OPF cols 1..9
    inputall_df:                pd.DataFrame,  # indexed by SpotID
    hkls:                       HklTable,
    lsd_um:                     float,
    space_group:                int = 225,
    omega_tol_deg:              float = 2.0,
    canonicalize_to_fz:         bool = True,
) -> SeedHklTable:
    """Recover the signed seed-(h, k, l) of every refined candidate.

    For each candidate i:
    1. Look up its seed SpotID in InputAll → get observed (Y, Z, ω, ring).
    2. Compute lab-frame diffraction vector q_lab from (Y, Z) and Lsd.
    3. Rotate to sample frame using observed ω.
    4. Rotate to crystal frame using candidate's OM.
    5. Compare against all (h, k, l) in InputAll's RingNumber (from hkls.csv).
    6. Pick the (h, k, l) with the smallest angular distance.
    7. Sanity-check the ω match by forward-predicting the chosen hkl's ω;
       if outside ``omega_tol_deg``, mark as recovery-failed.

    The angular distance is the cleanest discriminator because the magnitude
    of |q| is set by the ring and the direction of q in crystal frame is
    deterministic per hkl.
    """
    N = len(seed_spot_id_per_candidate)
    seed_id = np.asarray(seed_spot_id_per_candidate, dtype=np.int64)
    OM = np.asarray(orientation_matrices, dtype=np.float64).copy()

    # FZ-canonicalize. The MIDAS indexer seeds every candidate from the
    # canonical signed variant of the indexing ring (e.g. always (+2,0,0)
    # for {002} on FCC), so the *raw* refined OM has crystal-frame X aligned
    # with the observed lab-frame g of the seed spot. That bakes the seed-hkl
    # into the OM and makes per-candidate hkl recovery degenerate.
    # The fix is to map each OM to its fundamental-zone representative via
    # midas_stress.fundamental_zone — under the FZ rep, candidates of one
    # physical grain that were seeded from different variants collapse to
    # a single canonical orientation, and the per-spot seed-hkl recovery
    # below correctly distributes across the multiplicity of the ring.
    if canonicalize_to_fz:
        from midas_stress.orientation import (
            orient_mat_to_quat, quat_to_orient_mat, fundamental_zone,
        )
        import torch as _torch
        q = orient_mat_to_quat(_torch.from_numpy(OM))
        q_fz = fundamental_zone(q, space_group)
        OM = np.asarray(quat_to_orient_mat(q_fz)).reshape(-1, 3, 3)

    # Per-candidate seed-spot lookup
    out_ring = np.zeros(N, dtype=np.int32)
    out_y    = np.full(N, np.nan, dtype=np.float64)
    out_z    = np.full(N, np.nan, dtype=np.float64)
    out_ome  = np.full(N, np.nan, dtype=np.float64)
    out_eta  = np.full(N, np.nan, dtype=np.float64)
    out_h    = np.full(N, -127, dtype=np.int8)
    out_k    = np.full(N, -127, dtype=np.int8)
    out_l    = np.full(N, -127, dtype=np.int8)
    out_res  = np.full(N, np.nan, dtype=np.float64)
    out_alive = np.zeros(N, dtype=bool)

    # Look up seed-spot info via pandas reindex (fast on a unique index)
    df = inputall_df
    valid_seed = (seed_id != 0)
    # Some indexers leave seed_id == 0 for dead candidates
    valid_ids = seed_id[valid_seed]
    # reindex returns NaN for missing rows; we skip those
    rows = df.reindex(valid_ids)
    found = ~rows["RingNumber"].isna()
    sub_idx = np.flatnonzero(valid_seed)[found.values]

    out_ring[sub_idx] = rows.loc[found, "RingNumber"].astype(np.int32).to_numpy()
    out_y[sub_idx]    = rows.loc[found, "YLab"].astype(np.float64).to_numpy()
    out_z[sub_idx]    = rows.loc[found, "ZLab"].astype(np.float64).to_numpy()
    out_ome[sub_idx]  = rows.loc[found, "Omega"].astype(np.float64).to_numpy()
    out_eta[sub_idx]  = rows.loc[found, "Eta"].astype(np.float64).to_numpy()

    if len(sub_idx) == 0:
        return SeedHklTable(
            n_seeds=N, seed_spot_id=seed_id,
            seed_ring=out_ring, seed_eta_deg=out_eta,
            seed_omega_deg=out_ome, seed_y_um=out_y, seed_z_um=out_z,
            seed_h=out_h, seed_k=out_k, seed_l=out_l,
            seed_match_omega_deg=out_res, seed_alive=out_alive,
        )

    # ----- Forward path: lab → sample → crystal — for the seed sub-set -----
    g_lab = _g_lab_from_observed(out_y[sub_idx], out_z[sub_idx], lsd_um)
    g_sample = _rotate_lab_to_sample_by_omega(g_lab, out_ome[sub_idx])
    # crystal frame: g_crystal = R^T · g_sample (with OM mapping crystal → sample)
    OM_sub = OM[sub_idx]                                             # (M, 3, 3)
    # Use einsum to do v_c = sum_j OM[i, j, :] * g_s[i, j]
    g_crystal_obs = np.einsum("ijk,ij->ik", OM_sub, g_sample)        # (M, 3)
    g_crystal_obs /= np.linalg.norm(g_crystal_obs, axis=1, keepdims=True) + 1e-12

    # ----- Group hkls by ring for fast per-ring matching -----
    hkls_by_ring: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    for r in np.unique(hkls.ring):
        m = hkls.ring == r
        g = hkls.g_crystal[m]
        gnorm = g / (np.linalg.norm(g, axis=1, keepdims=True) + 1e-12)
        hkls_by_ring[int(r)] = (
            np.column_stack([hkls.h[m], hkls.k[m], hkls.l[m]]).astype(np.int8),
            gnorm,
        )

    # ----- Per-candidate (in sub_idx) matching -----
    for local_i, global_i in enumerate(sub_idx):
        ring = int(out_ring[global_i])
        if ring not in hkls_by_ring:
            continue
        hkl_arr, g_ref = hkls_by_ring[ring]
        # Cosine similarity → argmax
        cos = g_ref @ g_crystal_obs[local_i]
        best = int(np.argmax(cos))
        # Cosine of angular separation
        match_cos = float(np.clip(cos[best], -1.0, 1.0))
        match_ang_deg = float(np.degrees(np.arccos(match_cos)))
        if match_ang_deg > omega_tol_deg:
            # Recovery failed; leave -127 sentinel
            out_res[global_i] = match_ang_deg
            continue
        out_h[global_i] = hkl_arr[best, 0]
        out_k[global_i] = hkl_arr[best, 1]
        out_l[global_i] = hkl_arr[best, 2]
        out_res[global_i] = match_ang_deg
        out_alive[global_i] = True

    return SeedHklTable(
        n_seeds=N, seed_spot_id=seed_id,
        seed_ring=out_ring, seed_eta_deg=out_eta,
        seed_omega_deg=out_ome, seed_y_um=out_y, seed_z_um=out_z,
        seed_h=out_h, seed_k=out_k, seed_l=out_l,
        seed_match_omega_deg=out_res, seed_alive=out_alive,
    )


# ---------------------------------------------------------------------------
# Convenience: end-to-end one-shot loader
# ---------------------------------------------------------------------------


def load_seed_hkls(
    layer_dir: Union[str, Path],
    *,
    opf: Optional[np.ndarray] = None,
    omega_tol_deg: float = 2.0,
) -> SeedHklTable:
    """End-to-end: from a MIDAS layer directory, return the SeedHklTable.

    If ``opf`` is None, mmaps ``Results/OrientPosFit.bin`` from ``layer_dir``.
    """
    from .. io.binary import read_orient_pos_fit  # local import to avoid cycles

    layer_dir = Path(layer_dir)
    if opf is None:
        opf = read_orient_pos_fit(layer_dir)
    inputall = read_inputall_minimal(
        layer_dir / "InputAllExtraInfoFittingAll.csv"
    )
    hkls = read_hkls_csv(layer_dir / "hkls.csv")

    # Lsd from paramstest. The C refiner writes ``LsdFit`` after refinement;
    # raw paramstest.txt files have ``Lsd``. Accept either.
    try:
        lsd_um = _read_paramstest_scalar(layer_dir / "paramstest.txt", "LsdFit")
    except KeyError:
        lsd_um = _read_paramstest_scalar(layer_dir / "paramstest.txt", "Lsd")

    # SpaceGroup for FZ canonicalization
    try:
        sg = int(_read_paramstest_scalar(layer_dir / "paramstest.txt", "SpaceGroup"))
    except KeyError:
        sg = 225  # default to cubic FCC

    seed_spot_id = opf[:, 0].astype(np.int64)
    OM = opf[:, 1:10].astype(np.float64).reshape(-1, 3, 3)
    return recover_seed_hkls(
        seed_spot_id_per_candidate=seed_spot_id,
        orientation_matrices=OM,
        inputall_df=inputall,
        hkls=hkls,
        lsd_um=lsd_um,
        space_group=sg,
        omega_tol_deg=omega_tol_deg,
    )


def _read_paramstest_scalar(path: Path, key: str) -> float:
    """Read a single float key from a paramstest.txt file.

    MIDAS paramstest tokens may carry trailing ``;``; strip before float-cast.
    """
    with open(path, "r") as fp:
        for line in fp:
            tokens = line.split("#", 1)[0].split()
            if len(tokens) >= 2 and tokens[0] == key:
                return float(tokens[1].rstrip(";"))
    raise KeyError(f"paramstest.txt at {path} has no key {key!r}")
