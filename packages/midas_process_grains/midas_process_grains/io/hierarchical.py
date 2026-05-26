"""Hierarchical grain table emitter (v4).

Combines outputs from Stages 1+2+3+4+5+6 into a single
``GrainsV4.csv`` and a JSON sidecar that exposes the parent-child
relationships (twin families, sub-grain pairs).

Output schema
=============

Leaf table (``GrainsV4.csv``): one row per primitive grain (leaf node).

  GrainID, RepSeed, X, Y, Z, O11..O33, eps_11..eps_23,
  GrainRadius_naive, GrainRadius_NNLS,
  Confidence,
  hkl_n_observed, hkl_n_expected, hkl_coverage,
  hkl_dup_count, splits_emerged_from,
  trust_tier_strict, trust_tier_loose,
  twin_partner_id, twin_type, twin_family_id,
  subgrain_partner_id

Parent / rollup table (``GrainsV4_families.csv``): one row per
**parent grain** — either a twin family (multiple leaves sharing a
``twin_family_id``) OR a singleton leaf that is in no family. This is
the "coarse-grained grain" view: iterating this file gives one entry
per physical grain after collapsing twin variants. Schema:

  ParentID, ParentType (twin / singleton), MemberCount, MemberGrainIDs,
  X_um, Y_um, Z_um,                 # volume-weighted mean over members
  O11..O33,                          # rotation-mean OM (hemisphere-aligned, then renormalized)
  Confidence,                        # member-mean
  TotalVolume_NNLS_um3, EquivalentRadius_um,
  trust_tier_strict, trust_tier_loose,  # max of member tiers
  hkl_coverage                       # max of member coverages

JSON sidecar (``GrainsV4.meta.json``): full metadata —
- algorithm version + git SHA (if available)
- per-stage timings
- trust scheme parameters
- indexing-ring / geometry parameters
- summary counts (n_leaf, n_twin_families, n_subgrain_pairs)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd


@dataclass
class V4HierarchicalEmissionConfig:
    """Knobs for the emitter."""

    package_version: str = "0.4.0-v4"
    trust_scheme: str = "strict"
    write_extended_columns: bool = True


def emit_v4_grains(
    out_dir: Union[str, Path],
    *,
    # Per-leaf-grain core
    rep_seed_per_grain: np.ndarray,        # (N_g,) — seed index of the cluster rep
    positions_um:       np.ndarray,        # (N_g, 3)
    orient_mats_3x3:    np.ndarray,        # (N_g, 3, 3)
    strain_voigt:       Optional[np.ndarray] = None,   # (N_g, 6) — ε11,ε22,ε33,ε12,ε13,ε23
    # Trust + physics from Stages 3/4
    hkl_n_observed:        np.ndarray = None,    # (N_g,) int — matched-based (primary)
    hkl_n_observed_seed:   Optional[np.ndarray] = None,  # (N_g,) int — seed-based (diagnostic)
    hkl_n_expected:        np.ndarray = None,    # (N_g,) int
    hkl_coverage:          np.ndarray = None,    # (N_g,) float
    hkl_dup_count:         np.ndarray = None,    # (N_g,) int
    splits_emerged_from:   np.ndarray = None,    # (N_g,) int
    trust_tier_strict:     np.ndarray = None,    # (N_g,) int8
    trust_tier_loose:      np.ndarray = None,    # (N_g,) int8
    trust_tier_sigma_aware: Optional[np.ndarray] = None,  # (N_g,) int8, optional
    confidence:            Optional[np.ndarray] = None,   # (N_g,) float
    # Stage 5 labels
    twin_partner_id:       Optional[np.ndarray] = None,
    twin_family_id:        Optional[np.ndarray] = None,
    twin_type:             Optional[List[str]] = None,
    subgrain_partner_id:   Optional[np.ndarray] = None,
    # Stage 6 volumes
    radius_naive_um:       Optional[np.ndarray] = None,
    radius_nnls_um:        Optional[np.ndarray] = None,
    radius_disc_um:        Optional[np.ndarray] = None,    # √(V/π) — lateral R for thin foils
    volume_nnls_um3:       Optional[np.ndarray] = None,
    sigma_R_nnls_um:       Optional[np.ndarray] = None,
    # Stage 7 position uncertainty (optional; midas_propagate)
    sigma_X_um:            Optional[np.ndarray] = None,
    sigma_Y_um:            Optional[np.ndarray] = None,
    sigma_Z_um:            Optional[np.ndarray] = None,
    n_spots_matched:       Optional[np.ndarray] = None,
    sigma_residual_rms_px: Optional[np.ndarray] = None,
    # Stage 8.5 drop policy
    drop_by_budget:        Optional[np.ndarray] = None,    # (N_g,) bool — per-grain v1
    drop_by_budget_family: Optional[np.ndarray] = None,    # (N_g,) bool — family-aware v2
    volume_recovery:       Optional[np.ndarray] = None,    # (N_g,) float = V_NNLS/V_naive
    family_quality:        Optional[np.ndarray] = None,    # (N_g,) float — family rank score
    family_V_um3:          Optional[np.ndarray] = None,    # (N_g,) float — V of the grain's family (max-aggregated)
    # Provenance
    cfg: Optional[V4HierarchicalEmissionConfig] = None,
    meta_extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Path]:
    """Write the v4 hierarchical grain output.

    Returns paths to the three emitted files.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = cfg or V4HierarchicalEmissionConfig()
    N_g = rep_seed_per_grain.shape[0]

    def _coerce(arr, default_dtype):
        if arr is None:
            return np.full(N_g, np.nan if default_dtype == float else -1, dtype=default_dtype)
        return np.asarray(arr)

    hkl_n_observed       = _coerce(hkl_n_observed,    np.int32)
    hkl_n_observed_seed  = _coerce(hkl_n_observed_seed, np.int32)
    hkl_n_expected       = _coerce(hkl_n_expected,    np.int32)
    hkl_coverage         = _coerce(hkl_coverage,      float)
    hkl_dup_count        = _coerce(hkl_dup_count,     np.int32)
    splits_emerged_from  = _coerce(splits_emerged_from, np.int32)
    trust_tier_strict    = _coerce(trust_tier_strict, np.int8)
    trust_tier_loose     = _coerce(trust_tier_loose,  np.int8)
    trust_tier_sigma_aware = _coerce(trust_tier_sigma_aware, np.int8)
    confidence           = _coerce(confidence,        float)
    twin_partner_id      = _coerce(twin_partner_id,   np.int64)
    twin_family_id       = _coerce(twin_family_id,    np.int64)
    twin_type            = twin_type if twin_type is not None else [""] * N_g
    subgrain_partner_id  = _coerce(subgrain_partner_id, np.int64)
    radius_naive_um      = _coerce(radius_naive_um,   float)
    radius_nnls_um       = _coerce(radius_nnls_um,    float)
    radius_disc_um       = _coerce(radius_disc_um,    float)
    volume_nnls_um3      = _coerce(volume_nnls_um3,   float)
    sigma_R_nnls_um      = _coerce(sigma_R_nnls_um,   float)
    sigma_X_um           = _coerce(sigma_X_um,        float)
    sigma_Y_um           = _coerce(sigma_Y_um,        float)
    sigma_Z_um           = _coerce(sigma_Z_um,        float)
    n_spots_matched      = _coerce(n_spots_matched,   np.int32)
    drop_by_budget       = (np.zeros(N_g, dtype=bool) if drop_by_budget is None
                            else np.asarray(drop_by_budget, dtype=bool))
    drop_by_budget_family = (np.zeros(N_g, dtype=bool) if drop_by_budget_family is None
                             else np.asarray(drop_by_budget_family, dtype=bool))
    volume_recovery      = _coerce(volume_recovery,   float)
    family_quality       = _coerce(family_quality,    float)
    family_V_um3         = _coerce(family_V_um3,      float)
    sigma_residual_rms_px= _coerce(sigma_residual_rms_px, float)

    if strain_voigt is None:
        strain_voigt = np.full((N_g, 6), np.nan, dtype=float)

    # ---- Leaf grain table ----
    leaf_cols = {
        "GrainID":              np.arange(N_g, dtype=np.int64),
        "RepSeed":              rep_seed_per_grain.astype(np.int64),
        "X":                    positions_um[:, 0],
        "Y":                    positions_um[:, 1],
        "Z":                    positions_um[:, 2],
    }
    for i in range(3):
        for j in range(3):
            leaf_cols[f"O{i+1}{j+1}"] = orient_mats_3x3[:, i, j]
    for ii, name in enumerate(("eps_11", "eps_22", "eps_33", "eps_12", "eps_13", "eps_23")):
        leaf_cols[name] = strain_voigt[:, ii]
    leaf_cols.update({
        "GrainRadius_naive":      radius_naive_um,
        "GrainRadius_NNLS":       radius_nnls_um,
        "GrainRadius_disc_um":    radius_disc_um,
        "sigma_R_NNLS_um":        sigma_R_nnls_um,
        "Confidence":             confidence,
        "sigma_X_um":             sigma_X_um,
        "sigma_Y_um":             sigma_Y_um,
        "sigma_Z_um":             sigma_Z_um,
        "n_spots_matched":        n_spots_matched,
        "sigma_residual_rms_px":  sigma_residual_rms_px,
        "hkl_n_observed":         hkl_n_observed,         # matched-based (primary)
        "hkl_n_observed_seed":    hkl_n_observed_seed,    # seed-based (diagnostic)
        "hkl_n_expected":         hkl_n_expected,
        "hkl_coverage":           hkl_coverage,
        "hkl_dup_count":          hkl_dup_count,
        "splits_emerged_from":    splits_emerged_from,
        "trust_tier_strict":      trust_tier_strict,
        "trust_tier_loose":       trust_tier_loose,
        "trust_tier_sigma_aware": trust_tier_sigma_aware,
        "twin_partner_id":        twin_partner_id,
        "twin_type":              twin_type,
        "twin_family_id":         twin_family_id,
        "subgrain_partner_id":    subgrain_partner_id,
        "volume_recovery":        volume_recovery,
        "drop_by_budget":         drop_by_budget.astype(np.int8),
        "drop_by_budget_family":  drop_by_budget_family.astype(np.int8),
        "family_quality":         family_quality,
        "family_V_um3":           family_V_um3,
    })
    leaf_df = pd.DataFrame(leaf_cols)
    leaf_path = out_dir / "GrainsV4.csv"
    leaf_df.to_csv(leaf_path, sep="\t", index=False, float_format="%.6g")

    # ---- Family / parent roll-up table ----
    # One row per PHYSICAL parent grain: a twin family if multiple leaves
    # share twin_family_id, OR a singleton leaf otherwise. Iterating this
    # file gives "n_parent_grains" entries — the coarse-grained count
    # after collapsing twin variants.
    fam_path = out_dir / "GrainsV4_families.csv"
    family_rows = _build_family_rollup(
        twin_family_id=twin_family_id,
        positions_um=positions_um,
        orient_mats_3x3=orient_mats_3x3,
        confidence=confidence,
        volume_nnls_um3=volume_nnls_um3,
        radius_naive_um=radius_naive_um,
        trust_tier_strict=trust_tier_strict,
        trust_tier_loose=trust_tier_loose,
        hkl_coverage=hkl_coverage,
    )
    fam_df = pd.DataFrame(family_rows)
    fam_df.to_csv(fam_path, sep="\t", index=False, float_format="%.6g")

    # ---- Metadata sidecar ----
    n_twin_pairs = int((twin_partner_id >= 0).sum() // 2) if twin_partner_id is not None else 0
    n_subgrain_pairs = int((subgrain_partner_id >= 0).sum() // 2) if subgrain_partner_id is not None else 0
    n_twin_fam = sum(1 for r in family_rows if r["ParentType"] == "twin")
    n_singleton_fam = sum(1 for r in family_rows if r["ParentType"] == "singleton")
    meta = {
        "package_version":  cfg.package_version,
        "trust_scheme":     cfg.trust_scheme,
        "n_leaf_grains":    int(N_g),
        "n_parent_grains":  int(len(family_rows)),
        "n_twin_families":  int(n_twin_fam),
        "n_singleton_parents": int(n_singleton_fam),
        "n_twin_pairs":     n_twin_pairs,
        "n_subgrain_pairs": n_subgrain_pairs,
    }
    if meta_extra is not None:
        meta.update(meta_extra)
    meta_path = out_dir / "GrainsV4.meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))

    return {"leaf": leaf_path, "families": fam_path, "meta": meta_path}


# ---------------------------------------------------------------------------
# Family / parent rollup
# ---------------------------------------------------------------------------


def _build_family_rollup(
    *,
    twin_family_id: np.ndarray,    # (N_g,) int64 — -1 if leaf is not in any family
    positions_um:   np.ndarray,    # (N_g, 3)
    orient_mats_3x3: np.ndarray,   # (N_g, 3, 3)
    confidence: np.ndarray,
    volume_nnls_um3: np.ndarray,
    radius_naive_um: np.ndarray,
    trust_tier_strict: np.ndarray,
    trust_tier_loose:  np.ndarray,
    hkl_coverage: np.ndarray,
) -> List[Dict[str, Any]]:
    """Build the ``GrainsV4_families.csv`` row list (parent grains).

    Each ROW is one physical parent grain:
      - ``ParentType = "twin"``: ``MemberCount`` ≥ 2 leaves share a
        ``twin_family_id`` (the Stage 5 twin-graph component).
      - ``ParentType = "singleton"``: leaf is not in any twin family;
        each such leaf becomes its own one-member parent row.

    Position is the volume-weighted mean. Orientation is the rotation-
    mean (hemisphere-aligned quaternion average, then renormalized to
    a unit quat → 3×3 OM). Trust tiers are the max across members.
    Iterating this file gives "n_parent_grains" entries.
    """
    N_g = len(twin_family_id)
    if N_g == 0:
        return []

    # Volume per leaf (NaN-safe): use NNLS if present, else naive radius^3
    vol = np.where(
        np.isfinite(volume_nnls_um3) & (volume_nnls_um3 > 0),
        volume_nnls_um3,
        (4.0 / 3.0) * np.pi * np.where(np.isfinite(radius_naive_um), radius_naive_um, 0.0) ** 3,
    )

    # Convert OM → quat once (we'll average within each parent group)
    # using midas_stress.orientation. Fallback to a local impl if import fails.
    try:
        import torch
        from midas_stress.orientation import orient_mat_to_quat, quat_to_orient_mat
        q_all = orient_mat_to_quat(torch.from_numpy(orient_mats_3x3)).numpy()
        _quat_to_om = lambda q: np.asarray(
            quat_to_orient_mat(torch.from_numpy(q[None, :]))
        ).reshape(3, 3)
    except Exception:
        q_all = _local_om_to_quat(orient_mats_3x3)
        _quat_to_om = _local_quat_to_om

    rows: List[Dict[str, Any]] = []
    parent_id = 0

    # Group leaves into twin families
    have_family = twin_family_id >= 0
    if have_family.any():
        unique_fids = np.unique(twin_family_id[have_family])
        for fid in unique_fids:
            members = np.flatnonzero(twin_family_id == fid)
            rows.append(_one_parent_row(
                parent_id=parent_id, parent_type="twin", members=members,
                positions_um=positions_um, orient_mats_3x3=orient_mats_3x3,
                q_all=q_all, _quat_to_om=_quat_to_om,
                confidence=confidence, vol=vol,
                trust_tier_strict=trust_tier_strict, trust_tier_loose=trust_tier_loose,
                hkl_coverage=hkl_coverage,
            ))
            parent_id += 1

    # Singleton leaves: every leaf that's NOT in any twin family becomes
    # its own one-member parent row.
    singleton_mask = ~have_family
    singleton_idx = np.flatnonzero(singleton_mask)
    for li in singleton_idx:
        rows.append(_one_parent_row(
            parent_id=parent_id, parent_type="singleton",
            members=np.array([li], dtype=np.int64),
            positions_um=positions_um, orient_mats_3x3=orient_mats_3x3,
            q_all=q_all, _quat_to_om=_quat_to_om,
            confidence=confidence, vol=vol,
            trust_tier_strict=trust_tier_strict, trust_tier_loose=trust_tier_loose,
            hkl_coverage=hkl_coverage,
        ))
        parent_id += 1

    return rows


def _one_parent_row(
    *,
    parent_id: int,
    parent_type: str,
    members: np.ndarray,
    positions_um: np.ndarray,
    orient_mats_3x3: np.ndarray,
    q_all: np.ndarray,
    _quat_to_om,
    confidence: np.ndarray,
    vol: np.ndarray,
    trust_tier_strict: np.ndarray,
    trust_tier_loose: np.ndarray,
    hkl_coverage: np.ndarray,
) -> Dict[str, Any]:
    """One row for the rollup table — geometric mean per parent."""
    n_m = len(members)
    w = vol[members]
    w_sum = float(w.sum())
    if w_sum > 0:
        wn = w / w_sum
        x_mean = float((positions_um[members, 0] * wn).sum())
        y_mean = float((positions_um[members, 1] * wn).sum())
        z_mean = float((positions_um[members, 2] * wn).sum())
    else:
        x_mean = float(np.mean(positions_um[members, 0]))
        y_mean = float(np.mean(positions_um[members, 1]))
        z_mean = float(np.mean(positions_um[members, 2]))

    # Rotation-mean OM: hemisphere-align member quats, average, renormalize
    if n_m == 1:
        om_mean = orient_mats_3x3[members[0]]
    else:
        q_mem = q_all[members]
        ref = q_mem[0]
        signs = np.where((q_mem * ref).sum(axis=1) >= 0, 1.0, -1.0)
        q_mean = (signs[:, None] * q_mem).mean(axis=0)
        q_mean /= np.linalg.norm(q_mean) + 1e-12
        om_mean = _quat_to_om(q_mean)

    total_vol = float(w_sum)
    eq_r = (3 * total_vol / (4 * np.pi)) ** (1 / 3) if total_vol > 0 else float("nan")
    conf_mean = float(np.nanmean(confidence[members]))
    cov_max = float(np.nanmax(hkl_coverage[members]))
    tts_max = int(np.nanmax(trust_tier_strict[members]))
    ttl_max = int(np.nanmax(trust_tier_loose[members]))

    row: Dict[str, Any] = {
        "ParentID":              int(parent_id),
        "ParentType":            parent_type,
        "MemberCount":           int(n_m),
        "MemberGrainIDs":        ",".join(str(int(g)) for g in members),
        "X_um":                  x_mean,
        "Y_um":                  y_mean,
        "Z_um":                  z_mean,
    }
    for i in range(3):
        for j in range(3):
            row[f"O{i+1}{j+1}"] = float(om_mean[i, j])
    row.update({
        "Confidence":            conf_mean,
        "TotalVolume_NNLS_um3":  total_vol,
        "EquivalentRadius_um":   eq_r,
        "trust_tier_strict":     tts_max,
        "trust_tier_loose":      ttl_max,
        "hkl_coverage":          cov_max,
    })
    return row


def _local_om_to_quat(OMs: np.ndarray) -> np.ndarray:
    """Fallback: 3×3 → quat (w, x, y, z) via Shepperd's method, batched."""
    OMs = np.asarray(OMs, dtype=np.float64)
    tr = OMs[:, 0, 0] + OMs[:, 1, 1] + OMs[:, 2, 2]
    q = np.zeros((OMs.shape[0], 4), dtype=np.float64)
    pos = tr > 0
    if pos.any():
        s = np.sqrt(tr[pos] + 1.0) * 2.0
        q[pos, 0] = 0.25 * s
        q[pos, 1] = (OMs[pos, 2, 1] - OMs[pos, 1, 2]) / s
        q[pos, 2] = (OMs[pos, 0, 2] - OMs[pos, 2, 0]) / s
        q[pos, 3] = (OMs[pos, 1, 0] - OMs[pos, 0, 1]) / s
    rest = ~pos
    for i in np.flatnonzero(rest):
        M = OMs[i]
        d = np.diag(M)
        k = int(np.argmax(d))
        if k == 0:
            s = np.sqrt(1.0 + M[0, 0] - M[1, 1] - M[2, 2]) * 2.0
            q[i] = [(M[2, 1] - M[1, 2]) / s, 0.25 * s, (M[0, 1] + M[1, 0]) / s, (M[0, 2] + M[2, 0]) / s]
        elif k == 1:
            s = np.sqrt(1.0 + M[1, 1] - M[0, 0] - M[2, 2]) * 2.0
            q[i] = [(M[0, 2] - M[2, 0]) / s, (M[0, 1] + M[1, 0]) / s, 0.25 * s, (M[1, 2] + M[2, 1]) / s]
        else:
            s = np.sqrt(1.0 + M[2, 2] - M[0, 0] - M[1, 1]) * 2.0
            q[i] = [(M[1, 0] - M[0, 1]) / s, (M[0, 2] + M[2, 0]) / s, (M[1, 2] + M[2, 1]) / s, 0.25 * s]
    return q


def _local_quat_to_om(q: np.ndarray) -> np.ndarray:
    """Fallback: quat (w, x, y, z) → 3×3."""
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z),     2 * (x * y - z * w),     2 * (x * z + y * w)],
        [    2 * (x * y + z * w), 1 - 2 * (x * x + z * z),     2 * (y * z - x * w)],
        [    2 * (x * z - y * w),     2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])
