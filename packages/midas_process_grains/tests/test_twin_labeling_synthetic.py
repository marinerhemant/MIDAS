"""End-to-end synthetic validation of twin labeling across crystal systems.

These tests plant known twin pairs at the **grain-orientation level** (the
input to Stage 5's :func:`label_twins`), then assert that the labeller
finds every planted pair with the correct twin-type name. They cover
the three open validation gaps left by the real-data audits:

A. **Deformed HCP {10-12} twins** — real Bucsek 220 N CP-Ti data is
   elastic-regime and contains no {10-12} twin pairs; synthetic planted
   pairs close the end-to-end loop for the HCP operators.

B. **Tetragonal {101} twins** — none of our four real datasets is
   tetragonal; synthetic planted pairs validate the {101} (L1₀ FePt-
   style) operator end-to-end.

C. **Multi-phase invocation** — the dispatcher picks ONE space group
   per run, so multi-phase samples require per-phase invocation. The
   test exercises this pattern: label twins on an FCC sub-population
   and an HCP sub-population independently, verify union is correct.

D. **User-supplied orthorhombic twin** — the dispatcher has no
   defaults for orthorhombic/monoclinic/triclinic. Users must pass
   ``twin_relations=[my_op]`` directly. This test validates the
   custom-operator code path.
"""

from __future__ import annotations

import math
import numpy as np
import pytest
import torch

from midas_stress.orientation import (
    orient_mat_to_quat, quat_to_orient_mat, misorientation_quat_batch,
)
from midas_process_grains.compute.twin_label import label_twins
from midas_process_grains.compute.twins import (
    TwinRelation,
    default_hcp_twin_relations,
    default_tetragonal_twin_relations,
    default_cubic_twin_relations,
)


def _quat_mul(a, b):
    aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack([
        aw*bw - ax*bx - ay*by - az*bz,
        aw*bx + ax*bw + ay*bz - az*by,
        aw*by - ax*bz + ay*bw + az*bx,
        aw*bz + ax*by - ay*bx + az*bw,
    ], axis=-1)


def _random_quats(n: int, seed: int) -> np.ndarray:
    """Uniform random unit quats in the upper hemisphere (w >= 0)."""
    rng = np.random.default_rng(seed)
    q = rng.normal(size=(n, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    q *= np.sign(q[:, 0:1])
    return q


# ---------------------------------------------------------------------------
# A. Deformed HCP synthetic: planted {10-12} tension twin pairs
# ---------------------------------------------------------------------------


def test_hcp_planted_1012_twin_pairs_are_detected():
    """Plant 50 random HCP parent grains + their {10-12} tension twins, and
    confirm Stage 5 labels EVERY planted pair as the right HCP twin type.

    This validates the end-to-end HCP twin code path with realistic
    refiner-noise (~0.05° quat noise per grain).
    """
    n_parents = 50
    c_over_a = 1.587   # Ti
    parent_quats = _random_quats(n_parents, seed=11)
    T = default_hcp_twin_relations(c_over_a=c_over_a, systems=("tension_1012",))[0].quaternion

    twin_quats = _quat_mul(parent_quats, T)
    twin_quats /= np.linalg.norm(twin_quats, axis=1, keepdims=True)

    # Stack parents and twins; add a tiny refiner-noise jitter so this is
    # NOT a pathologically-perfect test. 0.002 quaternion noise ≈ 0.23°
    # angular noise, well within the 1° tolerance used below.
    rng = np.random.default_rng(99)
    noise = 0.002 * rng.normal(size=(2 * n_parents, 4))
    quats = np.concatenate([parent_quats, twin_quats], axis=0) + noise
    quats /= np.linalg.norm(quats, axis=1, keepdims=True)

    twin_partner, twin_family, twin_type, n_pairs = label_twins(
        grain_quats=quats,
        space_group=194,
        c_over_a=c_over_a,
        tol_deg=1.0,   # widen vs default 0.5° to accommodate jitter + HCP refiner noise
    )

    # Every parent (index i in [0, n_parents)) should point to its twin
    # (index i + n_parents). Or the reverse — the labeller picks the
    # nearest partner; if it's symmetric we accept either direction.
    n_paired_parents = 0
    for i in range(n_parents):
        if twin_partner[i] == i + n_parents or twin_partner[i + n_parents] == i:
            n_paired_parents += 1
    assert n_paired_parents >= int(0.95 * n_parents), (
        f"only {n_paired_parents} of {n_parents} planted HCP {{10-12}} twin pairs "
        f"detected (≥95% expected)"
    )
    # And the labelled type should be the HCP tension {10-12} variant
    labelled_types = [t for t in twin_type if t]
    hcp_1012 = sum(1 for t in labelled_types if "HCP_tension_1012" in t)
    assert hcp_1012 >= int(0.95 * len(labelled_types)), (
        f"{hcp_1012} of {len(labelled_types)} labels are HCP_tension_1012; "
        f"expected ≥95%"
    )


def test_hcp_planted_compression_1011_pairs_are_detected():
    """Same as above but for the {10-11} compression twin (~57° in Ti)."""
    n_parents = 30
    c_over_a = 1.587
    parent_quats = _random_quats(n_parents, seed=22)
    T = default_hcp_twin_relations(c_over_a=c_over_a, systems=("compression_1011",))[0].quaternion

    twin_quats = _quat_mul(parent_quats, T)
    twin_quats /= np.linalg.norm(twin_quats, axis=1, keepdims=True)

    quats = np.concatenate([parent_quats, twin_quats], axis=0)

    twin_partner, twin_family, twin_type, n_pairs = label_twins(
        grain_quats=quats,
        space_group=194,
        c_over_a=c_over_a,
        hcp_systems_override=None,
        tol_deg=0.5,
    ) if False else label_twins(
        grain_quats=quats, space_group=194,
        c_over_a=c_over_a, tol_deg=0.5,
    )
    paired = sum(
        1 for i in range(n_parents)
        if twin_partner[i] == i + n_parents or twin_partner[i + n_parents] == i
    )
    assert paired >= int(0.95 * n_parents), (
        f"only {paired} of {n_parents} planted HCP compression {{10-11}} pairs detected"
    )
    n_1011 = sum(1 for t in twin_type if "compression_1011" in t)
    assert n_1011 >= int(0.9 * 2 * paired)


# ---------------------------------------------------------------------------
# B. Tetragonal planted {101} twin pairs (L1₀ FePt-style)
# ---------------------------------------------------------------------------


def test_tetragonal_planted_101_pairs_are_detected_fept():
    """Plant 50 random tetragonal parents + their {101} twins for FePt-like
    c/a = 0.967; verify label_twins identifies the planted pairs."""
    n_parents = 50
    c_over_a = 0.967
    parent_quats = _random_quats(n_parents, seed=33)
    T = default_tetragonal_twin_relations(c_over_a=c_over_a, systems=("twin_101",))[0].quaternion
    twin_quats = _quat_mul(parent_quats, T)
    twin_quats /= np.linalg.norm(twin_quats, axis=1, keepdims=True)

    quats = np.concatenate([parent_quats, twin_quats], axis=0)

    twin_partner, twin_family, twin_type, n_pairs = label_twins(
        grain_quats=quats,
        space_group=123,   # P4/mmm tetragonal
        c_over_a=c_over_a,
        tol_deg=0.5,
    )
    paired = sum(
        1 for i in range(n_parents)
        if twin_partner[i] == i + n_parents or twin_partner[i + n_parents] == i
    )
    assert paired >= int(0.95 * n_parents), (
        f"only {paired} of {n_parents} planted tetragonal {{101}} pairs detected"
    )
    n_101 = sum(1 for t in twin_type if "twin_101" in t)
    assert n_101 >= int(0.9 * 2 * paired)


# ---------------------------------------------------------------------------
# C. Multi-phase pattern: per-phase invocation
# ---------------------------------------------------------------------------


def test_multi_phase_per_phase_invocation_finds_phase_specific_twins():
    """Multi-phase samples need per-phase ``label_twins`` calls because the
    dispatcher picks ONE space-group default per invocation. Verify the
    pattern: split the grain list by phase, label each phase
    independently, union the results.
    """
    n_fcc = 30
    n_hcp = 20
    c_over_a = 1.587

    # FCC sub-population: plant Σ3 <111> twins
    fcc_parents = _random_quats(n_fcc, seed=44)
    fcc_T = default_cubic_twin_relations(include=("Sigma3",))[0].quaternion
    fcc_twins = _quat_mul(fcc_parents, fcc_T)
    fcc_twins /= np.linalg.norm(fcc_twins, axis=1, keepdims=True)
    fcc_quats = np.concatenate([fcc_parents, fcc_twins], axis=0)

    # HCP sub-population: plant {10-12} tension twins
    hcp_parents = _random_quats(n_hcp, seed=55)
    hcp_T = default_hcp_twin_relations(c_over_a=c_over_a, systems=("tension_1012",))[0].quaternion
    hcp_twins = _quat_mul(hcp_parents, hcp_T)
    hcp_twins /= np.linalg.norm(hcp_twins, axis=1, keepdims=True)
    hcp_quats = np.concatenate([hcp_parents, hcp_twins], axis=0)

    # Phase 1: FCC
    fcc_partner, _, fcc_types, _ = label_twins(
        grain_quats=fcc_quats, space_group=225, tol_deg=0.5,
    )
    n_fcc_paired = sum(
        1 for i in range(n_fcc)
        if fcc_partner[i] == i + n_fcc or fcc_partner[i + n_fcc] == i
    )
    assert n_fcc_paired >= int(0.95 * n_fcc), (
        f"FCC phase: only {n_fcc_paired} of {n_fcc} Σ3 pairs detected"
    )
    assert all("Sigma3" in t for t in fcc_types if t)

    # Phase 2: HCP — independent invocation with its own SG + c/a
    hcp_partner, _, hcp_types, _ = label_twins(
        grain_quats=hcp_quats, space_group=194, c_over_a=c_over_a, tol_deg=0.5,
    )
    n_hcp_paired = sum(
        1 for i in range(n_hcp)
        if hcp_partner[i] == i + n_hcp or hcp_partner[i + n_hcp] == i
    )
    assert n_hcp_paired >= int(0.95 * n_hcp), (
        f"HCP phase: only {n_hcp_paired} of {n_hcp} {{10-12}} pairs detected"
    )
    assert all("HCP_tension_1012" in t for t in hcp_types if t)


# ---------------------------------------------------------------------------
# D. User-supplied orthorhombic twin operator
# ---------------------------------------------------------------------------


def _ortho_twin_180_about_b() -> TwinRelation:
    """Synthetic orthorhombic twin: 180° rotation about the b-axis.

    Common in some pseudo-orthorhombic mineral twin laws and used here as
    a stand-in for any user-defined operator the framework should accept.
    """
    return TwinRelation(
        name="OrthoCustom_180_b",
        quaternion=np.array([0.0, 0.0, 1.0, 0.0]),
        angle_deg=180.0,
        axis=(0.0, 1.0, 0.0),
    )


def test_user_supplied_orthorhombic_twin_is_detected():
    """The dispatcher returns ``[]`` for orthorhombic SGs (16-74). Users must
    pass ``twin_relations=[my_op]``. Verify that path works end-to-end."""
    n_parents = 25
    parent_quats = _random_quats(n_parents, seed=77)
    custom_tw = _ortho_twin_180_about_b()
    T = custom_tw.quaternion
    twin_quats = _quat_mul(parent_quats, T)
    twin_quats /= np.linalg.norm(twin_quats, axis=1, keepdims=True)
    quats = np.concatenate([parent_quats, twin_quats], axis=0)

    # SG 22 = F222 orthorhombic — the dispatcher would return [] here;
    # we explicitly pass our user operator.
    twin_partner, _, twin_type, _ = label_twins(
        grain_quats=quats,
        space_group=22,
        twin_relations=[custom_tw],
        tol_deg=0.5,
    )
    paired = sum(
        1 for i in range(n_parents)
        if twin_partner[i] == i + n_parents or twin_partner[i + n_parents] == i
    )
    assert paired >= int(0.95 * n_parents), (
        f"only {paired} of {n_parents} user-supplied orthorhombic pairs detected"
    )
    assert all("OrthoCustom_180_b" in t for t in twin_type if t)


def test_user_supplied_operator_overrides_dispatcher_default():
    """When ``twin_relations=`` is passed explicitly, it must REPLACE the
    dispatcher default (e.g. supplying just Σ3 for a cubic SG should
    NOT also pull in Σ9/Σ27/etc.)."""
    n_parents = 10
    parent_quats = _random_quats(n_parents, seed=88)
    # Construct a single 30°-around-z operator that's NOT a real CSL boundary
    half = math.radians(30.0 / 2)
    fake_tw = TwinRelation(
        name="FakeTwin_30z",
        quaternion=np.array([math.cos(half), 0.0, 0.0, math.sin(half)]),
        angle_deg=30.0, axis=(0.0, 0.0, 1.0),
    )
    twin_quats = _quat_mul(parent_quats, fake_tw.quaternion)
    twin_quats /= np.linalg.norm(twin_quats, axis=1, keepdims=True)
    quats = np.concatenate([parent_quats, twin_quats], axis=0)

    twin_partner, _, twin_type, _ = label_twins(
        grain_quats=quats,
        space_group=225,  # cubic — but we override with our fake op
        twin_relations=[fake_tw],
        tol_deg=0.5,
    )
    paired = sum(
        1 for i in range(n_parents)
        if twin_partner[i] == i + n_parents or twin_partner[i + n_parents] == i
    )
    assert paired >= int(0.95 * n_parents)
    assert all("FakeTwin_30z" in t for t in twin_type if t), (
        "user-supplied operator did not override dispatcher default"
    )
