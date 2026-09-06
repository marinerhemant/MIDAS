"""The orientation convention is a property of the voxel cloud, not a rule.

Two demk products from the same experiment need OPPOSITE conventions. An earlier
`polytype/ladder.py` docstring stated the transpose as universal; applied to the
other cloud it gives garbage. These tests pin both directions so the trap cannot
quietly come back.

Real-data arms are env-gated (`MIDAS_DEFECT_REAL_DATA=1`) like the rest of the
suite; the synthetic arms always run.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from midas_defect.bragg_diffuse import (check_orientation_convention,
                                        enumerate_hkls, on_lattice_fraction,
                                        predicted_reflection_points)
from midas_defect.lattice import fcc_cu_crystal

REAL = os.environ.get("MIDAS_DEFECT_REAL_DATA") == "1"
FIX = Path(__file__).parent / "fixtures" / "demk_g1592_9r.npz"
BIGCLOUD = Path.home() / "Desktop/analysis/demk/fcc_reanalysis/cc3d/all_labels_qvox.npz"
GRAINS = Path.home() / "Desktop/analysis/demk/fcc_reanalysis/cc3d/Grains_L3786.csv"


def _rng_orientations(n, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        out.append(q * np.sign(np.linalg.det(q)))
    return np.asarray(out)


def _synth_cloud(oms, crystal, *, seed=0, sigma=0.01, q_max=6.0):
    """Voxels planted ON the lattice in the `U @ G` sense, U = the given matrices."""
    rng = np.random.default_rng(seed)
    G = (2 * np.pi / crystal.lattice.a) * enumerate_hkls(
        crystal, q_max_inv_A=q_max).astype(float)
    pts = np.concatenate([(U @ G.T).T for U in oms])
    q = pts + rng.normal(scale=sigma, size=pts.shape)
    return q, np.full(len(q), 100.0)


def test_detects_plain_convention():
    """A cloud built as U @ G must be reported as OM when U is passed as given."""
    crystal = fcc_cu_crystal()
    oms = _rng_orientations(4)
    q, I = _synth_cloud(oms, crystal)
    v = check_orientation_convention(q, I, oms, crystal, q_max_inv_A=6.5)
    assert v.decisive and v.convention == "OM", v.note
    assert v.frac_om > 0.9 and v.margin > 3.0


def test_detects_transposed_convention():
    """The SAME cloud, with the matrices handed over transposed, flips the verdict."""
    crystal = fcc_cu_crystal()
    oms = _rng_orientations(4, seed=1)
    q, I = _synth_cloud(oms, crystal, seed=1)
    v = check_orientation_convention(
        q, I, np.transpose(oms, (0, 2, 1)), crystal, q_max_inv_A=6.5)
    assert v.decisive and v.convention == "OM.T", v.note


def test_satellite_only_cloud_is_undecidable_not_confidently_wrong():
    """Off-lattice-by-construction voxels must return decisive=False.

    The failure this guards: a satellite cloud scores near zero BOTH ways, and a
    naive argmax would still name a winner. Undecidable is the honest answer.
    """
    crystal = fcc_cu_crystal()
    oms = _rng_orientations(3, seed=2)
    G111 = (2 * np.pi / crystal.lattice.a) * np.array([1.0, 1.0, 1.0])
    rng = np.random.default_rng(2)
    axis = oms[0] @ (G111 / np.linalg.norm(G111))
    n = np.arange(1, 6)[:, None] / 3.0                      # the n.G/3 rungs
    pts = (n * np.linalg.norm(G111)) * axis
    q = np.repeat(pts, 200, axis=0) + rng.normal(scale=0.01, size=(len(pts) * 200, 3))
    v = check_orientation_convention(q, np.full(len(q), 50.0), oms, crystal,
                                     q_max_inv_A=6.5)
    assert not v.decisive and v.convention is None, v.note
    assert "satellite" in v.note


@pytest.mark.skipif(not REAL or not BIGCLOUD.exists(),
                    reason="needs MIDAS_DEFECT_REAL_DATA=1 and the L3786 voxel cloud")
def test_real_all_labels_cloud_wants_OM_as_given():
    """all_labels_qvox.npz: the raw Grains.csv matrix, NOT transposed."""
    z = np.load(BIGCLOUD)
    rows = [l.split() for l in GRAINS.read_text().splitlines()
            if l.strip() and not l.startswith("%")]
    oms = np.array([[float(x) for x in r[1:10]] for r in rows]).reshape(-1, 3, 3)
    v = check_orientation_convention(z["q"].astype(float), z["I"].astype(float),
                                     oms, fcc_cu_crystal())
    assert v.decisive and v.convention == "OM", v.note
    assert v.frac_om > 0.9 > v.frac_om_t


@pytest.mark.skipif(not FIX.exists(), reason="fixture missing")
def test_real_ladder_fixture_wants_the_TRANSPOSE_by_the_axis_test():
    """demk_g1592_9r: the opposite convention, and on_lattice cannot see it.

    Two assertions, and the second is the point: the axis test picks OM.T, while
    the on-lattice test is blind here because the cloud is satellites.
    """
    d = np.load(FIX, allow_pickle=True)
    OM = np.asarray(d["OM"], float)
    q = np.asarray(d["q"], float)
    I = np.asarray(d["intensity"], float)
    hkl = np.asarray(d["hkl_axis"], float)
    bright = I >= np.percentile(I, 99.0)

    perp = {}
    for name, U in (("OM", OM), ("OM.T", OM.T)):
        ax = U @ hkl
        ax = ax / np.linalg.norm(ax)
        qb = q[bright]
        perp[name] = float(np.median(
            np.linalg.norm(qb - np.outer(qb @ ax, ax), axis=1)))
    assert perp["OM.T"] < 0.5 < perp["OM"], perp        # 0.21 vs 4.47 as measured

    crystal = fcc_cu_crystal(a=float(d["a_fcc"]))
    v = check_orientation_convention(q, I, OM[None], crystal)
    assert not v.decisive, (
        "on_lattice_fraction must NOT claim a verdict on a satellite cloud; "
        f"got {v.note}")
