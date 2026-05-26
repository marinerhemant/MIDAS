"""Hierarchical emitter smoke tests."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from midas_process_grains.io.hierarchical import emit_v4_grains


def test_emit_leaf_only(tmp_path: Path):
    """Minimum-input emission: just rep seeds + positions + OMs.
    Default columns fill in with NaN/-1 sentinels and the file writes."""
    N = 4
    paths = emit_v4_grains(
        tmp_path,
        rep_seed_per_grain=np.arange(N, dtype=np.int64),
        positions_um=np.zeros((N, 3)),
        orient_mats_3x3=np.tile(np.eye(3), (N, 1, 1)),
    )
    assert paths["leaf"].exists()
    assert paths["families"].exists()
    assert paths["meta"].exists()
    meta = json.loads(paths["meta"].read_text())
    assert meta["n_leaf_grains"] == N


def test_twin_family_rollup(tmp_path: Path):
    """Two grains in twin family 0 → one twin-family row + one singleton row.

    The families table emits one row per PHYSICAL parent grain: a twin
    family for the two-member group, plus a singleton row for the
    grain-3 leaf that isn't in any family. Iterating this file gives
    "n_parent_grains" entries (2 here, after collapsing the twin pair).
    """
    N = 3
    twin_family = np.array([0, 0, -1], dtype=np.int64)
    twin_partner = np.array([1, 0, -1], dtype=np.int64)
    vols = np.array([100.0, 200.0, 50.0])
    paths = emit_v4_grains(
        tmp_path,
        rep_seed_per_grain=np.arange(N, dtype=np.int64),
        positions_um=np.zeros((N, 3)),
        orient_mats_3x3=np.tile(np.eye(3), (N, 1, 1)),
        twin_family_id=twin_family,
        twin_partner_id=twin_partner,
        volume_nnls_um3=vols,
    )
    import pandas as pd
    fam = pd.read_csv(paths["families"], sep="\t")
    assert len(fam) == 2, f"expected 1 twin family + 1 singleton, got {len(fam)} rows"
    twin_row = fam[fam.ParentType == "twin"].iloc[0]
    singleton_row = fam[fam.ParentType == "singleton"].iloc[0]
    assert twin_row["MemberCount"] == 2
    assert twin_row["TotalVolume_NNLS_um3"] == 300.0
    assert singleton_row["MemberCount"] == 1
    assert singleton_row["TotalVolume_NNLS_um3"] == 50.0


def test_singletons_only_family_table(tmp_path: Path):
    """When no twin families exist, every leaf becomes its own singleton
    parent row — the rollup file has n_leaf_grains entries."""
    N = 2
    paths = emit_v4_grains(
        tmp_path,
        rep_seed_per_grain=np.arange(N, dtype=np.int64),
        positions_um=np.zeros((N, 3)),
        orient_mats_3x3=np.tile(np.eye(3), (N, 1, 1)),
    )
    import pandas as pd
    fam = pd.read_csv(paths["families"], sep="\t")
    assert len(fam) == N
    assert (fam["ParentType"] == "singleton").all()
    assert (fam["MemberCount"] == 1).all()


def test_family_rollup_uses_volume_weighted_position(tmp_path: Path):
    """Two twin-family members at different positions, weighted by NNLS volume,
    must produce a volume-weighted mean position on the rollup row."""
    N = 2
    paths = emit_v4_grains(
        tmp_path,
        rep_seed_per_grain=np.arange(N, dtype=np.int64),
        positions_um=np.array([[100.0, 0.0, 0.0], [300.0, 0.0, 0.0]]),
        orient_mats_3x3=np.tile(np.eye(3), (N, 1, 1)),
        twin_family_id=np.array([0, 0], dtype=np.int64),
        twin_partner_id=np.array([1, 0], dtype=np.int64),
        volume_nnls_um3=np.array([100.0, 300.0]),  # 25% / 75%
    )
    import pandas as pd
    fam = pd.read_csv(paths["families"], sep="\t")
    assert len(fam) == 1
    # 100*0.25 + 300*0.75 = 25 + 225 = 250
    assert abs(fam["X_um"].iloc[0] - 250.0) < 0.5


def test_family_rollup_rotation_mean_om_normalizes(tmp_path: Path):
    """The rollup OM for a family with two slightly-rotated members must
    still be a valid rotation (orthonormal, det = +1)."""
    import torch
    # Two OMs differing by ~10° around Z
    def rot_z(deg):
        th = np.deg2rad(deg)
        c, s = np.cos(th), np.sin(th)
        return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])

    OMs = np.stack([rot_z(0.0), rot_z(10.0)])
    paths = emit_v4_grains(
        tmp_path,
        rep_seed_per_grain=np.arange(2, dtype=np.int64),
        positions_um=np.zeros((2, 3)),
        orient_mats_3x3=OMs,
        twin_family_id=np.array([0, 0], dtype=np.int64),
        twin_partner_id=np.array([1, 0], dtype=np.int64),
    )
    import pandas as pd
    fam = pd.read_csv(paths["families"], sep="\t")
    O = np.array([[fam[f"O{i}{j}"].iloc[0] for j in (1, 2, 3)] for i in (1, 2, 3)])
    # Orthonormal + det = +1
    np.testing.assert_allclose(O @ O.T, np.eye(3), atol=1e-6)
    assert abs(np.linalg.det(O) - 1.0) < 1e-6


def test_family_meta_parent_grain_count(tmp_path: Path):
    """meta.json must record n_parent_grains = n_twin_families + n_singleton_parents."""
    import json
    N = 5
    # leaves 0+1 in family 0, leaves 2+3+4 are singletons
    paths = emit_v4_grains(
        tmp_path,
        rep_seed_per_grain=np.arange(N, dtype=np.int64),
        positions_um=np.zeros((N, 3)),
        orient_mats_3x3=np.tile(np.eye(3), (N, 1, 1)),
        twin_family_id=np.array([0, 0, -1, -1, -1], dtype=np.int64),
        twin_partner_id=np.array([1, 0, -1, -1, -1], dtype=np.int64),
    )
    meta = json.loads(paths["meta"].read_text())
    assert meta["n_leaf_grains"] == 5
    assert meta["n_twin_families"] == 1
    assert meta["n_singleton_parents"] == 3
    assert meta["n_parent_grains"] == 4
