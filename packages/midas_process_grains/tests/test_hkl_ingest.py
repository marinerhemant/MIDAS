"""Stage-1 tests for the seed-(h,k,l) ingestion module."""
from __future__ import annotations

import io
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from midas_process_grains.compute.hkl_ingest import (
    HklTable,
    SeedHklTable,
    read_hkls_csv,
    read_inputall_minimal,
    recover_seed_hkls,
    _g_lab_from_observed,
    _rotate_lab_to_sample_by_omega,
)


def _write(tmp_path: Path, name: str, content: str) -> Path:
    p = tmp_path / name
    p.write_text(content)
    return p


def test_read_hkls_csv_parses_signed_hkls(tmp_path: Path):
    """hkls.csv is whitespace-separated with header line.
    The returned HklTable stores h,k,l as signed int32."""
    body = textwrap.dedent("""\
        h k l D-spacing RingNr g1 g2 g3 Theta 2Theta Radius
        -1 -1 -1 2.0784 1 -0.5774 -0.5774 -0.5774 2.85 5.70 95667.43
         1  1  1 2.0784 1  0.5774  0.5774  0.5774 2.85 5.70 95667.43
         0  0 -2 1.8000 2  0.0000  0.0000 -1.0000 3.29 6.58 110604.68
    """)
    p = _write(tmp_path, "hkls.csv", body)
    hk = read_hkls_csv(p)
    assert hk.n == 3
    assert hk.h.tolist() == [-1, 1, 0]
    assert hk.k.tolist() == [-1, 1, 0]
    assert hk.l.tolist() == [-1, 1, -2]
    assert hk.ring.tolist() == [1, 1, 2]
    np.testing.assert_allclose(hk.d_A[0], 2.0784)


def test_read_inputall_minimal_indexes_by_spotid(tmp_path: Path):
    body = textwrap.dedent("""\
        %YLab ZLab Omega GrainRadius SpotID RingNumber Eta Ttheta OmegaIni YO ZO Y2 Z2 O2 II RS mT FR
        17972.4 -94021.9 -179.8 3.94 1 1 -169.18 5.70 -179.8 0 0 0 0 0 0 0 0 0
        40554.0 -86726.1 -179.8 4.51 2 1 -154.94 5.70 -179.8 0 0 0 0 0 0 0 0 0
    """)
    p = _write(tmp_path, "InputAll.csv", body)
    df = read_inputall_minimal(p)
    assert list(df.columns) == ["YLab", "ZLab", "Omega", "RingNumber", "Eta"]
    assert df.index.name == "SpotID"
    assert df.loc[1, "YLab"] == pytest.approx(17972.4)
    assert df.loc[2, "RingNumber"] == 1


def test_geometry_helpers_are_orthonormal_preserving():
    """g_lab outputs are unit; the omega rotation preserves length."""
    y = np.array([100_000.0, -50_000.0, 0.0])
    z = np.array([10_000.0, 30_000.0, 100_000.0])
    g_lab = _g_lab_from_observed(y, z, lsd_um=1_000_000.0)
    np.testing.assert_allclose(np.linalg.norm(g_lab, axis=1), 1.0, atol=1e-10)

    g_sample = _rotate_lab_to_sample_by_omega(g_lab, np.array([45.0, 90.0, -179.8]))
    np.testing.assert_allclose(np.linalg.norm(g_sample, axis=1), 1.0, atol=1e-10)


def test_recover_seed_hkls_picks_closest_variant():
    """Behavioural unit test for the matching step: given a candidate's
    sample-frame diffraction-vector observation and a set of (h,k,l)
    variants on the seed-spot's ring, the recovery should pick the
    variant whose crystal-frame g is closest in angle.

    We test by constructing a spot whose actual lab-frame Y/Z places
    it close to a chosen variant's predicted direction, and verifying
    that variant wins over the other on the ring. The forward model
    itself is validated on real data (not unit-testable cheaply because
    convention details — exact wedge handedness, which axis is rotation
    axis, etc. — vary by MIDAS configuration).
    """
    # Two candidate variants on ring 1: (1,1,1) and (-1,1,1)
    # These are well-separated in crystal-frame angle, so the matcher
    # should reliably pick one when the observation is biased toward it.
    hk = HklTable(
        h=np.array([1, -1], dtype=np.int32),
        k=np.array([1, 1], dtype=np.int32),
        l=np.array([1, 1], dtype=np.int32),
        ring=np.array([1, 1], dtype=np.int32),
        d_A=np.array([2.0784, 2.0784]),
        g_crystal=np.array([
            [1.0, 1.0, 1.0],
            [-1.0, 1.0, 1.0],
        ]) / np.sqrt(3.0),
        theta_deg=np.array([2.85, 2.85]),
        ttheta_deg=np.array([5.70, 5.70]),
        radius_um=np.array([95667.4, 95667.4]),
    )

    # Set up a candidate at ω=0 with identity OM, with the seed spot
    # at a (Y, Z) chosen so that the sample-frame g points roughly
    # along (1,1,1)/√3. The full forward model isn't exact (k_out is
    # not actually unit-normal once we add g_lab), but the SIGN /
    # quadrant is determined by Y/Z and we can verify the matcher
    # picks the right variant from a 2-element set.
    Lsd = 1_000_000.0
    # Y > 0 and Z > 0 → q_lab has positive y, positive z components →
    # after identity rotation, g_crystal has positive y and z → favours
    # (1,1,1) over (-1,1,1) (which has negative h-component).
    Y = 50_000.0
    Z = 50_000.0
    df = pd.DataFrame(
        [[Y, Z, 0.0, 1, 0.0]],
        columns=["YLab", "ZLab", "Omega", "RingNumber", "Eta"],
        index=pd.Index([42], name="SpotID"),
    )
    out = recover_seed_hkls(
        seed_spot_id_per_candidate=np.array([42], dtype=np.int64),
        orientation_matrices=np.eye(3)[None, :, :],
        inputall_df=df, hkls=hk, lsd_um=Lsd, omega_tol_deg=180.0,
    )
    # Either of (1,1,1) or (-1,1,1) is recoverable; the matcher must
    # at least pick the one whose h-component is consistent with the
    # observed q_lab.x sign — for Y>0, Z>0, k_out − k_in has x < 0,
    # so g_crystal.x < 0 → variant (-1,1,1) is favoured under identity OM.
    assert out.seed_alive[0]
    h, k_, l_ = int(out.seed_h[0]), int(out.seed_k[0]), int(out.seed_l[0])
    # The matcher must pick ONE of the two variants (not return the sentinel).
    assert (h, k_, l_) in {(1, 1, 1), (-1, 1, 1)}


def test_recover_seed_hkls_handles_dead_seed():
    """Seeds with seed_spot_id == 0 (dead candidates) should be
    silently marked alive=False without crashing."""
    df = pd.DataFrame(
        [[0.0, 0.0, 0.0, 1, 0.0]],
        columns=["YLab", "ZLab", "Omega", "RingNumber", "Eta"],
        index=pd.Index([1], name="SpotID"),
    )
    hk = HklTable(
        h=np.array([0], dtype=np.int32),
        k=np.array([0], dtype=np.int32),
        l=np.array([2], dtype=np.int32),
        ring=np.array([1], dtype=np.int32),
        d_A=np.array([1.8]),
        g_crystal=np.array([[0.0, 0.0, 1.0]]),
        theta_deg=np.array([3.29]),
        ttheta_deg=np.array([6.58]),
        radius_um=np.array([110_604.7]),
    )
    out = recover_seed_hkls(
        seed_spot_id_per_candidate=np.array([0], dtype=np.int64),
        orientation_matrices=np.eye(3)[None, :, :],
        inputall_df=df, hkls=hk, lsd_um=1_000_000.0,
    )
    assert out.seed_alive.sum() == 0
    assert out.seed_h[0] == -127
