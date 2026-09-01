"""Grains.csv / SpotMatrix.csv fixtures for the shipped ``midas-uq`` CLI.

The CLI's two readers were positional and header-token-specific:

* ``_parse_grains_csv`` looked only for a ``%GrainID`` header line. Every file
  written by ``midas_process_grains.io.csv`` (i.e. every mode except
  ``c_parity``) starts ``%ID``, so ``header_cols`` stayed ``None`` and the next
  line raised ``TypeError: 'NoneType' object is not iterable``.
* ``_parse_spot_matrix`` used ``np.loadtxt(skiprows=1)`` with no ``Matched``
  filter, so predicted-but-never-found reflections (-1 SpotID/RingNr, NaN
  observations) entered every count: ``n_spots``, the ``--min-spots`` gate,
  the half-half split sizes and the jackknife row index.

There was no fixture for either file in this package, which is why both
survived.
"""
from __future__ import annotations

import numpy as np
import pytest

from midas_uq.cli import (
    _build_grain_state,
    _grain_observations,
    _grain_spot_ids,
    _parse_grains_csv,
    _parse_spot_matrix,
)

# ---------------------------------------------------------------- fixtures

_PREAMBLE = (
    "%NumGrains 2\n%BeamCenter 0.0 0.0\n%BeamThickness 0.0\n"
    "%GlobalPosition 0.0\n%NumPhases 1\n%PhaseInfo\n%\tSpaceGroup:225\n"
    "%\tLattice Parameter:3.590280 3.590280 3.590280 90.000000 90.000000 90.000000\n"
)
_OM = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]

_GRAINS_53_COLS = (
    ["O11", "O12", "O13", "O21", "O22", "O23", "O31", "O32", "O33",
     "X", "Y", "Z", "a", "b", "c", "alpha", "beta", "gamma",
     "DiffPos", "DiffOme", "DiffAngle", "GrainRadius", "Confidence"]
    + [f"eFab{i}{j}" for i in (1, 2, 3) for j in (1, 2, 3)]
    + [f"eKen{i}{j}" for i in (1, 2, 3) for j in (1, 2, 3)]
    + ["RMSErrorStrain", "PhaseNr", "Eul0", "Eul1", "Eul2",
       "DiffPosPre", "DiffOmePre", "DiffAnglePre",
       "DiffPosPost", "DiffOmePost", "DiffAnglePost"]
)

#: (GrainID, a, X, GrainRadius, Confidence)
_G_ROWS = ((1, 3.5910, 10.0, 8.6, 0.991), (2, 3.5895, -20.0, 25.0, 0.640))


def _grains_53(token: str = "ID") -> str:
    hdr = "%" + token + "\t" + "\t".join(_GRAINS_53_COLS) + "\n"
    lines = []
    for gid, a, x, rad, conf in _G_ROWS:
        vals = ([gid] + _OM + [x, 2.0 * gid, 3.0 * gid]
                + [a, a, a, 90.0, 90.0, 90.0]
                + [251.3, 0.072, 0.15, rad, conf]
                + [100.0] * 9 + [200.0] * 9
                + [12.5, 1, 0.1, 0.2, 0.3]
                + [281.3, 0.092, 0.17, 251.3, 0.072, 0.15])
        assert len(vals) == 53
        lines.append("\t".join(repr(v) if isinstance(v, float) else str(v)
                               for v in vals))
    return _PREAMBLE + hdr + "\n".join(lines) + "\n"


def _grains_21() -> str:
    """Legacy width: no per-grain lattice, only the Voigt strain block."""
    cols = ["O11", "O12", "O13", "O21", "O22", "O23", "O31", "O32", "O33",
            "X", "Y", "Z", "E11", "E22", "E33", "E12", "E13", "E23",
            "GrainRadius", "Confidence"]
    hdr = "%ID\t" + "\t".join(cols) + "\n"
    lines = []
    for gid, _a, x, rad, conf in _G_ROWS:
        vals = ([gid] + _OM + [x, 2.0 * gid, 3.0 * gid]
                + [1e-3, 2e-3, 3e-3, 0.0, 0.0, 0.0] + [rad, conf])
        lines.append("\t".join(str(v) for v in vals))
    return _PREAMBLE + hdr + "\n".join(lines) + "\n"


_SM_28_COLS = ["SpotID", "Omega", "DetectorHor", "DetectorVert", "OmeRaw",
               "Eta", "RingNr", "YLab", "ZLab", "Theta", "StrainError",
               "Matched", "theorSpotID", "theorRingNr", "theorEta", "YExp",
               "ZExp", "OmegaExp", "DiffLen", "DiffOme", "InternalAngle",
               "YExpPost", "ZExpPost", "OmegaExpPost", "DiffLenPost",
               "DiffOmePost", "InternalAnglePost"]

#: (GrainID, SpotID, Omega, RingNr, YLab, ZLab, Theta). The ``Eta`` column is
#: written as a MICRON value (-20945 etc.) exactly as the Python
#: ProcessGrains writes it on 1-ID data, so a reader that trusts it as an
#: angle is caught.
_SM_MATCHED = (
    (1, 101, 10.0, 1, -48891.6, -21135.2, 1.5),
    (1, 102, 20.0, 1, 52362.7, 12815.5, 1.5),
    (1, 103, 30.0, 2, -49924.4, 19139.3, 2.0),
    (2, 104, 40.0, 2, -51541.2, 35444.5, 2.0),
)
_ETA_MICRONS = (-20945.1855, 13017.3164, 19351.5723, 35619.7031)


def _spotmatrix_28(n_unmatched: int = 2) -> str:
    hdr = "%GrainID\t" + "\t".join(_SM_28_COLS) + "\n"
    lines = []
    for (gid, sid, ome, ring, ylab, zlab, theta), eta_um in zip(
            _SM_MATCHED, _ETA_MICRONS):
        v = ([gid, sid, ome, 100.0, 200.0, ome, eta_um, ring, ylab, zlab,
              theta, 1.0e-4, 1] + [sid, ring, 44.0] + [1.0] * 12)
        assert len(v) == 28
        lines.append("\t".join(str(x) for x in v))
    for k in range(n_unmatched):
        v = ([1, -1] + ["nan"] * 5 + [-1] + ["nan"] * 4 + [0]
             + [900 + k, 3, 44.0] + ["nan"] * 12)
        assert len(v) == 28
        lines.append("\t".join(str(x) for x in v))
    # SpotMatrix.csv is written with newline='\t\n'.
    return hdr + "\n".join(line + "\t" for line in lines) + "\n"


def _spotmatrix_12() -> str:
    hdr = ("%GrainID\tSpotID\tOmega\tDetectorHor\tDetectorVert\tOmeRaw\tEta\t"
           "RingNr\tYLab\tZLab\tTheta\tStrainError\n")
    lines = []
    for (gid, sid, ome, ring, ylab, zlab, theta), eta_um in zip(
            _SM_MATCHED, _ETA_MICRONS):
        v = [gid, sid, ome, 100.0, 200.0, ome, eta_um, ring, ylab, zlab,
             theta, 1.0e-4]
        lines.append("\t".join(str(x) for x in v))
    return hdr + "\n".join(lines) + "\n"


# ---------------------------------------------------------------- Grains.csv


@pytest.mark.parametrize("token", ["ID", "GrainID"])
def test_grains_both_header_tokens(tmp_path, token):
    """``%ID`` used to leave header_cols None -> TypeError on the next line."""
    p = tmp_path / "Grains.csv"
    p.write_text(_grains_53(token=token))
    g = _parse_grains_csv(p)
    assert g["GrainID"].tolist() == [1.0, 2.0]
    np.testing.assert_allclose(g["GrainRadius"], [8.6, 25.0])
    np.testing.assert_allclose(g["Confidence"], [0.991, 0.640])


def test_grains_53col_state_uses_named_lattice(tmp_path):
    p = tmp_path / "Grains.csv"
    p.write_text(_grains_53())
    g = _parse_grains_csv(p)
    st = _build_grain_state(g, 0)
    np.testing.assert_allclose(st.latc.numpy(),
                               [3.5910, 3.5910, 3.5910, 90.0, 90.0, 90.0])
    np.testing.assert_allclose(st.pos.numpy(), [10.0, 2.0, 3.0])
    # Identity orientation -> zero Euler triple.
    np.testing.assert_allclose(st.euler_rad.numpy(), [0.0, 0.0, 0.0], atol=1e-12)


def test_grains_21col_lattice_seeded_from_strain(tmp_path):
    """A legacy file has no a/b/c columns; seed them from the header lattice."""
    p = tmp_path / "Grains.csv"
    p.write_text(_grains_21())
    g = _parse_grains_csv(p)
    np.testing.assert_allclose(g["a"], [3.59028 * 1.001] * 2)
    np.testing.assert_allclose(g["b"], [3.59028 * 1.002] * 2)
    np.testing.assert_allclose(g["c"], [3.59028 * 1.003] * 2)
    np.testing.assert_allclose(g["alpha"], [90.0, 90.0])
    st = _build_grain_state(g, 0)
    assert st.latc.numpy()[0] == pytest.approx(3.59028 * 1.001)


def test_grains_21col_microstrain_block_refuses_to_guess(tmp_path):
    """Do not silently build a lattice from a block that is not strain."""
    text = _grains_21().replace("0.001\t0.002\t0.003", "1000.0\t2000.0\t3000.0")
    p = tmp_path / "Grains.csv"
    p.write_text(text)
    with pytest.raises(ValueError, match="not a dimensionless strain"):
        _parse_grains_csv(p)


# ------------------------------------------------------------ SpotMatrix.csv


def test_spot_matrix_drops_unmatched(tmp_path):
    p = tmp_path / "SpotMatrix.csv"
    p.write_text(_spotmatrix_28(n_unmatched=3))
    s = _parse_spot_matrix(p)
    assert s["n_rows_total"] == 7
    assert s["n_rows_unmatched"] == 3
    assert len(s["grain_id"]) == 4
    assert -1 not in s["spot_id"].tolist()
    for key in ("omega_deg", "eta_deg", "theta_deg"):
        assert np.isfinite(s[key]).all(), key


def test_spot_matrix_eta_recomputed_from_lab_frame(tmp_path):
    """The SpotMatrix ``Eta`` column is not reliably an angle.

    On 1-ID Cu output written by the Python ProcessGrains it holds MICRONS;
    the CLI was multiplying that by pi/180 and calling it eta.
    """
    p = tmp_path / "SpotMatrix.csv"
    p.write_text(_spotmatrix_28())
    s = _parse_spot_matrix(p)
    ylab = np.array([r[4] for r in _SM_MATCHED])
    zlab = np.array([r[5] for r in _SM_MATCHED])
    np.testing.assert_allclose(s["eta_deg"],
                               np.degrees(np.arctan2(-ylab, zlab)))
    assert np.abs(s["eta_deg"]).max() <= 180.0
    # None of the raw micron values survives.
    for bad in _ETA_MICRONS:
        assert not np.any(np.isclose(s["eta_deg"], bad))


def test_spot_matrix_12col_keeps_every_row(tmp_path):
    p = tmp_path / "SpotMatrix.csv"
    p.write_text(_spotmatrix_12())
    s = _parse_spot_matrix(p)
    assert len(s["grain_id"]) == 4
    assert s["n_rows_unmatched"] == 0


# ------------------------------------------------------------ downstream use


def test_observation_count_is_matched_spots_only(tmp_path):
    """``n_spots`` and the ``--min-spots`` gate are computed from obs.shape[0].

    Grain 1 has 3 matched spots; the 4 unmatched rows also carry GrainID 1,
    so the gate used to see 7.
    """
    p = tmp_path / "SpotMatrix.csv"
    p.write_text(_spotmatrix_28(n_unmatched=4))
    s = _parse_spot_matrix(p)
    obs = _grain_observations(s, 1)
    assert obs.shape == (3, 3)
    assert np.isfinite(obs.numpy()).all()
    assert _grain_observations(s, 2).shape == (1, 3)
    assert _grain_observations(s, 99) is None


def test_spot_ids_align_with_observation_rows(tmp_path):
    """Jackknife influence index k must name a real SpotID."""
    p = tmp_path / "SpotMatrix.csv"
    p.write_text(_spotmatrix_28(n_unmatched=2))
    s = _parse_spot_matrix(p)
    obs = _grain_observations(s, 1)
    sids = _grain_spot_ids(s, 1)
    assert len(sids) == obs.shape[0]
    assert sids.tolist() == [101, 102, 103]
    # Row k's 2-theta must be the 2-theta of spot sids[k].
    theta_by_sid = {r[1]: r[6] for r in _SM_MATCHED}
    for k, sid in enumerate(sids):
        assert float(obs[k, 0]) == pytest.approx(
            np.deg2rad(2.0 * theta_by_sid[int(sid)]))


def test_half_half_split_sees_equal_real_evidence(tmp_path):
    """``half_half_spots`` splits ``randperm(n)`` in half.

    With unmatched rows in, n counted phantoms, so the two halves held
    unequal amounts of real evidence. After the filter every row is real.
    """
    p = tmp_path / "SpotMatrix.csv"
    p.write_text(_spotmatrix_28(n_unmatched=5))
    s = _parse_spot_matrix(p)
    obs = _grain_observations(s, 1)
    n = obs.shape[0]
    assert n == 3
    assert np.isfinite(obs.numpy()).all()


# ------------------------------------------------- half/half split coverage

def _capture_split_sizes(n_obs, n_splits=4, seed=3):
    """Run ``half_half_spots`` with the fit stubbed out, returning, per split,
    the observation indices each half received.

    The split is inline in ``half_half_spots``, so the only way to see it
    without paying for four L-BFGS refinements is to intercept the fit.
    Column 0 of the synthetic observations is the row index, so the stub can
    recover exactly which rows reached it.
    """
    import torch

    from midas_uq import spots as _spots
    from midas_uq._common import GrainState

    obs = torch.zeros((n_obs, 3), dtype=torch.float64)
    obs[:, 0] = torch.arange(n_obs, dtype=torch.float64)
    init = GrainState(torch.zeros(3, dtype=torch.float64),
                      torch.tensor([3.6, 3.6, 3.6, 90.0, 90.0, 90.0],
                                   dtype=torch.float64))

    seen: list[list[int]] = []

    def _stub(model, observations, init_state, loss, **kw):
        seen.append([int(v) for v in observations[:, 0].tolist()])
        return init_state.clone()

    real = _spots._fit_grain_spots
    _spots._fit_grain_spots = _stub
    try:
        _spots.half_half_spots(None, init, obs, n_splits=n_splits, seed=seed)
    finally:
        _spots._fit_grain_spots = real

    assert len(seen) == 2 * n_splits
    return [(seen[2 * k], seen[2 * k + 1]) for k in range(n_splits)]


def test_half_half_odd_spot_count_uses_every_spot():
    """An ODD spot count must still put every observation in exactly one half.

    ``perm[half:2*half]`` dropped the last permuted spot on odd n, so one
    observation constrained neither fit -- a different one on every split, so
    it never showed up as a reproducible failure.
    """
    n = 11
    for half_a, half_b in _capture_split_sizes(n):
        assert not set(half_a) & set(half_b), "a spot landed in both halves"
        assert sorted(half_a + half_b) == list(range(n)), (
            f"spots {sorted(set(range(n)) - set(half_a + half_b))} reached "
            f"neither half")
        assert abs(len(half_a) - len(half_b)) == 1


def test_half_half_even_spot_count_splits_exactly():
    n = 10
    for half_a, half_b in _capture_split_sizes(n):
        assert len(half_a) == len(half_b) == 5
        assert sorted(half_a + half_b) == list(range(n))


def test_half_half_rejects_degenerate_spot_count():
    """n < 2 cannot form two halves; both were silently empty before."""
    import torch

    from midas_uq import spots as _spots
    from midas_uq._common import GrainState

    init = GrainState(torch.zeros(3, dtype=torch.float64),
                      torch.tensor([3.6, 3.6, 3.6, 90.0, 90.0, 90.0],
                                   dtype=torch.float64))
    with pytest.raises(ValueError, match="at least 2 observations"):
        _spots.half_half_spots(None, init, torch.zeros((1, 3),
                                                       dtype=torch.float64))
