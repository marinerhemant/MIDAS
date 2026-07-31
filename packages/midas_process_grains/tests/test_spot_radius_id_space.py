"""GrainRadius must be averaged over the RIGHT spots.

Found by running the C reference ``FF_HEDM/src/ProcessGrains.c`` against the
python pipeline on identical refinement output (1-ID GE5 Au3_cubes_ff_000008,
2026-07-30). Same grains, same positions to 6 decimals, but:

    grain      C ProcessGrains      python
    80/1000        114.620659       20.775146   µm
    185/1177        99.962738       17.160936   µm

Cause: ``_load_spot_radius_by_id`` built its SpotID → radius lookup from
``Radius_*.csv``. Both that file and ``ExtraInfo.bin`` contain the same spots
numbered 1..N, but in DIFFERENT orders — ``calc_radius`` renumbers once, then
``bin_data`` sorts by (RingNumber, Omega, Eta) and renumbers again. Every id
downstream of the binner (FitBest, SpotMatrix, the refiner) is in the
ExtraInfo space, so the join silently averaged ~112 arbitrary spots and every
grain converged to the GLOBAL mean radius (~22 µm) instead of its own.

These tests pin the ID space, not just the arithmetic: a lookup that is
merely "some radii" would pass a mean-of-a-column check, so the fixture makes
the two orderings disagree and asserts the grain gets ITS OWN spots' radii.
"""

from __future__ import annotations

import numpy as np
import pytest

from midas_process_grains.pipeline import ProcessGrains


# ExtraInfo.bin layout: (N, 16) float64, col 3 = per-spot grain radius (µm),
# col 4 = SpotID. See midas_fit_grain/observations.py:96-102.
EI_NCOLS = 16
EI_RADIUS_COL = 3
EI_SPOTID_COL = 4


def _write_extra_info(path, spot_ids, radii):
    arr = np.zeros((len(spot_ids), EI_NCOLS), dtype=np.float64)
    arr[:, EI_SPOTID_COL] = spot_ids
    arr[:, EI_RADIUS_COL] = radii
    arr.tofile(path)


def _write_radius_csv(path, spot_ids, radii):
    """Radius_*.csv with the SAME radii but a DIFFERENT SpotID ordering —
    which is exactly the trap: reading it keyed by SpotID yields plausible
    numbers that belong to the wrong spots."""
    hdr = ("%SpotID IntegratedIntensity Omega YCen ZCen IMax MinOme MaxOme "
           "Radius Theta Eta DeltaOmega NImgs RingNr GrainVolume GrainRadius "
           "PowderIntensity SigmaR SigmaEta NrPx NrPxTot RawSumIntensity "
           "maskTouched FitRMSE OrigSpotID ReturnCode")
    rows = np.zeros((len(spot_ids), 26), dtype=np.float64)
    rows[:, 0] = spot_ids
    rows[:, 15] = radii
    rows[:, 24] = spot_ids
    np.savetxt(path, rows, header=hdr, comments="")


@pytest.fixture
def run_dir(tmp_path):
    """4 spots. ExtraInfo says spot 1 has radius 100; Radius_*.csv — in its
    own (reversed) order — says spot 1 has radius 4. Any lookup keyed off the
    wrong file gives 4."""
    ids = np.array([1, 2, 3, 4, 5, 6])
    ei_radii = np.array([100.0, 110.0, 2.0, 3.0, 2.5, 3.5])
    _write_extra_info(tmp_path / "ExtraInfo.bin", ids, ei_radii)
    # Same VALUES, reversed assignment to ids -> different physical meaning.
    _write_radius_csv(tmp_path / "Radius_StartNr_1_EndNr_9.csv",
                      ids, ei_radii[::-1])
    return tmp_path


def _lookup(run_dir):
    pg = ProcessGrains.__new__(ProcessGrains)
    pg.run_dir = run_dir
    pg.spot_radius_by_id = None
    return pg._load_spot_radius_by_id()


def test_lookup_comes_from_extra_info_not_radius_csv(run_dir):
    out = _lookup(run_dir)
    assert out is not None
    # ExtraInfo says 100.0 for SpotID 1; Radius_*.csv (reversed) says 3.5.
    assert out[1] == pytest.approx(100.0), (
        "spot radius lookup is keyed off Radius_*.csv, whose SpotID numbering "
        "is a different space from ExtraInfo.bin/FitBest/SpotMatrix"
    )
    assert out[2] == pytest.approx(110.0)
    assert out[3] == pytest.approx(2.0)
    assert out[6] == pytest.approx(3.5)


def test_grain_radius_is_grain_specific_not_the_global_mean(run_dir):
    """The failure signature of the bug: a grain made of the two BIG spots
    must not come out near the global mean."""
    out = _lookup(run_dir)
    grain_spots = np.array([1, 2])            # the two large spots
    grain_mean = out[grain_spots].mean()
    global_mean = out[1:].mean()
    assert grain_mean == pytest.approx(105.0)
    assert grain_mean > 2 * global_mean, (
        f"grain radius {grain_mean} collapsed toward the global mean "
        f"{global_mean} — the hallmark of an ID-space mismatch"
    )


def test_returns_none_without_extra_info(tmp_path):
    """No ExtraInfo.bin → keep the refiner's own meanRadius rather than
    inventing one from a file in the wrong ID space."""
    _write_radius_csv(tmp_path / "Radius_StartNr_1_EndNr_9.csv",
                      np.array([1, 2]), np.array([7.0, 8.0]))
    pg = ProcessGrains.__new__(ProcessGrains)
    pg.run_dir = tmp_path
    pg.spot_radius_by_id = None
    assert pg._load_spot_radius_by_id() is None


def test_malformed_extra_info_is_rejected(tmp_path):
    """A truncated ExtraInfo.bin must not be reshaped into garbage radii."""
    (tmp_path / "ExtraInfo.bin").write_bytes(
        np.zeros(16 * 3 + 5, dtype=np.float64).tobytes()
    )
    pg = ProcessGrains.__new__(ProcessGrains)
    pg.run_dir = tmp_path
    pg.spot_radius_by_id = None
    assert pg._load_spot_radius_by_id() is None
