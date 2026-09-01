"""Mode A: Indexer derives SpotsToIndex.csv from Grains.csv.

When `IndexerParams.isGrainsInput=True`, the C indexer reads grain
orientations from `Grains.csv` and writes a `SpotsToIndex.csv` derived
from those grains. `Indexer.load_observations` mirrors that path.

This test exercises the code branch directly without touching the
matching/pipeline (which is heavily covered elsewhere).
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import numpy as np
import pytest

from midas_index import Indexer, IndexerParams


_GRAINS_CSV = """\
%NumGrains 3
%BeamCenter 0.000000
%BeamThickness 200.000000
%GlobalPosition 0.000000
%NumPhases 1
%PhaseInfo
%	SpaceGroup:225
%	Lattice Parameter: 4.080000 4.080000 4.080000 90.000000 90.000000 90.000000
%GrainID	O11	O12	O13	O21	O22	O23	O31	O32	O33	X	Y	Z	a	b	c	alpha	beta	gamma	DiffPos	DiffOme	DiffAngle	GrainRadius
17\t1.0\t0.0\t0.0\t0.0\t1.0\t0.0\t0.0\t0.0\t1.0\t0.0\t0.0\t0.0\tx\ty\tz\tw\tv\tu\tt\ts\tr\t50.0
42\t0.0\t-1.0\t0.0\t1.0\t0.0\t0.0\t0.0\t0.0\t1.0\t10.0\t-5.0\t2.5\tx\ty\tz\tw\tv\tu\tt\ts\tr\t75.0
99\t1.0\t0.0\t0.0\t0.0\t0.7071\t-0.7071\t0.0\t0.7071\t0.7071\t-3.0\t1.5\t-1.0\tx\ty\tz\tw\tv\tu\tt\ts\tr\t60.0
"""


def _toy_params(grains_file: str = "Grains.csv") -> IndexerParams:
    """Build the minimal IndexerParams needed to exercise Mode A."""
    p = IndexerParams()
    p.Distance = 1_000_000.0
    p.Wavelength = 0.172979
    p.Rsample = 100.0
    p.Hbeam = 100.0
    p.px = 200.0
    p.SpaceGroup = 225
    p.LatticeConstant = (4.08, 4.08, 4.08, 90.0, 90.0, 90.0)
    p.StepsizePos = 100.0
    p.StepsizeOrient = 5.0
    p.MarginOme = 1.0
    p.MarginRad = 5000.0
    p.MarginRadial = 1000.0
    p.MarginEta = 100.0
    p.EtaBinSize = 1.0
    p.OmeBinSize = 1.0
    p.ExcludePoleAngle = 1.0
    p.MinMatchesToAcceptFrac = 0.1
    p.RingNumbers = [1]
    p.RingRadii = {1: 56000.0}
    p.OmegaRanges = [(-180.0, 180.0)]
    p.BoxSizes = [(-2_000_000.0, 2_000_000.0, -2_000_000.0, 2_000_000.0)]
    p.UseFriedelPairs = 0
    p.OutputFolder = "."
    p.isGrainsInput = True
    p.GrainsFileName = grains_file
    return p


def _toy_obs():
    obs = np.array(
        [[0.0, 56000.0, 0.0, 56000.0, 1.0, 1.0, 0.0, 1.5, 0.0]],
        dtype=np.float64,
    )
    return obs


def _toy_bins(obs):
    from midas_index.io import build_bin_index
    return build_bin_index(obs, eta_bin_size=1.0, ome_bin_size=1.0, n_rings=1)


def _toy_hkls():
    import math
    a = 4.08
    one_over_a = 1.0 / a
    hkls_real = np.array(
        [[one_over_a, one_over_a, one_over_a, 1.0, a / math.sqrt(3),
          0.05 * math.pi / 180.0, 56000.0]],
        dtype=np.float64,
    )
    hkls_int = np.array([[1, 1, 1, 1]], dtype=np.int64)
    return hkls_real, hkls_int


def test_mode_a_derives_spots_to_index_from_grains_csv(tmp_path: Path):
    """When isGrainsInput=True and SpotsToIndex.csv is absent,
    `Indexer.load_observations` must read Grains.csv and write
    `SpotsToIndex.csv` whose first column = grain IDs."""
    grains_path = tmp_path / "Grains.csv"
    grains_path.write_text(_GRAINS_CSV)

    params = _toy_params(grains_file=str(grains_path))
    obs = _toy_obs()
    bin_data, bin_ndata = _toy_bins(obs)
    hkls_real, hkls_int = _toy_hkls()

    sti = tmp_path / "SpotsToIndex.csv"
    assert not sti.exists(), "precondition: SpotsToIndex.csv should not exist"

    indexer = Indexer(params, device="cpu", dtype="float64")
    indexer.load_observations(
        cwd=tmp_path,
        spots=obs,
        bins=(bin_data, bin_ndata),
        hkls=(hkls_real, hkls_int),
        # spot_ids omitted -> trigger the Grains.csv derivation branch
    )

    assert sti.exists(), (
        "Indexer.load_observations failed to derive SpotsToIndex.csv from Grains.csv"
    )

    # File contents should reflect the 3 grain IDs (one per line, two ints
    # per line per the mode-A convention "newID origID").
    text_lines = [
        line for line in sti.read_text().splitlines() if line.strip()
    ]
    assert len(text_lines) == 3, (
        f"expected 3 SpotsToIndex.csv rows; got {len(text_lines)}: {text_lines!r}"
    )
    parsed = [list(map(int, line.split())) for line in text_lines]
    grain_ids = [row[0] for row in parsed]
    # Order matches Grains.csv input order (17, 42, 99).
    assert grain_ids == [17, 42, 99]

    # Loaded observations should expose the same spot IDs back to the
    # caller via the cached _observations dict.
    assert indexer._observations is not None
    spot_ids = indexer._observations["spot_ids"].tolist()
    assert spot_ids == [17, 42, 99]


def test_mode_a_skipped_when_spots_to_index_already_exists(tmp_path: Path):
    """If SpotsToIndex.csv already exists, mode A must NOT overwrite it
    even when isGrainsInput=True (matches C semantics — explicit file
    on disk wins)."""
    grains_path = tmp_path / "Grains.csv"
    grains_path.write_text(_GRAINS_CSV)

    sti = tmp_path / "SpotsToIndex.csv"
    sti.write_text("777\n888\n")  # pre-existing list

    params = _toy_params(grains_file=str(grains_path))
    obs = _toy_obs()
    bin_data, bin_ndata = _toy_bins(obs)
    hkls_real, hkls_int = _toy_hkls()

    indexer = Indexer(params, device="cpu", dtype="float64")
    indexer.load_observations(
        cwd=tmp_path, spots=obs, bins=(bin_data, bin_ndata),
        hkls=(hkls_real, hkls_int),
    )

    # The existing 2-row file is preserved, NOT overwritten with 3 grain rows.
    text_lines = [
        line for line in sti.read_text().splitlines() if line.strip()
    ]
    assert text_lines == ["777", "888"]
    assert indexer._observations["spot_ids"].tolist() == [777, 888]


def test_mode_a_relative_grains_path_resolved_against_cwd(tmp_path: Path):
    """A relative GrainsFileName should resolve against cwd (per IndexerOMP
    behavior — it runs from the OutputFolder dir)."""
    (tmp_path / "Grains.csv").write_text(_GRAINS_CSV)

    params = _toy_params(grains_file="Grains.csv")    # relative
    obs = _toy_obs()
    bin_data, bin_ndata = _toy_bins(obs)
    hkls_real, hkls_int = _toy_hkls()

    indexer = Indexer(params, device="cpu", dtype="float64")
    indexer.load_observations(
        cwd=tmp_path, spots=obs, bins=(bin_data, bin_ndata),
        hkls=(hkls_real, hkls_int),
    )

    sti = tmp_path / "SpotsToIndex.csv"
    assert sti.exists()
    assert indexer._observations["spot_ids"].tolist() == [17, 42, 99]


# ---------------------------------------------------------------------------
# Mode A on the CURRENT file format.
#
# _GRAINS_CSV above is 23 columns under %GrainID. Grains.csv is 53 columns
# today and half the corpus is written under %ID (io/csv.py) rather than
# %GrainID (c_parity_emit.py), so neither the current width nor the other
# spelling was exercised through this path.
# ---------------------------------------------------------------------------

def _grains_csv_53(id_token: str, ids=(17, 42, 99)) -> str:
    names = (
        [f"O{i}{j}" for i in (1, 2, 3) for j in (1, 2, 3)]
        + ["X", "Y", "Z", "a", "b", "c", "alpha", "beta", "gamma",
           "DiffPos", "DiffOme", "DiffAngle", "GrainRadius", "Confidence"]
        + [f"eFab{i}{j}" for i in (1, 2, 3) for j in (1, 2, 3)]
        + [f"eKen{i}{j}" for i in (1, 2, 3) for j in (1, 2, 3)]
        + ["RMSErrorStrain", "PhaseNr", "Eul0", "Eul1", "Eul2",
           "DiffPosPre", "DiffOmePre", "DiffAnglePre",
           "DiffPosPost", "DiffOmePost", "DiffAnglePost"]
    )
    out = (f"%NumGrains {len(ids)}\n%BeamCenter 0.000000\n"
           "%BeamThickness 200.000000\n%GlobalPosition 0.000000\n"
           "%NumPhases 1\n%PhaseInfo\n%\tSpaceGroup:225\n"
           "%\tLattice Parameter: 4.080000 4.080000 4.080000 "
           "90.000000 90.000000 90.000000\n")
    out += "%" + "\t".join([id_token] + names) + "\n"
    for k, gid in enumerate(ids):
        row = ([str(gid)]
               + [f"{v:.6f}" for v in (1, 0, 0, 0, 1, 0, 0, 0, 1)]
               + [f"{float(k):.6f}"] * 3
               + [f"{v:.6f}" for v in (4.08, 4.08, 4.08, 90.0, 90.0, 90.0)]
               + ["0.000000", "0.000000", "0.000000",
                  f"{50.0 + k:.6f}", "1.000000"]
               + ["0.000000"] * 19
               + ["1", "0.100000", "0.200000", "0.300000"]
               + ["0.000000"] * 6)
        assert len(row) == 53, len(row)
        # The C writer ends every row in a tab; keep that here.
        out += "\t".join(row) + "\t\n"
    return out


@pytest.mark.parametrize("id_token", ["GrainID", "ID"])
def test_mode_a_on_current_53_column_grains_csv(tmp_path: Path, id_token):
    grains_path = tmp_path / "Grains.csv"
    grains_path.write_text(_grains_csv_53(id_token))

    params = _toy_params(grains_file=str(grains_path))
    obs = _toy_obs()
    bin_data, bin_ndata = _toy_bins(obs)
    hkls_real, hkls_int = _toy_hkls()

    indexer = Indexer(params, device="cpu", dtype="float64")
    indexer.load_observations(
        cwd=tmp_path, spots=obs, bins=(bin_data, bin_ndata),
        hkls=(hkls_real, hkls_int),
    )

    sti = tmp_path / "SpotsToIndex.csv"
    assert sti.exists()
    grain_ids = [int(l.split()[0]) for l in sti.read_text().splitlines()
                 if l.strip()]
    assert grain_ids == [17, 42, 99]
    assert indexer._observations["spot_ids"].tolist() == [17, 42, 99]


@pytest.mark.parametrize("id_token", ["GrainID", "ID"])
def test_grains_reader_gets_radius_from_a_53_column_file(tmp_path: Path,
                                                         id_token):
    """GrainRadius is column 22 at every width so far, but the file has been
    widened four times; the reader now takes the index from the header."""
    from midas_index.io import read_grains_csv

    grains_path = tmp_path / "Grains.csv"
    grains_path.write_text(_grains_csv_53(id_token))
    g = read_grains_csv(grains_path)
    assert g["ids"].tolist() == [17, 42, 99]
    np.testing.assert_allclose(g["radii"], [50.0, 51.0, 52.0])
    np.testing.assert_allclose(g["orient_mat"][0], np.eye(3))
