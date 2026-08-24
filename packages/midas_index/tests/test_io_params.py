"""Tests for read_params (paramstest.txt parser)."""

import textwrap

import pytest

from midas_index.io import read_params


def _write(tmp_path, body):
    p = tmp_path / "paramstest.txt"
    p.write_text(textwrap.dedent(body).lstrip())
    return p


def test_minimal(tmp_path):
    p = read_params(_write(tmp_path, """
        Wavelength 0.172979
        Distance 1000000
        SpaceGroup 225
        LatticeConstant 4.08 4.08 4.08 90 90 90
        StepsizePos 5
        StepsizeOrient 0.5
        MarginOme 0.5
        MarginRadius 200
        MarginRadial 200
        MarginEta 1
        EtaBinSize 0.1
        OmeBinSize 0.1
        ExcludePoleAngle 1
        MinMatchesToAcceptFrac 0.6
        OmegaRange -180 180
        BoxSize -1500000 1500000 -1500000 1500000
        OutputFolder /tmp/out
    """))
    assert p.Wavelength == pytest.approx(0.172979)
    assert p.Distance == pytest.approx(1_000_000.0)
    assert p.SpaceGroup == 225
    assert p.LatticeConstant == (4.08, 4.08, 4.08, 90.0, 90.0, 90.0)
    assert p.StepsizeOrient == 0.5
    assert p.MarginRad == 200.0
    assert p.MarginRadial == 200.0
    assert p.OmegaRanges == [(-180.0, 180.0)]
    assert p.BoxSizes == [(-1_500_000.0, 1_500_000.0, -1_500_000.0, 1_500_000.0)]
    assert p.UseFriedelPairs == 0
    assert p.OutputFolder == "/tmp/out"
    assert p.isGrainsInput is False


def test_aliases_collapse_to_canonical_field(tmp_path):
    p = read_params(_write(tmp_path, """
        Lsd 999000
        LatticeParameter 3.6 3.6 3.6 90 90 90
        MarginRadius 150
        StepSizeOrient 0.25
        Completeness 0.75
        MinEta 1.5
    """))
    assert p.Distance == 999_000.0           # Lsd  -> Distance
    assert p.LatticeConstant[0] == 3.6        # LatticeParameter -> LatticeConstant
    assert p.MarginRad == 150.0               # MarginRadius -> MarginRad
    assert p.StepsizeOrient == 0.25           # StepSizeOrient -> StepsizeOrient
    assert p.MinMatchesToAcceptFrac == 0.75   # Completeness -> MinMatchesToAcceptFrac
    assert p.ExcludePoleAngle == 1.5          # MinEta -> ExcludePoleAngle


def test_repeated_keys_accumulate(tmp_path):
    p = read_params(_write(tmp_path, """
        RingNumbers 1
        RingNumbers 2
        RingNumbers 5
        RingsToExcludeFraction 3
        RingsToExcludeFraction 7
        RingRadii 56000
        RingRadii 81000
        RingRadii 142000
        OmegaRange -180 0
        OmegaRange 0 180
        BoxSize -1 1 -1 1
        BoxSize -2 2 -2 2
    """))
    assert p.RingNumbers == [1, 2, 5]
    assert p.RingsToReject == [3, 7]
    # RingRadii is sparse-by-ring-index per IndexerOMP.c:1535-1538
    assert p.RingRadii == {1: 56000.0, 2: 81000.0, 5: 142000.0}
    assert p.get_ring_radius(2) == 81000.0
    assert p.get_ring_radius(99) == 0.0
    assert p.highest_ring_nr() == 5
    assert len(p.OmegaRanges) == 2
    assert len(p.BoxSizes) == 2


def test_grains_file_sets_mode_a(tmp_path):
    p = read_params(_write(tmp_path, """
        GrainsFile Grains.csv
        UseFriedelPairs 1
    """))
    assert p.isGrainsInput is True
    assert p.GrainsFileName == "Grains.csv"
    assert p.UseFriedelPairs == 1


def test_unknown_key_emits_warning(tmp_path):
    path = _write(tmp_path, """
        Wavelength 0.17
        Bogus 42
    """)
    with pytest.warns(UserWarning, match="Bogus"):
        read_params(path)


def test_blank_and_comment_lines_skipped(tmp_path):
    p = read_params(_write(tmp_path, """

        # this is a comment
        Wavelength 0.17

        # another
        Distance 999
    """))
    assert p.Wavelength == 0.17
    assert p.Distance == 999.0


def test_big_det_size_zero_is_still_ignored(tmp_path):
    """0 means "no mask", which the Python backends can honour honestly."""
    p = read_params(_write(tmp_path, """
        BigDetSize 0
        Wavelength 0.17
    """))
    assert p.Wavelength == 0.17
    # No attribute leak for BigDet
    assert not hasattr(p, "BigDetSize")


def test_big_det_size_nonzero_is_refused_not_ignored(tmp_path):
    """Changed 2026-08-23. This test previously asserted the opposite.

    ``BigDetSize > 0`` now enables the detector active-area mask in the C
    indexer: a reflection predicted onto a dead pixel or off the panel leaves
    BOTH sides of the completeness ratio. The torch and numba backends do not
    implement it.

    Silently ignoring it would report a completeness computed against a
    denominator the user did not ask for — and that number is
    indistinguishable from a correct one, so nothing downstream could catch
    it. Refusing is the only honest option while the two backends differ.
    """
    with pytest.raises(ValueError, match="only the C indexer"):
        read_params(_write(tmp_path, """
            BigDetSize 8192
            Wavelength 0.17
        """))


def test_confidence_metric_raw_is_accepted(tmp_path):
    p = read_params(_write(tmp_path, """
        ConfidenceMetric raw
        Wavelength 0.17
    """))
    assert p.Wavelength == 0.17


@pytest.mark.parametrize("metric", ["filtered", "weighted"])
def test_confidence_metric_non_raw_is_refused(tmp_path, metric):
    """Weighting changes the SCALE of completeness, including at the gate.

    A backend that quietly returned the unweighted ratio would be reporting a
    different quantity under the same name, and ``Completeness`` /
    ``MinMatchesToAcceptFrac`` would then be compared against it.
    """
    with pytest.raises(ValueError, match="only in the C indexer"):
        read_params(_write(tmp_path, f"""
            ConfidenceMetric {metric}
            Wavelength 0.17
        """))


def test_confidence_metric_garbage_is_refused(tmp_path):
    with pytest.raises(ValueError, match="raw\\|filtered\\|weighted"):
        read_params(_write(tmp_path, """
            ConfidenceMetric sometimes
            Wavelength 0.17
        """))
