"""``MinSeedGrainRadius`` — the PF seed-strength floor.

Under ``OneSolPerVox`` only the highest-completeness solution per voxel
survives, but every spot on ``RingToIndex`` inside the beam gate is still tried
as a seed. On bt_1id_jun25b s1/L9 that was 27.3 M orientation trials (796
core-hours) to keep 361 orientations, and the spots that actually won came from
grains ~2.1x the bulk median ``GrainRadius``.

The floor skips weak spots **as seeds only**. They stay in ``ObsSpotsLab`` and
``CompareSpots`` can still match them, so no grain's completeness can move —
that separation is the whole point, and is what a plain ``RingThresh`` raise
could not give.

``GrainRadius`` rather than raw intensity: ``midas_transforms/radius`` divides
the spot's integrated intensity by the ring's ``powder_int`` and ``m_hkl``, so
structure factor and multiplicity cancel and the value is comparable across
rings. A raw cut conflates a small grain with a low-|F| reflection of a large
one.
"""

import textwrap

from midas_index.io import read_params


def _write(tmp_path, extra=""):
    body = textwrap.dedent("""
        Wavelength 0.172979
        Distance 1000000
        SpaceGroup 166
        LatticeConstant 2.87522 2.87522 14.204 90 90 120
        StepsizePos 5
        StepsizeOrient 0.5
        MarginOme 0.5
        MarginRadius 200
        MarginRadial 200
        MarginEta 1
        EtaBinSize 0.1
        OmeBinSize 0.1
        ExcludePoleAngle 1
        RingToIndex 5
    """).lstrip() + extra
    p = tmp_path / "paramstest.txt"
    p.write_text(body)
    return p


def test_absent_defaults_to_zero(tmp_path):
    # 0 must mean "historical seed set", so an old parameter file is unchanged.
    p = read_params(_write(tmp_path))
    assert p.MinSeedGrainRadius == 0.0


def test_parsed_from_paramstest(tmp_path):
    p = read_params(_write(tmp_path, "MinSeedGrainRadius 19.2\n"))
    assert p.MinSeedGrainRadius == 19.2


def test_trailing_semicolon_flavour(tmp_path):
    # paramstest_comp.txt is written with trailing ';' by _emit_c_omp_paramstest.
    p = read_params(_write(tmp_path, "MinSeedGrainRadius 24.300000;\n"))
    assert p.MinSeedGrainRadius == 24.3


def test_registered_for_both_paths(tmp_path):
    """Scope means "the shared ReadParams accepts it", not "it has an effect".

    FF and PF go through one parser, so declaring this PF-only would hide a key
    the parser does accept — which is what test_pf_equals_ff_plus_beamsize
    guards. The PF-only *effect* belongs in the notes.
    """
    from midas_params.registry import by_name

    spec = by_name().get("MinSeedGrainRadius")
    assert spec is not None
    assert "GrainRadius" in (spec.notes or "")


def test_emitted_to_the_c_paramstest_when_set_in_memory(tmp_path):
    """The binary reads this only from the file.

    ``RingToIndex`` and ``ScanPosTol`` are overridden the same way for the same
    reason: an in-memory value that never reaches the file would leave the
    floor silently doing nothing, which looks exactly like "the lever gained
    us nothing".
    """
    from midas_index.indexer import Indexer

    pp = _write(tmp_path, "MinSeedGrainRadius 5.0\n")
    ind = Indexer.from_param_file(pp)
    ind.params.MinSeedGrainRadius = 19.2
    out = ind._emit_c_omp_paramstest(pp)

    lines = [ln.strip() for ln in out.read_text().splitlines()]
    hits = [ln for ln in lines if ln.startswith("MinSeedGrainRadius")]
    assert len(hits) == 1, f"stale copy left behind: {hits}"
    assert hits[0].startswith("MinSeedGrainRadius 19.2")


def test_zero_is_not_emitted(tmp_path):
    # Absent key => the C default (0.0) => byte-identical historical behaviour.
    from midas_index.indexer import Indexer

    pp = _write(tmp_path)
    ind = Indexer.from_param_file(pp)
    out = ind._emit_c_omp_paramstest(pp)
    assert not any(ln.strip().startswith("MinSeedGrainRadius")
                   for ln in out.read_text().splitlines())


# --- SeedDropWeakestFrac: the fraction form, which is the reportable one -----

def test_frac_absent_defaults_to_zero(tmp_path):
    p = read_params(_write(tmp_path))
    assert p.SeedDropWeakestFrac == 0.0


def test_frac_parsed(tmp_path):
    p = read_params(_write(tmp_path, "SeedDropWeakestFrac 0.5\n"))
    assert p.SeedDropWeakestFrac == 0.5


def test_frac_emitted_to_the_c_paramstest(tmp_path):
    from midas_index.indexer import Indexer

    pp = _write(tmp_path)
    ind = Indexer.from_param_file(pp)
    ind.params.SeedDropWeakestFrac = 0.5
    out = ind._emit_c_omp_paramstest(pp)
    hits = [ln.strip() for ln in out.read_text().splitlines()
            if ln.strip().startswith("SeedDropWeakestFrac")]
    assert len(hits) == 1 and hits[0].startswith("SeedDropWeakestFrac 0.5")


def test_both_knobs_can_coexist(tmp_path):
    # The C takes the stricter of the two; neither may silently weaken the
    # other, so both must survive the round trip.
    from midas_index.indexer import Indexer

    pp = _write(tmp_path)
    ind = Indexer.from_param_file(pp)
    ind.params.MinSeedGrainRadius = 10.0
    ind.params.SeedDropWeakestFrac = 0.5
    txt = ind._emit_c_omp_paramstest(pp).read_text()
    assert "MinSeedGrainRadius 10" in txt
    assert "SeedDropWeakestFrac 0.5" in txt


def test_frac_is_registered_and_preferred(tmp_path):
    from midas_params.registry import by_name

    spec = by_name().get("SeedDropWeakestFrac")
    assert spec is not None
    assert "UNCALIBRATED" in (spec.notes or "")
