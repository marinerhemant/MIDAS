"""Validator tests — check that real mistakes get caught."""

from __future__ import annotations

import textwrap
from pathlib import Path as FsPath

import pytest

from midas_params import Path, Severity
from midas_params.validator import validate


MIDAS_ROOT = FsPath(__file__).resolve().parents[3]
FF_EXAMPLE = MIDAS_ROOT / "FF_HEDM" / "Example" / "Parameters.txt"
NF_EXAMPLE = MIDAS_ROOT / "NF_HEDM" / "Example" / "ps_au.txt"


# ─── Cross-field rule coverage ───────────────────────────────────────────────


def test_omega_step_direction_error(tmp_path):
    fn = tmp_path / "p.txt"
    fn.write_text(textwrap.dedent("""
        OmegaStart 0
        OmegaEnd 180
        OmegaStep -0.25
    """).strip())
    r = validate(str(fn), Path.FF)
    rules = {i.rule for i in r.errors}
    assert "omega_step_direction" in rules


def test_omega_step_direction_ok(tmp_path):
    fn = tmp_path / "p.txt"
    fn.write_text(textwrap.dedent("""
        OmegaStart 180
        OmegaEnd -180
        OmegaStep -0.25
    """).strip())
    r = validate(str(fn), Path.FF)
    assert not any(i.rule == "omega_step_direction" for i in r.errors)


def test_startnr_gt_endnr_error(tmp_path):
    fn = tmp_path / "p.txt"
    fn.write_text("StartNr 100\nEndNr 50\n")
    r = validate(str(fn), Path.FF)
    assert any(i.rule == "startnr_le_endnr" for i in r.errors)


def test_overallring_not_in_ringthresh(tmp_path):
    fn = tmp_path / "p.txt"
    fn.write_text(textwrap.dedent("""
        RingThresh 1 100
        RingThresh 2 100
        OverAllRingToIndex 7
    """).strip())
    r = validate(str(fn), Path.FF)
    matches = [i for i in r.errors if i.rule == "overallring_in_ringthresh"]
    assert matches, "Should catch OverAllRingToIndex not in RingThresh"
    assert "7" in matches[0].message


def test_nf_lsd_count_mismatch(tmp_path):
    fn = tmp_path / "p.txt"
    fn.write_text(textwrap.dedent("""
        nDistances 3
        Lsd 8000
        Lsd 10000
        BC 1000 10
        BC 1000 10
        BC 1000 10
    """).strip())
    r = validate(str(fn), Path.NF)
    # Lsd count (2) should not equal nDistances (3)
    errs = [i for i in r.errors if i.rule == "nf_multi_entry_count_matches" and i.key == "Lsd"]
    assert errs, "Should catch Lsd count mismatch"
    assert "2 entries" in errs[0].message and "nDistances=3" in errs[0].message


def test_nf_bc_count_ok(tmp_path):
    fn = tmp_path / "p.txt"
    fn.write_text(textwrap.dedent("""
        nDistances 2
        Lsd 8000
        Lsd 10000
        BC 985 17
        BC 985 24
    """).strip())
    r = validate(str(fn), Path.NF)
    assert not any(i.rule == "nf_multi_entry_count_matches" for i in r.errors)


def test_nf_omega_range_multi_entries_allowed(tmp_path):
    """OmegaRange may appear any number of times without triggering the count rule."""
    fn = tmp_path / "p.txt"
    fn.write_text(textwrap.dedent("""
        nDistances 2
        Lsd 8000
        Lsd 10000
        BC 985 17
        BC 985 24
        OmegaRange -180 0
        OmegaRange 0 180
        OmegaRange -90 90
    """).strip())
    r = validate(str(fn), Path.NF)
    assert not any(
        i.rule == "nf_multi_entry_count_matches" and i.key == "OmegaRange"
        for i in r.errors + r.warnings
    ), "OmegaRange should not be count-checked"


# ─── OmegaRange / BoxSize arity and bounds ───────────────────────────────────


def test_omega_range_arity_wrong(tmp_path):
    fn = tmp_path / "p.txt"
    fn.write_text("OmegaRange -180 0 180\n")  # 3 values
    r = validate(str(fn), Path.FF)
    assert any(i.rule == "omega_range_arity" for i in r.errors)


def test_omega_range_arity_correct(tmp_path):
    fn = tmp_path / "p.txt"
    fn.write_text("OmegaRange -180 180\n")
    r = validate(str(fn), Path.FF)
    assert not any(i.rule == "omega_range_arity" for i in r.errors)


def test_omega_range_reversed(tmp_path):
    fn = tmp_path / "p.txt"
    fn.write_text("OmegaRange 180 -180\n")  # reversed
    r = validate(str(fn), Path.FF)
    assert any(i.rule == "omega_range_ordered" for i in r.errors)


def test_box_size_arity_wrong(tmp_path):
    fn = tmp_path / "p.txt"
    fn.write_text("BoxSize -1000 1000 -1000\n")  # 3 values
    r = validate(str(fn), Path.FF)
    assert any(i.rule == "box_size_arity" for i in r.errors)


def test_box_size_ordered_wrong(tmp_path):
    fn = tmp_path / "p.txt"
    fn.write_text("BoxSize 1000 -1000 -1000 1000\n")  # Ymin > Ymax
    r = validate(str(fn), Path.FF)
    assert any(i.rule == "box_size_ordered" for i in r.errors)


def test_omega_range_within_scan_error(tmp_path):
    fn = tmp_path / "p.txt"
    fn.write_text(textwrap.dedent("""
        OmegaStart 0
        OmegaEnd 180
        OmegaStep 0.25
        OmegaRange -90 270
        BoxSize -1000 1000 -1000 1000
    """).strip())
    r = validate(str(fn), Path.FF)
    errs = [i for i in r.errors if i.rule == "omega_range_within_scan"]
    assert errs
    assert "outside the scanned ω range" in errs[0].message


def test_omega_range_within_scan_tolerance(tmp_path):
    """Window on exact scan boundary should not trigger due to tolerance."""
    fn = tmp_path / "p.txt"
    fn.write_text(textwrap.dedent("""
        OmegaStart 180
        OmegaEnd -180
        OmegaStep -0.25
        OmegaRange -180 180
        BoxSize -1000 1000 -1000 1000
    """).strip())
    r = validate(str(fn), Path.FF)
    assert not any(i.rule == "omega_range_within_scan" for i in r.errors)


def test_omega_range_missing_required(tmp_path):
    fn = tmp_path / "p.txt"
    fn.write_text("OmegaStart 180\nOmegaEnd -180\nOmegaStep -0.25\n")
    r = validate(str(fn), Path.FF)
    assert any(i.key == "OmegaRange" and i.rule == "required_key_missing"
               for i in r.errors)


def test_box_size_is_optional_with_default(tmp_path):
    """BoxSize has a default of ±1e6 so it's no longer required; absence is fine."""
    fn = tmp_path / "p.txt"
    fn.write_text("OmegaStart 180\nOmegaEnd -180\nOmegaStep -0.25\n")
    r = validate(str(fn), Path.FF)
    assert not any(i.key == "BoxSize" and i.rule == "required_key_missing"
                   for i in r.errors)


# ─── Per-key validators ──────────────────────────────────────────────────────


def test_space_group_out_of_range(tmp_path):
    fn = tmp_path / "p.txt"
    fn.write_text("SpaceGroup 999\n")
    r = validate(str(fn), Path.FF)
    assert any(i.rule == "space_group_range" for i in r.errors)


def test_space_group_default_smell_info(tmp_path):
    fn = tmp_path / "p.txt"
    fn.write_text("SpaceGroup 225\n")
    r = validate(str(fn), Path.FF)
    info = [i for i in r.issues if i.rule == "space_group_default_smell"]
    assert info and info[0].severity == Severity.INFO


def test_wavelength_plausible_warning(tmp_path):
    fn = tmp_path / "p.txt"
    fn.write_text("Wavelength 100\n")  # 100 Å is way too long
    r = validate(str(fn), Path.FF)
    assert any(i.rule == "wavelength_plausible" for i in r.warnings)


def test_missing_file_error(tmp_path):
    fn = tmp_path / "p.txt"
    fn.write_text("MaskFile /definitely/does/not/exist.tif\n")
    r = validate(str(fn), Path.FF)
    assert any(i.rule == "file_exists" for i in r.errors)


def test_typo_suggestion(tmp_path):
    fn = tmp_path / "p.txt"
    fn.write_text("Completenes 0.8\n")  # typo
    r = validate(str(fn), Path.FF)
    unknown = [i for i in r.warnings if i.rule == "unknown_key"]
    assert unknown
    assert unknown[0].suggestion is not None
    assert "Completeness" in unknown[0].suggestion


# ─── Silent-failure scenarios (the point of the validator) ──────────────────


def test_ff_rotation_direction_with_mismatched_range(tmp_path):
    """Reversed OmegaStep + OmegaRange outside the actual (reversed) scan:
    both errors should fire."""
    fn = tmp_path / "p.txt"
    fn.write_text(textwrap.dedent("""
        OmegaStart 0
        OmegaEnd 180
        OmegaStep -0.25
        OmegaRange -180 0
    """).strip())
    r = validate(str(fn), Path.FF)
    rules = {i.rule for i in r.errors}
    assert "omega_step_direction" in rules
    # -180..0 is outside scan direction [0,180] (but reversed step means effective reach of -45)
    # We at minimum want the direction error to fire, and the OmegaRange check to fire too.
    assert "omega_range_within_scan" in rules


def test_nf_too_many_frames_specified(tmp_path):
    """User claims more NrFilesPerDistance than actually exist — catch by
    checking the file at the computed last index doesn't exist."""
    fn = tmp_path / "p.txt"
    # Create a synthetic NF dataset: 3 real frames, user claims 100
    datadir = tmp_path / "data"
    datadir.mkdir()
    (datadir / "sample_000001.tif").touch()
    (datadir / "sample_000002.tif").touch()
    (datadir / "sample_000003.tif").touch()
    fn.write_text(textwrap.dedent(f"""
        DataDirectory {datadir}
        OrigFileName sample
        extOrig tif
        RawStartNr 1
        NrFilesPerDistance 100
        nDistances 1
        Lsd 8000
        BC 1000 10
        Padding 6
    """).strip())
    r = validate(str(fn), Path.NF)
    rules = {i.rule for i in r.errors}
    assert "frames_exist_on_disk" in rules, \
        f"Should catch missing frames (files 4..100 don't exist). Got: {rules}"


def test_ff_wrong_filestem_caught(tmp_path):
    """User wrote FileStem that doesn't match actual files → frames don't resolve."""
    fn = tmp_path / "p.txt"
    rawdir = tmp_path / "raw"
    rawdir.mkdir()
    (rawdir / "actual_000001.ge3").touch()
    (rawdir / "actual_000002.ge3").touch()
    fn.write_text(textwrap.dedent(f"""
        RawFolder {rawdir}
        FileStem typo
        Ext .ge3
        Padding 6
        StartNr 1
        EndNr 2
    """).strip())
    r = validate(str(fn), Path.FF)
    assert any(i.rule == "frames_exist_on_disk" for i in r.errors)


def test_ff_correct_files_present_passes(tmp_path):
    """Correctly-matching FileStem + files → no frames_exist_on_disk error."""
    fn = tmp_path / "p.txt"
    rawdir = tmp_path / "raw"
    rawdir.mkdir()
    for i in range(1, 11):
        (rawdir / f"good_{i:06d}.ge3").touch()
    fn.write_text(textwrap.dedent(f"""
        RawFolder {rawdir}
        FileStem good
        Ext .ge3
        Padding 6
        StartNr 1
        EndNr 10
    """).strip())
    r = validate(str(fn), Path.FF)
    assert not any(i.rule == "frames_exist_on_disk" for i in r.errors)


def test_missing_datadirectory_nf(tmp_path):
    fn = tmp_path / "p.txt"
    fn.write_text("DataDirectory /definitely/not/a/real/path\n")
    r = validate(str(fn), Path.NF)
    assert any(i.rule == "directory_exists" for i in r.errors)


def test_nf_seed_orientations_missing(tmp_path):
    fn = tmp_path / "p.txt"
    fn.write_text("SeedOrientations /tmp/definitely_not_there_seeds.txt\n")
    r = validate(str(fn), Path.NF)
    assert any(i.rule == "file_exists" and i.key == "SeedOrientations"
               for i in r.errors)


def test_ff_extension_without_dot_still_finds_files(tmp_path):
    """Ext specified as "ge3" (no dot) should still resolve files."""
    fn = tmp_path / "p.txt"
    rawdir = tmp_path / "raw"
    rawdir.mkdir()
    for i in range(1, 4):
        (rawdir / f"sample_{i:06d}.ge3").touch()
    fn.write_text(textwrap.dedent(f"""
        RawFolder {rawdir}
        FileStem sample
        Ext ge3
        Padding 6
        StartNr 1
        EndNr 3
    """).strip())
    r = validate(str(fn), Path.FF)
    assert not any(i.rule == "frames_exist_on_disk" for i in r.errors)


def test_boundary_frames_checked(tmp_path):
    """Last file missing should be caught even if first is present.

    FF file numbering on disk uses StartFileNrFirstLayer + NrFilesPerSweep,
    NOT StartNr/EndNr (those are frame indices within multi-frame containers).
    """
    fn = tmp_path / "p.txt"
    rawdir = tmp_path / "raw"
    rawdir.mkdir()
    for i in [1, 2, 3]:
        (rawdir / f"sample_{i:06d}.ge3").touch()
    # User claims a 10-file sweep starting at 1, but only files 1..3 exist
    fn.write_text(textwrap.dedent(f"""
        RawFolder {rawdir}
        FileStem sample
        Ext .ge3
        Padding 6
        StartFileNrFirstLayer 1
        NrFilesPerSweep 10
    """).strip())
    r = validate(str(fn), Path.FF)
    assert any(i.rule == "frames_exist_on_disk" for i in r.errors)


def test_single_multiframe_ge_file_passes(tmp_path):
    """A real FF scenario: one GE container file at StartFileNrFirstLayer=410,
    containing 1440 internal frames (StartNr=1 through EndNr=1440). The
    file-existence check should find the ONE on-disk file, not look for
    1440 imaginary files."""
    fn = tmp_path / "p.txt"
    rawdir = tmp_path / "raw"
    rawdir.mkdir()
    # Single multi-frame GE file at 410
    (rawdir / f"sample_{410:06d}.ge3").touch()
    fn.write_text(textwrap.dedent(f"""
        RawFolder {rawdir}
        FileStem sample
        Ext .ge3
        Padding 6
        StartFileNrFirstLayer 410
        NrFilesPerSweep 1
        StartNr 1
        EndNr 1440
    """).strip())
    r = validate(str(fn), Path.FF)
    frame_errors = [i for i in r.errors if i.rule == "frames_exist_on_disk"]
    assert not frame_errors, \
        f"Should find the single GE file, got: {[i.message for i in frame_errors]}"


def test_omega_range_multiple_entries_each_checked(tmp_path):
    """Multiple OmegaRange entries — each gets its own arity and bounds check."""
    fn = tmp_path / "p.txt"
    fn.write_text(textwrap.dedent("""
        OmegaStart 0
        OmegaEnd 180
        OmegaStep 0.25
        OmegaRange 0 90
        OmegaRange 90 180
        OmegaRange -90 45
    """).strip())
    r = validate(str(fn), Path.FF)
    # First two are in-range, third is out-of-range
    within_scan = [i for i in r.errors if i.rule == "omega_range_within_scan"]
    assert len(within_scan) == 1
    assert "entry 3" in within_scan[0].message


def test_nf_multiple_lsd_but_one_bc(tmp_path):
    """Mismatched multi-entry counts for Lsd vs BC."""
    fn = tmp_path / "p.txt"
    fn.write_text(textwrap.dedent("""
        nDistances 2
        Lsd 8000
        Lsd 10000
        BC 985 17
    """).strip())
    r = validate(str(fn), Path.NF)
    bc_errs = [i for i in r.errors if i.rule == "nf_multi_entry_count_matches" and i.key == "BC"]
    assert bc_errs


# ─── Integration: Example files ──────────────────────────────────────────────


@pytest.mark.skipif(not FF_EXAMPLE.exists(), reason="FF Example not present")
def test_ff_example_parses_without_type_errors():
    """FF Example should have no type-coercion errors (semantic missing-key
    errors are expected since the Example is forward-sim focused)."""
    r = validate(str(FF_EXAMPLE), Path.FF)
    type_issues = [i for i in r.issues if i.rule == "type_coercion"]
    assert not type_issues, type_issues


@pytest.mark.skipif(not NF_EXAMPLE.exists(), reason="NF Example not present")
def test_nf_example_parses_without_type_errors():
    r = validate(str(NF_EXAMPLE), Path.NF)
    type_issues = [i for i in r.issues if i.rule == "type_coercion"]
    assert not type_issues, type_issues


@pytest.mark.skipif(not NF_EXAMPLE.exists(), reason="NF Example not present")
def test_nf_example_passes_multi_entry_rule():
    """The NF Example has nDistances=2 with 2 Lsd and 2 BC entries — should pass."""
    r = validate(str(NF_EXAMPLE), Path.NF)
    assert not any(i.rule == "nf_multi_entry_count_matches" for i in r.errors), \
        [i for i in r.errors if i.rule == "nf_multi_entry_count_matches"]


# ─── RI path coverage ────────────────────────────────────────────────────────


def test_ri_param_file_exercises_new_keys_cleanly(tmp_path):
    """A realistic RI parameter file with the keys IntegratorZarrOMP and
    DetectorMapper read should produce zero 'unknown key' warnings."""
    fn = tmp_path / "ri.txt"
    fn.write_text(textwrap.dedent("""
        NrPixels 2048
        px 200
        Lsd 1000000
        BC 1024 1024
        Wavelength 0.22291
        LatticeConstant 4.08 4.08 4.08 90 90 90
        SpaceGroup 225
        StartNr 1
        EndNr 100
        RMin 10
        RMax 1500
        RBinSize 0.25
        EtaMin -180
        EtaMax 180
        EtaBinSize 5
        OmegaStart 0
        OmegaStep 0.25
        OmegaSumFrames 10
        DoPeakFit 1
        MultiplePeaks 1
        PeakLocation 245.3
        PeakLocation 347.1
        FitROIPadding 25
        SNIPIterations 50
        AutoDetectPeaks 0
        SolidAngleCorrection 1
        PolarizationCorrection 1
        PolarizationFraction 0.99
        p0 0
        p1 0
        p2 0
        SumImages 0
        SaveIndividualFrames 1
        Normalize 1
    """).strip())
    r = validate(str(fn), Path.RI)
    unknowns = [i.key for i in r.warnings if i.rule == "unknown_key"]
    assert not unknowns, f"Unexpected unknown keys on RI path: {unknowns}"


def test_p_coefficients_scoped_away_from_nf():
    """A defense-in-depth registry assertion: p0..p14 and tolP0..tolP14 are
    FF/PF/RI-only. The validator doesn't currently emit a 'wrong path' warning
    for known-but-non-applicable keys, but the scoping here ensures the
    wizard, diagnose, and any future per-path warning don't suggest them to
    NF users."""
    from midas_params.registry import by_name
    by = by_name()
    for i in range(15):
        assert Path.NF not in by[f"p{i}"].applies_to
        assert Path.NF not in by[f"tolP{i}"].applies_to
