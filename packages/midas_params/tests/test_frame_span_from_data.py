"""The frame span belongs to the dataset, not to the parameter file.

For a multi-frame container (GE, HDF5, Zarr) every frame lives in one file, so
StartNr..EndNr is just the length of that file's frame axis. Demanding the user
restate it creates a second place to be wrong, and getting it wrong invents a
scan extent: a real 1440-frame HDF5 scan written as `NrFilesPerSweep 1` was read
as a ONE-frame scan, which reported a -180..180 OmegaRange as reaching only
-179.75 and flagged a correct parameter file as out of bounds.
"""

from __future__ import annotations

import textwrap

import pytest

from midas_params import Path, Severity
from midas_params.crossfield import data_carries_frame_span, is_frame_per_file
from midas_params.discovery import discover_from_file
from midas_params.validator import validate


# One HDF5 file holding every frame — the case that failed.
MULTIFRAME = """
    Ext .vrx.h5
    NrFilesPerSweep 1
    OmegaStart -180
    OmegaStep 0.25
    OmegaRange -180 180
"""

# One frame per file — here the file count really is the frame count.
FRAME_PER_FILE = """
    Ext .tif
    NrFilesPerSweep 1440
    OmegaStart -180
    OmegaStep 0.25
    OmegaRange -180 180
"""


def _write(tmp_path, body, name="p.txt"):
    fn = tmp_path / name
    fn.write_text(textwrap.dedent(body).strip())
    return str(fn)


# ─── the format predicate ────────────────────────────────────────────────────


def test_ge_and_hdf5_are_multiframe_containers():
    # A GE container with 1440 frames is a single file, so it carries its span.
    for ext in (".ge3", ".ge5", ".h5", ".hdf5", ".vrx.h5", ".zarr", ".zip"):
        assert not is_frame_per_file({"Ext": ext}), ext
        assert data_carries_frame_span({"Ext": ext}), ext


def test_single_frame_formats_do_not_carry_the_span():
    for ext in (".tif", ".tiff", ".edf", ".cbf"):
        assert is_frame_per_file({"Ext": ext}), ext
        assert not data_carries_frame_span({"Ext": ext}), ext


def test_unknown_extension_is_not_assumed_to_carry_the_span():
    # No Ext at all: we know nothing, so nothing is derived.
    assert not data_carries_frame_span({})


# ─── required-key behaviour ──────────────────────────────────────────────────


def test_multiframe_container_does_not_require_startnr_endnr(tmp_path):
    r = validate(_write(tmp_path, MULTIFRAME), Path.FF)
    missing = {i.key for i in r.issues if i.rule == "required_key_missing"}
    assert "StartNr" not in missing
    assert "EndNr" not in missing


def test_multiframe_container_says_it_is_deriving_them(tmp_path):
    r = validate(_write(tmp_path, MULTIFRAME), Path.FF)
    derived = {i.key: i for i in r.issues if i.rule == "derived_from_data"}
    assert {"StartNr", "EndNr"} <= set(derived)
    # Silence would be worse than an error: the user must be able to see that a
    # value they did not supply is being taken from the file.
    for issue in derived.values():
        assert issue.severity is Severity.INFO


def test_frame_per_file_still_requires_the_span(tmp_path):
    # Nothing carries the frame count here, so the parameter file must.
    r = validate(_write(tmp_path, FRAME_PER_FILE), Path.FF)
    missing = {i.key for i in r.issues if i.rule == "required_key_missing"}
    assert "StartNr" in missing
    assert "EndNr" in missing


# ─── the scan-extent regression ──────────────────────────────────────────────


def test_multiframe_scan_extent_is_not_inferred_from_file_count(tmp_path):
    """`NrFilesPerSweep 1` means one FILE, not one frame."""
    r = validate(_write(tmp_path, MULTIFRAME), Path.FF)
    rules = {i.rule for i in r.errors}
    assert "omega_range_within_scan" not in rules


def test_frame_per_file_scan_extent_still_checked(tmp_path):
    """Where the file count IS the frame count, the check must still bite."""
    body = """
        Ext .tif
        StartNr 1
        EndNr 100
        OmegaStart 0
        OmegaStep 0.25
        OmegaRange 0 180
    """
    r = validate(_write(tmp_path, body), Path.FF)
    # 100 frames x 0.25 deg = 25 deg of scan; a 0..180 window is out of bounds.
    assert "omega_range_within_scan" in {i.rule for i in r.errors}


def test_discovery_reads_the_frame_count_off_the_hdf5_shape(tmp_path):
    """The frame axis was being read and thrown away."""
    h5py = pytest.importorskip("h5py")
    import numpy as np

    fn = tmp_path / "scan_000001.h5"
    with h5py.File(fn, "w") as f:
        # (frames, z, y) — the first axis is the span we need.
        f.create_dataset("exchange/data", data=np.zeros((37, 8, 9), dtype="uint16"))

    res = discover_from_file(fn)
    assert res.extracted.get("nFramesInContainer") == 37
    assert res.extracted.get("StartNr") == 1
    assert res.extracted.get("EndNr") == 37
    # detector shape must not be confused with the frame axis
    assert res.extracted.get("NrPixelsZ") == 8
    assert res.extracted.get("NrPixelsY") == 9


def test_discovery_reads_the_frame_count_off_the_zarr_shape(tmp_path):
    zarr = pytest.importorskip("zarr")
    import numpy as np

    fn = tmp_path / "scan.zarr"
    z = zarr.open(str(fn), mode="w")
    z.create_dataset("exchange/data", data=np.zeros((23, 4, 5), dtype="uint16"))

    res = discover_from_file(fn)
    assert res.extracted.get("nFramesInContainer") == 23
    assert res.extracted.get("StartNr") == 1
    assert res.extracted.get("EndNr") == 23
    assert res.extracted.get("NrPixelsZ") == 4
    assert res.extracted.get("NrPixelsY") == 5


def test_zarr_span_subtracts_the_recorded_skipframe(tmp_path):
    """A MIDAS zarr records SkipFrame, so the usable span is exact.

    The real case: exchange/data is 1441 long with SkipFrame 1, and peakfit
    reports nFrames 1440. Deriving 1..1441 would put every frame one ω step out.
    """
    zarr = pytest.importorskip("zarr")
    import numpy as np

    fn = tmp_path / "scan.zarr"
    z = zarr.open(str(fn), mode="w")
    z.create_dataset("exchange/data", data=np.zeros((1441, 2, 2), dtype="uint16"))
    z.create_dataset("analysis/process/analysis_parameters/SkipFrame",
                     data=np.array([1]))

    res = discover_from_file(fn)
    assert res.extracted["nFramesInContainer"] == 1441   # the array really is 1441
    assert res.extracted["EndNr"] == 1440                # but only 1440 are usable
    assert res.extracted["SkipFrame"] == 1
    # exact now, so no hedge is needed
    assert not any("SkipFrame" in w for w in res.warnings), res.warnings


def test_derived_span_flags_that_skipframe_is_not_applied(tmp_path):
    """1441 frames with SkipFrame 1 is 1..1440 or 2..1441 — one ω step apart."""
    h5py = pytest.importorskip("h5py")
    import numpy as np

    fn = tmp_path / "scan_000001.h5"
    with h5py.File(fn, "w") as f:
        f.create_dataset("exchange/data", data=np.zeros((1441, 2, 2), dtype="uint16"))

    res = discover_from_file(fn)
    assert res.extracted["EndNr"] == 1441
    assert any("SkipFrame" in w for w in res.warnings), res.warnings


def test_explicit_span_still_defines_the_extent(tmp_path):
    """An HDF5 run that does state its span is checked against that span."""
    body = """
        Ext .h5
        NrFilesPerSweep 1
        StartNr 1
        EndNr 1440
        OmegaStart -180
        OmegaStep 0.25
        OmegaRange -180 180
    """
    r = validate(_write(tmp_path, body), Path.FF)
    assert "omega_range_within_scan" not in {i.rule for i in r.errors}
