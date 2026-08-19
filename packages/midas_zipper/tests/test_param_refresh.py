"""Tests for refreshing an existing .MIDAS.zip's analysis parameters.

The bug being covered: ``zip_convert`` reuses any archive it finds and every
downstream stage reads geometry from the zarr, so editing ``tx`` and re-running
into the same result folder kept the OLD value with no warning. These tests pin
that a refresh takes, that it refuses the parameters the stored frames depend
on, and that a ``zip -u`` which silently does nothing is reported as a failure
rather than a success.
"""

from __future__ import annotations

import shutil

import numpy as np
import pytest

zarr = pytest.importorskip("zarr")

from midas_zipper.ff_zip import (create_zarr_structure, parse_parameter_file,
                                 write_analysis_parameters)
from midas_zipper.param_refresh import (ANALYSIS_PATH, BakedInParamChanged,
                                        ParamRefreshError,
                                        coerce_analysis_params,
                                        diff_analysis_params,
                                        refresh_analysis_params)

pytestmark = pytest.mark.skipif(shutil.which("zip") is None,
                                reason="Info-ZIP 'zip' not on PATH")

BASE_PARAMS = """\
tx 0.0
ty -0.15
tz 0.02
Lsd 1000000.0
BC 1024.0 1024.5
Wavelength 0.123
SkipFrame 1
Padding 6
RingThresh 1 100
RingThresh 2 80
RingThresh 3 60
LatticeConstant 4.078 4.078 4.078 90 90 90
SpaceGroup 225
ImTransOpt 0
MinPeakSNR 3.0
Wedge 0.0
MaskFile /some/mask.tif
"""


def _write_params(tmp_path, text, name="Parameters.txt"):
    p = tmp_path / name
    p.write_text(text)
    return p


def _build_zip(tmp_path, param_file, name="d.MIDAS.zip"):
    """Build an archive through the real create path."""
    cfg = parse_parameter_file(str(param_file))
    fn = tmp_path / name
    store = zarr.ZipStore(str(fn), mode="w")
    root = zarr.group(store=store, overwrite=True)
    groups = create_zarr_structure(root)
    write_analysis_parameters(groups, cfg)
    root["exchange"].create_dataset("data", data=np.zeros((8, 32, 32), np.uint16))
    store.close()
    return fn


def _read(fn, key):
    return np.asarray(zarr.open(str(fn), "r")[f"{ANALYSIS_PATH}/{key}"][...])


# ── coercion is shared with the create path ─────────────────────────────────
def test_coercion_matches_what_the_create_path_wrote(tmp_path):
    """A freshly built archive must diff clean against its own parameter file.

    If the refresh path typed anything differently from the create path, every
    run would 'refresh' keys nobody touched.
    """
    pf = _write_params(tmp_path, BASE_PARAMS)
    fn = _build_zip(tmp_path, pf)
    refreshable, baked_in, _ = diff_analysis_params(fn, pf)
    assert refreshable == []
    assert baked_in == []


def test_bc_fans_out_to_ycen_zcen():
    params, _ = coerce_analysis_params({"BC": [1024.0, 1024.5]})
    assert sorted(p.key for p in params) == ["YCen", "ZCen"]
    assert all(p.source_key == "BC" for p in params)


def test_omega_step_lands_in_the_measurement_group():
    params, _ = coerce_analysis_params({"OmegaStep": 0.25})
    assert params[0].key == "step"
    assert params[0].path.startswith("measurement/")


# ── the refresh itself ───────────────────────────────────────────────────────
def test_scalar_geometry_refresh_takes(tmp_path):
    pf = _write_params(tmp_path, BASE_PARAMS)
    fn = _build_zip(tmp_path, pf)
    assert _read(fn, "tx") == pytest.approx(0.0)

    _write_params(tmp_path, BASE_PARAMS.replace("tx 0.0", "tx -0.2670"))
    report = refresh_analysis_params(fn, pf)

    assert report.changed_keys == ["tx"]
    assert _read(fn, "tx") == pytest.approx(-0.2670)
    # neighbours untouched
    assert _read(fn, "ty") == pytest.approx(-0.15)
    assert _read(fn, "Lsd") == pytest.approx(1000000.0)


def test_bc_edit_reaches_both_ycen_and_zcen(tmp_path):
    pf = _write_params(tmp_path, BASE_PARAMS)
    fn = _build_zip(tmp_path, pf)
    _write_params(tmp_path, BASE_PARAMS.replace("BC 1024.0 1024.5",
                                                "BC 1030.0 1019.0"))
    refresh_analysis_params(fn, pf)
    assert _read(fn, "YCen") == pytest.approx(1030.0)
    assert _read(fn, "ZCen") == pytest.approx(1019.0)


def test_shape_change_is_handled(tmp_path):
    """RingThresh growing from 3 rings to 5 must replace, not append.

    This is the case a plain zarr append cannot do -- ZipStore raises
    NotImplementedError on a shape change -- so it pins the zip -u path.
    """
    pf = _write_params(tmp_path, BASE_PARAMS)
    fn = _build_zip(tmp_path, pf)
    assert _read(fn, "RingThresh").shape == (3, 2)

    _write_params(tmp_path, BASE_PARAMS.replace(
        "RingThresh 3 60", "RingThresh 3 60\nRingThresh 4 40\nRingThresh 5 25"))
    refresh_analysis_params(fn, pf)

    got = _read(fn, "RingThresh")
    assert got.shape == (5, 2)
    assert got[4].tolist() == [5.0, 25.0]


def test_zero_value_is_written_not_dropped(tmp_path):
    """Zero equals zarr's fill value, so the chunk can be omitted entirely.

    Without write_empty_chunks the .zarray would be replaced while the OLD
    chunk stayed in the archive -- the parameter would read back unchanged.
    Setting a correction to 0 to disable it is exactly when this matters.
    """
    pf = _write_params(tmp_path, BASE_PARAMS)
    fn = _build_zip(tmp_path, pf)
    _write_params(tmp_path, BASE_PARAMS.replace("MinPeakSNR 3.0", "MinPeakSNR 0"))
    refresh_analysis_params(fn, pf)
    assert _read(fn, "MinPeakSNR") == pytest.approx(0.0)


def test_string_parameter_refresh(tmp_path):
    pf = _write_params(tmp_path, BASE_PARAMS)
    fn = _build_zip(tmp_path, pf)
    _write_params(tmp_path, BASE_PARAMS.replace("/some/mask.tif", "/other/m.tif"))
    refresh_analysis_params(fn, pf)
    assert b"/other/m.tif" in bytes(_read(fn, "MaskFile").flatten()[0])


def test_refresh_is_idempotent(tmp_path):
    pf = _write_params(tmp_path, BASE_PARAMS)
    fn = _build_zip(tmp_path, pf)
    _write_params(tmp_path, BASE_PARAMS.replace("tx 0.0", "tx -0.2670"))
    first = refresh_analysis_params(fn, pf)
    second = refresh_analysis_params(fn, pf)
    assert first.changed_keys == ["tx"]
    assert second.applied == []          # nothing left to do, no zip call


def test_frames_survive_a_refresh(tmp_path):
    pf = _write_params(tmp_path, BASE_PARAMS)
    fn = _build_zip(tmp_path, pf)
    _write_params(tmp_path, BASE_PARAMS.replace("tx 0.0", "tx -0.2670"))
    refresh_analysis_params(fn, pf)
    data = zarr.open(str(fn), "r")["exchange/data"]
    assert data.shape == (8, 32, 32)


def test_refresh_within_the_same_second_still_takes(tmp_path):
    """`zip -u` skips a file that is not newer than the archived entry.

    Built and refreshed back to back, the naive call exits 12 and changes
    nothing. The staged mtimes are forced past the archive's, so this must
    still land -- it is the regression that made the whole mechanism unsafe.
    """
    pf = _write_params(tmp_path, BASE_PARAMS)
    fn = _build_zip(tmp_path, pf)                 # no sleep: same wall second
    _write_params(tmp_path, BASE_PARAMS.replace("Wedge 0.0", "Wedge 0.35"))
    refresh_analysis_params(fn, pf)
    assert _read(fn, "Wedge") == pytest.approx(0.35)


# ── refusals ─────────────────────────────────────────────────────────────────
def test_baked_in_key_is_refused(tmp_path):
    pf = _write_params(tmp_path, BASE_PARAMS)
    fn = _build_zip(tmp_path, pf)
    _write_params(tmp_path, BASE_PARAMS.replace("SkipFrame 1", "SkipFrame 2"))
    with pytest.raises(BakedInParamChanged) as ei:
        refresh_analysis_params(fn, pf)
    assert ei.value.keys == ["SkipFrame"]
    # and the archive is untouched
    assert _read(fn, "SkipFrame") == 1


def test_baked_in_refusal_blocks_the_safe_keys_too(tmp_path):
    """A mixed edit must not half-apply: the run needs a rebuild, not a patch."""
    pf = _write_params(tmp_path, BASE_PARAMS)
    fn = _build_zip(tmp_path, pf)
    _write_params(tmp_path, BASE_PARAMS.replace("SkipFrame 1", "SkipFrame 2")
                                       .replace("tx 0.0", "tx -0.2670"))
    with pytest.raises(BakedInParamChanged):
        refresh_analysis_params(fn, pf)
    assert _read(fn, "tx") == pytest.approx(0.0)


def test_allow_baked_in_applies_them(tmp_path):
    pf = _write_params(tmp_path, BASE_PARAMS)
    fn = _build_zip(tmp_path, pf)
    _write_params(tmp_path, BASE_PARAMS.replace("SkipFrame 1", "SkipFrame 2"))
    report = refresh_analysis_params(fn, pf, allow_baked_in=True)
    assert "SkipFrame" in report.changed_keys
    assert _read(fn, "SkipFrame") == 2


def test_dry_run_writes_nothing(tmp_path):
    pf = _write_params(tmp_path, BASE_PARAMS)
    fn = _build_zip(tmp_path, pf)
    _write_params(tmp_path, BASE_PARAMS.replace("tx 0.0", "tx -0.2670"))
    report = refresh_analysis_params(fn, pf, dry_run=True)
    assert report.changed_keys == ["tx"]
    assert _read(fn, "tx") == pytest.approx(0.0)


def test_verification_turns_a_silent_noop_into_an_error(tmp_path, monkeypatch):
    """If the rewrite does not take, the caller must hear about it."""
    import midas_zipper.param_refresh as pr

    pf = _write_params(tmp_path, BASE_PARAMS)
    fn = _build_zip(tmp_path, pf)
    _write_params(tmp_path, BASE_PARAMS.replace("tx 0.0", "tx -0.2670"))

    class _FakeOK:
        returncode = 0
        stdout = stderr = ""

    monkeypatch.setattr(pr.subprocess, "run", lambda *a, **k: _FakeOK())
    with pytest.raises(ParamRefreshError, match="did not take"):
        refresh_analysis_params(fn, pf)


def test_nonzero_zip_exit_is_an_error(tmp_path, monkeypatch):
    import midas_zipper.param_refresh as pr

    pf = _write_params(tmp_path, BASE_PARAMS)
    fn = _build_zip(tmp_path, pf)
    _write_params(tmp_path, BASE_PARAMS.replace("tx 0.0", "tx -0.2670"))

    class _Fake12:
        returncode = 12
        stdout = stderr = ""

    monkeypatch.setattr(pr.subprocess, "run", lambda *a, **k: _Fake12())
    with pytest.raises(ParamRefreshError, match="nothing to do"):
        refresh_analysis_params(fn, pf)
