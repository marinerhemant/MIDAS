"""check_zip_vs_params: a zip must not silently disagree with the parameter file (#75 follow-up).

Multi-detector runs read every panel's zip as built and never refresh it from
Parameters.txt, so a Wedge or OmegaStart in the file that the zip does not hold
was ignored with no message. These tests pin the check, the message, the
key-restricted refresh it points to, and that OmegaStart reaches the dataset
peakfit actually reads (scan_parameters/start).
"""

from __future__ import annotations

import shutil

import numpy as np
import pytest

zarr = pytest.importorskip("zarr")

from midas_zipper.ff_zip import (create_zarr_structure, parse_parameter_file,
                                 write_analysis_parameters)
from midas_zipper.param_refresh import (MEASUREMENT_PATH, SCAN_LEVEL_KEYS,
                                        check_zip_vs_params,
                                        mismatch_message,
                                        refresh_analysis_params)

pytestmark = pytest.mark.skipif(shutil.which("zip") is None,
                                reason="Info-ZIP 'zip' not on PATH")

PARAMS = """\
Lsd 3300000.0
BC 2300.0 2150.0
tx 15.0
Wavelength 0.172979
SkipFrame 1
OmegaStart -30.0
OmegaStep 0.25
Wedge 0.0
SpaceGroup 225
LatticeConstant 5.4 5.4 5.4 90 90 90
RingThresh 1 100
"""


def _params(tmp_path, text, name="Parameters.txt"):
    p = tmp_path / name
    p.write_text(text)
    return p


def _zip(tmp_path, param_file, name="p1.MIDAS.zip"):
    cfg = parse_parameter_file(str(param_file))
    fn = tmp_path / name
    store = zarr.ZipStore(str(fn), mode="w")
    root = zarr.group(store=store, overwrite=True)
    write_analysis_parameters(create_zarr_structure(root), cfg)
    root["exchange"].create_dataset("data", data=np.zeros((4, 8, 8), np.uint16))
    store.close()
    return fn


def _start(fn):
    return float(np.asarray(zarr.open(str(fn), "r")[f"{MEASUREMENT_PATH}/start"][...]).flat[0])


def test_matching_file_and_zip_report_nothing(tmp_path):
    pf = _params(tmp_path, PARAMS)
    assert check_zip_vs_params(_zip(tmp_path, pf), pf) == []


def test_wedge_and_omegastart_set_in_the_file_are_reported(tmp_path):
    fn = _zip(tmp_path, _params(tmp_path, PARAMS))
    pf = _params(tmp_path, PARAMS.replace("Wedge 0.0", "Wedge -0.296")
                 .replace("OmegaStart -30.0", "OmegaStart -30.65"))
    got = {c.source_key for c in check_zip_vs_params(fn, pf)}
    assert got == {"Wedge", "OmegaStart"}


def test_per_panel_geometry_is_not_a_scan_level_difference(tmp_path):
    fn = _zip(tmp_path, _params(tmp_path, PARAMS))
    pf = _params(tmp_path, PARAMS.replace("Lsd 3300000.0", "Lsd 3420000.0")
                 .replace("tx 15.0", "tx 105.0"))
    assert check_zip_vs_params(fn, pf) == []
    assert {c.source_key for c in check_zip_vs_params(fn, pf, keys=None)} == {"Lsd", "tx"}


def test_keys_the_file_does_not_set_are_never_reported(tmp_path):
    fn = _zip(tmp_path, _params(tmp_path, PARAMS))
    pf = _params(tmp_path, "Lsd 3300000.0\n")
    assert check_zip_vs_params(fn, pf, keys=None) == []


def test_text_round_trip_noise_is_not_a_difference(tmp_path):
    fn = _zip(tmp_path, _params(tmp_path, PARAMS))
    pf = _params(tmp_path, PARAMS.replace("Wavelength 0.172979", "Wavelength 0.1729790000000001"))
    assert check_zip_vs_params(fn, pf) == []


def test_omegastart_edit_reaches_the_dataset_peakfit_reads(tmp_path):
    fn = _zip(tmp_path, _params(tmp_path, PARAMS))
    assert _start(fn) == pytest.approx(-30.0 - 1 * 0.25)
    pf = _params(tmp_path, PARAMS.replace("OmegaStart -30.0", "OmegaStart -30.65"))
    changes = check_zip_vs_params(fn, pf)
    assert {c.full_key.rsplit("/", 1)[-1] for c in changes} == {"OmegaStart", "start"}

    refresh_analysis_params(fn, pf, keys=["OmegaStart"])
    assert _start(fn) == pytest.approx(-30.65 - 0.25)
    assert check_zip_vs_params(fn, pf) == []


def test_keys_restricted_refresh_leaves_the_panels_own_geometry(tmp_path):
    fn = _zip(tmp_path, _params(tmp_path, PARAMS))
    pf = _params(tmp_path, PARAMS.replace("Wedge 0.0", "Wedge -0.296")
                 .replace("Lsd 3300000.0", "Lsd 9999999.0"))
    report = refresh_analysis_params(fn, pf, keys=["Wedge"])
    assert report.changed_keys == ["Wedge"]
    root = zarr.open(str(fn), "r")
    assert float(root["analysis/process/analysis_parameters/Wedge"][0]) == pytest.approx(-0.296)
    assert float(root["analysis/process/analysis_parameters/Lsd"][0]) == pytest.approx(3300000.0)


def test_message_names_the_keys_and_both_ways_out(tmp_path):
    fn = _zip(tmp_path, _params(tmp_path, PARAMS))
    pf = _params(tmp_path, PARAMS.replace("Wedge 0.0", "Wedge -0.296"))
    msg = mismatch_message({str(fn): check_zip_vs_params(fn, pf)}, pf,
                           accept_hint="pass --accept-zip-params to keep")
    assert "Wedge" in msg and "p1.MIDAS.zip" in msg
    assert "midas-refresh-zarr-params" in msg and "--keys Wedge" in msg
    assert "edit Parameters.txt" in msg and "--accept-zip-params" in msg
    assert "Nothing was run" in msg


def test_scan_level_keys_are_the_ones_stages_read_from_the_zip():
    assert {"Wedge", "OmegaStart"} <= set(SCAN_LEVEL_KEYS)
    assert not {"Lsd", "BC", "tx", "ty", "tz"} & set(SCAN_LEVEL_KEYS)


def test_per_panel_thresholds_and_ring_radius_are_not_scan_level():
    from midas_zipper.param_refresh import PANEL_LEVEL_KEYS
    assert {"RingThresh", "MaxRingRad"} == set(PANEL_LEVEL_KEYS)
    assert not set(PANEL_LEVEL_KEYS) & set(SCAN_LEVEL_KEYS)
