"""tx handling in the paramstest / zarr geometry readers ([KL-066]).

The C FitSetupZarr never wrote ``txFit``; a missing tx used to become tx = 0
silently. These tests pin: precedence (explicit override > txFit > tx >
fallback > 0), a loud warning when tx is absent, the Parameters.txt fallback
used by ``build_model_from_paramstest``, and ``geometry_from_zarr`` reading the
zarr's ``tx``.
"""
import warnings

import numpy as np
import pytest

from midas_pf_odf import io
from midas_pf_odf.io import (
    _tx_fallback_from_parameters_txt,
    geometry_from_paramstest,
    parse_paramstest,
)
from tests.test_io import _PARAMSTEST


def _geom(tmp_path, extra="", **kw):
    p = tmp_path / "paramstest.txt"
    p.write_text(_PARAMSTEST + extra)
    return geometry_from_paramstest(
        parse_paramstest(p), n_pixels_y=1475, n_pixels_z=1679, n_frames=1440, **kw)


def _tx_warnings(rec):
    return [w for w in rec if "tx" in str(w.message) and "paramstest" in str(w.message)]


def test_txfit_in_file_is_used_without_warning(tmp_path):
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        g = _geom(tmp_path, "txFit 0.25\n")
    assert g.tx == 0.25
    assert not _tx_warnings(rec)


def test_plain_tx_key_is_used(tmp_path):
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        g = _geom(tmp_path, "tx 0.3\n")
    assert g.tx == 0.3 and not _tx_warnings(rec)


def test_txfit_wins_over_tx_key(tmp_path):
    assert _geom(tmp_path, "tx 0.3\ntxFit 0.25\n").tx == 0.25


def test_missing_tx_defaults_to_zero_with_loud_warning(tmp_path):
    with pytest.warns(UserWarning, match="neither txFit nor tx"):
        g = _geom(tmp_path)
    assert g.tx == 0.0


def test_fallback_used_with_warning(tmp_path):
    with pytest.warns(UserWarning, match="fallback tx = 0.114081"):
        g = _geom(tmp_path, tx_fallback_deg=0.114081)
    assert g.tx == pytest.approx(0.114081)


def test_file_value_beats_fallback_silently(tmp_path):
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        g = _geom(tmp_path, "txFit 0.25\n", tx_fallback_deg=0.114081)
    assert g.tx == 0.25 and not _tx_warnings(rec)


def test_explicit_override_always_wins(tmp_path):
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        g = _geom(tmp_path, "txFit 0.25\n", tx_deg=-0.1, tx_fallback_deg=0.114081)
    assert g.tx == -0.1 and not _tx_warnings(rec)
    assert _geom(tmp_path, tx_deg=0.0).tx == 0.0      # explicit zero is honoured


def test_parameters_txt_fallback_parsing(tmp_path):
    assert _tx_fallback_from_parameters_txt(tmp_path) is None            # no file
    (tmp_path / "Parameters.txt").write_text("Lsd 834247.9   ### x\nty -0.19\n")
    assert _tx_fallback_from_parameters_txt(tmp_path) is None            # no tx key
    (tmp_path / "Parameters.txt").write_text(
        "Lsd 834247.9\ntx 0.114081       ### TILT OF THE DETECTOR AROUND X RAY BEAM\nty -0.19\n")
    assert _tx_fallback_from_parameters_txt(tmp_path) == pytest.approx(0.114081)


def test_build_model_passes_parameters_txt_fallback_and_override(tmp_path, monkeypatch):
    """build_model_from_paramstest hands the sibling Parameters.txt tx (and the
    explicit override) to geometry_from_paramstest. Everything after the
    geometry (hkls.csv etc.) is absent here, so stop right there."""
    (tmp_path / "paramstest.txt").write_text(_PARAMSTEST)
    (tmp_path / "Parameters.txt").write_text("tx 0.114081\n")
    seen = {}

    class _Stop(Exception):
        pass

    def _spy(params, **kw):
        seen.update(kw)
        raise _Stop

    monkeypatch.setattr(io, "geometry_from_paramstest", _spy)
    with pytest.raises(_Stop):
        io.build_model_from_paramstest(tmp_path, n_pixels_y=1475, n_pixels_z=1679, n_frames=1440)
    assert seen["tx_fallback_deg"] == pytest.approx(0.114081) and seen["tx_deg"] is None
    seen.clear()
    with pytest.raises(_Stop):
        io.build_model_from_paramstest(tmp_path, n_pixels_y=1475, n_pixels_z=1679, n_frames=1440,
                                       tx_deg=0.0)
    assert seen["tx_deg"] == 0.0


def _zarr_params(with_tx):
    d = {k: np.array([v]) for k, v in dict(
        Lsd=834247.9, YCen=1017.18, ZCen=1024.52, PixelSize=200.0, numFilesPerScan=1440,
        numPxY=2048, numPxZ=2048, MinEta=6.0, Wavelength=0.162897, ty=-0.19, tz=0.138,
        Wedge=0.0, _omega_start=180.0, _omega_step=-0.25).items()}
    if with_tx:
        d["tx"] = np.array([0.114081])
    return d


def test_geometry_from_zarr_reads_tx(monkeypatch):
    monkeypatch.setattr(io, "read_zarr_params", lambda p: _zarr_params(True))
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        g = io.geometry_from_zarr("x.zip")
    assert g.tx == pytest.approx(0.114081)
    assert not [w for w in rec if "tx" in str(w.message)]


def test_geometry_from_zarr_without_tx_warns(monkeypatch):
    monkeypatch.setattr(io, "read_zarr_params", lambda p: _zarr_params(False))
    with pytest.warns(UserWarning, match="no 'tx'"):
        g = io.geometry_from_zarr("x.zip")
    assert g.tx == 0.0
