"""`load_esrf_parameters` against real files written by real writers.

This is the path that will meet real data first, and it shipped untested. These
build genuine .mat files -- scipy for the classic format, h5py laid out the way
MATLAB v7.3 writes a struct -- rather than mocking the reader's input.
"""
import numpy as np
import pytest

from midas_dct_tt.esrf import (ESRFGeometry, esrf_default_detgeo,
                               load_esrf_parameters)
from midas_dct_tt.conventions import DCT_OMEGA_SIGN_AERO, DCT_OMEGA_SIGN_CCW


def _params(rotdir=(0.0, 0.0, -1.0), beamdir=(1.0, 0.0, 0.0)):
    dg = esrf_default_detgeo(2048, 2048, pixelsize_mm=1.4e-3, dist_mm=12.5)
    detgeo = {k: (np.asarray(v, dtype=float) if isinstance(v, (tuple, list)) else v)
              for k, v in dg.items()}
    return {"labgeo": {"beamdir": np.asarray(beamdir, dtype=float),
                       "rotdir": np.asarray(rotdir, dtype=float),
                       "labunit": "mm", "rotpos": np.zeros(3)},
            "detgeo": detgeo,
            "samgeo": {"orig": np.zeros(3)}}


def _write_v5(path, params):
    from scipy.io import savemat
    savemat(str(path), {"parameters": params})
    return path


def _write_v73(path, params):
    """Lay a struct out the way MATLAB -v7.3 does: nested groups, arrays transposed."""
    import h5py
    with h5py.File(path, "w") as f:
        root = f.create_group("parameters")
        for sub, d in params.items():
            g = root.create_group(sub)
            for k, v in d.items():
                if isinstance(v, str):
                    g.create_dataset(k, data=np.frombuffer(v.encode("utf-16-le"),
                                                           dtype=np.uint16))
                else:
                    a = np.asarray(v, dtype=float)
                    g.create_dataset(k, data=a.T if a.ndim > 1 else a)
    return path


@pytest.mark.parametrize("writer", [_write_v5, _write_v73])
def test_round_trip_through_a_real_mat_file(tmp_path, writer):
    p = writer(tmp_path / "parameters.mat", _params())
    geo = load_esrf_parameters(p)
    assert geo.omega_sign() == DCT_OMEGA_SIGN_AERO
    det, normal = geo.detector()
    assert det.pixel_um == pytest.approx(1.4)          # mm -> um
    assert det.shape == (2048, 2048)
    assert det.distance_um == pytest.approx(12500.0)   # mm -> um
    assert normal[0] == pytest.approx(1.0, abs=1e-9)   # outward, along the beam


@pytest.mark.parametrize("writer", [_write_v5, _write_v73])
def test_format_is_detected_from_magic_bytes_not_the_extension(tmp_path, writer):
    """A v7.3 file is HDF5; a classic one is not. The reader must sniff, because
    both are named .mat."""
    p = writer(tmp_path / "geometry.dat", _params())   # deliberately not .mat
    geo = load_esrf_parameters(p)
    assert geo.detector()[0].shape == (2048, 2048)


def test_ccw_stage_gives_the_other_sign(tmp_path):
    p = _write_v5(tmp_path / "p.mat", _params(rotdir=(0.0, 0.0, 1.0)))
    assert load_esrf_parameters(p).omega_sign() == DCT_OMEGA_SIGN_CCW


def test_foreign_beam_direction_is_rejected_at_load(tmp_path):
    """check_frame runs during load, so a file in another frame fails loudly
    rather than producing a plausible reconstruction in the wrong one."""
    p = _write_v5(tmp_path / "p.mat", _params(beamdir=(0.0, 1.0, 0.0)))
    with pytest.raises(ValueError, match=r"not \[1,0,0\]"):
        load_esrf_parameters(p)


def test_hand_built_geometry_needs_no_file(tmp_path):
    """The documented fallback: if the parser mis-reads a real file, the conversion
    path still works from a dict."""
    geo = ESRFGeometry(labgeo={"beamdir": (1, 0, 0), "rotdir": (0, 0, -1)},
                       detgeo=esrf_default_detgeo(512, 512, 2e-3, 8.0))
    geo.check_frame()
    assert geo.detector()[0].distance_um == pytest.approx(8000.0)
