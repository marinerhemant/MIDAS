"""Saturated regions are recorded (AllPeaks_PS_sat.bin), never silently dropped.

A region with a pixel over IntSat is not fitted. It used to vanish; its faint
edges in neighbouring omega frames then survived as tiny unflagged spots that
grain matching accepted (AlON 1-ID, 2026-09-22). Now:

* ``AllPeaks_PS.bin`` holds exactly the fitted (unsaturated) peaks, as before;
* ``AllPeaks_PS_sat.bin`` holds one centroid row per saturated region, same
  29-col layout, ``returnCode = -2``, SpotIDs continuing after the frame's
  fitted peaks so ``(frame, SpotID)`` is unique across both files.
"""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest
import zarr

from midas_peakfit.compat.reference_decoder import read_ps
from midas_peakfit.connected import Region
from midas_peakfit.orchestrator import run
from midas_peakfit.postfit import build_saturated_rows
from midas_peakfit.seeds import SATURATED_RETURN_CODE, saturated_region


def _zarr_with_int_sat(tmp: Path, int_sat: float) -> Path:
    """The conftest synthetic scan (peaks 1500/2200 in frame 0, 3000 in frame 1)
    with a chosen UpperBoundThreshold."""
    zip_path = tmp / "sat.MIDAS.zip"
    nF, NZ, NY = 3, 256, 256
    Yg, Zg = np.meshgrid(np.arange(NY, dtype=float), np.arange(NZ, dtype=float), indexing="xy")

    def g(y0, z0, amp, sig=4.0):
        return amp * np.exp(-((Yg - y0) ** 2 + (Zg - z0) ** 2) / (2 * sig * sig))

    data = np.zeros((nF, NZ, NY), dtype=np.uint16)
    data[0] = (g(60, 70, 1500) + g(180, 200, 2200) + 5).astype(np.uint16)
    data[1] = (g(128, 128, 3000) + 5).astype(np.uint16)
    data[2] = 5
    with zarr.ZipStore(str(zip_path), mode="w") as store:
        root = zarr.open_group(store=store, mode="w")
        root.create_dataset("exchange/data", data=data, chunks=(1, NZ, NY))
        ap = root.require_group("analysis/process/analysis_parameters")
        sp = root.require_group("measurement/process/scan_parameters")
        for k, v in (("YCen", 128.0), ("ZCen", 128.0), ("PixelSize", 200.0),
                     ("Lsd", 1e6), ("Wavelength", 0.18), ("RhoD", NY * 200.0),
                     ("Width", 10000.0), ("UpperBoundThreshold", int_sat)):
            ap.create_dataset(k, data=np.array([v]))
        for k, v in (("DoFullImage", 1), ("MinNrPx", 3), ("MaxNrPx", 10000), ("MaxNPeaks", 20)):
            ap.create_dataset(k, data=np.array([v], dtype=np.int32))
        ap.create_dataset("RingThresh", data=np.array([[1, 50.0]]))
        ap.create_dataset("ResultFolder", data=np.bytes_(str(tmp).encode()))
        sp.create_dataset("start", data=np.array([0.0]))
        sp.create_dataset("step", data=np.array([1.0]))
        sp.create_dataset("doPeakFit", data=np.array([1], dtype=np.int32))
    return zip_path


@pytest.fixture
def tmp():
    d = Path(tempfile.mkdtemp(prefix="midas_peakfit_sat_"))
    try:
        yield d
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_saturated_regions_go_to_the_sibling_file_not_the_fit(tmp):
    zp = _zarr_with_int_sat(tmp, 2000.0)     # 2200 and 3000 peaks saturate; 1500 does not
    out = run(str(zp), block_nr=0, n_blocks=1, num_procs=1,
              result_folder_cli=str(tmp), device="cpu", dtype="float64",
              producer="thread")
    assert out["n_saturated"] == 2
    ps = read_ps(out["ps_path"])
    sat = read_ps(out["sat_ps_path"])

    # Main file: only the fitted, unsaturated peak (frame 0 at Y=60, Z=70).
    assert list(ps.n_peaks) == [1, 0, 0]
    assert np.hypot(ps.rows_per_frame[0][0, 3] - 60, ps.rows_per_frame[0][0, 4] - 70) < 1.0
    assert ps.rows_per_frame[0][0, 18] != SATURATED_RETURN_CODE

    # Sibling: one centroid row per saturated region, flagged, ID after the fitted ones.
    assert list(sat.n_peaks) == [1, 1, 0]
    r0, r1 = sat.rows_per_frame[0][0], sat.rows_per_frame[1][0]
    assert r0[18] == SATURATED_RETURN_CODE and r1[18] == SATURATED_RETURN_CODE
    assert np.hypot(r0[3] - 180, r0[4] - 200) < 0.5
    assert np.hypot(r1[3] - 128, r1[4] - 128) < 0.5
    assert r0[0] == 2 and r1[0] == 1          # frame 0 has 1 fitted peak before it
    assert r0[5] > 2000 and r0[1] == pytest.approx(r0[26])   # IMax clipped-high; II = raw sum
    assert Path(out["sat_ps_path"]).with_name("AllPeaks_PX_sat.bin").exists()


def test_no_saturation_writes_an_empty_sibling_and_an_unchanged_main_file(tmp):
    zp = _zarr_with_int_sat(tmp, 14000.0)
    out = run(str(zp), block_nr=0, n_blocks=1, num_procs=1,
              result_folder_cli=str(tmp), device="cpu", dtype="float64",
              producer="thread")
    assert out["n_saturated"] == 0
    assert list(read_ps(out["sat_ps_path"]).n_peaks) == [0, 0, 0]
    assert list(read_ps(out["ps_path"]).n_peaks)[:2] == [2, 1]


def test_saturated_row_geometry_matches_the_fitted_row_convention():
    """YCen/ZCen of a saturated row must land on the pixel centroid in the same
    detector frame as fitted rows (YCen = row, ZCen = col)."""
    rows = np.repeat(np.arange(40, 45), 5).astype(np.int32)
    cols = np.tile(np.arange(90, 95), 5).astype(np.int32)
    vals = np.full(rows.size, 100.0)
    vals[12] = 20000.0                       # the saturated pixel, at (42, 92)
    reg = Region(id=1, pixel_rows=rows, pixel_cols=cols, intensities=vals,
                 raw_sum=float(vals.sum()), threshold=10.0)
    s = saturated_region(reg, np.zeros((128, 128)), None, Ycen=64.0, Zcen=64.0, panels=[])
    r = build_saturated_rows([s], omega=1.5, Ycen=64.0, Zcen=64.0, spot_id_start=7)[0]
    assert r[0] == 7 and r[2] == 1.5 and r[18] == SATURATED_RETURN_CODE
    assert r[3] == pytest.approx(42.0, abs=1e-9) and r[4] == pytest.approx(92.0, abs=1e-9)
    assert (r[13], r[14]) == (42.0, 92.0)
    assert r[1] == pytest.approx(vals.sum()) and r[10] == 25
