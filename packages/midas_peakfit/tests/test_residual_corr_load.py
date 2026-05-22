"""Regression: peakfit must index the Stage-4 residual map as [Y, Z].

The map on disk is (NrPixelsZ, NrPixelsY) z-major — element (z, y) holds
ΔR(Y=y, Z=z) — the convention written by
midas_calibrate_v2.compat.to_integrate.write_residual_correction_from_spline.
geometry.compute_rt_eta adds it as ``Rt[A=Y, B=Z] += residualMap[A, B]``, so
``residualMap`` must be indexed [Y, Z]. A plain ``reshape(NrPixelsY, NrPixelsZ)``
silently transposes the correction — invisible on square detectors, wrong on
non-square. This test uses a NON-SQUARE detector and an asymmetric ΔR(Y, Z) so
the transpose is caught, and runs the REAL v2 writer → REAL load_corrections.
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import zarr

from midas_peakfit.zarr_io import parse_zarr_params, load_corrections


def _load_v2_writer():
    p = (Path(__file__).resolve().parents[2]
         / "midas_calibrate_v2" / "midas_calibrate_v2" / "compat" / "to_integrate.py")
    if not p.exists():
        return None
    spec = importlib.util.spec_from_file_location("_v2_to_integrate_pf", p)
    m = importlib.util.module_from_spec(spec)
    sys.modules["_v2_to_integrate_pf"] = m
    try:
        spec.loader.exec_module(m)
    except Exception:
        return None
    return getattr(m, "write_residual_correction_from_spline", None)


def _build_nonsquare_zarr(zip_path: Path, NZ: int, NY: int, rcm_path: Path):
    data = np.full((1, NZ, NY), 5, dtype=np.uint16)
    with zarr.ZipStore(str(zip_path), mode="w") as store:
        root = zarr.open_group(store=store, mode="w")
        root.create_dataset("exchange/data", data=data, chunks=(1, NZ, NY))
        ap = root.require_group("analysis/process/analysis_parameters")
        sp = root.require_group("measurement/process/scan_parameters")
        ap.create_dataset("YCen", data=np.array([NY / 2.0]))
        ap.create_dataset("ZCen", data=np.array([NZ / 2.0]))
        ap.create_dataset("PixelSize", data=np.array([150.0]))
        ap.create_dataset("Lsd", data=np.array([1e6]))
        ap.create_dataset("Wavelength", data=np.array([0.18]))
        ap.create_dataset("RhoD", data=np.array([NY * 150.0]))
        ap.create_dataset("Width", data=np.array([10000.0]))
        ap.create_dataset("DoFullImage", data=np.array([1], dtype=np.int32))
        ap.create_dataset("RingThresh", data=np.array([[1, 50.0]]))
        ap.create_dataset("MinNrPx", data=np.array([3], dtype=np.int32))
        ap.create_dataset("MaxNrPx", data=np.array([10000], dtype=np.int32))
        ap.create_dataset("MaxNPeaks", data=np.array([20], dtype=np.int32))
        ap.create_dataset("UpperBoundThreshold", data=np.array([14000.0]))
        ap.create_dataset("ResidualCorrectionMap", data=np.bytes_(str(rcm_path).encode()))
        sp.create_dataset("start", data=np.array([0.0]))
        sp.create_dataset("step", data=np.array([1.0]))
        sp.create_dataset("doPeakFit", data=np.array([1], dtype=np.int32))


def test_residual_map_indexed_YZ(tmp_path):
    writer = _load_v2_writer()
    if writer is None:
        pytest.skip("midas_calibrate_v2.compat.to_integrate not importable")

    NY, NZ, px = 5, 7, 150.0  # NON-square: a transpose gives a wrong value

    def spline_predict(ys, zs):  # ΔR in µm, asymmetric in Y vs Z
        return 30.0 * np.sin(2 * np.pi * zs / NZ) + 12.0 * np.cos(2 * np.pi * ys / NY)

    rcm = tmp_path / "residual_corr.bin"
    writer(spline_predict, NrPixelsY=NY, NrPixelsZ=NZ, px_mean_um=px, out_path=rcm)

    zpath = tmp_path / "nonsquare.MIDAS.zip"
    _build_nonsquare_zarr(zpath, NZ, NY, rcm)

    p = parse_zarr_params(str(zpath))
    assert (p.NrPixelsY, p.NrPixelsZ) == (NY, NZ)
    load_corrections(str(zpath), p)

    assert p.residualMap is not None
    assert p.residualMap.shape == (NY, NZ), (
        f"residualMap must be [Y, Z] = {(NY, NZ)} to match Rt[A=Y, B=Z]; "
        f"got {p.residualMap.shape}"
    )
    for y in range(NY):
        for z in range(NZ):
            truth = spline_predict(np.array([y]), np.array([z]))[0] / px
            assert abs(p.residualMap[y, z] - truth) < 1e-12, (
                f"residualMap[{y},{z}]={p.residualMap[y, z]} != ΔR(Y={y},Z={z})"
                f"/px={truth} — transpose/indexing bug"
            )
