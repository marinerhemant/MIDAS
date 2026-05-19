"""Byte-level regression tests against C-binary goldens.

These tests compare the output of each midas-transforms stage against the
C reference binary's output, run on the same inputs.

Goldens are generated locally by ``tests/test_ff_hedm.py --no-cleanup`` and
copied into ``tests/data/c_goldens/`` (see ``c_goldens/README.md``). They
are not committed (~550 MB).

Marked ``@pytest.mark.slow``; run with::

    pytest -m slow tests/test_regression_vs_c.py -v
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest


GOLDENS = Path(__file__).parent / "data" / "c_goldens"


@pytest.fixture(scope="module")
def goldens_dir():
    if not GOLDENS.exists() or not (GOLDENS / "Spots.bin").exists():
        pytest.skip(
            f"C goldens not found at {GOLDENS}. "
            f"Generate via: python tests/test_ff_hedm.py -nCPUs 8 --no-cleanup, "
            f"then copy LayerNr_1/* into c_goldens/. "
            f"See c_goldens/README.md."
        )
    return GOLDENS


# ---------------------------------------------------------------------------
# Stage 4: bin_data — Spots.bin / ExtraInfo.bin / Data.bin / nData.bin
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_bin_data_byte_exact_spots_bin(tmp_path: Path, goldens_dir: Path):
    """Run bin_data on the C-generated InputAll/Extra/paramstest, and compare
    Spots.bin byte-for-byte.

    Spots.bin = N x 9 float64. Cols 0..7 are pass-through from InputAll.csv;
    col 8 is RadiusDistIdeal = sqrt(YLab² + ZLab²) - RingRadii[ring_nr].
    """
    from midas_transforms.bin_data import bin_data

    # Stage inputs into tmp.
    for fn in ("InputAll.csv", "InputAllExtraInfoFittingAll.csv", "paramstest.txt"):
        shutil.copy2(goldens_dir / fn, tmp_path / fn)
    bin_data(result_folder=tmp_path, device="cpu", dtype="float64")

    from midas_transforms.io import binary as bio
    new_spots = bio.read_spots_bin(tmp_path / "Spots.bin")
    ref_spots = bio.read_spots_bin(goldens_dir / "Spots.bin")
    assert new_spots.shape == ref_spots.shape
    np.testing.assert_array_equal(new_spots, ref_spots,
        err_msg="Spots.bin must be byte-exact (N x 9 float64).")


@pytest.mark.slow
def test_bin_data_byte_exact_extrainfo_bin(tmp_path: Path, goldens_dir: Path):
    from midas_transforms.bin_data import bin_data
    for fn in ("InputAll.csv", "InputAllExtraInfoFittingAll.csv", "paramstest.txt"):
        shutil.copy2(goldens_dir / fn, tmp_path / fn)
    bin_data(result_folder=tmp_path, device="cpu", dtype="float64")

    from midas_transforms.io import binary as bio
    new_e = bio.read_extrainfo_bin(tmp_path / "ExtraInfo.bin")
    ref_e = bio.read_extrainfo_bin(goldens_dir / "ExtraInfo.bin")
    assert new_e.shape == ref_e.shape
    np.testing.assert_array_equal(new_e, ref_e,
        err_msg="ExtraInfo.bin must be byte-exact (N x 16 float64).")


@pytest.mark.slow
def test_bin_data_byte_exact_data_ndata(tmp_path: Path, goldens_dir: Path):
    """Data.bin and nData.bin must be byte-exact — these define the contract
    with the indexer."""
    from midas_transforms.bin_data import bin_data
    for fn in ("InputAll.csv", "InputAllExtraInfoFittingAll.csv", "paramstest.txt"):
        shutil.copy2(goldens_dir / fn, tmp_path / fn)
    bin_data(result_folder=tmp_path, device="cpu", dtype="float64")

    from midas_transforms.io import binary as bio
    new_data = bio.read_data_bin(tmp_path / "Data.bin")
    ref_data = bio.read_data_bin(goldens_dir / "Data.bin")
    new_nd = bio.read_ndata_bin(tmp_path / "nData.bin")
    ref_nd = bio.read_ndata_bin(goldens_dir / "nData.bin")
    assert new_nd.shape == ref_nd.shape, f"nData shape: {new_nd.shape} vs {ref_nd.shape}"
    assert new_data.shape == ref_data.shape, f"Data shape: {new_data.shape} vs {ref_data.shape}"
    np.testing.assert_array_equal(new_nd, ref_nd,
        err_msg="nData.bin must be byte-exact (M x 2 int32).")
    np.testing.assert_array_equal(new_data, ref_data,
        err_msg="Data.bin must be byte-exact (T int32, ring/eta/ome major order).")


# ---------------------------------------------------------------------------
# Stage 2: calc_radius — Radius_*.csv
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_calc_radius_close_to_c(tmp_path: Path, goldens_dir: Path):
    """Run calc_radius on the C-generated Result.csv and compare to C
    Radius.csv with a relaxed tolerance (the C version uses single-key qsort
    on Eta and prints with %f — we accept atol=1e-3 µm / px on coords)."""
    from midas_transforms.params import read_zarr_params
    from midas_transforms.radius import calc_radius
    from midas_transforms.io import csv as csv_io

    zarr_path = goldens_dir / "Au_FF_000001_pf.analysis.MIDAS.zip"
    if not zarr_path.exists():
        pytest.skip("Zarr archive not found among goldens")
    zp = read_zarr_params(zarr_path)
    shutil.copy2(goldens_dir / "Result_StartNr_1_EndNr_1440.csv", tmp_path / "Result_StartNr_1_EndNr_1440.csv")
    shutil.copy2(goldens_dir / "hkls.csv", tmp_path / "hkls.csv")

    calc_radius(result_folder=tmp_path, zarr_params=zp,
                start_nr=1, end_nr=1440, write=True,
                device="cpu", dtype="float64")

    new = csv_io.read_radius_csv(tmp_path / "Radius_StartNr_1_EndNr_1440.csv")
    ref = csv_io.read_radius_csv(goldens_dir / "Radius_StartNr_1_EndNr_1440.csv")
    # Row count must match exactly (ring filter is identical).
    assert new.shape == ref.shape, f"shape mismatch: {new.shape} vs {ref.shape}"
    # Column-wise tolerance (looser on derived columns 14, 15 = GrainVolume/Radius).
    np.testing.assert_allclose(new[:, :14], ref[:, :14], atol=1e-3, rtol=1e-5,
        err_msg="Radius cols 0..13 (geometry + ring) must match C within tight tolerance.")


# ---------------------------------------------------------------------------
# Stage 3: fit_setup — InputAll.csv, InputAllExtraInfoFittingAll.csv
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_fit_setup_close_to_c(tmp_path: Path, goldens_dir: Path):
    """Run fit_setup on C-generated Radius.csv and compare InputAll.csv.

    Tolerance is looser than bin_data because the per-spot tilt+distortion
    transform involves trig/atan2 and accumulates float error. We accept
    1e-2 µm absolute on lab-frame coordinates (sub-pixel)."""
    from midas_transforms.fit_setup import fit_setup
    from midas_transforms.params import read_zarr_params
    from midas_transforms.io import csv as csv_io

    zarr_path = goldens_dir / "Au_FF_000001_pf.analysis.MIDAS.zip"
    if not zarr_path.exists():
        pytest.skip("Zarr archive not found among goldens")
    zp = read_zarr_params(zarr_path)
    shutil.copy2(goldens_dir / "Radius_StartNr_1_EndNr_1440.csv", tmp_path / "Radius_StartNr_1_EndNr_1440.csv")
    shutil.copy2(goldens_dir / "hkls.csv", tmp_path / "hkls.csv")

    fit_setup(result_folder=tmp_path, zarr_params=zp,
              start_nr=1, end_nr=1440, do_fit=False, write=True,
              device="cpu", dtype="float64")

    new = csv_io.read_inputall_csv(tmp_path / "InputAll.csv")
    ref = csv_io.read_inputall_csv(goldens_dir / "InputAll.csv")
    assert new.shape == ref.shape
    # Cols 0,1 = YLab, ZLab (µm) — tolerance 1e-2 µm
    # Cols 2 = Omega (deg) — tolerance 1e-3 deg
    # Cols 3 = GrainRadius (µm) — pass-through, exact
    # Cols 4 = SpotID — must be exact integer
    # Cols 5 = RingNumber — pass-through, exact
    # Cols 6,7 = Eta, Ttheta — tolerance 1e-3 deg
    np.testing.assert_array_equal(new[:, 4].astype(int), ref[:, 4].astype(int),
        err_msg="SpotID must match exactly.")
    np.testing.assert_array_equal(new[:, 5].astype(int), ref[:, 5].astype(int),
        err_msg="RingNumber must match exactly.")
    np.testing.assert_allclose(new[:, [0, 1]], ref[:, [0, 1]], atol=1e-2,
        err_msg="YLab, ZLab (µm) within 1e-2 µm of C output.")
    np.testing.assert_allclose(new[:, 2], ref[:, 2], atol=1e-3,
        err_msg="Omega within 1e-3 deg of C output.")
    np.testing.assert_allclose(new[:, [6, 7]], ref[:, [6, 7]], atol=1e-3,
        err_msg="Eta, Ttheta within 1e-3 deg of C output.")


def test_fit_setup_inmemory_paramstest_carries_ring_radii(
    tmp_path: Path, goldens_dir: Path
):
    """The in-memory ``FitSetupResult.paramstest`` (consumed by the unified
    ``Pipeline`` / midas-pipeline) MUST carry RingRadii and the filtered
    RingNumbers — identical to the on-disk paramstest.txt.

    Regression for the FF pipeline bug where ``Pipeline.run()`` used the
    RingRadii-less ``zarr_params.to_paramstest()`` view: the binner then
    dropped every spot (no ring had radius > 0), wrote a zero-byte Data.bin,
    and indexing failed to mmap it.
    """
    from midas_transforms.fit_setup import fit_setup
    from midas_transforms.params import read_zarr_params, read_paramstest

    zarr_path = goldens_dir / "Au_FF_000001_pf.analysis.MIDAS.zip"
    if not zarr_path.exists():
        pytest.skip("Zarr archive not found among goldens")
    zp = read_zarr_params(zarr_path)
    shutil.copy2(goldens_dir / "Radius_StartNr_1_EndNr_1440.csv",
                 tmp_path / "Radius_StartNr_1_EndNr_1440.csv")
    shutil.copy2(goldens_dir / "hkls.csv", tmp_path / "hkls.csv")

    res = fit_setup(result_folder=tmp_path, zarr_params=zp,
                    start_nr=1, end_nr=1440, do_fit=False, write=True,
                    device="cpu", dtype="float64")

    pt_disk = read_paramstest(tmp_path / "paramstest.txt")
    pt_mem = res.paramstest
    assert pt_mem is not None
    # In-memory and on-disk paramstest must agree on the binning-critical keys.
    assert list(pt_mem.RingNumbers) == list(pt_disk.RingNumbers)
    # paramstest.txt rounds RingRadii to %.6f on write; compare to that precision.
    np.testing.assert_allclose(pt_mem.RingRadii, pt_disk.RingRadii, atol=1e-6)
    # And RingRadii must be non-empty + positive so the binner keeps spots.
    assert len(pt_mem.RingRadii) == len(pt_mem.RingNumbers) > 0
    assert all(r > 0 for r in pt_mem.RingRadii)


# ---------------------------------------------------------------------------
# Stage 1: merge_overlapping_peaks — Result_*.csv
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_merge_byte_exact_to_c(tmp_path: Path, goldens_dir: Path):
    """Run merge_overlapping_peaks on the C-generated AllPeaks_PS.bin and
    compare Result_*.csv against the C ``MergeOverlappingPeaksAllZarr`` golden.

    Frame-by-frame mutual-nearest is fully deterministic given a stable
    sort key on the per-frame Eta-sorted peak list (which we provide via
    ``np.lexsort((np.arange(N), eta))``). With identical input, identical
    margin (OverlapLength=2.0 from Zarr), and identical algorithm, output
    must match the C reference row-for-row at full float64 precision.
    """
    from midas_transforms.merge import merge_overlapping_peaks
    from midas_transforms.io import csv as csv_io

    ps_bin = goldens_dir / "AllPeaks_PS.bin"
    if not ps_bin.exists():
        pytest.skip(f"AllPeaks_PS.bin not found at {ps_bin}")
    zarr_path = goldens_dir / "Au_FF_000001_pf.analysis.MIDAS.zip"
    if not zarr_path.exists():
        pytest.skip(f"Zarr archive not found at {zarr_path}")

    merge_overlapping_peaks(
        zarr_path=zarr_path,
        allpeaks_ps_bin=ps_bin,
        result_folder=tmp_path,
        out_dir=tmp_path,
        start_nr=1, end_nr=1440,
        write=True,
        device="cpu", dtype="float64",
    )
    new = csv_io.read_result_csv(tmp_path / "Result_StartNr_1_EndNr_1440.csv")
    ref = csv_io.read_result_csv(goldens_dir / "Result_StartNr_1_EndNr_1440.csv")

    assert new.shape == ref.shape, (
        f"merge row count: new={new.shape[0]} vs C={ref.shape[0]}"
    )

    # Cols 0 (SpotID) and 1..16 (numeric) — full f64 tolerance.
    # SpotID must match exactly; the rest at 1e-9 absolute (CSV %lf
    # round-trip introduces ~1e-7 noise at 6 decimals on µm-scale values).
    np.testing.assert_array_equal(
        new[:, 0].astype(int), ref[:, 0].astype(int),
        err_msg="SpotID must match exactly.",
    )
    np.testing.assert_allclose(
        new[:, 1:], ref[:, 1:], atol=1e-4, rtol=0,
        err_msg=(
            "Result.csv numeric columns must match C output within CSV "
            "round-trip precision (~1e-4 µm on positions, ~1e-6 on intensities)."
        ),
    )


PX_GOLDENS = Path(__file__).parent / "data" / "c_goldens_px"


@pytest.fixture(scope="module")
def px_goldens_dir():
    if not PX_GOLDENS.exists() or not (PX_GOLDENS / "AllPeaks_PX.bin").exists():
        pytest.skip(
            f"PX-mode C goldens not found at {PX_GOLDENS}. "
            f"Generate via: python tests/test_ff_hedm.py -nCPUs 8 "
            f"--no-cleanup --px-overlap, then copy LayerNr_1/* and "
            f"Temp/AllPeaks_P{{S,X}}.bin into c_goldens_px/."
        )
    return PX_GOLDENS


@pytest.mark.slow
def test_merge_pixel_overlap_byte_exact_to_c(tmp_path: Path, px_goldens_dir: Path):
    """Run pixel-overlap merge against the C goldens generated with
    UsePixelOverlap=1.

    On the FF_HEDM/Example dataset the C centroid and C pixel-overlap
    outputs are byte-identical (peaks are sparse enough that both
    matchers find the same 20 cluster pairs). Our corrected
    pixel-overlap algorithm tracks Eta-sort permutations across both
    arrays — a fix of an indexing bug in the C reference — so on this
    dataset we should match C output. Synthetic-divergence tests live
    in `test_merge_pixel_overlap_synthetic` (non-slow).
    """
    from midas_transforms.merge import merge_overlapping_peaks
    from midas_transforms.io import csv as csv_io

    ps_bin = px_goldens_dir / "AllPeaks_PS.bin"
    px_bin = px_goldens_dir / "AllPeaks_PX.bin"
    zarr_path = px_goldens_dir / "Au_FF_000001_pf.analysis.MIDAS.zip"
    if not (ps_bin.exists() and px_bin.exists() and zarr_path.exists()):
        pytest.skip("PX-mode goldens incomplete")

    merge_overlapping_peaks(
        zarr_path=zarr_path,
        allpeaks_ps_bin=ps_bin,
        allpeaks_px_bin=px_bin,
        result_folder=tmp_path,
        out_dir=tmp_path,
        start_nr=1, end_nr=1440,
        write=True,
        device="cpu", dtype="float64",
    )
    new = csv_io.read_result_csv(tmp_path / "Result_StartNr_1_EndNr_1440.csv")
    ref = csv_io.read_result_csv(px_goldens_dir / "Result_StartNr_1_EndNr_1440.csv")
    assert new.shape == ref.shape, (
        f"px-overlap merge row count: new={new.shape[0]} vs C={ref.shape[0]}"
    )
    np.testing.assert_array_equal(
        new[:, 0].astype(int), ref[:, 0].astype(int),
        err_msg="SpotID must match exactly.",
    )
    np.testing.assert_allclose(
        new[:, 1:], ref[:, 1:], atol=1e-4, rtol=0,
        err_msg=(
            "px-overlap Result.csv numeric cols must match C output within "
            "CSV round-trip precision (~1e-4 µm on positions, ~1e-6 on intensities)."
        ),
    )


@pytest.mark.slow
def test_merge_intensity_conservation(tmp_path: Path, goldens_dir: Path):
    """Total integrated intensity is invariant under merge."""
    from midas_transforms.merge import merge_overlapping_peaks
    from midas_transforms.io import csv as csv_io
    from midas_transforms.io import zarr_io as zio

    ps_bin = goldens_dir / "AllPeaks_PS.bin"
    if not ps_bin.exists():
        pytest.skip(f"AllPeaks_PS.bin not found")
    zarr_path = goldens_dir / "Au_FF_000001_pf.analysis.MIDAS.zip"

    merge_overlapping_peaks(
        zarr_path=zarr_path, allpeaks_ps_bin=ps_bin,
        result_folder=tmp_path, out_dir=tmp_path,
        start_nr=1, end_nr=1440, write=True,
        device="cpu", dtype="float64",
    )
    new = csv_io.read_result_csv(tmp_path / "Result_StartNr_1_EndNr_1440.csv")

    # Sum of input IntegratedIntensity (over peaks with II >= 1).
    input_total = 0.0
    for frame in zio.read_allpeaks_ps_frames(ps_bin):
        if frame.shape[0] == 0:
            continue
        ii = frame[:, 1]
        input_total += float(ii[ii >= 1.0].sum())

    output_total = float(new[:, 1].sum())
    np.testing.assert_allclose(output_total, input_total, rtol=1e-9)
