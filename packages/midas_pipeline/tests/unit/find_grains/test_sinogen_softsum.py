"""Soft sino assembly tests (P7).

Covers the ``emit_softsum`` and ``soft_weight_fn`` extensions to
:func:`midas_pipeline.find_grains.generate_sinograms_tolerance`:

- Default behavior: no softsum file written, ``sino_paths["softsum"]``
  absent.  The 4 existing variants are bit-exact with the legacy path.
- ``emit_softsum=True`` (no weight fn) → softsum cell is the **sum** of
  all matching spot intensities, vs the existing ``raw`` cell which
  carries only the **max**.
- ``soft_weight_fn=...`` → softsum cell is the weighted sum
  ``Σ w(ω_diff, η_diff) · IntegratedIntensity(s)``; verified against a
  hand-computed Gaussian weighting.
- File shape matches the standard variants (``n_grains × max_h × n_scans``,
  float64).
- The ``stages/sinogen.py`` verifier recognizes ``sinos_softsum_*.bin``
  as a known variant (glob includes it).
"""

from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import math
from pathlib import Path

import numpy as np
import pytest

from midas_pipeline.find_grains import (
    SpotData, SpotList, generate_sinograms_tolerance,
)


def _spot_list_one_grain(*, omega_vals=(0.0, 1.0, 2.0)):
    return SpotList(
        spots=[
            SpotData(
                omega=om, eta=10.0 + idx * 0.05, ring_nr=1,
                merged_id=idx + 1, scan_nr=0, grain_nr=0, spot_nr=idx,
            )
            for idx, om in enumerate(omega_vals)
        ],
        max_n_hkls=len(omega_vals),
    )


def _all_spots_with_doubled_match(omega_vals, n_scans, intensity_a=100.0,
                                  intensity_b=40.0, dω=0.05):
    """For each scan/spot we emit TWO observed candidates within ω-tolerance.

    The first carries ``intensity_a`` at omega = sd.omega; the second
    carries ``intensity_b`` at omega = sd.omega + dω.  Same scan, same
    ring, etas matching the signature.  This means the max-pool will
    pick intensity_a (it's bigger) while the softsum should accumulate
    intensity_a + intensity_b * w(dω).
    """
    rows = []
    sid = 1
    for sc in range(n_scans):
        for idx, om in enumerate(omega_vals):
            base_eta = 10.0 + idx * 0.05
            rows.append([0.0, 0.0, om, intensity_a, sid, 1, base_eta, 5.0, 1.0, sc])
            sid += 1
            rows.append([0.0, 0.0, om + dω, intensity_b, sid, 1, base_eta, 5.0, 1.0, sc])
            sid += 1
    return np.asarray(rows, dtype=np.float64)


# -------------------------------------------------------------- back-compat


def test_back_compat_no_softsum_when_not_requested(tmp_path):
    sl = _spot_list_one_grain()
    all_spots = _all_spots_with_doubled_match(
        [s.omega for s in sl.spots], n_scans=2,
    )
    out = generate_sinograms_tolerance(
        spot_list=sl, n_unique=1, all_spots=all_spots, n_scans=2,
        tol_ome=0.5, tol_eta=0.1, output_dir=tmp_path,
    )
    assert "softsum" not in out.sino_paths
    assert not list(Path(tmp_path).glob("sinos_softsum_*.bin"))


def test_back_compat_raw_unchanged_with_softsum_flag(tmp_path):
    """Adding emit_softsum=True does NOT alter the raw / max-pool sino."""
    sl = _spot_list_one_grain()
    all_spots = _all_spots_with_doubled_match(
        [s.omega for s in sl.spots], n_scans=2,
    )
    a = tmp_path / "a"; a.mkdir()
    b = tmp_path / "b"; b.mkdir()
    out_a = generate_sinograms_tolerance(
        spot_list=sl, n_unique=1, all_spots=all_spots, n_scans=2,
        tol_ome=0.5, tol_eta=0.1, output_dir=a,
    )
    out_b = generate_sinograms_tolerance(
        spot_list=sl, n_unique=1, all_spots=all_spots, n_scans=2,
        tol_ome=0.5, tol_eta=0.1, output_dir=b,
        emit_softsum=True,
    )
    assert (a / Path(out_a.sino_paths["raw"]).name).read_bytes() == \
           (b / Path(out_b.sino_paths["raw"]).name).read_bytes()


# -------------------------------------------------------------- softsum


def test_emit_softsum_uniform_sums_all_matching(tmp_path):
    """Without a weight fn, softsum cell = Σ intensity over matching spots
    (uniform weights = 1.0)."""
    sl = _spot_list_one_grain()
    omegas = [s.omega for s in sl.spots]
    n_scans = 3
    all_spots = _all_spots_with_doubled_match(
        omegas, n_scans=n_scans, intensity_a=100.0, intensity_b=40.0,
    )
    out = generate_sinograms_tolerance(
        spot_list=sl, n_unique=1, all_spots=all_spots, n_scans=n_scans,
        tol_ome=0.5, tol_eta=0.1, output_dir=tmp_path,
        emit_softsum=True,
    )
    assert "softsum" in out.sino_paths
    softsum_path = Path(out.sino_paths["softsum"])
    raw_path = Path(out.sino_paths["raw"])
    assert softsum_path.exists()
    assert softsum_path.name == f"sinos_softsum_{out.n_grains}_{out.max_n_hkls}_{out.n_scans}.bin"

    raw = np.frombuffer(raw_path.read_bytes(), dtype=np.float64).reshape(
        out.n_grains, out.max_n_hkls, out.n_scans
    )
    softsum = np.frombuffer(softsum_path.read_bytes(), dtype=np.float64).reshape(
        out.n_grains, out.max_n_hkls, out.n_scans
    )
    # Each cell has two matching observed spots (intensity 100 + 40 = 140).
    np.testing.assert_allclose(softsum, np.full_like(raw, 140.0), atol=1e-12)
    # Raw is the max — 100 per cell.
    np.testing.assert_allclose(raw, np.full_like(raw, 100.0), atol=1e-12)


def test_soft_weight_fn_gaussian_matches_hand_computed(tmp_path):
    """With a Gaussian ω-weight, softsum = 100·exp(0) + 40·exp(-Δω²/(2σ²))."""
    sl = _spot_list_one_grain()
    omegas = [s.omega for s in sl.spots]
    n_scans = 2
    dω = 0.05
    all_spots = _all_spots_with_doubled_match(
        omegas, n_scans=n_scans, intensity_a=100.0, intensity_b=40.0, dω=dω,
    )

    sigma = 0.1
    def fn(ome_d, eta_d):
        # Only ω matters here (η_d = 0 between paired spots and the signature).
        return np.exp(-(ome_d * ome_d) / (2.0 * sigma * sigma))

    out = generate_sinograms_tolerance(
        spot_list=sl, n_unique=1, all_spots=all_spots, n_scans=n_scans,
        tol_ome=0.5, tol_eta=0.1, output_dir=tmp_path,
        soft_weight_fn=fn,
    )
    assert "softsum" in out.sino_paths
    softsum = np.frombuffer(
        Path(out.sino_paths["softsum"]).read_bytes(), dtype=np.float64,
    ).reshape(out.n_grains, out.max_n_hkls, out.n_scans)

    # Expected per cell:
    #   spot A: ω_diff = 0       → weight = 1 → contrib 100
    #   spot B: ω_diff = 0.05    → weight = exp(-0.0025 / 0.02) = exp(-0.125)
    w_B = math.exp(-(dω * dω) / (2.0 * sigma * sigma))
    expected_cell = 100.0 + 40.0 * w_B
    np.testing.assert_allclose(
        softsum, np.full_like(softsum, expected_cell), atol=1e-9,
    )


def test_softsum_file_dtype_and_shape(tmp_path):
    sl = _spot_list_one_grain(omega_vals=(0.0, 1.0, 2.0, 3.0))
    n_scans = 5
    all_spots = _all_spots_with_doubled_match(
        [s.omega for s in sl.spots], n_scans=n_scans,
    )
    out = generate_sinograms_tolerance(
        spot_list=sl, n_unique=1, all_spots=all_spots, n_scans=n_scans,
        tol_ome=0.5, tol_eta=0.1, output_dir=tmp_path,
        emit_softsum=True,
    )
    p = Path(out.sino_paths["softsum"])
    raw_bytes = p.read_bytes()
    # n_grains * max_n_hkls * n_scans * 8 bytes (float64)
    assert len(raw_bytes) == out.n_grains * out.max_n_hkls * out.n_scans * 8


# -------------------------------------------------------------- stage glob


def test_sinogen_stage_globs_softsum_variant(monkeypatch):
    """The stages/sinogen.py glob list must include 'softsum'."""
    from midas_pipeline.stages import sinogen as sg
    # Inspect the variant set the stage uses (read straight from the source).
    src = Path(sg.__file__).read_text()
    assert '"softsum"' in src, (
        "stages/sinogen.py must include 'softsum' in its sinos_paths glob"
    )
