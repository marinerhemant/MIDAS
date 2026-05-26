"""Unit tests for compute.volume_consistency."""

import math
import numpy as np
import pytest

from midas_process_grains.compute.volume_consistency import (
    compute_volume_consistency, consistency_as_meta_dict,
    VolumeConsistencyResult,
)


def test_empty_grain_list_returns_zero_total_volume():
    res = compute_volume_consistency(radius_um=np.array([]))
    assert res.n_grains == 0
    assert res.sum_v_grain_um3 == 0.0
    assert res.packing_fraction is None


def test_total_volume_matches_sum_of_sphere_volumes():
    R = np.array([1.0, 2.0, 3.0])
    res = compute_volume_consistency(radius_um=R)
    expected = sum((4.0 / 3.0) * math.pi * r ** 3 for r in R)
    assert abs(res.sum_v_grain_um3 - expected) < 1e-9


def test_packing_fraction_against_known_sample_volume():
    R = np.array([1.0])    # V = 4π/3 ≈ 4.189 µm³
    res = compute_volume_consistency(
        radius_um=R, v_sample_um3=1000.0,
    )
    expected_packing = (4.0 / 3.0) * math.pi / 1000.0
    assert abs(res.packing_fraction - expected_packing) < 1e-9


def test_sample_volume_derived_from_extents():
    R = np.array([1.0])
    res = compute_volume_consistency(
        radius_um=R, sample_extents_um=(1200.0, 1000.0, 200.0),
    )
    assert abs(res.v_sample_um3 - 2.4e8) < 1.0


def test_fraction_in_box_with_centered_sample():
    """Three grains at (0,0,0), (700,0,0) [outside X], (0,0,0) → 2/3 in box
    when sample = 1000×1000×1000 (half-extent = 500)."""
    R = np.array([0.5, 0.5, 0.5])
    pos = np.array([
        [0.0, 0.0, 0.0],
        [700.0, 0.0, 0.0],   # OUTSIDE X (700 > 500)
        [0.0, 0.0, 0.0],
    ])
    res = compute_volume_consistency(
        radius_um=R, positions_um=pos,
        sample_extents_um=(1000.0, 1000.0, 1000.0),
    )
    assert res.fraction_in_box == pytest.approx(2.0 / 3.0)


def test_meta_dict_round_trips_all_fields():
    R = np.array([1.0, 2.0])
    res = compute_volume_consistency(
        radius_um=R, v_sample_um3=100.0,
    )
    d = consistency_as_meta_dict(res)
    assert d["n_grains"] == 2
    assert d["v_sample_um3"] == 100.0
    assert d["packing_fraction"] is not None
    assert d["median_r_um"] == 1.5


def test_radius_correction_cylinder_fallback_when_no_radius_csv(tmp_path):
    """When ``Radius_*.csv`` is unavailable, the helper falls back to the
    cylinder formula ``Vgauge_legacy = Hbeam·π·Rsample²``. With
    ``Hbeam=Rsample=1000`` and ``Vsample 2.4e8`` in paramstest the
    correction factor must be ``(2.4e8 / π·10⁹)^(1/3) ≈ 0.424``.
    """
    from midas_process_grains.v4_pipeline import _compute_radius_correction

    ps_file = tmp_path / "paramstest.txt"
    ps_file.write_text(
        "Hbeam 1000.0;\n"
        "Rsample 1000.0;\n"
        "Vsample 240000000.0;\n"
    )
    log_lines = []
    corr = _compute_radius_correction(ps_file, lambda *a, **k: log_lines.append(a))
    expected = (2.4e8 / (math.pi * 1e9)) ** (1 / 3)
    assert abs(corr - expected) < 1e-6
    assert abs(corr - 1.0 / 2.357) < 1e-3   # ≈ 0.424


def test_radius_correction_back_solves_legacy_vgauge_from_radius_csv(tmp_path):
    """The helper's primary path back-solves ``Vgauge_legacy`` from a
    synthetic ``Radius_*.csv``. The Xuan production case has legacy
    Vgauge = 1e5 µm³ (from an upstream ``Vsample 100000`` line that the
    cylinder fallback DOESN'T see), and the user wants the truth
    Vsample = 2.4e8 µm³. Expected correction = (2.4e8/1e5)^(1/3) = 13.467.
    """
    from midas_process_grains.v4_pipeline import _compute_radius_correction

    ps_file = tmp_path / "paramstest.txt"
    ps_file.write_text(
        "Hbeam 1000.0;\nRsample 1000.0;\nVsample 240000000.0;\n"
    )
    # Synthesize a minimal Radius_*.csv that back-solves to Vgauge = 1e5.
    # GrainVolume = 0.5 · m_hkl · ΔΘ · cos(Θ) · Vgauge · IntInt / (NImgs·PowderInt)
    # Pick: Theta=5deg, Eta=90deg, DeltaOmega=0.1deg, IntInt=1000, NImgs=1,
    # PowderInt=1e6, m_hkl=1.  ΔΘ = d2r·(rad2deg·asin(sin(Th)·cos(dω)+cos(Th)·1·sin(dω)) − Th)
    theta_d = 5.0; eta_d = 90.0; dom_d = 0.1; intint = 1000.0; nimgs = 1.0
    powder = 1.0e6; vgauge_target = 1.0e5
    d2r = math.pi/180
    arg = math.sin(theta_d*d2r)*math.cos(dom_d*d2r) + math.cos(theta_d*d2r)*math.sin(dom_d*d2r)
    dTheta = d2r * (math.degrees(math.asin(arg)) - theta_d)
    grain_vol = 0.5 * 1.0 * dTheta * math.cos(theta_d*d2r) * vgauge_target * intint / (nimgs*powder)
    radius_file = tmp_path / "Radius_StartNr_1_EndNr_2.csv"
    header = ("SpotID IntegratedIntensity Omega(degrees) YCen(px) ZCen(px) IMax "
              "MinOme(degrees) MaxOme(degress) Radius(px) Theta(degrees) "
              "Eta(degrees) DeltaOmega NImgs RingNr GrainVolume GrainRadius "
              "PowderIntensity SigmaR SigmaEta NrPx NrPxTot RawSumIntensity "
              "maskTouched FitRMSE\n")
    row = (f"1 {intint} 0 0 0 0 0 0 200 {theta_d} {eta_d} {dom_d} {nimgs} 1 "
           f"{grain_vol} 0.1 {powder} 0 0 1 1 {intint} 0 0\n")
    radius_file.write_text(header + row + row)  # two identical rows
    (tmp_path / "hkls.csv").write_text("h k l ds RN g1 g2 g3 Theta 2Theta Radius\n"
                                       "1 1 1 2.0 1 0 0 0 5 10 100\n")
    log_lines = []
    corr = _compute_radius_correction(ps_file, lambda *a, **k: log_lines.append(a),
                                       layer_dir=tmp_path)
    expected = (2.4e8 / 1.0e5) ** (1.0 / 3.0)
    assert abs(corr - expected) / expected < 1e-2
    assert abs(corr - 13.389) < 0.05   # ≈ 13.389 (= 2400^(1/3); the Xuan correction)


def test_no_correction_when_vsample_is_zero(tmp_path):
    """When Vsample is missing or 0, the legacy fallback is
    in effect already — correction factor must be exactly 1.0."""
    from midas_process_grains.v4_pipeline import _compute_radius_correction
    ps_file = tmp_path / "paramstest.txt"
    ps_file.write_text("Hbeam 1000.0;\nRsample 1000.0;\n")
    corr = _compute_radius_correction(ps_file, lambda *a, **k: None)
    assert corr == 1.0
