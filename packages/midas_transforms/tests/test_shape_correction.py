"""Grain volumes rescaled onto the measured illuminated volume.

The arithmetic here is four lines. The tests are not, because the failure mode
this module exists to prevent — correcting the numerator and leaving the powder
denominator raw — is invisible: it scales every volume by <1/A>, uniformly, in
the direction the user expects, with no symptom anywhere in the output.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from midas_transforms.radius.shape_correction import (
    GaugeVolume,
    correct_grain_volumes,
    normalise_per_ring,
    volume_to_radius,
)


# ---------------------------------------------------------------- V_gauge

def test_the_search_bound_branch_is_what_the_reference_run_uses():
    """``ff_refiner_prepost/result/LayerNr_1/paramstest.txt`` has no Vsample,
    so V_gauge = 2000 * pi * 2000^2 = 2.513e10 um^3 — two search bounds."""
    g = GaugeVolume(hbeam_um=2000.0, rsample_um=2000.0)
    assert g.value_um3 == pytest.approx(2000.0 * math.pi * 2000.0 ** 2)
    assert "SEARCH BOUNDS" in g.source


def test_vsample_overrides_the_search_bounds():
    g = GaugeVolume(hbeam_um=2000.0, rsample_um=2000.0, vsample_um3=1.0e8)
    assert g.value_um3 == 1.0e8
    assert g.source == "Vsample"


def test_disc_model_wins_over_both():
    g = GaugeVolume(hbeam_um=2000.0, rsample_um=2000.0, vsample_um3=1.0e8,
                    disc_model=1, disc_area_um2=1234.0)
    assert g.value_um3 == 1234.0


def test_the_calibration_template_defaults_are_flagged():
    """Not an error — most runs are like this. But the reported absolute grain
    size then carries a number from a template, and that has to be visible."""
    assert GaugeVolume(hbeam_um=1000.0, rsample_um=1000.0).is_template_default
    assert GaugeVolume(hbeam_um=2000.0, rsample_um=2000.0,
                       vsample_um3=50_000_000.0).is_template_default
    assert not GaugeVolume(hbeam_um=2000.0, rsample_um=2000.0).is_template_default


def test_gauge_volume_reads_a_real_parameter_file(tmp_path):
    p = tmp_path / "paramstest.txt"
    p.write_text(
        "RingThresh 1 100;\nHbeam 2000.000000;\nRsample 2000.000000;\n"
        "LatticeConstant 3.6 3.6 3.6 90 90 90;\n"
    )
    g = GaugeVolume.from_param_file(p)
    assert g.hbeam_um == 2000.0 and g.rsample_um == 2000.0
    assert g.vsample_um3 == 0.0
    assert g.value_um3 == pytest.approx(2.513e10, rel=1e-3)


def test_a_parameter_file_with_no_gauge_at_all_is_refused(tmp_path):
    p = tmp_path / "p.txt"
    p.write_text("Lsd 1000000;\nWavelength 0.2066;\n")
    with pytest.raises(ValueError, match="V_gauge cannot be reproduced"):
        GaugeVolume.from_param_file(p)


# ------------------------------------------- the powder double-counting guard

def test_a_uniform_correction_is_EXACTLY_one():
    """N1 from the plan, and the reason this branch is special-cased.

    ``f / f.mean()`` is not guaranteed to give 1.0 for an arbitrary constant,
    and 'bit-identical' has to mean bit-identical.
    """
    for c in (0.5, 1.0, 0.37, 1e-3, 0.8412327319):
        f = np.full(7, c)
        out = normalise_per_ring(f)
        assert np.all(out == 1.0), f"c={c} gave {out}"


def test_a_constant_correction_leaves_radii_BIT_IDENTICAL():
    """The invariant that makes the whole thing safe: if absorption is uniform
    it is already in the powder reference, so it must cancel exactly."""
    rng = np.random.default_rng(0)
    v = rng.uniform(1e3, 1e6, 500)
    g = GaugeVolume(hbeam_um=2000.0, rsample_um=2000.0)
    v_illum = g.value_um3          # ratio exactly 1

    base, r_base, _ = correct_grain_volumes(v, gauge=g,
                                            illuminated_volume_um3=v_illum)
    corr, r_corr, rep = correct_grain_volumes(
        v, gauge=g, illuminated_volume_um3=v_illum,
        spot_correction=np.full(500, 0.317),
    )
    assert np.array_equal(r_corr, r_base)
    assert np.array_equal(corr, base)
    assert rep.per_spot_applied


def test_the_guard_HAS_POWER_the_unnormalised_version_inflates_by_mean_inv_A():
    """The test above passes for a correction that was never applied, so it
    proves nothing alone. This is the companion: skip the per-ring
    normalisation and the inflation appears, at the predicted size."""
    rng = np.random.default_rng(1)
    mu_d = rng.uniform(0.2, 0.8, 4000)      # a real spread of path lengths
    A = np.exp(-mu_d)                        # transmitted fraction
    f_raw = 1.0 / A                          # the naive "divide it out"
    f_norm = normalise_per_ring(f_raw)

    inflation = f_raw.mean() / f_norm.mean()
    assert inflation == pytest.approx(np.mean(1.0 / A), rel=1e-9)
    assert inflation > 1.4                   # ~1.6x in volume as predicted
    assert inflation ** (1 / 3) > 1.12       # ~17 % in radius
    assert f_norm.mean() == pytest.approx(1.0, abs=1e-12)


def test_normalisation_is_per_ring_not_global():
    """Rings differ in 2-theta, so their mean absorption differs; a global mean
    would leave a ring-to-ring bias that reads as anisotropic grain size."""
    f = np.array([1.0, 2.0, 10.0, 20.0])
    ring = np.array([1, 1, 2, 2])
    out = normalise_per_ring(f, ring)
    assert out[:2].mean() == pytest.approx(1.0)
    assert out[2:].mean() == pytest.approx(1.0)
    assert out[0] == pytest.approx(out[2])   # same relative position in-ring


def test_a_zero_or_negative_correction_is_refused():
    with pytest.raises(ValueError, match="infinitely large"):
        normalise_per_ring(np.array([1.0, 0.0, 1.0]))
    with pytest.raises(ValueError, match="<= 0"):
        normalise_per_ring(np.array([1.0, -0.5, 1.0]))


def test_ring_index_shape_must_match():
    with pytest.raises(ValueError, match="ring_index shape"):
        normalise_per_ring(np.ones(4), np.array([1, 2]))


# ------------------------------------------------------------ the global term

def test_the_global_scale_is_the_ratio_and_radius_is_its_cube_root():
    v = np.array([1000.0, 8000.0])
    _, r, rep = correct_grain_volumes(
        v, gauge=1.0e9, illuminated_volume_um3=1.0e8,
    )
    assert rep.volume_scale == pytest.approx(0.1)
    assert rep.radius_scale == pytest.approx(0.1 ** (1 / 3))
    # radius still equals the cube-root formula on the corrected volume
    assert r == pytest.approx(volume_to_radius(v * 0.1))


def test_the_reference_run_numbers_give_the_factor_the_plan_predicts():
    """FF reference run: V_gauge 2.513e10 from search bounds. A 1 mm rod lit
    over 200 um is 1.571e8 — grain radii overstated by (1/r)^(1/3) ~ 5x."""
    g = GaugeVolume(hbeam_um=2000.0, rsample_um=2000.0)
    v_illum = math.pi * 500.0 ** 2 * 200.0
    _, _, rep = correct_grain_volumes(np.array([1e4]), gauge=g,
                                      illuminated_volume_um3=v_illum)
    assert rep.volume_scale < 0.01
    assert 1.0 / rep.radius_scale > 4.5
    assert rep.gauge_is_template_default is False
    assert any("global term only" in w for w in rep.warnings)


def test_a_zero_illuminated_volume_is_refused_with_the_likely_cause():
    with pytest.raises(ValueError, match="beam slab missed the sample in z"):
        correct_grain_volumes(np.array([1.0]), gauge=1e9,
                              illuminated_volume_um3=0.0)


def test_packing_above_one_is_reported_not_clipped():
    """Grains cannot occupy more than the volume that was lit. When they
    appear to, something upstream is wrong and clipping would hide it."""
    v = np.full(100, 3.0e6)
    _, _, rep = correct_grain_volumes(v, gauge=1.0e8,
                                      illuminated_volume_um3=1.0e7)
    assert rep.packing_fraction == pytest.approx(3.0)
    assert any("impossible" in w for w in rep.warnings)


def test_packing_below_one_raises_no_alarm():
    v = np.full(100, 1.0e3)
    _, _, rep = correct_grain_volumes(v, gauge=1.0e8,
                                      illuminated_volume_um3=1.0e7)
    assert rep.packing_fraction < 1.0
    assert not any("impossible" in w for w in rep.warnings)


def test_disc_model_radius_uses_the_area_formula():
    v = np.array([math.pi * 25.0])
    r = volume_to_radius(v, disc_model=1)
    assert r[0] == pytest.approx(5.0)


def test_negative_volumes_keep_their_sign():
    """A spot below the local background gives a negative volume; the legacy
    code sign-preserves the cube root and so must this."""
    r = volume_to_radius(np.array([-1000.0, 1000.0]))
    assert r[0] == pytest.approx(-r[1])


# --------------------------------------------------------- end-to-end shape

def test_a_SampleShape_drives_the_correction_end_to_end():
    from midas_transforms.geometry import SampleShape

    s = SampleShape.cylinder(diameter_um=1000.0, height_um=2000.0,
                             pixel_size_um=20.0)
    g = GaugeVolume(hbeam_um=2000.0, rsample_um=2000.0)
    v_illum = s.illuminated_volume_um3(beam_height_um=200.0)

    radii_legacy = np.array([31.4, 26.1, 40.4])          # the run's p50/p25/p75
    volumes = (4.0 / 3.0) * math.pi * radii_legacy ** 3
    _, r_cor, rep = correct_grain_volumes(volumes, gauge=g,
                                          illuminated_volume_um3=v_illum)
    assert np.all(r_cor < radii_legacy)
    assert rep.volume_scale == pytest.approx(v_illum / g.value_um3)
    assert "V_gauge" in rep.summary() and "V_illum" in rep.summary()


def test_packing_fraction_is_INVARIANT_under_the_global_term():
    """A check that cannot fail, pinned so nobody reads it as one that can.

    ``sum(V*s) / (V_gauge*s) == sum(V) / V_gauge``, so the packing fraction is
    identical before and after the rescale and says nothing about whether
    V_illum is right. On the FF reference run it reads 0.0653 either way --
    which is the 6.5 % already measured against V_gauge, not a second
    confirmation of it.
    """
    rng = np.random.default_rng(4)
    v = rng.uniform(1e3, 1e6, 300)
    g = GaugeVolume(hbeam_um=2000.0, rsample_um=2000.0)
    packs = [
        correct_grain_volumes(v, gauge=g, illuminated_volume_um3=vi)[2]
        .packing_fraction
        for vi in (g.value_um3, 1.0e8, 1.571e8, 1.0e6)
    ]
    assert all(p == pytest.approx(packs[0], rel=1e-12) for p in packs)
    assert packs[0] == pytest.approx(v.sum() / g.value_um3, rel=1e-12)


def test_the_reference_run_reproduces_its_own_6_5_percent():
    """End to end against the number in the plan, read from the real file."""
    from pathlib import Path

    run = Path.home() / "Desktop/analysis/ff_refiner_prepost/result/LayerNr_1"
    if not (run / "Grains.csv").is_file():
        pytest.skip(f"{run} not present on this machine")

    rows = (run / "Grains.csv").read_text().splitlines()
    hdr_i = next(i for i, r in enumerate(rows) if "GrainRadius" in r)
    col = rows[hdr_i].lstrip("%").split().index("GrainRadius")
    radii = np.array([float(r.split()[col]) for r in rows[hdr_i + 1:] if r.strip()])
    assert radii.size == 6112

    g = GaugeVolume.from_param_file(run / "paramstest.txt")
    assert g.value_um3 == pytest.approx(2.513e10, rel=1e-3)
    volumes = (4.0 / 3.0) * math.pi * radii ** 3
    assert volumes.sum() / g.value_um3 == pytest.approx(0.065, abs=0.003)
