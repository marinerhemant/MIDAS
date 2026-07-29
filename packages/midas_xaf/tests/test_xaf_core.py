"""Core tests for midas-xaf: geometry gates, forward orchestration, metrics.

Device tests parametrise over CPU and any available accelerator (MPS/CUDA), per
the project rule that new torch code carries CPU + accelerator + autograd tests.
"""
import math

import numpy as np
import pytest
import torch

from midas_xaf import (XAFConfig, XAFForwardModel, make_sample, geometry,
                       metrics, merge, reconstruct, micromech, synth, indexing,
                       pipeline, budget, structure)


def _all_devices():
    """Devices for pure-tensor XAF ops (masks) -- these support MPS."""
    devs = ["cpu"]
    if torch.backends.mps.is_available():
        devs.append("mps")
    if torch.cuda.is_available():
        devs.append("cuda")
    return devs


def _forward_devices():
    """Devices for the full forward path.  MPS is excluded: the upstream
    ``midas_diffract`` forward model hardcodes float64 detector tilts, which
    MPS does not support.  CUDA (float64-capable) is fine."""
    devs = ["cpu"]
    if torch.cuda.is_available():
        devs.append("cuda")
    return devs


def _small_cfg(**kw):
    base = dict(material="zirconia_monoclinic", n_grains=6, energy_keV=80.0,
                seed=0)
    base.update(kw)
    return XAFConfig(**base)


# --------------------------------------------------------------------------- #
#  Geometry / access gates                                                    #
# --------------------------------------------------------------------------- #
def test_wedge_mask_windows():
    cfg = _small_cfg(opening_full_deg=15.0)
    om = torch.deg2rad(torch.tensor([0.0, 7.0, 8.0, 45.0, 90.0, 183.0, 270.0]))
    m = geometry.wedge_mask(om, cfg)
    # wedge half = 7.5 deg -> 0,7,90,183(=180+3),270 inside; 8,45 outside
    assert m.tolist() == [True, True, False, False, True, True, True]


def test_exit_cone_reduces_to_tth_cap_at_wedge_centre():
    """At omega=0 the cone gate must equal the 2theta<=half disk."""
    cfg = _small_cfg(opening_full_deg=15.0)

    class SD:  # minimal stand-in with the fields the mask reads
        pass
    sd = SD()
    n = 200
    sd.two_theta = torch.linspace(0.0, math.radians(12.0), n)
    sd.eta = torch.linspace(-math.pi, math.pi, n)
    sd.omega = torch.zeros(n)             # wedge centre
    cone = geometry.exit_aperture_mask(sd, cfg)
    cap = torch.rad2deg(sd.two_theta) <= cfg.opening_half_deg
    assert torch.equal(cone, cap)


def test_exit_cone_shadowing_is_omega_eta_asymmetric():
    """Off wedge-centre, a spot's accessibility depends on eta (shadowing)."""
    cfg = _small_cfg(opening_full_deg=15.0)

    class SD:
        pass
    sd = SD()
    tt = math.radians(6.0)
    off = math.radians(5.0)   # 5 deg into the wedge
    # eta=+90 (toward +tilt) should clip earlier than eta=-90
    sd.two_theta = torch.tensor([tt, tt])
    sd.eta = torch.tensor([math.pi / 2, -math.pi / 2])
    sd.omega = torch.tensor([off, off])
    m = geometry.exit_aperture_mask(sd, cfg)
    assert m[0].item() != m[1].item()   # asymmetric under eta sign


# --------------------------------------------------------------------------- #
#  Forward orchestration                                                       #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("device", _all_devices())
def test_geometry_masks_run_on_device(device):
    """The XAF access gates are pure-tensor ops and must run on every device."""
    cfg = _small_cfg(device=device)

    class SD:
        pass
    sd = SD()
    n = 64
    sd.two_theta = torch.linspace(0.0, math.radians(10.0), n, device=device)
    sd.eta = torch.linspace(-math.pi, math.pi, n, device=device)
    sd.omega = torch.linspace(-math.pi, math.pi, n, device=device)
    sd.valid = torch.ones(n, device=device)
    w = geometry.wedge_mask(sd.omega, cfg)
    e = geometry.exit_aperture_mask(sd, cfg)
    a = geometry.accessible_mask(sd, cfg)
    assert w.device.type == torch.device(device).type
    assert bool((a == (w & e & (sd.valid > 0.5))).all())


@pytest.mark.parametrize("device", _forward_devices())
def test_forward_runs_and_counts_are_physical(device):
    cfg = _small_cfg(device=device)
    fwd = XAFForwardModel(cfg)
    grains = make_sample(cfg)
    sim = fwd.simulate(grains)
    spg = metrics.spots_per_grain(sim)
    # per-grain path (not the N^2 cross product): tens, not thousands
    assert spg.max() < 1000
    assert spg.sum() == len(sim.table)
    assert 0.0 <= metrics.friedel_completeness(sim) <= 1.0


def test_cone_gate_removes_shadowed_spots():
    from dataclasses import replace
    cfg = _small_cfg()
    n_cone = len(XAFForwardModel(cfg).simulate(make_sample(cfg)).table)
    cfg_cap = replace(cfg, exit_model="tth_cap")
    n_cap = len(XAFForwardModel(cfg_cap).simulate(make_sample(cfg_cap)).table)
    assert n_cone < n_cap   # shadowing strictly removes spots


# --------------------------------------------------------------------------- #
#  Determinability metric (autograd)                                          #
# --------------------------------------------------------------------------- #
def test_strain_jacobian_autograd_full_rank():
    cfg = _small_cfg(n_grains=4)
    fwd = XAFForwardModel(cfg)
    grains = make_sample(cfg)
    d = metrics.grain_strain_determinability(
        fwd, grains.euler[0:1], grains.position[0:1], grains.strain[0])
    assert d.n_spots > 6
    assert d.singular_values.shape == (6,)
    assert d.full_rank           # all six strain DOF observable from merged data


def test_cross_axis_merge_improves_worst_strain_direction():
    cfg = _small_cfg(n_grains=8)
    fwd = XAFForwardModel(cfg)
    grains = make_sample(cfg)
    gain = metrics.cross_axis_gain(fwd, grains)
    # orthogonal second mounting should not hurt, and generally helps
    assert gain["median_s_min_merged"] >= gain["median_s_min_single"] - 1e-9


# --------------------------------------------------------------------------- #
#  Beam modes + position localisation                                         #
# --------------------------------------------------------------------------- #
def test_beam_modes_build_scan_config():
    from dataclasses import replace
    box = XAFForwardModel(_small_cfg(beam_mode="box"))
    assert box.scan_config is None
    line = XAFForwardModel(_small_cfg(beam_mode="line", beam_size_um=3.0))
    assert line.scan_config is not None
    assert len(line.scan_config.beam_positions) > 1
    # union over beam positions must recover the box spot set (same coverage)
    cfg_box = _small_cfg(beam_mode="box", n_grains=5)
    cfg_line = replace(cfg_box, beam_mode="line", beam_size_um=3.0)
    n_box = len(XAFForwardModel(cfg_box).simulate(make_sample(cfg_box)).table)
    n_line = len(XAFForwardModel(cfg_line).simulate(make_sample(cfg_line)).table)
    assert n_box == n_line


def test_position_localization_finite_and_scanning_better():
    cfg = _small_cfg(n_grains=6, beam_mode="line", beam_size_um=3.0)
    fwd = XAFForwardModel(cfg)
    grains = make_sample(cfg)
    loc = metrics.position_localization(fwd, grains)
    # box position must be finite (grain COM IS observable from pixel positions)
    assert math.isfinite(loc["box_friedel_position_um"])
    # point-focus beam localises better than the diffraction geometry alone
    assert loc["scanning_position_um"] < loc["box_friedel_position_um"]


def test_position_determinability_uses_pixel_observables():
    """Position must be observable -- regression against using 2theta/eta."""
    cfg = _small_cfg(n_grains=3)
    fwd = XAFForwardModel(cfg)
    grains = make_sample(cfg)
    d = metrics.grain_determinability(
        fwd, grains.euler[0:1], grains.position[0:1], grains.strain[0],
        wrt="position")
    assert d.full_rank and math.isfinite(d.precision_worst)


# --------------------------------------------------------------------------- #
#  Phase 2: registration + reconstruction                                     #
# --------------------------------------------------------------------------- #
def test_mat2euler_is_inverse_of_euler2mat():
    import midas_diffract as md
    rng = np.random.default_rng(0)
    for _ in range(5):
        e = torch.tensor(rng.uniform(0.1, 3.0, size=3))
        M = md.HEDMForwardModel.euler2mat(e)
        M2 = md.HEDMForwardModel.euler2mat(reconstruct.mat2euler_zxz(M))
        assert float((M2 - M).abs().max()) < 1e-6


def test_reconstruction_clean_median_recovers_truth():
    """Noise-free, tight seed: the median grain recovers truth well.

    A minority of grains have a reflection near Ewald tangency (near-infinite
    d(omega)/d(orientation)) and fall outside the local LM basin -- a genuine
    stiffness, not a bug -- so we assert on the robust median, and that the
    merged fit is at least as good as a single mounting.
    """
    cfg = _small_cfg(n_grains=8, opening_full_deg=20.0)
    fwd = XAFForwardModel(cfg)
    grains = make_sample(cfg)
    st = reconstruct.recovery_study(fwd, grains, n_grains=8, noise=False, seed=0)
    assert st["median_misori_deg_merged"] < 0.05
    assert st["median_strain_err_ue_merged"] < 200.0
    assert (st["median_misori_deg_merged"]
            <= st["median_misori_deg_single"] + 1e-6)


def test_spot_overlap_increases_with_grain_count():
    cfg_lo = _small_cfg(n_grains=20, material="ruby", opening_full_deg=20.0)
    cfg_hi = _small_cfg(n_grains=200, material="ruby", opening_full_deg=20.0)
    o_lo = metrics.spot_overlap(
        XAFForwardModel(cfg_lo).simulate(make_sample(cfg_lo)), dy_px=6, dz_px=6,
        domega_deg=1.0)["overlap_fraction"]
    o_hi = metrics.spot_overlap(
        XAFForwardModel(cfg_hi).simulate(make_sample(cfg_hi)), dy_px=6, dz_px=6,
        domega_deg=1.0)["overlap_fraction"]
    assert o_hi > o_lo
    assert 0.0 <= o_lo <= 1.0 and 0.0 <= o_hi <= 1.0


def test_pilatus_gaps_remove_spots_but_preserve_friedel():
    """Gaps cost spots, but the point-symmetric Pilatus tiling keeps Friedel
    pairs intact (a mate maps gap->gap, live->live)."""
    from dataclasses import replace
    cfg = _small_cfg(n_grains=15, material="ruby", opening_full_deg=20.0,
                     px_um=172.0, n_pixels_y=1475, n_pixels_z=1679)
    sim_perfect = XAFForwardModel(cfg).simulate(make_sample(cfg))
    cfg_gap = replace(cfg, detector_type="pilatus2m")
    sim_gap = XAFForwardModel(cfg_gap).simulate(make_sample(cfg_gap))
    assert len(sim_gap.table) < len(sim_perfect.table)          # gaps cost spots
    assert metrics.friedel_completeness(sim_gap) >= 0.98        # pairs preserved


def test_three_mountings_give_orthogonal_axes_and_help():
    """R_x(90)/R_y(90) remounts yield 3 mutually orthogonal rotation axes, and
    the third mounting does not worsen strain determinability."""
    ortho = (((1., 0, 0), 90.0), ((0, 1., 0), 90.0))
    z = np.array([0.0, 0.0, 1.0])
    cfg3 = _small_cfg(material="ruby", opening_full_deg=20.0, n_grains=5,
                      n_mountings=3, remount_specs=ortho)
    axes = [geometry.mounting_matrix(cfg3, m).T @ z for m in range(3)]  # sample frame
    for i in range(3):
        for j in range(i + 1, 3):
            assert abs(float(np.dot(axes[i], axes[j]))) < 1e-9
    # third mounting improves (or at least does not hurt) strain determinability
    fwd = XAFForwardModel(cfg3)
    grains = make_sample(cfg3)
    d2 = metrics.population_strain_determinability(fwd, grains, mountings=[0, 1])
    d3 = metrics.population_strain_determinability(fwd, grains, mountings=[0, 1, 2])
    assert d3["median_strain_precision_ue"] <= d2["median_strain_precision_ue"] * 1.02


def test_slit_beam_breaks_multi_mounting_mergeability():
    """A box beam covering the sample is 100% mergeable; a slit smaller than the
    sample collapses the fraction of grains seen in all three mountings."""
    ortho = (((1., 0, 0), 90.0), ((0, 1., 0), 90.0))
    cfg = _small_cfg(material="garnet_pyrope", n_grains=400, sample_radius_um=300.0,
                     n_mountings=3, remount_specs=ortho)
    fwd = XAFForwardModel(cfg)
    grains = make_sample(cfg)
    box = metrics.beam_mergeability(fwd, grains, half_y_um=350.0, half_z_um=350.0)
    slit = metrics.beam_mergeability(fwd, grains, half_y_um=50.0, half_z_um=50.0)
    assert box["mergeable_fraction"] > 0.95
    # slit lights a similar fraction per mounting but far fewer in all three
    assert slit["mergeable_fraction"] < 0.15
    assert slit["mergeable_fraction"] < min(slit["per_mounting_fraction"])


def test_large_cell_cubic_is_spot_rich():
    """Garnet (large cubic cell) gives far more spots than small-cell MgO."""
    def spg(mat):
        cfg = _small_cfg(material=mat, opening_full_deg=15.0, energy_keV=71.0,
                         n_grains=6)
        sim = XAFForwardModel(cfg).simulate(make_sample(cfg))
        return float(np.median(metrics.spots_per_grain(sim)))
    assert spg("garnet_pyrope") > 5 * spg("mgo")


def test_structure_factors_and_transmission():
    """|F|^2 keeps more measurable garnet reflections than MgO; higher energy
    transmits better."""
    from midas_xaf.crystal import build_reflections
    wl = 12.398 / 71.0
    def measurable(mat):
        hk, th, hi = build_reflections(mat, wl, 7.5)
        I = structure.reflection_intensities(mat, hi.numpy(), th.numpy(), wl)
        return int((I > 0.01).sum())
    assert measurable("garnet_pyrope") > 4 * measurable("mgo")
    t_lo = structure.dac_transmission(35.0)["T_total"]
    t_hi = structure.dac_transmission(90.0)["T_total"]
    assert t_hi > t_lo


def test_friedel_seeded_indexer_finds_grains():
    """Vector-space indexer recovers grains from the unlabelled cloud."""
    cfg = _small_cfg(material="ringwoodite", opening_full_deg=15.0,
                     energy_keV=80.0, n_grains=6)
    fwd = XAFForwardModel(cfg)
    grains = make_sample(cfg)
    d = synth.make_measured_spots(cfg, grains, fwd=fwd, seed=1)
    res = indexing.friedel_seeded_index(fwd, d["spots"], n_seed_spots=35,
                                        min_matched=15)
    assert res["n_found"] >= 3
    assert max(res["matched"]) >= 20        # a real grain scores high


def test_digital_twin_and_indexing_uniqueness():
    """Sparse merged spots uniquely index orientation (true >> random match)."""
    cfg = _small_cfg(material="ruby", opening_full_deg=20.0, n_grains=4)
    grains = make_sample(cfg)
    fwd = XAFForwardModel(cfg)
    d = synth.make_measured_spots(cfg, grains, fwd=fwd, seed=1)
    assert d["summary"]["n_total"] > 0
    u = indexing.orientation_uniqueness(fwd, grains, d["spots"], n_random=80, seed=0)
    assert u["frac_indexable"] == 1.0
    assert u["median_margin"] > 10


def test_pipeline_recovers_from_measured_spots():
    """Full chain (twin -> assign -> refine) recovers grains from measured data."""
    cfg = _small_cfg(material="ruby", opening_full_deg=20.0, n_grains=5)
    grains = make_sample(cfg)
    res = pipeline.run_pipeline(cfg, grains, seed=1)
    assert res.frac_recovered >= 0.8
    assert res.median_misorientation_mdeg < 200.0
    assert res.median_assignment_purity > 0.9


def test_budget_point_focus_costs_more_than_box():
    box = budget.beamtime_estimate(_small_cfg(beam_mode="box"))
    pt = budget.beamtime_estimate(_small_cfg(beam_mode="point", beam_size_um=3.0))
    assert pt["scans_per_mounting"] > 1
    assert pt["total_hours"] > box["total_hours"]


def test_corundum_anisotropic_strain_is_physical():
    cfg = _small_cfg(material="ruby", n_grains=4)
    grains = make_sample(cfg)
    C = micromech.corundum_stiffness_GPa()
    strains = micromech.grain_strains_from_stress_aniso(
        grains.euler, np.diag([100.0, 0.0, -100.0]), C)
    assert strains.shape == (4, 6)
    assert np.isfinite(strains.numpy()).all()
    assert float(strains.abs().max()) * 1e6 < 2000.0    # ~100 MPa -> few 100 ue


def test_micromech_recovers_deviatoric_stress():
    """Apply a deviatoric load, reconstruct, and recover the macro stress."""
    cfg = _small_cfg(material="ruby", opening_full_deg=20.0)
    stress = micromech.deviatoric_load(500.0, "x") + micromech.deviatoric_load(-500.0, "z")
    res = micromech.micromech_study(cfg, stress, E_GPa=400.0, nu=0.23,
                                    n_grains=6, seed=0, noise=False)
    assert res["n_recovered"] >= 3
    # deviatoric (shear-relevant) recovered to a fraction of the 1 GPa load
    assert res["deviatoric_stress_error_MPa"] < 200.0


def test_fiducial_registration_needs_three_markers():
    """Two markers are rotationally degenerate; three resolve the remount."""
    import numpy as _np
    true_R = geometry.mounting_matrix(_small_cfg(), 1)
    pool = _np.array([[10.0, 5, -3], [-8, 12, 6], [4, -9, 11], [0, 7, -14]])
    two = merge.register_fiducials(true_R, pool[:2], sigma_um=0.5, seed=0)
    three = merge.register_fiducials(true_R, pool[:3], sigma_um=0.5, seed=0)
    assert two["degenerate"] and not three["degenerate"]
    # three non-collinear markers register far better than two
    assert three["angle_error_deg"] < two["angle_error_deg"]


# --------------------------------------------------------------------------- #
#  Autonomy driver (active experiment design)                                 #
# --------------------------------------------------------------------------- #
def test_autonomy_active_beats_uniform():
    """The D-optimal greedy schedule reaches the final precision in no more
    acquisitions than a uniform schedule, and its curve is monotone."""
    from midas_xaf import autonomy
    cfg = _small_cfg(material="garnet_pyrope", opening_full_deg=23.0,
                     n_grains=3, n_mountings=2, seed=1)
    grains = make_sample(cfg)
    centers = list(np.arange(-180.0, 180.0, 60.0))       # 6/mounting -> 12
    res = autonomy.benchmark(cfg, grains, wedge_centers_deg=centers,
                             wedge_half_deg=8.0, n_random=5)
    assert len(res["labels"]) == 12
    act = res["active"]
    # monotone non-increasing information => precision never worsens
    assert all(b <= a * (1 + 1e-9) for a, b in zip(act, act[1:]))
    # greedy is at least as fast as uniform to the target
    assert res["n_active"] is not None
    assert res["n_uniform"] is None or res["n_active"] <= res["n_uniform"]
    # and both end at the same full-menu precision
    assert act[-1] == pytest.approx(res["uniform"][-1], rel=1e-6)
