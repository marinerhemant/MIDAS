"""midas_defect.raster and midas_defect.synthetic: the raster composition, end to end.

Every test drives the SAME code path a real 6-ID-C/HPCAT raster would: raw frames -> the real
ingest chain -> find_domains -> completeness/honesty/ab gates -> refine_cell_joint. Nothing here
mocks a package function; the synthetic frames are the only thing that isn't real data, and they
are built from a KNOWN planted answer (`meta`/`PositionTruth`) so every test can grade itself.
"""
from __future__ import annotations

import json

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from midas_defect.geometry import Geometry
from midas_defect.raster import assemble_raster_results, reduce_one_position, reduce_raster_block
from midas_defect.synthetic import synthetic_dac_raster, synthetic_position_frames

A, B_SPLIT, C = 4.0, 3.9, 10.0                    # a genuinely fictional cell, not any real material
SG = 123                                          # P4/mmm
SIGMA_RTN = (0.02, 0.02, 0.02)


def _geom(n_pix=768, lsd_um=120_000.0, px_um=172.0):
    return Geometry(lsd_um=lsd_um, bcy_px=n_pix / 2, bcz_px=n_pix / 2, px_um=px_um,
                    wavelength_A=0.4246, n_pix_y=n_pix, n_pix_z=n_pix,
                    omega_first_deg=-19.5, omega_step_deg=1.0, n_frames=40, label="synthetic")


def _one_domain_frames(a, b, c, seed=0, euler=(15.0, 25.0, -8.0), **kw):
    geom = _geom()
    U = Rotation.from_euler("zyx", list(euler), degrees=True).as_matrix()
    frames, truth = synthetic_position_frames(
        [U], a=a, b=b, c=c, space_group_number=SG, geom=geom, hmax=8, kmax=8, lmax=14,
        seed=seed, **kw)
    return geom, frames, truth, U


# ---------------------------------------------------------------------------
# synthetic_position_frames: the frames are internally consistent with the geometry
# ---------------------------------------------------------------------------

def test_synthetic_frames_reflections_are_on_detector_and_in_range():
    geom, frames, truth, U = _one_domain_frames(A, B_SPLIT, C)
    assert frames.shape == (geom.n_frames, geom.n_pix_z, geom.n_pix_y)
    assert len(truth.reflections) > 50, "too few planted reflections for a meaningful test"
    for r in truth.reflections:
        assert 0 <= r.row < geom.n_pix_z
        assert 0 <= r.col < geom.n_pix_y
        assert -0.5 <= r.frame <= geom.n_frames - 0.5


def test_synthetic_frames_pedestal_survives_build_mask():
    """A pedestal at or below build_mask's low_count_threshold masks nearly everything --
    a real trap (build_mask flags a pixel whose per-frame MEDIAN is persistently low), caught
    once while building this module. Pin it so a future default change cannot reintroduce it."""
    from midas_defect.ingest import build_mask
    geom, frames, truth, U = _one_domain_frames(A, B_SPLIT, C)
    m = build_mask(frames)
    mask = m.mask if hasattr(m, "mask") else m
    assert mask.mean() < 0.05, "the default pedestal should leave almost the whole detector live"


# ---------------------------------------------------------------------------
# reduce_one_position: recovers the planted cell, and its gates can FAIL
# ---------------------------------------------------------------------------

def test_reduce_one_position_recovers_planted_cell_and_split():
    geom, frames, truth, U = _one_domain_frames(A, B_SPLIT, C, seed=0)
    res = reduce_one_position(frames, geom, a=A, c=C, space_group_number=SG,
                              sigma_rtn=SIGMA_RTN, n_bootstrap=200)
    assert res.quotable, res.notes
    assert len(res.domains.domains) == 1
    dom = res.domains.domains[0]
    # a <-> b is a gauge choice for a tetragonal-extinction cell distorted orthorhombically
    # (the manual's own point: "a common sign across domains is physically impossible" because
    # the label is arbitrary) -- so check the SET of lengths, not which one is called "a".
    assert sorted([dom.lat.a, dom.lat.b]) == pytest.approx(sorted([A, B_SPLIT]), abs=0.01)
    assert dom.lat.c == pytest.approx(C, abs=0.01)
    delta, sigma, z = res.split_pct
    planted_delta = 100.0 * (A - B_SPLIT) / B_SPLIT
    assert abs(delta) == pytest.approx(abs(planted_delta), abs=0.3)
    assert z > 5, "a 2.6% split on ~300 reflections should be highly significant"


def test_reduce_one_position_omega_sign_is_decisive_and_correct():
    geom, frames, truth, U = _one_domain_frames(A, B_SPLIT, C, seed=0)
    res = reduce_one_position(frames, geom, a=A, c=C, space_group_number=SG,
                              sigma_rtn=SIGMA_RTN, n_bootstrap=20)
    assert res.omega_sign.chosen_sign == 1
    assert res.omega_sign.decisive
    assert res.omega_sign.n_explained[1] > 0
    assert res.omega_sign.n_explained[-1] == 0


def test_reduce_one_position_completeness_has_almost_no_missed_or_absent():
    geom, frames, truth, U = _one_domain_frames(A, B_SPLIT, C, seed=0)
    res = reduce_one_position(frames, geom, a=A, c=C, space_group_number=SG,
                              sigma_rtn=SIGMA_RTN, n_bootstrap=20)
    counts = res.completeness[0].counts
    total = sum(counts.values())
    assert counts["INDEXED"] / total > 0.9
    assert counts["MASKED"] == 0


def test_decoy_test_rejects_a_deliberately_wrong_cell():
    """The honesty gate must have teeth: an inflated cell must NOT pass, or the gate is inert."""
    geom, frames, truth, U = _one_domain_frames(A, B_SPLIT, C, seed=0)
    res = reduce_one_position(frames, geom, a=A, c=C, space_group_number=SG,
                              sigma_rtn=SIGMA_RTN, n_bootstrap=20, decoy_fractions=(0.05, -0.05))
    dt = res.decoy[0]
    assert dt["verdict"] == "informative"
    assert dt["real_passes"]
    assert not dt["decoys_passing"], f"a 5% decoy should not pass: {dt}"


def test_ab_gate_fails_when_only_weak_partners_exist():
    """With few reflections, every a/b pair should rest on <= 1 spot on one side -- the gate
    must refuse to claim a splitting, not claim one anyway. If this ever passes on a small
    detector, the gate has stopped being able to fail."""
    small_geom = _geom(n_pix=256, lsd_um=150_000.0)
    U2 = Rotation.from_euler("zyx", [15.0, 25.0, -8.0], degrees=True).as_matrix()
    frames2, truth2 = synthetic_position_frames([U2], a=A, b=B_SPLIT, c=C,
                                                space_group_number=SG, geom=small_geom,
                                                hmax=3, kmax=3, lmax=5, seed=0)
    res = reduce_one_position(frames2, small_geom, a=A, c=C, space_group_number=SG,
                              sigma_rtn=SIGMA_RTN, n_bootstrap=20)
    assert not res.quotable
    assert not any(g.get("passed_to_refine") for g in res.ab_gates)


def test_reduce_one_position_no_splitting_is_flagged_as_provisional():
    """a == b: the pipeline must not manufacture a splitting, and must say the single-position
    answer cannot rule out the radial-systematic artifact on its own."""
    geom, frames, truth, U = _one_domain_frames(A, A, C, seed=1)
    res = reduce_one_position(frames, geom, a=A, c=C, space_group_number=SG,
                              sigma_rtn=SIGMA_RTN, n_bootstrap=100)
    delta, sigma, z = res.split_pct
    assert abs(delta) < 0.5, "no planted split should not recover a large one"
    if abs(z) > 3:
        assert any("radial-systematic" in n for n in res.notes)


def test_reduce_one_position_no_spots_returns_empty_not_an_exception():
    geom = _geom()
    frames = np.full((geom.n_frames, geom.n_pix_z, geom.n_pix_y), 150.0, dtype=np.float32)
    rng = np.random.default_rng(0)
    frames = rng.poisson(frames).astype(np.float32)     # pure background, no reflections
    res = reduce_one_position(frames, geom, a=A, c=C, space_group_number=SG,
                              sigma_rtn=SIGMA_RTN)
    assert res.n_spots == 0
    assert not res.quotable
    assert not res.domains.domains


# ---------------------------------------------------------------------------
# reduce_raster_block / assemble_raster_results: sharding and idempotence
# ---------------------------------------------------------------------------

def _four_point_raster():
    geom = _geom()

    def orientations_of_point(p, rng):
        U = Rotation.from_euler("zyx", [15.0 + p, 25.0, -8.0], degrees=True).as_matrix()
        return [U], [f"domain0_p{p}"]

    return geom, synthetic_dac_raster(a=A, b=B_SPLIT, c=C, space_group_number=SG, geom=geom,
                                      hmax=8, kmax=8, lmax=14, n_points=4,
                                      orientations_of_point=orientations_of_point, seed=0)


def test_reduce_raster_block_shards_are_disjoint_and_complete(tmp_path):
    geom, raster = _four_point_raster()
    points = list(range(4))
    shard0 = reduce_raster_block(raster.loader, geom, points, tmp_path, block_nr=0, n_blocks=2,
                                 a=A, c=C, space_group_number=SG, sigma_rtn=SIGMA_RTN,
                                 n_bootstrap=20)
    shard1 = reduce_raster_block(raster.loader, geom, points, tmp_path, block_nr=1, n_blocks=2,
                                 a=A, c=C, space_group_number=SG, sigma_rtn=SIGMA_RTN,
                                 n_bootstrap=20)
    assert set(shard0) & set(shard1) == set()
    assert set(shard0) | set(shard1) == set(points)
    rows = assemble_raster_results(tmp_path, points)
    assert all(r is not None for r in rows)
    assert all(r["quotable"] for r in rows)


def test_reduce_raster_block_rerun_is_idempotent(tmp_path):
    geom, raster = _four_point_raster()
    points = [0, 1]
    reduce_raster_block(raster.loader, geom, points, tmp_path, a=A, c=C,
                        space_group_number=SG, sigma_rtn=SIGMA_RTN, n_bootstrap=20)
    first = assemble_raster_results(tmp_path, points)
    reduce_raster_block(raster.loader, geom, points, tmp_path, a=A, c=C,
                        space_group_number=SG, sigma_rtn=SIGMA_RTN, n_bootstrap=20)
    second = assemble_raster_results(tmp_path, points)
    for r1, r2 in zip(first, second):
        assert r1["split_pct"] == r2["split_pct"]


def test_assemble_raster_results_leaves_unrun_points_as_none(tmp_path):
    geom, raster = _four_point_raster()
    reduce_raster_block(raster.loader, geom, [0, 2], tmp_path, a=A, c=C,
                        space_group_number=SG, sigma_rtn=SIGMA_RTN, n_bootstrap=20)
    rows = assemble_raster_results(tmp_path, [0, 1, 2, 3])
    assert rows[0] is not None and rows[2] is not None
    assert rows[1] is None and rows[3] is None


def test_reduce_raster_block_rejects_invalid_sharding(tmp_path):
    geom, raster = _four_point_raster()
    with pytest.raises(ValueError):
        reduce_raster_block(raster.loader, geom, [0], tmp_path, block_nr=2, n_blocks=2,
                            a=A, c=C, space_group_number=SG, sigma_rtn=SIGMA_RTN)


# ---------------------------------------------------------------------------
# Two fixes found running this module against real 2604 raster data (2026-09-12):
# ring detection must use the UNSUBTRACTED image, and row-seeding must rank by real
# intensity, not a blob's frame span. Neither surfaces on the homogeneous synthetic
# scenes above -- pinned here so neither regresses silently.
# ---------------------------------------------------------------------------

def test_powder_ring_detection_uses_the_unsubtracted_image():
    """detect_powder_rings on the background-SUBTRACTED stack sees almost nothing -- removing
    azimuthally-uniform intensity is the background step's job. On a real, heavily gasket/anvil
    -contaminated 2604 position this left 87% of a genuine ring un-flagged and drowned the
    crystal's own row-seed pool (6 rings -> 2.2% flagged, vs 59 rings -> 21.8% on the
    unsubtracted median). Pinned two ways: the unsubtracted image must not detect FEWER rings,
    and reduce_one_position must still find the real domain cleanly with a strong ring present."""
    from midas_defect.geometry import detector_angle_maps
    from midas_defect.ingest import build_mask, subtract_background, detect_powder_rings

    geom, frames, truth, U = _one_domain_frames(A, B_SPLIT, C, seed=2,
                                                powder_two_theta_deg=(6.0,), powder_amplitude=250.0)
    tth, az = detector_angle_maps(geom)
    m = build_mask(frames)
    mask = m.mask if hasattr(m, "mask") else m
    sub = subtract_background(frames, tth, az, mask)
    rings_sub = detect_powder_rings(sub.max(axis=0), tth, mask, azimuth_deg=az)
    rings_raw = detect_powder_rings(np.median(frames, axis=0), tth, mask, azimuth_deg=az)
    n_sub = len(getattr(rings_sub, "centre_deg", rings_sub))
    n_raw = len(getattr(rings_raw, "centre_deg", rings_raw))
    assert n_raw >= max(n_sub, 1), (
        f"a synthetic ring this strong must be detected on the unsubtracted image ({n_raw} rings), "
        f"and not fewer than the subtracted one sees ({n_sub})")

    res = reduce_one_position(frames, geom, a=A, c=C, space_group_number=SG,
                              sigma_rtn=SIGMA_RTN, n_bootstrap=20)
    assert res.domains.domains, f"the crystal domain must still be found with a powder ring present: {res.notes}"
    dom = res.domains.domains[0]
    assert sorted([dom.lat.a, dom.lat.b]) == pytest.approx(sorted([A, B_SPLIT]), abs=0.02)


def test_row_seeding_intensity_is_integrated_counts_not_frame_span(monkeypatch):
    """`find_lattice_rows` ranks its seed pool by `argsort(-intensity)[:n_seed]` (`rows.py`).
    reduce_one_position must pass `spots.integrated` (real brightness) to that ranking, not
    `spots.n_frames` (how many frames a blob spans) -- on real 2604 data the latter let dim,
    many-frame gasket/anvil debris outrank the crystal's own bright reflections and
    find_domains found nothing at either omega sign."""
    import midas_defect.raster as raster_mod

    seen = {}
    orig = raster_mod.find_domains

    def _capture(q, intensity, row, col, frame, **kw):
        seen["intensity"] = np.asarray(intensity).copy()
        seen["n_calls"] = seen.get("n_calls", 0) + 1
        return orig(q, intensity, row, col, frame, **kw)

    monkeypatch.setattr(raster_mod, "find_domains", _capture)
    geom, frames, truth, U = _one_domain_frames(A, B_SPLIT, C, seed=0)
    reduce_one_position(frames, geom, a=A, c=C, space_group_number=SG,
                        sigma_rtn=SIGMA_RTN, n_bootstrap=20)

    assert seen.get("n_calls", 0) >= 1, "find_domains was never called"
    from midas_defect.raster import _ingest_position
    spots, *_ = _ingest_position(frames, geom)
    assert seen["intensity"] == pytest.approx(spots.integrated.values.astype(float)), (
        "reduce_one_position must rank the row-seed pool by spots.integrated (real brightness), "
        "not spots.n_frames (frame span) -- see this test's docstring")
