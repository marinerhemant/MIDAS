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


def test_orientation_envelope_is_off_by_default_and_optional_when_on():
    """SKETCH, 2026-09-13: the orientation envelope (seed_index.bootstrap_orientation_uncertainty,
    wired into reduce_one_position via orientation_uncertainty=). Default must stay None per
    domain -- this is not free (it re-runs the orientation refit n_orientation_boot times per
    domain) and must never fire silently. When turned on, it must return real numbers with the
    documented keys, and the bootstrap mean orientation must sit close to the domain's own
    point-estimate fit (both are estimates of the same planted U)."""
    geom, frames, truth, U = _one_domain_frames(A, B_SPLIT, C, seed=0)

    res_off = reduce_one_position(frames, geom, a=A, c=C, space_group_number=SG,
                                  sigma_rtn=SIGMA_RTN, n_bootstrap=20)
    assert res_off.orientation_envelope == [None]

    res_on = reduce_one_position(frames, geom, a=A, c=C, space_group_number=SG,
                                 sigma_rtn=SIGMA_RTN, n_bootstrap=20,
                                 orientation_uncertainty=True, n_orientation_boot=15)
    assert len(res_on.orientation_envelope) == 1
    oe = res_on.orientation_envelope[0]
    assert oe is not None
    for k in ("U_mean", "U_bootstraps", "pair_angle_mean_deg",
             "pair_angle_p95_deg", "pair_angle_max_deg", "n_boot", "keep_n"):
        assert k in oe
    assert oe["n_boot"] == 15
    # the bootstrap mean orientation and the domain's own point-estimate fit are both
    # estimates of the SAME planted U -- they must agree to a small angle, not merely both
    # exist. This is the actual correctness check, not just "did it run".
    dom = res_on.domains.domains[0]
    delta_U = dom.U @ oe["U_mean"].T
    angle_deg = np.degrees(np.arccos(np.clip((np.trace(delta_U) - 1) / 2, -1, 1)))
    assert angle_deg < 2.0, f"bootstrap mean U disagrees with the point estimate by {angle_deg} deg"
    # a well-populated, noise-free-ish synthetic domain should show a TIGHT envelope --
    # this is what makes the envelope a real signal rather than a number that's always large.
    assert oe["pair_angle_p95_deg"] < 1.0


def test_cell_bootstrap_samples_gives_individual_a_b_c_ranges():
    """refine_cell_joint's bootstrap loop already computes every resample's full cell, then
    used to discard everything but the std. cell_bootstrap_samples now carries the raw
    (n_kept, 6) array through; cell_sigma_bootstrap is the compact per-parameter summary of
    the SAME data. Both must be populated together, agree with each other (sigma ==
    samples.std), and land near the known planted a/b/c on a clean synthetic domain."""
    geom, frames, truth, U = _one_domain_frames(A, B_SPLIT, C, seed=0)
    res = reduce_one_position(frames, geom, a=A, c=C, space_group_number=SG,
                              sigma_rtn=SIGMA_RTN, n_bootstrap=50)
    assert res.refined_cell is not None
    assert res.cell_bootstrap_samples is not None
    assert res.cell_sigma_bootstrap is not None
    samples = res.cell_bootstrap_samples
    assert samples.shape[1] == 6, "one column each for a, b, c, alpha, beta, gamma"
    assert samples.shape[0] > 4
    for j in range(3):
        assert samples[:, j].std(ddof=1) == pytest.approx(res.cell_sigma_bootstrap[j], rel=1e-9)
    # a <-> b is a gauge choice (same reasoning as test_reduce_one_position_recovers_
    # planted_cell_and_split above) -- check the SET of means, not which one lands in
    # column 0.
    a_mean, b_mean = samples[:, 0].mean(), samples[:, 1].mean()
    assert sorted([a_mean, b_mean]) == pytest.approx(sorted([A, B_SPLIT]), abs=0.05)


def test_to_dict_drops_bulky_bootstrap_arrays_but_keeps_summaries():
    """A raster of hundreds of positions writes one JSON file per point
    (reduce_raster_block) -- the full orientation U_bootstraps and cell_bootstrap_samples
    arrays belong on the live, single-position object (for 05's own plotting) and must NOT
    be replicated into every position's JSON file. The compact summaries
    (cell_sigma_bootstrap, and each domain's pair_angle_*/U_mean/independent_resamples) must
    survive into the JSON form, since a raster-wide map is built from exactly those."""
    geom, frames, truth, U = _one_domain_frames(A, B_SPLIT, C, seed=0)
    res = reduce_one_position(frames, geom, a=A, c=C, space_group_number=SG,
                              sigma_rtn=SIGMA_RTN, n_bootstrap=20,
                              orientation_uncertainty=True, n_orientation_boot=10)
    d = res.to_dict()
    assert d["cell_bootstrap_samples"] is None
    assert d["cell_sigma_bootstrap"] is not None
    oe = d["orientation_envelope"][0]
    assert oe is not None
    assert "U_bootstraps" not in oe, "the bulky per-resample U array must not reach JSON"
    for k in ("U_mean", "pair_angle_mean_deg", "pair_angle_p95_deg", "pair_angle_max_deg",
             "n_boot", "keep_n", "n_distinct_subsets", "independent_resamples"):
        assert k in oe
    json.dumps(d, default=str)  # must actually serialize, not just look dict-shaped


def test_orientation_envelope_skips_domains_with_too_few_reflections():
    """bootstrap_orientation_uncertainty's own resample floor (keep_n = max(4, 0.7*n)) means
    fewer than 4 centroids makes it try to choose 4 distinct items from a smaller population
    without replacement, which numpy raises on -- confirmed directly below. reduce_one_position
    guards this with `len(centroids) >= 4` before ever calling it (see raster.py); this test
    pins WHY that guard exists, so it cannot be "simplified" away as defensive dead code later."""
    from midas_defect.seed_index import bootstrap_orientation_uncertainty

    too_few = list(zip([(1, 0, 0), (0, 1, 0), (0, 0, 1)], np.eye(3)))
    with pytest.raises(ValueError):
        bootstrap_orientation_uncertainty(too_few, np.eye(3), a=A, c=C, n_boot=5)


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


def test_quotable_is_false_when_the_real_fit_fails_the_decoy_test(monkeypatch):
    """`decoy_test` has THREE verdicts (`informative`, `uninformative`, `real_fails`), and
    `quotable` must require the first one. Found on real La3Ni2O7 2601 data (2026-09-13,
    positions 332 and 364): `fit_passed_decoy` checked `verdict != "uninformative"`, which is
    also true for `real_fails` (the real fit itself does not clear the acceptance threshold,
    a stronger failure than a decoy merely tying it) -- so a domain whose own fit failed
    outright was reported `quotable: True`. Force `real_fails` on an otherwise-quotable
    scenario and pin that `quotable` is False and a note names the real cause."""
    import midas_defect.raster as raster_mod

    geom, frames, truth, U = _one_domain_frames(A, B_SPLIT, C, seed=0)

    real_decoy_test = raster_mod.decoy_test

    def _force_real_fails(score_of, cell, decoys, *, threshold):
        result = real_decoy_test(score_of, cell, decoys, threshold=threshold)
        result = dict(result, real_passes=False, verdict="real_fails")
        return result

    monkeypatch.setattr(raster_mod, "decoy_test", _force_real_fails)
    res = reduce_one_position(frames, geom, a=A, c=C, space_group_number=SG,
                              sigma_rtn=SIGMA_RTN, n_bootstrap=20)

    assert any(g.get("passed_to_refine") for g in res.ab_gates), \
        "test setup: the ab gate chain itself must still pass so only the decoy verdict differs"
    assert res.decoy[0]["verdict"] == "real_fails"
    assert not res.quotable, "a domain whose own fit fails the decoy test must not be quotable"
    assert any("did not clear the acceptance threshold" in n for n in res.notes)


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
    # 2026-09-13: was points=list(range(4)) -- shard disjointness/union is a property of the
    # BLOCK-NR ARITHMETIC (point_indices[block_nr::n_blocks]), not of how many points there
    # are or of reduce_one_position's own physics. 2 points still forces both shards non-empty
    # (n_blocks=2 -> shard0=[0], shard1=[1]) and cuts this test's cost roughly in half (it was
    # the single slowest test in the suite at ~84s, almost entirely reduce_one_position calls).
    geom, raster = _four_point_raster()
    points = [0, 1]
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
    # 2026-09-13: was points=[0, 1] -- idempotency (does a second, independent
    # reduce_one_position call on the SAME point reproduce split_pct exactly) is a per-point
    # property; a second point adds cost without adding coverage. 1 point still forces a real
    # re-computation (not a cache hit) for the actual determinism check, and halves this
    # test's cost (~85s, the single slowest test alongside the sharding test above).
    geom, raster = _four_point_raster()
    points = [0]
    reduce_raster_block(raster.loader, geom, points, tmp_path, a=A, c=C,
                        space_group_number=SG, sigma_rtn=SIGMA_RTN, n_bootstrap=20)
    first = assemble_raster_results(tmp_path, points)
    reduce_raster_block(raster.loader, geom, points, tmp_path, a=A, c=C,
                        space_group_number=SG, sigma_rtn=SIGMA_RTN, n_bootstrap=20)
    second = assemble_raster_results(tmp_path, points)
    for r1, r2 in zip(first, second):
        assert r1["split_pct"] == r2["split_pct"]


def test_assemble_raster_results_leaves_unrun_points_as_none(tmp_path):
    # 2026-09-13: was reducing [0, 2] and reading back [0, 1, 2, 3] -- the "ran vs never
    # attempted" distinction only needs ONE real reduction and ONE never-requested point to
    # prove; a second real point (2) added a full reduce_one_position call (~20s) with no
    # additional coverage of this function's own None-filling behavior.
    geom, raster = _four_point_raster()
    reduce_raster_block(raster.loader, geom, [0], tmp_path, a=A, c=C,
                        space_group_number=SG, sigma_rtn=SIGMA_RTN, n_bootstrap=20)
    rows = assemble_raster_results(tmp_path, [0, 1])
    assert rows[0] is not None
    assert rows[1] is None


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


# ---------------------------------------------------------------------------
# raster_wide_asymmetry_sign_test: the sign-aware alternative to the magnitude-only
# check /verify REFUTED on real 2604 data (2026-09-12, claim a1c6f1f505cb).
# ---------------------------------------------------------------------------

def test_raster_wide_asymmetry_sign_test_null_is_not_significant():
    """A raster whose domains' a/b signs are UNRELATED to their own h/k dominance --
    built from independent synthetic positions -- must not be flagged, however many points."""
    from midas_defect.raster import raster_wide_asymmetry_sign_test
    rng = np.random.default_rng(2)
    rows = []
    for p in range(30):
        h_dom = bool(rng.integers(0, 2))
        b_p = A + 0.1 if h_dom else A - 0.1              # sign(a-b) tied to h_dom BY DESIGN...
        if rng.integers(0, 2):                            # ...half the time, randomly flip it
            b_p = 2 * A - b_p                             # so the two are UNCORRELATED overall
        hkl = np.array([[3, 1, l] for l in range(-8, 9)]) if h_dom else \
              np.array([[1, 3, l] for l in range(-8, 9)])
        rows.append(dict(domains=[dict(hkl=hkl.tolist(), a=A, b=b_p)]))
    r = raster_wide_asymmetry_sign_test(rows)
    assert r["n_domains_pooled"] == 30
    assert r["p_value"] > 0.05, f"an independent null must not read as significant: {r}"


def test_raster_wide_asymmetry_sign_test_catches_a_planted_correlation():
    """The positive control: plant EXACTLY the documented mechanism (h-dominant domains
    consistently fitted a>b) and confirm it is caught, pooled across positions."""
    from midas_defect.raster import raster_wide_asymmetry_sign_test
    rng = np.random.default_rng(3)
    rows, none_row = [], None
    for p in range(25):
        h_dom = bool(rng.integers(0, 2))
        a_p, b_p = (4.0, 3.9) if h_dom else (3.9, 4.0)    # PLANTED: tracks h_dom every time
        hkl = np.array([[3, 1, l] for l in range(-8, 9)]) if h_dom else \
              np.array([[1, 3, l] for l in range(-8, 9)])
        rows.append(dict(domains=[dict(hkl=hkl.tolist(), a=a_p, b=b_p)]))
    rows.insert(5, None)                                  # an unfinished shard -- must be skipped
    r = raster_wide_asymmetry_sign_test(rows)
    assert r["n_domains_pooled"] == 25
    # planted: h-dominant domains get a>b -- "opposite sense" in asymmetry_sign_test's naming.
    assert r["direction"] == -1
    assert r["p_value"] < 1e-4, f"a perfect planted correlation must be caught: {r}"


def test_raster_wide_asymmetry_sign_test_pools_multiple_domains_per_position():
    """A position with several domains contributes all of them, not just one."""
    from midas_defect.raster import raster_wide_asymmetry_sign_test
    hkl_h = [[3, 1, 0]] * 5
    hkl_k = [[1, 3, 0]] * 5
    rows = [dict(domains=[dict(hkl=hkl_h, a=4.0, b=3.9), dict(hkl=hkl_k, a=3.9, b=4.0)])] * 10
    r = raster_wide_asymmetry_sign_test(rows)
    assert r["n_domains_pooled"] == 20
