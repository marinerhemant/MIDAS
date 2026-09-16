"""midas_defect.spatial_coherence: recovering a real grain a position's own
free search missed, using an orientation found at another position -- and
making sure the null can actually FAIL (no shared grain -> no recovery).

Every test drives the real reduce_one_position/refine_to_convergence chain on
synthetic frames built from a KNOWN planted answer, same convention as
test_raster.py."""
from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from midas_defect.geometry import Geometry
from midas_defect.synthetic import synthetic_position_frames
from midas_defect.spatial_coherence import (
    recover_domains_across_raster, morans_i, permutation_null_morans_i,
    planted_identical_cell_delta, RecoveredDomain, PositionCoherence, RasterCoherenceResult,
    best_domain_per_position, grid_quantity, spatial_coherence_report, SpatialCoherenceReport,
    cluster_permutation_null_morans_i, exact_rank_p_value,
    values_to_grid, grid_coherence_report,
)
from midas_hkls.cell_constrained import refine_cell_joint, refine_cell_radial

A, B_SPLIT, C = 4.0, 3.9, 10.0        # a genuinely fictional cell, not any real material
SG = 123                              # P4/mmm
SIGMA_RTN = (0.02, 0.02, 0.02)


def _disk_loader(p: int, base_dir) -> np.ndarray:
    """A loader for the n_workers>1 (spawn) path: MUST be a plain, module-level, picklable
    callable, not a closure over an in-memory frames dict (the pattern every other test in this
    file uses, which does not survive `spawn`) -- this is what a real caller's own loader needs
    to look like for parallel use. Bind `base_dir` via `functools.partial`, itself picklable."""
    import numpy as _np
    from pathlib import Path as _Path
    return _np.load(_Path(base_dir) / f"frame_{p}.npy")


def _geom(n_pix=768):
    return Geometry(lsd_um=120_000.0, bcy_px=n_pix / 2, bcz_px=n_pix / 2, px_um=172.0,
                    wavelength_A=0.4246, n_pix_y=n_pix, n_pix_z=n_pix,
                    omega_first_deg=-19.5, omega_step_deg=1.0, n_frames=40, label="synthetic")


def test_recovers_a_grain_too_sparse_for_free_search_to_find_on_its_own():
    """Position 1: U0, full hkl box -- free search finds it natively (this is
    the ALREADY-validated path, test_raster.py covers it; not re-asserted
    here beyond a sanity count). Position 2: the SAME U0, but a hkl box small
    enough (9 reflections) that free search finds NOTHING there -- checked as
    a precondition of this test being meaningful, not assumed. Recovery must
    find it anyway, seeded from position 1."""
    geom = _geom()
    U0 = Rotation.from_euler("zyx", [15.0, 25.0, -8.0], degrees=True).as_matrix()

    frames = {}
    frames[1], _ = synthetic_position_frames([U0], a=A, b=B_SPLIT, c=C, space_group_number=SG,
                                              geom=geom, hmax=8, kmax=8, lmax=14, seed=0)
    frames[2], truth2 = synthetic_position_frames([U0], a=A, b=B_SPLIT, c=C, space_group_number=SG,
                                                   geom=geom, hmax=1, kmax=1, lmax=2, seed=1)
    assert len(truth2.reflections) < 15, "test setup: position 2 must be genuinely sparse"

    def loader(p):
        return frames[p]

    result = recover_domains_across_raster(
        loader, geom, [1, 2], a=A, c=C, space_group_number=SG, sigma_rtn=SIGMA_RTN,
        dedup_threshold_deg=1.0, n_null_draws=8, rng=np.random.default_rng(0))

    pc2 = result.per_position[2]
    assert len(pc2.res.domains.domains) == 0, \
        "test setup: position 2's OWN free search must find nothing, or this isn't testing recovery"
    assert len(pc2.recovered) == 1, f"expected exactly 1 recovered domain at position 2, got {len(pc2.recovered)}"
    rec = pc2.recovered[0]
    assert rec.n == len(truth2.reflections)
    assert rec.origin_point == 1


def test_no_shared_grain_means_no_recovery():
    """Two positions, two UNRELATED random orientations, no shared grain at
    all. The null must be able to FAIL: recovery should find nothing."""
    geom = _geom()
    U1 = Rotation.from_euler("zyx", [12.0, -30.0, 40.0], degrees=True).as_matrix()
    U2 = Rotation.from_euler("zyx", [-55.0, 8.0, 17.0], degrees=True).as_matrix()

    frames = {}
    frames[1], _ = synthetic_position_frames([U1], a=A, b=B_SPLIT, c=C, space_group_number=SG,
                                              geom=geom, hmax=8, kmax=8, lmax=14, seed=2)
    frames[2], _ = synthetic_position_frames([U2], a=A, b=B_SPLIT, c=C, space_group_number=SG,
                                              geom=geom, hmax=8, kmax=8, lmax=14, seed=3)

    def loader(p):
        return frames[p]

    result = recover_domains_across_raster(
        loader, geom, [1, 2], a=A, c=C, space_group_number=SG, sigma_rtn=SIGMA_RTN,
        dedup_threshold_deg=1.0, n_null_draws=8, rng=np.random.default_rng(0))

    total_recovered = sum(len(pc.recovered) for pc in result.per_position.values())
    assert total_recovered == 0, \
        f"no shared grain was planted, but {total_recovered} domain(s) were 'recovered' -- the null failed to fail"
    # both positions' own native domain should still be there, untouched
    assert len(result.per_position[1].res.domains.domains) == 1
    assert len(result.per_position[2].res.domains.domains) == 1


def test_to_domain_data_feeds_both_joint_and_radial_fits():
    """The whole point of the design: one recovery pipeline, ONE DomainData
    list, both refine_cell_joint and refine_cell_radial consume it directly.

    NOTE 2026-09-14: this test failed once, in isolation, with a pandas
    "20 columns passed, passed data had 13 columns" error inside
    ingest.find_blobs_3d's DataFrame construction (SPOT_COLUMNS has 20
    fields; something produced a 13-field row) -- not reproduced across 4
    subsequent runs (3 isolated + the full file). Likely a rare edge case in
    blob-splitting on the deliberately sparse (9-reflection) position 2
    frames, not something this module's own code touches. Flagged, not
    chased -- if this reappears, it is an ingest.py bug, not a
    spatial_coherence.py one."""
    geom = _geom()
    U0 = Rotation.from_euler("zyx", [15.0, 25.0, -8.0], degrees=True).as_matrix()
    frames = {}
    frames[1], _ = synthetic_position_frames([U0], a=A, b=B_SPLIT, c=C, space_group_number=SG,
                                              geom=geom, hmax=8, kmax=8, lmax=14, seed=0)
    frames[2], _ = synthetic_position_frames([U0], a=A, b=B_SPLIT, c=C, space_group_number=SG,
                                              geom=geom, hmax=1, kmax=1, lmax=2, seed=1)

    def loader(p):
        return frames[p]

    result = recover_domains_across_raster(
        loader, geom, [1, 2], a=A, c=C, space_group_number=SG, sigma_rtn=SIGMA_RTN,
        dedup_threshold_deg=1.0, n_null_draws=8, rng=np.random.default_rng(0))

    doms = result.to_domain_data()
    assert len(doms) >= 2   # >=1 native at position 1, >=1 recovered at position 2
    labels = [d.label for d in doms]
    assert any("native" in l for l in labels)
    assert any("from_p1" in l for l in labels)

    # two_pi=True: this project's q convention throughout is q = 2*pi/d (see
    # reference_midas_q_convention_two_pi.md) -- the same convention
    # refine_to_convergence/match_mask use internally by default.
    fj = refine_cell_joint(doms, system="orthorhombic", cell0=(A, A, C, 90., 90., 90.), two_pi=True)
    fr = refine_cell_radial(doms, system="orthorhombic", cell0=(A, A, C, 90., 90., 90.), two_pi=True)
    assert abs(fj.cell[0] - A) < 0.15 and abs(fj.cell[1] - B_SPLIT) < 0.15
    assert abs(fr.cell[0] - A) < 0.15 and abs(fr.cell[1] - B_SPLIT) < 0.15


def test_null_is_suspect_flag_logic():
    """Direct, deterministic check of the FLAG's own logic. The phenomenon it
    detects is real and was reproduced exactly (notebook 06's synthetic demo,
    position 3: a 307-reflection native domain, 2 of 20 random-orientation
    null draws basin-hopped all the way to the full domain via
    refine_to_convergence's own iterate-refit loop, giving a null threshold
    of 307.0 -- a ~10% false-full-convergence rate for THAT position, not a
    one-off fluke: confirmed by replaying the exact sequential RNG state
    recover_domains_across_raster produces for its 4th position). That
    reproduction is RNG-state-coupled to this module's internal call order
    and too slow for a routine test (a full reduce_one_position + 20 seeded
    refits), so this test instead checks the flag's decision logic directly
    against both the found real values and a clean control."""
    class _FakeDomain:
        def __init__(self, n):
            self.n = n

    class _FakeDomains:
        def __init__(self, ns):
            self.domains = [_FakeDomain(n) for n in ns]

    class _FakeRes:
        def __init__(self, ns):
            self.domains = _FakeDomains(ns)

    from midas_defect.spatial_coherence import PositionCoherence

    # the exact real values reproduced from notebook 06's position 3
    pc_bad = PositionCoherence(point=3, res=_FakeRes([307]), q_all=np.zeros((1, 3)),
                               null_counts=[0]*17 + [13, 307, 307], null_threshold=307.0)
    assert pc_bad.null_is_suspect

    # a clean position (e.g. p0 in the same notebook run: null_threshold=0.4, native n=321)
    pc_ok = PositionCoherence(point=0, res=_FakeRes([321]), q_all=np.zeros((1, 3)),
                              null_counts=[0]*19 + [8], null_threshold=0.4)
    assert not pc_ok.null_is_suspect

    # no domains at all -- must not divide by zero or misfire
    pc_empty = PositionCoherence(point=9, res=_FakeRes([]), q_all=np.zeros((1, 3)),
                                 null_counts=[0]*20, null_threshold=0.0)
    assert not pc_empty.null_is_suspect


def test_index_asymmetry_is_suspect_flag_logic():
    """Direct check of `.index_asymmetry_is_suspect`'s decision logic, added
    2026-09-15 after a real horizontal-vs-vertical spatial-coherence claim on
    2604_25K (33 vs 20 recoveries) was REFUTED by /verify: the null gate
    passed trivially (threshold ~0 everywhere) while index_asymmetry sat at
    the documented 2604 systematic (ratio 0.0/inf) in 56-80% of domains,
    raster-wide -- a failure mode the null alone cannot see. Uses hand-built
    `ab_gate` dicts (the same shapes `_ab_gate`/`raster.py`'s own gate
    produce) rather than driving the full pipeline -- this is the flag's
    logic in isolation, matching `test_null_is_suspect_flag_logic` above."""
    def _rec(ab_gate):
        return RecoveredDomain(point=0, origin_point=1, n=10, U=np.eye(3),
                               cell=(4.0, 4.0, 10.0), hkl=np.zeros((10, 3)),
                               g=np.zeros((10, 3)), nearest_pass1_deg=5.0,
                               null_threshold=0.0, ab_gate=ab_gate)

    # the 2604 raster's own documented signature: extreme ratio, far side
    assert _rec({"stage": "index_asymmetry", "reason": {"ratio": 0.0}}).index_asymmetry_is_suspect
    assert _rec({"stage": "index_asymmetry", "reason": {"ratio": float("inf")}}).index_asymmetry_is_suspect
    # a healthy, roughly-balanced population is not suspect
    assert not _rec({"stage": "index_asymmetry", "reason": {"ratio": 1.1}}).index_asymmetry_is_suspect
    # earlier gate stages (rank < 2, or a partner pair resting on <= 1 spot) are suspect too --
    # they never even reach index_asymmetry, which is worse, not better
    assert _rec({"stage": "ab_separable", "reason": "rank < 2"}).index_asymmetry_is_suspect
    assert _rec({"stage": "partner_multiplicity", "reason": "..."}).index_asymmetry_is_suspect
    # an empty/never-computed gate must not misfire
    assert not _rec({}).index_asymmetry_is_suspect


def test_recovered_domain_carries_an_ab_gate_computed_from_its_own_hkl():
    """Integration check: `recover_domains_across_raster` must populate
    `ab_gate` on every real `RecoveredDomain` it returns, computed from that
    recovery's OWN claimed `hkl` -- not left as the dataclass default `{}`.
    Reuses the sparse-recovery scenario from
    `test_recovers_a_grain_too_sparse_for_free_search_to_find_on_its_own`
    (position 2's free search finds nothing; recovery must find it anyway,
    seeded from position 1) since that is already a known-good recovery to
    gate."""
    geom = _geom()
    U0 = Rotation.from_euler("zyx", [15.0, 25.0, -8.0], degrees=True).as_matrix()

    frames = {}
    frames[1], _ = synthetic_position_frames([U0], a=A, b=B_SPLIT, c=C, space_group_number=SG,
                                              geom=geom, hmax=8, kmax=8, lmax=14, seed=0)
    frames[2], truth2 = synthetic_position_frames([U0], a=A, b=B_SPLIT, c=C, space_group_number=SG,
                                                   geom=geom, hmax=1, kmax=1, lmax=2, seed=1)

    def loader(p):
        return frames[p]

    result = recover_domains_across_raster(
        loader, geom, [1, 2], a=A, c=C, space_group_number=SG, sigma_rtn=SIGMA_RTN,
        dedup_threshold_deg=1.0, n_null_draws=8, rng=np.random.default_rng(0))

    rec = result.per_position[2].recovered[0]
    assert rec.ab_gate, "ab_gate must be populated on a real recovery, not left at the empty default"
    assert rec.ab_gate["stage"] in ("ab_separable", "partner_multiplicity", "index_asymmetry")
    assert isinstance(rec.index_asymmetry_is_suspect, (bool, np.bool_))
    # summary() must surface a suspect recovery without raising, whichever way this one landed
    assert isinstance(result.summary(), str)


def test_cache_dir_round_trips_and_second_call_matches_first(tmp_path):
    """A cache_dir write-through: first call ingests fresh and writes
    position_{p}.h5 per point; second call reads those back and must give
    IDENTICAL native domain counts -- the whole point of caching is that it
    changes speed, not the answer."""
    geom = _geom()
    U0 = Rotation.from_euler("zyx", [15.0, 25.0, -8.0], degrees=True).as_matrix()
    frames = {}
    frames[1], _ = synthetic_position_frames([U0], a=A, b=B_SPLIT, c=C, space_group_number=SG,
                                              geom=geom, hmax=8, kmax=8, lmax=14, seed=0)

    def loader(p):
        return frames[p]

    cache_dir = tmp_path / "cache"
    r1 = recover_domains_across_raster(loader, geom, [1], a=A, c=C, space_group_number=SG,
                                       sigma_rtn=SIGMA_RTN, dedup_threshold_deg=1.0, n_null_draws=5,
                                       rng=np.random.default_rng(0), cache_dir=str(cache_dir))
    assert (cache_dir / "position_1.h5").exists()

    r2 = recover_domains_across_raster(loader, geom, [1], a=A, c=C, space_group_number=SG,
                                       sigma_rtn=SIGMA_RTN, dedup_threshold_deg=1.0, n_null_draws=5,
                                       rng=np.random.default_rng(0), cache_dir=str(cache_dir))
    n1 = [d.n for d in r1.per_position[1].res.domains.domains]
    n2 = [d.n for d in r2.per_position[1].res.domains.domains]
    assert n1 == n2 and len(n1) > 0


def test_dedup_threshold_controls_how_readily_domains_merge():
    """Unit-level check on the dedup step alone (no synthetic frames needed):
    two orientations 1.5deg apart merge at a 2deg threshold and stay separate
    at a 1deg one."""
    from midas_defect.spatial_coherence import _dedup

    U0 = Rotation.from_euler("zyx", [0.0, 0.0, 0.0], degrees=True).as_matrix()
    U1 = Rotation.from_euler("zyx", [1.5, 0.0, 0.0], degrees=True).as_matrix()
    candidates = [dict(point=1, dom_idx=0, U=U0, n=50, branch="row"),
                 dict(point=2, dom_idx=0, U=U1, n=40, branch="row")]

    assert len(_dedup(candidates, 1.0, SG)) == 2
    assert len(_dedup(candidates, 2.0, SG)) == 1


# --------------------------------------------------- Step 8: spatial coherence as evidence

def _clustered_grid(n=10, seed=0):
    """A smooth, spatially-clustered field: a low-frequency 2-D sinusoid plus small noise --
    Moran's I should clearly detect this as clustered."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:n, 0:n]
    field = np.sin(2 * np.pi * xx / n) + np.cos(2 * np.pi * yy / n)
    return field + rng.normal(0, 0.05, field.shape)


def _random_grid(n=10, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(0, 1, (n, n))


def test_morans_i_separates_a_clustered_field_from_random_noise():
    clustered = morans_i(_clustered_grid())
    random_ = morans_i(_random_grid())
    assert np.isfinite(clustered) and np.isfinite(random_)
    assert clustered > 0.5          # a smooth low-frequency field is strongly positively spatially correlated
    assert abs(random_) < 0.3       # IID noise should sit near zero, well below the clustered field


def test_morans_i_returns_nan_below_20_finite_cells():
    small = np.full((4, 4), np.nan)
    small[0, 0] = small[1, 1] = small[2, 2] = 1.0   # 3 finite cells, well under the 20 floor
    assert np.isnan(morans_i(small))


def test_permutation_null_flags_the_clustered_grid_not_the_random_one():
    rng = np.random.default_rng(1)
    clustered_result = permutation_null_morans_i(_clustered_grid(), n_perm=200, rng=rng)
    random_result = permutation_null_morans_i(_random_grid(), n_perm=200,
                                              rng=np.random.default_rng(1))
    assert clustered_result.clustered is True
    assert random_result.clustered is False
    assert clustered_result.observed_i > clustered_result.null_p95
    assert len(clustered_result.null_distribution) == 200


def test_permutation_null_reports_none_not_false_when_untestable():
    # Fewer than 20 finite cells -- morans_i itself is NaN here. "Could not be tested" must
    # come back distinguishable from "tested, found not clustered" (found directly on real S5
    # data: a run with only 10/900 finite cells printed `clustered: false` for both grids,
    # which reads exactly like a genuine negative result).
    sparse = np.full((10, 10), np.nan)
    sparse[0, 0] = sparse[1, 1] = sparse[2, 2] = 1.0
    result = permutation_null_morans_i(sparse, n_perm=50, rng=np.random.default_rng(0))
    assert np.isnan(result.observed_i)
    assert result.clustered is None


class _FakeDomain:
    """Minimal stand-in for `midas_defect.domains.Domain` -- only the fields
    `planted_identical_cell_delta` actually reads (`.hkl`, `.U`, `.claim`)."""
    def __init__(self, hkl, U, n_claim):
        self.hkl = hkl
        self.U = U
        self.claim = np.ones(n_claim, dtype=bool)


def _random_hkl(n, rng, hmax=6):
    hkl = rng.integers(-hmax, hmax + 1, size=(n, 3))
    return hkl[np.any(hkl != 0, axis=1)][:n] if len(hkl) >= n else _random_hkl(n * 2, rng, hmax)


def test_planted_identical_cell_delta_is_small_when_the_truth_really_is_a_equals_b():
    """The control's whole point: if a==b really is the truth, refitting the planted q's
    (with only fit noise, no real splitting) must recover a small delta, not a spurious one."""
    rng = np.random.default_rng(0)
    A_TRUE, C_TRUE = 4.0, 10.0
    domains_by_position = {}
    for p in range(12):
        U = Rotation.random(random_state=rng.integers(0, 2**31)).as_matrix()
        hkl = _random_hkl(15, rng)
        domains_by_position[p] = [_FakeDomain(hkl, U, len(hkl))]

    deltas = planted_identical_cell_delta(
        domains_by_position, a=A_TRUE, c=C_TRUE, noise=1e-4,
        rng=np.random.default_rng(1))

    assert set(deltas.keys()) <= set(domains_by_position.keys())
    assert len(deltas) >= 8   # most of the 12 should fit successfully
    assert all(d < 0.5 for d in deltas.values()), \
        f"a planted a==b control should not manufacture a large split: {deltas}"


def test_planted_identical_cell_delta_filters_sparse_domains():
    rng = np.random.default_rng(2)
    hkl_sparse = _random_hkl(3, rng)
    domains_by_position = {0: [_FakeDomain(hkl_sparse, np.eye(3), len(hkl_sparse))]}
    deltas = planted_identical_cell_delta(
        domains_by_position, a=4.0, c=10.0, noise=1e-4,
        rng=np.random.default_rng(3), min_reflections=6)
    assert deltas == {}   # too few reflections to fit -- must be skipped, not crash or fabricate


def test_planted_identical_cell_delta_accepts_dict_shaped_domains_too():
    """`assemble_raster_results`/`PositionResult.to_dict()` hand back domains as plain dicts
    (hkl/U/n_claim keys), not live Domain objects with attribute access -- 06's own Step 2/5
    cells already work with exactly this shape, so this function must too."""
    rng = np.random.default_rng(4)
    hkl = _random_hkl(15, rng)
    U = Rotation.random(random_state=rng.integers(0, 2**31)).as_matrix()
    domains_by_position = {0: [dict(hkl=hkl.tolist(), U=U.tolist(), n_claim=len(hkl))]}
    deltas = planted_identical_cell_delta(
        domains_by_position, a=4.0, c=10.0, noise=1e-4, rng=np.random.default_rng(5))
    assert 0 in deltas
    assert deltas[0] < 0.5


# --------------------------------------------------- n_workers: parallel pass 1 + pass 2

def test_parallel_matches_sequential_on_real_data(tmp_path):
    """The load-bearing guarantee for n_workers>1: identical native+recovered counts to the
    sequential path on real data, not merely "runs without error". Reuses the sparse-recovery
    scenario (position 2's free search finds nothing on its own; recovery must find it anyway,
    seeded from position 1) since that already exercises both pass 1 and pass 2 meaningfully.
    Frames are written to disk (not an in-memory dict) because `_disk_loader` must be a plain,
    module-level, picklable callable for `spawn` to use it -- a closure over a local dict, the
    pattern every OTHER test in this file uses, does not survive spawn at all."""
    import functools
    geom = _geom()
    U0 = Rotation.from_euler("zyx", [15.0, 25.0, -8.0], degrees=True).as_matrix()
    frames1, _ = synthetic_position_frames([U0], a=A, b=B_SPLIT, c=C, space_group_number=SG,
                                           geom=geom, hmax=8, kmax=8, lmax=14, seed=0)
    frames2, truth2 = synthetic_position_frames([U0], a=A, b=B_SPLIT, c=C, space_group_number=SG,
                                                geom=geom, hmax=1, kmax=1, lmax=2, seed=1)
    np.save(tmp_path / "frame_1.npy", frames1)
    np.save(tmp_path / "frame_2.npy", frames2)
    loader = functools.partial(_disk_loader, base_dir=tmp_path)

    kwargs = dict(a=A, c=C, space_group_number=SG, sigma_rtn=SIGMA_RTN,
                 dedup_threshold_deg=1.0, n_null_draws=8, rng=np.random.default_rng(0))
    seq = recover_domains_across_raster(loader, geom, [1, 2], **kwargs)
    par = recover_domains_across_raster(loader, geom, [1, 2], n_workers=2,
                                        rng=np.random.default_rng(0), **{k: v for k, v in kwargs.items() if k != "rng"})

    for p in (1, 2):
        assert len(seq.per_position[p].res.domains.domains) == len(par.per_position[p].res.domains.domains)
        assert len(seq.per_position[p].recovered) == len(par.per_position[p].recovered)
    assert len(seq.per_position[2].recovered) == 1   # the actual recovery, same as the
                                                      # sequential-only test above expects
    assert par.per_position[2].recovered[0].n == len(truth2.reflections) and \
          seq.per_position[2].recovered[0].n == len(truth2.reflections)


# --------------------------------------------------- generic quantity + report

def test_best_domain_per_position_excludes_suspect_recoveries_by_default():
    """Direct logic check, matching test_index_asymmetry_is_suspect_flag_logic's own style --
    the whole point of this function existing is to make the S5 finding (90.7% of raw
    recoveries suspect, inflating Moran's I from 0.147 to 0.239) the DEFAULT-SAFE behavior
    rather than something every caller must remember to filter by hand."""
    class _FakeNative:
        def __init__(self, hkl, U, n):
            self.hkl = hkl; self.U = U; self.n = n; self.claim = np.ones(len(hkl), bool)

    class _FakeDomains:
        def __init__(self, domains):
            self.domains = domains

    class _FakeRes:
        def __init__(self, domains):
            self.domains = _FakeDomains(domains)

    native = _FakeNative(np.zeros((3, 3)), np.eye(3), 3)
    clean_rec = RecoveredDomain(point=0, origin_point=1, n=20, U=np.eye(3), cell=(4., 4., 10.),
                                hkl=np.zeros((20, 3)), g=np.zeros((20, 3)), nearest_pass1_deg=5.,
                                null_threshold=0., ab_gate={"stage": "index_asymmetry",
                                                            "reason": {"ratio": 1.1}})
    suspect_rec = RecoveredDomain(point=0, origin_point=2, n=50, U=np.eye(3), cell=(4., 4., 10.),
                                  hkl=np.zeros((50, 3)), g=np.zeros((50, 3)), nearest_pass1_deg=5.,
                                  null_threshold=0., ab_gate={"stage": "index_asymmetry",
                                                              "reason": {"ratio": 0.0}})
    q_all = np.zeros((3, 3))   # must match native.claim's length (3 reflections)
    pc = PositionCoherence(point=0, res=_FakeRes([native]), q_all=q_all, null_counts=[],
                           null_threshold=0.0, recovered=[clean_rec, suspect_rec])
    result = RasterCoherenceResult(dedup_threshold_deg=1.0, candidates=[], per_position={0: pc})

    # default: the suspect recovery (n=50, would otherwise win as "largest") must be excluded,
    # so the winner is the clean recovery (n=20), not native (n=3) either
    best = best_domain_per_position(result)
    assert best[0]["n"] == 20 and best[0]["source"] == "recovered_from_p1"

    # explicitly allowing suspect recoveries picks the largest overall, unfiltered
    best_unfiltered = best_domain_per_position(result, exclude_index_asymmetry_suspect=False)
    assert best_unfiltered[0]["n"] == 50


def test_grid_quantity_places_values_and_leaves_none_as_nan():
    domains_by_position = {0: dict(n=7), 3: dict(n=2), 31: dict(n=9)}   # n_fast=30: p31 -> row1,col1
    grid = grid_quantity(domains_by_position, lambda d: d["n"] if d["n"] > 3 else None,
                         n_fast=30, n_pos=60)
    assert grid.shape == (2, 30)
    assert grid[0, 0] == 7.0
    assert np.isnan(grid[0, 3])     # n=2 <= 3 -> quantity_fn returned None -> NaN
    assert grid[1, 1] == 9.0


def test_spatial_coherence_report_reports_control_spread_not_one_point():
    """The whole point of `control_sweep` being a list: a single control draw crossing (or
    not crossing) its own null is not decisive at n=1. Builds an observed grid with real
    spatial structure and a control_fn that returns pure noise each draw (by construction,
    should rarely cluster) to check the report's OWN bookkeeping (fraction crossed, "exceeds
    all controls"), not re-derive the S5 science finding."""
    rng_truth = np.random.default_rng(0)
    n = 10
    yy, xx = np.mgrid[0:n, 0:n]
    clustered_field = np.sin(2 * np.pi * xx / n) + np.cos(2 * np.pi * yy / n)
    domains_by_position = {i * n + j: dict(value=clustered_field[i, j])
                          for i in range(n) for j in range(n)}

    def quantity_fn(d):
        return d["value"]

    def control_fn(dbp, rng):
        return {p: rng.normal(0, 0.1) for p in dbp}

    report = spatial_coherence_report(domains_by_position, quantity_fn, n_fast=n, n_pos=n*n,
                                      control_fn=control_fn, n_control_draws=8,
                                      rng=np.random.default_rng(1))
    assert isinstance(report, SpatialCoherenceReport)
    assert report.observed.clustered   # the planted sinusoid must be detected
    assert len(report.control_sweep) == 8
    frac = report.control_exceeded_own_null_fraction
    assert frac is not None and 0.0 <= frac <= 1.0
    # pure per-position noise, no spatial structure at all -- the real observed signal should
    # exceed every one of 8 noise draws
    assert report.observed_exceeds_all_controls() is True
    assert "control:" in report.summary()


def test_spatial_coherence_report_without_control_fn_is_observed_only():
    domains_by_position = {i: dict(v=float(i % 3)) for i in range(25)}
    report = spatial_coherence_report(domains_by_position, lambda d: d["v"], n_fast=5, n_pos=25)
    assert report.control_sweep == []
    assert report.control_exceeded_own_null_fraction is None
    assert report.observed_exceeds_all_controls() is None
    assert "observed:" in report.summary()


# --------------------------------------------------- effective sample size / exact p-value

def test_exact_rank_p_value_matches_hand_computed_formula():
    # observed exceeds all 3 draws -> (0 exceed-or-equal + 1) / (3 + 1) = 0.25
    assert exact_rank_p_value(10.0, [1.0, 2.0, 3.0]) == pytest.approx(0.25)
    # observed exceeded by all 3 -> (3 + 1) / (3 + 1) = 1.0 (never significant)
    assert exact_rank_p_value(0.0, [1.0, 2.0, 3.0]) == pytest.approx(1.0)
    # observed ties one draw -> both the tie AND the genuinely larger draw count as ">="
    assert exact_rank_p_value(2.0, [1.0, 2.0, 3.0]) == pytest.approx(0.75)
    # the real S5 case: exceeds all 11 -> 1/12
    assert exact_rank_p_value(0.19, [0.01] * 11) == pytest.approx(1 / 12)
    assert np.isnan(exact_rank_p_value(1.0, []))


def test_cluster_permutation_null_only_explores_cluster_relabelings():
    """The whole point of this function: with only K underlying clusters, there are only K!
    possible ways to relabel which fitted value lands on which footprint -- far fewer distinct
    outcomes than the plain cell-level null's (which reassigns each of N cells independently
    and can realize many more distinct spatial arrangements). Checked directly: with 4
    clusters, the cluster-aware null's OWN observed Moran's I must be reproducible by AT LEAST
    ONE of the 4!=24 explicit relabelings (i.e. it is genuinely permuting whole footprints, not
    doing a finer-grained cell shuffle that could land outside that set) -- and, contrasted
    with the plain per-cell null on the SAME grid, the two null DISTRIBUTIONS must differ,
    since they are answering different questions."""
    rng = np.random.default_rng(0)
    n = 12
    grid = np.full((n, n), np.nan)
    cluster_ids = np.full((n, n), np.nan, dtype=object)
    # 4 blocks, each a contiguous quadrant, each with its own RANDOM (not spatially patterned)
    # value -- i.e. the ONLY reason any cell resembles its neighbor is shared block membership,
    # never real cross-block spatial structure.
    block_values = rng.normal(size=4)
    half = n // 2
    quadrants = [(slice(0, half), slice(0, half)), (slice(0, half), slice(half, n)),
                (slice(half, n), slice(0, half)), (slice(half, n), slice(half, n))]
    for k, (rs, cs) in enumerate(quadrants):
        grid[rs, cs] = block_values[k]
        cluster_ids[rs, cs] = k

    plain = permutation_null_morans_i(grid, n_perm=300, rng=np.random.default_rng(1))
    clustered_run = cluster_permutation_null_morans_i(grid, cluster_ids, n_perm=300,
                                                       rng=np.random.default_rng(1))

    # every draw in the cluster-aware null must be EXACTLY reproducible by relabeling the 4
    # quadrants with some permutation of block_values -- i.e. it really is a whole-footprint
    # relabeling, not a finer cell-level shuffle that happens to look similar
    import itertools
    possible_arrangements = set()
    for perm in itertools.permutations(block_values):
        g = np.full((n, n), np.nan)
        for k, (rs, cs) in enumerate(quadrants):
            g[rs, cs] = perm[k]
        possible_arrangements.add(round(morans_i(g), 9))
    null_values = {round(v, 9) for v in clustered_run.null_distribution if np.isfinite(v)}
    assert null_values <= possible_arrangements, \
        "cluster-aware null produced a Moran's I no whole-footprint relabeling could produce"

    # the two nulls answer different questions -- their distributions should differ (the plain
    # null has far more distinct achievable values, since it isn't constrained to K! outcomes)
    n_distinct_plain = len({round(v, 6) for v in plain.null_distribution if np.isfinite(v)})
    n_distinct_clustered = len(null_values)
    assert n_distinct_clustered <= 24              # <= 4! -- the whole point of this function
    assert n_distinct_plain > n_distinct_clustered  # the plain null is far less constrained


def test_best_domain_per_position_reports_origin_for_native_and_recovered():
    class _FakeNative:
        def __init__(self, hkl, U, n):
            self.hkl = hkl; self.U = U; self.n = n; self.claim = np.ones(len(hkl), bool)

    class _FakeDomains:
        def __init__(self, domains):
            self.domains = domains

    class _FakeRes:
        def __init__(self, domains):
            self.domains = _FakeDomains(domains)

    # a real, non-degenerate, BALANCED hkl set -- needs rank>=2 (ab_separable), at least one
    # a/b-sensitive pair backed by >=2 spots on EACH side (partner_multiplicity), and a
    # near-1 h/k ratio (index_asymmetry) or it gets excluded as suspect by default, defeating
    # the point of this test (an all-zero or thin array fails exactly this way -- caught by
    # running it, not assumed).
    native_hkl = np.array([[1, 0, 0], [1, 0, 1], [0, 1, 0], [0, 1, 1],
                           [2, 1, 0], [2, 1, 1], [1, 2, 0], [1, 2, 1]], float)
    native = _FakeNative(native_hkl, np.eye(3), 8)
    q_all = np.zeros((8, 3))
    rec = RecoveredDomain(point=7, origin_point=3, n=10, U=np.eye(3), cell=(4., 4., 10.),
                          hkl=np.zeros((10, 3)), g=np.zeros((10, 3)), nearest_pass1_deg=5.,
                          null_threshold=0., ab_gate={"stage": "index_asymmetry",
                                                      "reason": {"ratio": 1.0}})
    pc7 = PositionCoherence(point=7, res=_FakeRes([]), q_all=q_all, null_counts=[],
                            null_threshold=0.0, recovered=[rec])
    pc9 = PositionCoherence(point=9, res=_FakeRes([native]), q_all=q_all, null_counts=[],
                            null_threshold=0.0, recovered=[])
    result = RasterCoherenceResult(dedup_threshold_deg=1.0, candidates=[],
                                   per_position={7: pc7, 9: pc9})
    best = best_domain_per_position(result)
    assert best[7]["origin"] == 3          # recovered -- origin is where it was FOUND, not p
    assert best[9]["origin"] == 9          # native -- a domain is its own origin


def test_spatial_coherence_report_flags_small_effective_n_behind_many_positions():
    """4 raster cells, but only 2 underlying origins (2 cells each) -- n_positions=4,
    n_effective_clusters must report 2, not 4."""
    domains_by_position = {
        0: dict(v=1.0, origin=100), 1: dict(v=1.0, origin=100),
        2: dict(v=5.0, origin=200), 3: dict(v=5.0, origin=200),
    }
    report = spatial_coherence_report(domains_by_position, lambda d: d["v"], n_fast=2, n_pos=4,
                                      n_perm=20)
    assert report.n_positions == 4
    assert report.n_effective_clusters == 2
    assert report.cluster_observed is not None
    assert "INDEPENDENT domain" in report.summary()


# --------------------------------------------------- grid-only entry points (no Domain/
# RasterCoherenceResult required -- for a caller whose spatial maps come from their own,
# different pipeline)

def test_values_to_grid_matches_grid_quantity_identity():
    values = {0: 7.0, 3: None, 31: 9.0}    # n_fast=30: p31 -> row1, col1; p3 explicitly None
    grid = values_to_grid(values, n_fast=30, n_pos=60)
    assert grid.shape == (2, 30)
    assert grid[0, 0] == 7.0
    assert np.isnan(grid[0, 3])             # explicit None -> NaN, same convention as grid_quantity
    assert np.isnan(grid[1, 0])             # point 30 never mentioned at all -> also NaN
    assert grid[1, 1] == 9.0


def test_grid_coherence_report_matches_the_domain_based_report_on_the_same_data():
    """The whole point of the refactor: a caller who already has plain grids (their own a/b/c
    maps, computed a different way) must get IDENTICAL numbers out of `grid_coherence_report`
    as `spatial_coherence_report` gives from the equivalent domains_by_position/quantity_fn/
    control_fn -- they are the same statistics, just reached from a different starting point."""
    rng_truth = np.random.default_rng(0)
    n = 10
    yy, xx = np.mgrid[0:n, 0:n]
    clustered_field = np.sin(2 * np.pi * xx / n) + np.cos(2 * np.pi * yy / n)
    domains_by_position = {i * n + j: dict(value=clustered_field[i, j])
                          for i in range(n) for j in range(n)}

    def quantity_fn(d):
        return d["value"]

    def control_fn(dbp, rng):
        return {p: rng.normal(0, 0.1) for p in dbp}

    domain_report = spatial_coherence_report(domains_by_position, quantity_fn, n_fast=n, n_pos=n*n,
                                             control_fn=control_fn, n_control_draws=5,
                                             rng=np.random.default_rng(1))

    # A student with their own a/b/c grid does not have `domains_by_position` at all -- they
    # have the grid (here, built by hand from the SAME values to prove the two paths agree) and
    # whatever control grid(s) their own pipeline produced.
    my_own_grid = values_to_grid({p: d["value"] for p, d in domains_by_position.items()},
                                 n_fast=n, n_pos=n*n)
    # match control_fn's own draw sequence exactly (spatial_coherence_report advances ONE shared
    # rng across draws; redo that here so the two reports see the identical control values)
    shared_rng = np.random.default_rng(1)
    my_own_control_grids = [values_to_grid(control_fn(domains_by_position, shared_rng),
                                           n_fast=n, n_pos=n*n)
                            for _ in range(5)]

    grid_report = grid_coherence_report(my_own_grid, control_grids=my_own_control_grids)

    assert grid_report.observed.observed_i == pytest.approx(domain_report.observed.observed_i)
    assert grid_report.observed.clustered == domain_report.observed.clustered
    assert len(grid_report.control_sweep) == len(domain_report.control_sweep)
    for a, b in zip(grid_report.control_sweep, domain_report.control_sweep):
        assert a.observed_i == pytest.approx(b.observed_i)


def test_grid_coherence_report_with_cluster_id_grid_flags_small_effective_n():
    """Same effective-sample-size check as `spatial_coherence_report`'s own version, but reached
    the way a caller with no `domains_by_position` would: plain value grid + plain id grid."""
    grid = np.array([[1.0, 1.0], [5.0, 5.0]])
    cluster_ids = np.array([[100, 100], [200, 200]], dtype=object)
    report = grid_coherence_report(grid, cluster_id_grid=cluster_ids, n_perm=20)
    assert report.n_positions == 4
    assert report.n_effective_clusters == 2
    assert report.cluster_observed is not None
    assert "INDEPENDENT domain" in report.summary()


def test_grid_coherence_report_with_only_observed_grid_is_observed_only():
    domains_by_position = {i: dict(v=float(i % 3)) for i in range(25)}
    grid = values_to_grid({p: d["v"] for p, d in domains_by_position.items()}, n_fast=5, n_pos=25)
    report = grid_coherence_report(grid)
    assert report.control_sweep == []
    assert report.n_effective_clusters is None
    assert report.cluster_observed is None
    assert "observed:" in report.summary()
