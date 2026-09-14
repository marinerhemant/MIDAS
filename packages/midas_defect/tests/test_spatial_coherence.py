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
from midas_defect.spatial_coherence import recover_domains_across_raster
from midas_hkls.cell_constrained import refine_cell_joint, refine_cell_radial

A, B_SPLIT, C = 4.0, 3.9, 10.0        # a genuinely fictional cell, not any real material
SG = 123                              # P4/mmm
SIGMA_RTN = (0.02, 0.02, 0.02)


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
