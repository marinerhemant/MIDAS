"""Tests for defect_tests: forbidden-reflection, rod enrichment, fault-alpha."""
import numpy as np

from midas_defect.lattice import fcc_cu_crystal
from midas_defect.bragg_diffuse import predicted_reflection_points
from midas_defect.defect_tests import (
    forbidden_reflection_test, rod_family_enrichment, fault_probability_alpha,
    fault_rod_alignment, CUBIC_FAMILIES, _forbidden_hkls, _g_crystal,
)

CR = fcc_cu_crystal()
A = 3.6356


def _allowed_cloud(U):
    return predicted_reflection_points(U, CR, q_max_inv_A=8.0).numpy()


def test_forbidden_no_excess_when_clean():
    """Intensity only on allowed reflections ⇒ no forbidden excess."""
    U = np.eye(3)[None]
    P = _allowed_cloud(U)
    I = np.full(len(P), 100.0)
    ft = forbidden_reflection_test(P, I, U, CR)
    assert abs(ft.excess_median) < 1e-6
    assert ft.n_grains_excess == 0


def test_forbidden_detects_planted_intensity():
    """Plant intensity at forbidden (mixed-parity) positions ⇒ excess > 0."""
    U = np.eye(3)[None]
    P = _allowed_cloud(U)
    fhkl = _forbidden_hkls(CR, q_max_inv_A=8.0, q_min_inv_A=1.5)
    Pf = (U[0] @ _g_crystal(fhkl, CR).T).T
    q = np.vstack([P, Pf])
    I = np.concatenate([np.full(len(P), 100.0), np.full(len(Pf), 100.0)])
    ft = forbidden_reflection_test(q, I, U, CR)
    assert ft.excess_median > 0.0


def test_rod_enrichment_picks_planted_family():
    """Rods placed only along <111> ⇒ <111> enriched, others not."""
    U = np.eye(3)[None]
    P = _allowed_cloud(U)
    G111 = (2 * np.pi / A) * CUBIC_FAMILIES["<111>"]
    rods = []
    for v in G111:
        d = v / np.linalg.norm(v)
        for t in (0.15, 0.25, 0.35):
            rods.append(v + t * d)
    rods = np.array(rods)
    q = np.vstack([P, rods])
    I = np.concatenate([np.full(len(P), 100.0), np.full(len(rods), 30.0)])
    # disable the bright pre-filter: here the planted rods are dimmer than the
    # Bragg cloud by construction (real rod cores are bright, so the default
    # filter is correct for real data — see test_real_data_regression).
    re = rod_family_enrichment(q, I, U, CR, bright_percentile=0.0)
    assert re.enrichment["<111>"] > 2.0
    assert re.enrichment["<110>"] < 1.0
    assert re.enrichment["<100>"] < 1.0


def test_fault_alpha_increases_with_rod_intensity():
    U = np.eye(3)[None]
    P = _allowed_cloud(U)
    G111 = _g_crystal(np.array([[1, 1, 1], [1, 1, -1], [1, -1, 1], [-1, 1, 1],
                                [-1, -1, -1], [-1, -1, 1], [-1, 1, -1], [1, -1, -1]]), CR)
    def alpha_with(rod_I):
        rods = []
        for v in G111:
            d = v / np.linalg.norm(v)
            for t in np.arange(0.10, 0.45, 0.03):
                rods += [v + t * d, v - t * d]
        rods = np.array(rods)
        q = np.vstack([P, rods])
        I = np.concatenate([np.full(len(P), 100.0), np.full(len(rods), rod_I)])
        return fault_probability_alpha(q, I, U, CR).alpha_median
    assert alpha_with(50.0) > alpha_with(1.0)


def test_fault_rod_alignment_detects_planted_rods():
    """Diffuse intensity placed along <111> through Bragg points ⇒ along/perp > 1."""
    U = np.eye(3)[None]
    P = _allowed_cloud(U)
    # use the {111}-type Bragg points (|q| ~ 3) and lay diffuse voxels along
    # their <111> axis at off-lattice offsets (>0.10), q>1.5
    F111 = CUBIC_FAMILIES["<111>"]
    G111 = (2 * np.pi / A) * F111
    rod = []
    for v in G111:
        d = v / np.linalg.norm(v)
        for t in (0.15, 0.25, 0.35):
            rod += [v + t * d, v - t * d]
    rod = np.array(rod)
    q = np.vstack([P, rod])
    I = np.concatenate([np.full(len(P), 100.0), np.full(len(rod), 50.0)])
    fr = fault_rod_alignment(q, I, U, CR)
    assert fr.along_over_perp_median > 1.5

    # isotropic diffuse shell (no preferred axis) ⇒ along/perp ~ 1
    rng = np.random.default_rng(0)
    for v in G111:
        sph = rng.normal(size=(40, 3)); sph /= np.linalg.norm(sph, axis=1, keepdims=True)
        rod = np.vstack([rod, v + 0.25 * sph])
    q2 = np.vstack([P, rod])
    I2 = np.concatenate([np.full(len(P), 100.0), np.full(len(rod), 50.0)])
    fr2 = fault_rod_alignment(q2, I2, U, CR)
    assert fr2.along_over_perp_median < fr.along_over_perp_median
