"""The (h,k) rod observable must be normalised by the SAME rod's Bragg nodes (2026-09-10 fold-in)."""
import numpy as np

from midas_defect.rod_profile import RodProfile, centred_L_nodes, diffuse_to_bragg

L = np.arange(-16.0, 16.0 + 1e-9, 0.02)


def _rod(nodes, *, floor=200.0, node_height=1e4, scale=1.0, seed=0, nan_frac=0.0):
    rng = np.random.default_rng(seed)
    inten = np.full(len(L), floor)
    for n in nodes:
        inten += node_height * np.exp(-0.5 * ((L - n) / 0.05) ** 2)
    inten = scale * inten + rng.normal(0, 10.0, len(L))
    if nan_frac:
        inten[rng.random(len(L)) < nan_frac] = np.nan
    return RodProfile(L=L, intensity=inten, n_valid_px=np.ones(len(L)), omega_deg=np.zeros(len(L)),
                      dropped={})


def _control(seed=1):
    return RodProfile(L=L, intensity=np.random.default_rng(seed).normal(0, 10.0, len(L)),
                      n_valid_px=np.ones(len(L)), omega_deg=np.zeros(len(L)), dropped={})


def test_centring_rules_put_nodes_at_the_right_L():
    assert set(centred_L_nodes(1, 1, -4, 4, "I")) == {-4, -2, 0, 2, 4}
    assert set(centred_L_nodes(1, 0, -4, 4, "I")) == {-3, -1, 1, 3}      # opposite parity to (1,1)
    assert centred_L_nodes(1, 0, -4, 4, "F").size == 0                   # mixed h,k parity: no nodes
    assert set(centred_L_nodes(1, 1, -4, 4, "F")) == {-3, -1, 1, 3}
    assert len(centred_L_nodes(2, 1, -4, 4, "P")) == 9


def test_ratio_and_significance_on_a_clean_rod():
    nodes = centred_L_nodes(1, 1, -16, 16, "I")
    r = diffuse_to_bragg(_rod(nodes), _control(), nodes)
    assert r["usable"], r["reason"]
    assert abs(r["ratio"] - 0.02) < 0.004
    assert r["sigma"] > 15


def test_the_ratio_removes_the_structure_factor_and_the_raw_level_does_not():
    nodes = centred_L_nodes(1, 1, -16, 16, "I")
    weak, strong = (diffuse_to_bragg(_rod(nodes, scale=s), _control(), nodes) for s in (1.0, 7.0))
    assert strong["diffuse"] / weak["diffuse"] > 6.0          # raw level follows |F|^2
    assert abs(strong["ratio"] / weak["ratio"] - 1.0) < 0.05  # the ratio does not


def test_a_row_with_no_node_on_the_ewald_sphere_is_unusable_not_zero():
    nodes = centred_L_nodes(1, 1, -16, 16, "I")
    r = diffuse_to_bragg(_rod(nodes, node_height=0.0), _control(), nodes)
    assert not r["usable"] and "normaliser" in r["reason"]
    assert "ratio" not in r


def test_too_few_observable_points_is_unusable():
    nodes = centred_L_nodes(1, 1, -16, 16, "I")
    r = diffuse_to_bragg(_rod(nodes, nan_frac=0.97), _control(), nodes)
    assert not r["usable"]
