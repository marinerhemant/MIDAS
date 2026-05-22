"""midas_distortion parity + cross-backend gates.

The distortion model is correct iff its layout/mapping is correct; the
arithmetic is trivial. So we pin:

  HARD GATE — every additive term is BIT-IDENTICAL across the legacy v1 inline
    polynomial, the v1-layout kernel, and the v2-shim kernel (Δ=0). A mapping
    bug (wrong index/fold/phase/radial-power) surfaces here, not behind rounding.
  ASSEMBLED — the summed factor agrees to ≤ 8 ULP (pure IEEE-754 reassociation).
  CROSS-BACKEND — numpy and torch evaluate the same factor to ≤ 8 ULP.
"""
import numpy as np
import pytest

from midas_distortion import (
    distortion_factor, v1_term_layout, v2_term_layout,
    v1_to_v2_coeffs, v2_to_v1_coeffs, v2_coeffs_from_named,
    P_COEF_NAMES, V1_TO_V2_DISTORTION, V2_TO_V1_DISTORTION,
)

_DEG2RAD = 0.017453292519943295
_EPS = float(np.finfo(np.float64).eps)
_ULP_TOL = 8 * _EPS


def _grid_np():
    R = np.linspace(0.0, 1.3, 41)
    eta = np.linspace(-180.0, 180.0, 73)
    return np.meshgrid(R, eta, indexing="ij")


def _inline_terms(Rg, Eg, p):
    """The 9 additive terms of the legacy v1 polynomial, keyed by p-index."""
    eT = (90.0 - Eg) * _DEG2RAD
    return {
        0:  p[0] * Rg**2 * np.cos(2 * eT + _DEG2RAD * p[6]),
        1:  p[1] * Rg**4 * np.cos(4 * eT + _DEG2RAD * p[3]),
        2:  p[2] * Rg**2,
        4:  p[4] * Rg**6,
        5:  p[5] * Rg**4,
        7:  p[7] * Rg**4 * np.cos(eT + _DEG2RAD * p[8]),
        9:  p[9] * Rg**3 * np.cos(3 * eT + _DEG2RAD * p[10]),
        11: p[11] * Rg**5 * np.cos(5 * eT + _DEG2RAD * p[12]),
        13: p[13] * Rg**6 * np.cos(6 * eT + _DEG2RAD * p[14]),
    }


def _random_p(rng):
    p = rng.uniform(-5e-3, 5e-3, size=15)
    for ph in (3, 6, 8, 10, 12, 14):
        p[ph] = rng.uniform(-180.0, 180.0)
    return p


def test_v1_v2_maps_are_inverses():
    assert {v: k for k, v in V1_TO_V2_DISTORTION.items()} == V2_TO_V1_DISTORTION
    assert set(V1_TO_V2_DISTORTION.values()) == set(P_COEF_NAMES)
    assert sorted(V1_TO_V2_DISTORTION) == list(range(15))


def test_coeff_reindex_roundtrip_and_mapping():
    """v1→v2→v1 is identity, and each v2 slot pulls the right v1 index."""
    p = np.arange(15.0)
    v2 = v1_to_v2_coeffs(p)
    assert np.array_equal(v2_to_v1_coeffs(v2), p)
    for v1_idx, name in V1_TO_V2_DISTORTION.items():
        assert v2[P_COEF_NAMES.index(name)] == p[v1_idx]


def test_v2_coeffs_from_named():
    """v2 names populate the canonical vector; legacy p0..p14 is a fallback
    equal to v1_to_v2_coeffs; v2 names win on collision."""
    # pure v2 names
    d = {"iso_R2": -1.1e-3, "a2": 4.6e-4, "phi2": 33.0, "a4": -5.2e-4, "phi4": -6.87}
    v = v2_coeffs_from_named(d)
    for nm, val in d.items():
        assert v[P_COEF_NAMES.index(nm)] == val
    # legacy p path == v1_to_v2_coeffs
    pv = np.arange(15.0)
    assert np.array_equal(v2_coeffs_from_named({f"p{i}": pv[i] for i in range(15)}),
                          v1_to_v2_coeffs(pv))
    # v2 name wins over the same-slot legacy p
    mixed = {"p2": 99.0, "iso_R2": -1.1e-3}   # p2 and iso_R2 are the same slot
    assert v2_coeffs_from_named(mixed)[P_COEF_NAMES.index("iso_R2")] == -1.1e-3
    # None values skipped
    assert np.array_equal(v2_coeffs_from_named({"a2": None}), np.zeros(15))


def test_coeff_reindex_is_differentiable():
    """The v1→v2 gather must preserve autograd (transforms needs grads)."""
    torch = pytest.importorskip("torch")
    t = torch.arange(15.0, dtype=torch.float64, requires_grad=True)
    v1_to_v2_coeffs(t).sum().backward()
    assert t.grad is not None and torch.all(t.grad == 1.0)


def test_terms_bit_identical():
    """HARD GATE: per-term Δ=0 across inline / v1-kernel / v2-shim."""
    rng = np.random.default_rng(0)
    Rg, Eg = _grid_np()
    v1L, v2L = v1_term_layout(), v2_term_layout()
    name2v2 = {nm: i for i, nm in enumerate(P_COEF_NAMES)}
    v2_key = {k: name2v2[v] for k, v in V1_TO_V2_DISTORTION.items()}  # v1 amp idx → v2 amp idx
    eT = (90.0 - Eg) * _DEG2RAD
    for _ in range(300):
        p = _random_p(rng)
        p2 = v1_to_v2_coeffs(p)
        inl = _inline_terms(Rg, Eg, p)
        # kernel per-term, v1 layout
        kv1 = {}
        for t in v1L:
            rad = Rg ** t.radial_power
            kv1[t.coef_idx] = (p[t.coef_idx] * rad if t.fold == 0
                               else p[t.coef_idx] * rad * np.cos(t.fold * eT + p[t.phase_idx] * _DEG2RAD))
        # kernel per-term, v2 layout on shimmed coeffs
        kv2 = {}
        for t in v2L:
            rad = Rg ** t.radial_power
            kv2[t.coef_idx] = (p2[t.coef_idx] * rad if t.fold == 0
                               else p2[t.coef_idx] * rad * np.cos(t.fold * eT + p2[t.phase_idx] * _DEG2RAD))
        for k, term in inl.items():
            assert np.array_equal(term, kv1[k]), f"v1 kernel term p{k} != inline"
            assert np.array_equal(term, kv2[v2_key[k]]), f"v2 shim term p{k} != inline"


def test_assembled_within_ulp():
    rng = np.random.default_rng(1)
    Rg, Eg = _grid_np()
    for _ in range(300):
        p = _random_p(rng)
        p2 = v1_to_v2_coeffs(p)
        D_inline = sum(_inline_terms(Rg, Eg, p).values()) + 1.0
        D_v1 = distortion_factor(Rg, Eg, p, terms=v1_term_layout())
        D_v2 = distortion_factor(Rg, Eg, p2, terms=v2_term_layout())
        assert np.abs(D_inline - D_v1).max() <= _ULP_TOL
        assert np.abs(D_v1 - D_v2).max() <= _ULP_TOL


def test_zero_is_unity():
    Rg, Eg = _grid_np()
    z = np.zeros(15)
    for layout in (v1_term_layout(), v2_term_layout()):
        assert np.array_equal(distortion_factor(Rg, Eg, z, terms=layout), np.ones_like(Rg))


def test_cross_backend_numpy_torch():
    """numpy and torch evaluate the same factor to ≤ 8 ULP."""
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(2)
    Rg, Eg = _grid_np()
    Rt, Et = torch.from_numpy(Rg), torch.from_numpy(Eg)
    for _ in range(50):
        p = _random_p(rng)
        D_np = distortion_factor(Rg, Eg, p, terms=v1_term_layout())
        D_t = distortion_factor(Rt, Et, torch.from_numpy(p), terms=v1_term_layout())
        assert np.abs(D_np - D_t.numpy()).max() <= _ULP_TOL
