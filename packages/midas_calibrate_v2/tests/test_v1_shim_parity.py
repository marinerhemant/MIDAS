"""GATE: the v1→v2 distortion shim must reproduce the legacy path EXACTLY.

FIX-2 re-points peakfit/transforms at the shared v2 distortion kernel, reading
legacy ``p0..p14`` via a v1→v2 reindex shim. This is only safe if the v2-native
path reproduces the legacy result for v1 inputs — any drift would silently move
every corrected spot and break the C-parity tests.

The legacy distortion is the hardcoded polynomial in
``midas_calibrate.geometry_torch.pixel_to_REta_torch`` (lines 93-101). Both that
and the v2 kernel form the *same* multiplicative factor D(ρ,η) as a sum of 9
harmonic terms + 1. We pin two distinct claims, deliberately separated:

  HARD GATE (correctness) — the 9 individual terms are BIT-IDENTICAL
    (torch.equal, Δ=0) across all three computations. This proves the v1→v2
    coefficient/fold/phase/radial-power mapping is exact; a real mapping bug
    (wrong index, swapped phase, wrong fold) would show here as a nonzero Δ,
    NOT hide behind rounding.

  ASSEMBLED (numerics) — the summed factor agrees to ≤ 8 ULP. The kernel loops
    in layout order while the inline polynomial sums in source order; with an
    identical term set this difference is *only* IEEE-754 reassociation
    (measured worst case 4 ULP over 500 random inputs). At D≈1 that is ~1e-15
    relative on the corrected radius — sub-femtometer at any detector, and
    categorically distinct from a logic error.
"""
import numpy as np
import torch

from midas_calibrate_v2.forward.distortion import (
    distortion_factor,
    v1_term_layout,
    v2_term_layout,
)

_DEG2RAD = 0.017453292519943295
_EPS = float(np.finfo(np.float64).eps)
_ULP_TOL = 8 * _EPS  # reassociation headroom; measured worst case is 4 ULP

# v1 p-index → v2 canonical name (the lossless reindex; mirrors
# compat.from_v1.V1_TO_V2_DISTORTION and compat.to_v1.V2_TO_V1_DISTORTION).
V1_TO_V2 = {
    0: "a2", 1: "a4", 2: "iso_R2", 3: "phi4", 4: "iso_R6", 5: "iso_R4",
    6: "phi2", 7: "a1", 8: "phi1", 9: "a3", 10: "phi3", 11: "a5",
    12: "phi5", 13: "a6", 14: "phi6",
}
V2_NAMES = [
    "iso_R2", "iso_R4", "iso_R6", "a1", "phi1", "a2", "phi2", "a3", "phi3",
    "a4", "phi4", "a5", "phi5", "a6", "phi6",
]
_N2I = {nm: i for i, nm in enumerate(V2_NAMES)}


def _grid():
    """A realistic (ρ, η) sweep over a detector quadrant."""
    R = torch.linspace(0.0, 1.3, 41, dtype=torch.float64)         # ρ = R/RhoD
    eta = torch.linspace(-180.0, 180.0, 73, dtype=torch.float64)  # deg
    Rg, Eg = torch.meshgrid(R, eta, indexing="ij")
    return Rg, Eg


def _shim_v1_to_v2(p_v1: torch.Tensor) -> torch.Tensor:
    """Reindex a v1 p0..p14 vector into a v2-ordered coeff vector."""
    v2 = torch.zeros_like(p_v1)
    for v1_idx, v2_name in V1_TO_V2.items():
        v2[_N2I[v2_name]] = p_v1[v1_idx]
    return v2


def _inline_terms(Rg, Eg, p):
    """The 9 additive terms of geometry_torch.pixel_to_REta_torch:93-101,
    keyed by their originating p-index (verbatim transcription)."""
    eT = (90.0 - Eg) * _DEG2RAD
    return {
        0:  p[0] * Rg.pow(2) * torch.cos(2 * eT + _DEG2RAD * p[6]),
        1:  p[1] * Rg.pow(4) * torch.cos(4 * eT + _DEG2RAD * p[3]),
        2:  p[2] * Rg.pow(2),
        4:  p[4] * Rg.pow(6),
        5:  p[5] * Rg.pow(4),
        7:  p[7] * Rg.pow(4) * torch.cos(eT + _DEG2RAD * p[8]),
        9:  p[9] * Rg.pow(3) * torch.cos(3 * eT + _DEG2RAD * p[10]),
        11: p[11] * Rg.pow(5) * torch.cos(5 * eT + _DEG2RAD * p[12]),
        13: p[13] * Rg.pow(6) * torch.cos(6 * eT + _DEG2RAD * p[14]),
    }


def _kernel_terms(Rg, Eg, p, layout):
    """The per-term contributions distortion_factor would accumulate, keyed by
    the amplitude's coef_idx (so v1 and v2 layouts share a key space)."""
    eT = (90.0 - Eg) * _DEG2RAD
    out = {}
    for t in layout:
        amp = p[t.coef_idx]
        rad = Rg.pow(t.radial_power)
        if t.fold == 0:
            out[t.coef_idx] = amp * rad
        else:
            out[t.coef_idx] = amp * rad * torch.cos(t.fold * eT + p[t.phase_idx] * _DEG2RAD)
    return out


def _random_p(rng):
    p = torch.from_numpy(rng.uniform(-5e-3, 5e-3, size=15)).to(torch.float64)
    for ph in (3, 6, 8, 10, 12, 14):  # v1 phase slots get a degree-scale value
        p[ph] = float(rng.uniform(-180.0, 180.0))
    return p


def test_shim_terms_bit_identical():
    """HARD GATE: every distortion term matches across legacy-inline,
    v1-kernel, and v2-shim — exactly (Δ=0), for 500 random p0..p14."""
    rng = np.random.default_rng(0)
    Rg, Eg = _grid()
    v1L, v2L = v1_term_layout(), v2_term_layout()
    for _ in range(500):
        p = _random_p(rng)
        p2 = _shim_v1_to_v2(p)
        inl = _inline_terms(Rg, Eg, p)
        kv1 = _kernel_terms(Rg, Eg, p, v1L)        # keyed by v1 coef_idx
        kv2 = _kernel_terms(Rg, Eg, p2, v2L)       # keyed by v2 coef_idx
        # v2 amplitude slots, in the SAME fold order as the v1 p-index keys.
        v2_key = {0: _N2I["a2"], 1: _N2I["a4"], 2: _N2I["iso_R2"],
                  4: _N2I["iso_R6"], 5: _N2I["iso_R4"], 7: _N2I["a1"],
                  9: _N2I["a3"], 11: _N2I["a5"], 13: _N2I["a6"]}
        for k, term in inl.items():
            assert torch.equal(term, kv1[k]), f"v1 kernel term p{k} != inline (mapping bug)"
            assert torch.equal(term, kv2[v2_key[k]]), f"v2 shim term p{k} != inline (mapping bug)"


def test_shim_assembled_within_ulp():
    """ASSEMBLED: the summed factor D agrees across all three paths to ≤ 8 ULP
    (pure reassociation; the term set is proven identical above)."""
    rng = np.random.default_rng(1)
    Rg, Eg = _grid()
    eT = (90.0 - Eg) * _DEG2RAD
    for _ in range(500):
        p = _random_p(rng)
        p2 = _shim_v1_to_v2(p)
        inl = _inline_terms(Rg, Eg, p)
        D_inline = sum(inl.values()) + 1.0
        D_v1 = distortion_factor(Rg, Eg, p, terms=v1_term_layout())
        D_v2 = distortion_factor(Rg, Eg, p2, terms=v2_term_layout())
        assert (D_inline - D_v1).abs().max().item() <= _ULP_TOL
        assert (D_v1 - D_v2).abs().max().item() <= _ULP_TOL


def test_shim_zero_is_unity():
    """All-zero distortion → D ≡ 1 through every path (no spurious offset)."""
    Rg, Eg = _grid()
    z = torch.zeros(15, dtype=torch.float64)
    for layout in (v1_term_layout(), v2_term_layout()):
        D = distortion_factor(Rg, Eg, z, terms=layout)
        assert torch.equal(D, torch.ones_like(D))
