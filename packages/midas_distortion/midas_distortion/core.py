"""Canonical MIDAS radial-distortion model — the single source of truth.

This leaf holds the distortion *layout* (which coefficient drives which η-fold
at which radial power) and the v1↔v2 coefficient mapping. Bugs in a distortion
model live in the layout/mapping — a wrong index, swapped phase, wrong fold —
not in the trivial arithmetic. Centralising the layout here lets
``midas_calibrate_v2`` (calibration), ``midas_peakfit`` (numpy) and
``midas_transforms`` (torch) all evaluate the *same* model from one definition.

Model (multiplicative factor on the projected radius)::

    D(ρ, η) = 1
        + iso_R2·ρ² + iso_R4·ρ⁴ + iso_R6·ρ⁶                 (isotropic)
        + a1·ρ⁴·cos( η' + phi1)                              (1-fold; ρ⁴ is a
        + a2·ρ²·cos(2η' + phi2)                                v1-physics quirk)
        + a3·ρ³·cos(3η' + phi3)
        + a4·ρ⁴·cos(4η' + phi4)
        + a5·ρ⁵·cos(5η' + phi5)
        + a6·ρ⁶·cos(6η' + phi6)

with η' = 90° − η (degrees in, converted internally). ``distortion_factor`` is
backend-agnostic: pass numpy arrays or torch tensors and it dispatches cos /
ones_like / zeros_like on the input's own library, so the math is identical
across consumers (down to floating-point reassociation).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List

_DEG2RAD = 0.017453292519943295


# ───────────────────────────────────────────────────────────── canonical names
# 15 names, in the SAME order as the v2 p_coeffs vector slots.
P_COEF_NAMES: List[str] = [
    "iso_R2", "iso_R4", "iso_R6",   # 0,1,2  isotropic radial (no η)
    "a1", "phi1",                   # 3,4    1-fold (ρ⁴)
    "a2", "phi2",                   # 5,6    2-fold (ρ²)
    "a3", "phi3",                   # 7,8    3-fold (ρ³)
    "a4", "phi4",                   # 9,10   4-fold (ρ⁴)
    "a5", "phi5",                   # 11,12  5-fold (ρ⁵)
    "a6", "phi6",                   # 13,14  6-fold (ρ⁶)
]
PHASE_NAMES = ("phi1", "phi2", "phi3", "phi4", "phi5", "phi6")
ISO_NAMES = ("iso_R2", "iso_R4", "iso_R6")
AMP_NAMES = ("a1", "a2", "a3", "a4", "a5", "a6")
_NAME_TO_V2IDX = {nm: i for i, nm in enumerate(P_COEF_NAMES)}


# ─────────────────────────────────────────────────────────────── harmonic term
@dataclass
class HarmonicTerm:
    """One additive contribution to the multiplicative distortion D.

    Isotropic radial term ``c·ρⁿ`` → ``fold=0, phase_idx=-1``; polar harmonic
    ``c·ρⁿ·cos(k η' + φ)`` → ``fold=k``.
    """

    coef_idx: int      # index into p_coeffs for the amplitude
    phase_idx: int     # index into p_coeffs for the phase (-1 if isotropic)
    radial_power: int  # n in ρⁿ
    fold: int          # k in cos(k η' + φ); 0 = isotropic


def v2_term_layout() -> List[HarmonicTerm]:
    """v2 canonical ordering — fold-monotonic, isotropic terms grouped first."""
    return [
        HarmonicTerm(0, -1, 2, 0),   # iso_R2
        HarmonicTerm(1, -1, 4, 0),   # iso_R4
        HarmonicTerm(2, -1, 6, 0),   # iso_R6
        HarmonicTerm(3, 4, 4, 1),    # a1, phi1  (1-fold uses ρ⁴)
        HarmonicTerm(5, 6, 2, 2),    # a2, phi2
        HarmonicTerm(7, 8, 3, 3),    # a3, phi3
        HarmonicTerm(9, 10, 4, 4),   # a4, phi4
        HarmonicTerm(11, 12, 5, 5),  # a5, phi5
        HarmonicTerm(13, 14, 6, 6),  # a6, phi6
    ]


def v1_term_layout() -> List[HarmonicTerm]:
    """Legacy v1 ordering on the v1 p₀..p₁₄ vector (kept for paramstest parity).

    p₀ amp ρ²fold2(φ=p₆) | p₁ amp ρ⁴fold4(φ=p₃) | p₂ iso ρ² | p₄ iso ρ⁶ |
    p₅ iso ρ⁴ | p₇ amp ρ⁴fold1(φ=p₈) | p₉ amp ρ³fold3(φ=p₁₀) |
    p₁₁ amp ρ⁵fold5(φ=p₁₂) | p₁₃ amp ρ⁶fold6(φ=p₁₄).
    """
    return [
        HarmonicTerm(0, 6, 2, 2),
        HarmonicTerm(1, 3, 4, 4),
        HarmonicTerm(2, -1, 2, 0),
        HarmonicTerm(4, -1, 6, 0),
        HarmonicTerm(5, -1, 4, 0),
        HarmonicTerm(7, 8, 4, 1),
        HarmonicTerm(9, 10, 3, 3),
        HarmonicTerm(11, 12, 5, 5),
        HarmonicTerm(13, 14, 6, 6),
    ]


def extended_term_layout(max_fold: int = 8) -> List[HarmonicTerm]:
    """v2 base layout plus (a_k, phi_k) for k = 7..max_fold, each at radial ρᵏ.

    Caller must size the p_coeffs vector to ``15 + 2·(max_fold − 6)``.
    """
    if max_fold <= 6:
        return v2_term_layout()
    base = v2_term_layout()
    nxt = 15
    for k in range(7, max_fold + 1):
        base.append(HarmonicTerm(nxt, nxt + 1, k, k))
        nxt += 2
    return base


def extended_p_coef_names(max_fold: int = 8) -> List[str]:
    if max_fold <= 6:
        return list(P_COEF_NAMES)
    out = list(P_COEF_NAMES)
    for k in range(7, max_fold + 1):
        out.extend([f"a{k}", f"phi{k}"])
    return out


# ───────────────────────────────────────────────────────────────── v1 ↔ v2 map
# v1 p-index → v2 canonical name.
V1_TO_V2_DISTORTION: Dict[int, str] = {
    0: "a2", 1: "a4", 2: "iso_R2", 3: "phi4", 4: "iso_R6", 5: "iso_R4",
    6: "phi2", 7: "a1", 8: "phi1", 9: "a3", 10: "phi3", 11: "a5",
    12: "phi5", 13: "a6", 14: "phi6",
}
# v2 canonical name → v1 p-index.
V2_TO_V1_DISTORTION: Dict[str, int] = {v: k for k, v in V1_TO_V2_DISTORTION.items()}
# v2 canonical name → v1 p-name ("p0".."p14"), for paramstest field setting.
V2_TO_V1_PNAME: Dict[str, str] = {v: f"p{k}" for k, v in V1_TO_V2_DISTORTION.items()}


# Fixed gather permutations (pure indexing → differentiable + device-portable,
# unlike an in-place assignment loop). ``_PERM_V1_TO_V2[slot]`` is the v1 index
# feeding v2 slot ``slot``; ``_PERM_V2_TO_V1[v1_idx]`` is the v2 slot feeding it.
_PERM_V1_TO_V2 = [V2_TO_V1_DISTORTION[name] for name in P_COEF_NAMES]
_PERM_V2_TO_V1 = [_NAME_TO_V2IDX[V1_TO_V2_DISTORTION[i]] for i in range(15)]


def v1_to_v2_coeffs(p_v1):
    """Reindex a v1 p₀..p₁₄ coefficient vector into v2 canonical order.

    A pure gather (``p_v1[perm]``) — works for numpy arrays and torch tensors,
    returns the same type, and stays differentiable for autograd.
    """
    return p_v1[_PERM_V1_TO_V2]


def v2_to_v1_coeffs(p_v2):
    """Reindex a v2 canonical coefficient vector into v1 p₀..p₁₄ order."""
    return p_v2[_PERM_V2_TO_V1]


def v2_coeffs_from_named(named, *, default: float = 0.0):
    """Build the canonical 15-vector (v2 :data:`P_COEF_NAMES` order) from a
    mapping that uses **v2 names** (``iso_R2``, ``a1``, ``phi1``, …) — with
    legacy ``p0``..``p14`` accepted only as a fallback for old inputs.

    v2 names win over any same-slot ``pN`` (so a v2-named source is honoured
    exactly; a pure-v1 source still works). Returns a numpy float64[15].
    Missing entries default to ``default`` (0.0). ``None`` values are skipped.
    """
    import numpy as np
    out = np.full(15, float(default), dtype=np.float64)
    # legacy p-names first, so v2 names override on any collision
    for k, v in named.items():
        if v is None:
            continue
        if isinstance(k, str) and len(k) > 1 and k[0] == "p" and k[1:].isdigit():
            i = int(k[1:])
            if 0 <= i < 15:
                out[_NAME_TO_V2IDX[V1_TO_V2_DISTORTION[i]]] = float(v)
    for k, v in named.items():
        if v is None:
            continue
        if k in _NAME_TO_V2IDX:
            out[_NAME_TO_V2IDX[k]] = float(v)
    return out


# ─────────────────────────────────────────────────────────────────── evaluation
def _is_torch(x) -> bool:
    return type(x).__module__.split(".")[0] == "torch"


def _backend(x):
    """Return (cos, ones_like, zeros_like) bound to x's array library."""
    if _is_torch(x):
        import torch
        return torch.cos, torch.ones_like, torch.zeros_like
    import numpy as np
    return np.cos, np.ones_like, np.zeros_like


def _zeros_like(x):
    return _backend(x)[2](x)


def distortion_factor(R_norm, eta_deg, p_coeffs, *, terms: Iterable[HarmonicTerm] = None):
    """Multiplicative distortion factor D(ρ, η).

    Parameters
    ----------
    R_norm : ρ = R / RhoD, numpy array or torch tensor (broadcastable).
    eta_deg : azimuthal angle in degrees (atan2(-y, z) convention).
    p_coeffs : coefficient vector; indexing is defined by ``terms``
        (v2 order by default, v1 order if ``terms=v1_term_layout()``).
    terms : layout; defaults to :func:`v2_term_layout`.
    """
    if terms is None:
        terms = v2_term_layout()
    cos, ones_like, _ = _backend(R_norm)
    eta_T = (90.0 - eta_deg) * _DEG2RAD
    D = ones_like(R_norm)
    for t in terms:
        amp = p_coeffs[t.coef_idx]
        rad = R_norm ** t.radial_power
        if t.fold == 0:
            D = D + amp * rad
        else:
            phase = p_coeffs[t.phase_idx] * _DEG2RAD
            D = D + amp * rad * cos(t.fold * eta_T + phase)
    return D


def apply_distortion(R, eta_deg, p_coeffs, rho_d, *, terms: Iterable[HarmonicTerm] = None):
    """Multiplicatively apply the distortion to a projected radius ``R`` (same
    units as ``rho_d``)."""
    R_norm = R / rho_d
    return R * distortion_factor(R_norm, eta_deg, p_coeffs, terms=terms)
