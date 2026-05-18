"""Analytical radial distortion basis — extensible up to arbitrary harmonic order.

The v2 layout assigns one canonical name per coefficient, with amp/phase
pairs adjacent and η-folds running 0 → 1 → 2 → 3 → 4 → 5 → 6.  The model:

    D = 1
        + iso_R2 · ρ²                                         (isotropic ρ²)
        + iso_R4 · ρ⁴                                         (isotropic ρ⁴)
        + iso_R6 · ρ⁶                                         (isotropic ρ⁶)
        + a1 · ρ⁴ · cos( η' + phi1)                           (1-fold)
        + a2 · ρ² · cos(2η' + phi2)                           (2-fold)
        + a3 · ρ³ · cos(3η' + phi3)                           (3-fold)
        + a4 · ρ⁴ · cos(4η' + phi4)                           (4-fold)
        + a5 · ρ⁵ · cos(5η' + phi5)                           (5-fold)
        + a6 · ρ⁶ · cos(6η' + phi6)                           (6-fold)

with η' = 90° - η (in degrees on input; converted to radians internally).

The radial powers per fold follow the v1 model exactly (1-fold uses ρ⁴, not
ρ¹ — this is a v1-physics quirk preserved for parity).

The legacy v1 ordering (p₀…p₁₄ with phases scattered) is kept available
via :func:`v1_term_layout` for paramstest backward compatibility, and the
:mod:`midas_calibrate_v2.compat.from_v1` module maps v1 → v2 names.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import torch


_DEG2RAD = 0.017453292519943295


# ============================================================ canonical v2 names
#
# 15 names, in the SAME order as the p_coeffs vector slots.  Use this list
# to build a p_coeffs tensor from the unpacked parameter dict — no more
# scattered f"p{i}" string indexing.

P_COEF_NAMES: List[str] = [
    # Isotropic radial terms (no η dependence)
    "iso_R2",       # 0
    "iso_R4",       # 1
    "iso_R6",       # 2
    # 1-fold harmonic (radial power ρ⁴)
    "a1",           # 3  amplitude
    "phi1",         # 4  phase (deg)
    # 2-fold harmonic (radial power ρ²)
    "a2",           # 5
    "phi2",         # 6
    # 3-fold harmonic (radial power ρ³)
    "a3",           # 7
    "phi3",         # 8
    # 4-fold harmonic (radial power ρ⁴)
    "a4",           # 9
    "phi4",         # 10
    # 5-fold harmonic (radial power ρ⁵)
    "a5",           # 11
    "phi5",         # 12
    # 6-fold harmonic (radial power ρ⁶)
    "a6",           # 13
    "phi6",         # 14
]

# Indices of phase parameters within P_COEF_NAMES — needed by compat layer
# to apply the wider ±90° bound.
PHASE_NAMES = ("phi1", "phi2", "phi3", "phi4", "phi5", "phi6")
ISO_NAMES = ("iso_R2", "iso_R4", "iso_R6")
AMP_NAMES = ("a1", "a2", "a3", "a4", "a5", "a6")


# ----------------------------------------------------------- harmonic term

@dataclass
class HarmonicTerm:
    """One additive contribution to the multiplicative distortion D.

    Either an isotropic radial term ``c · ρⁿ`` (set ``fold=0``,
    ``phase_idx=-1``) or a polar harmonic ``c · ρⁿ cos(k η' + φ)``
    (set ``fold=k``).
    """

    coef_idx: int           # index into the p_coeffs vector for the amplitude
    phase_idx: int          # index into p_coeffs for the phase (-1 if isotropic)
    radial_power: int       # n in ρⁿ
    fold: int               # k in cos(k η' + φ); 0 = isotropic radial


def v2_term_layout() -> List[HarmonicTerm]:
    """v2 canonical ordering — fold-monotonic, isotropic terms grouped."""
    return [
        # Isotropic
        HarmonicTerm(coef_idx=0, phase_idx=-1, radial_power=2, fold=0),  # iso_R2
        HarmonicTerm(coef_idx=1, phase_idx=-1, radial_power=4, fold=0),  # iso_R4
        HarmonicTerm(coef_idx=2, phase_idx=-1, radial_power=6, fold=0),  # iso_R6
        # Harmonics: fold k uses ρ^max(k,2) with the v1-physics quirk that
        # 1-fold uses ρ⁴.
        HarmonicTerm(coef_idx=3,  phase_idx=4,  radial_power=4, fold=1),  # a1, phi1
        HarmonicTerm(coef_idx=5,  phase_idx=6,  radial_power=2, fold=2),  # a2, phi2
        HarmonicTerm(coef_idx=7,  phase_idx=8,  radial_power=3, fold=3),  # a3, phi3
        HarmonicTerm(coef_idx=9,  phase_idx=10, radial_power=4, fold=4),  # a4, phi4
        HarmonicTerm(coef_idx=11, phase_idx=12, radial_power=5, fold=5),  # a5, phi5
        HarmonicTerm(coef_idx=13, phase_idx=14, radial_power=6, fold=6),  # a6, phi6
    ]


def extended_term_layout(max_fold: int = 8) -> List[HarmonicTerm]:
    """Extension of the v2 basis to higher η-folds.

    Adds (a_k, phi_k) pairs for k = 7..max_fold, each with radial power
    ρᵏ.  Useful for diagnosing whether higher-order detector
    distortions (e.g., a 7-fold module-pattern artifact) are present
    in calibrant data — see Bayesian model comparison via BIC.

    Returns the v2 base layout plus extra terms; the caller is
    responsible for ensuring the p_coeffs vector is sized to match
    (15 + 2 × (max_fold - 6)).
    """
    if max_fold <= 6:
        return v2_term_layout()
    base = v2_term_layout()
    next_idx = 15
    for k in range(7, max_fold + 1):
        base.append(HarmonicTerm(
            coef_idx=next_idx, phase_idx=next_idx + 1,
            radial_power=k, fold=k,
        ))
        next_idx += 2
    return base


def extended_p_coef_names(max_fold: int = 8) -> List[str]:
    """v2 names extended with a7/phi7..a_k/phi_k for k > 6."""
    if max_fold <= 6:
        return list(P_COEF_NAMES)
    out = list(P_COEF_NAMES)
    for k in range(7, max_fold + 1):
        out.extend([f"a{k}", f"phi{k}"])
    return out


def v1_term_layout() -> List[HarmonicTerm]:
    """Legacy v1 ordering — kept for paramstest round-trip parity tests.

    The v1 mapping (preserved):
      p₀: amp, ρ²,  fold=2 (phase=p₆)        v2: a2
      p₁: amp, ρ⁴,  fold=4 (phase=p₃)        v2: a4
      p₂: amp, ρ²,  fold=0                    v2: iso_R2
      p₃: phase for p₁                        v2: phi4
      p₄: amp, ρ⁶,  fold=0                    v2: iso_R6
      p₅: amp, ρ⁴,  fold=0                    v2: iso_R4
      p₆: phase for p₀                        v2: phi2
      p₇: amp, ρ⁴,  fold=1 (phase=p₈)        v2: a1
      p₈: phase for p₇                        v2: phi1
      p₉: amp, ρ³,  fold=3 (phase=p₁₀)       v2: a3
      p₁₀: phase for p₉                       v2: phi3
      p₁₁: amp, ρ⁵,  fold=5 (phase=p₁₂)      v2: a5
      p₁₂: phase for p₁₁                      v2: phi5
      p₁₃: amp, ρ⁶,  fold=6 (phase=p₁₄)      v2: a6
      p₁₄: phase for p₁₃                      v2: phi6
    """
    return [
        HarmonicTerm(coef_idx=0,  phase_idx=6,  radial_power=2, fold=2),
        HarmonicTerm(coef_idx=1,  phase_idx=3,  radial_power=4, fold=4),
        HarmonicTerm(coef_idx=2,  phase_idx=-1, radial_power=2, fold=0),
        HarmonicTerm(coef_idx=4,  phase_idx=-1, radial_power=6, fold=0),
        HarmonicTerm(coef_idx=5,  phase_idx=-1, radial_power=4, fold=0),
        HarmonicTerm(coef_idx=7,  phase_idx=8,  radial_power=4, fold=1),
        HarmonicTerm(coef_idx=9,  phase_idx=10, radial_power=3, fold=3),
        HarmonicTerm(coef_idx=11, phase_idx=12, radial_power=5, fold=5),
        HarmonicTerm(coef_idx=13, phase_idx=14, radial_power=6, fold=6),
    ]


# ----------------------------------------------------------- evaluation

def distortion_factor(
    R_norm: torch.Tensor,           # ρ = R / RhoD (broadcastable)
    eta_deg: torch.Tensor,          # azimuthal angle in degrees, atan2(-y, z)
    p_coeffs: torch.Tensor,         # [N] amplitudes/phases (default N=15, v2 order)
    *,
    terms: Iterable[HarmonicTerm] = None,
) -> torch.Tensor:
    """Compute the multiplicative distortion factor D(ρ, η)."""
    if terms is None:
        terms = v2_term_layout()
    eta_T = (90.0 - eta_deg) * _DEG2RAD
    D = torch.ones_like(R_norm) if R_norm.ndim else R_norm.new_ones(())
    for term in terms:
        amp = p_coeffs[term.coef_idx]
        rad = R_norm.pow(term.radial_power)
        if term.fold == 0:
            D = D + amp * rad
        else:
            phase = p_coeffs[term.phase_idx] * _DEG2RAD
            D = D + amp * rad * torch.cos(term.fold * eta_T + phase)
    return D


def apply_distortion(
    R_um: torch.Tensor,         # raw projected R (μm or px — caller's units)
    eta_deg: torch.Tensor,
    p_coeffs: torch.Tensor,
    rho_d: torch.Tensor,        # px or μm; same units as R_um
    *,
    terms: Iterable[HarmonicTerm] = None,
) -> torch.Tensor:
    """Multiplicatively apply the distortion to a projected radius."""
    R_norm = R_um / rho_d
    return R_um * distortion_factor(R_norm, eta_deg, p_coeffs, terms=terms)


# ----------------------------------------------------------- helpers

def build_p_coeffs(unpacked: dict, *, dtype=None, device=None) -> torch.Tensor:
    """Stack the 15 v2-named distortion params from an unpacked dict.

    Centralises the v2 name list — call sites no longer need to know
    the order.  Missing names default to 0.0 (e.g. when a spec fixes a
    coefficient at zero by omission).
    """
    pieces = []
    ref = None
    for nm in P_COEF_NAMES:
        v = unpacked.get(nm)
        if v is None:
            pieces.append(None)
            continue
        if not isinstance(v, torch.Tensor):
            v = torch.as_tensor(v, dtype=dtype if dtype is not None else torch.float64,
                                  device=device if device is not None else "cpu")
        if ref is None:
            ref = v
        pieces.append(v)
    if ref is None:
        # Nothing supplied — return a 15-zero vector.
        return torch.zeros(15, dtype=dtype if dtype is not None else torch.float64,
                            device=device if device is not None else "cpu")
    out_dtype = dtype if dtype is not None else ref.dtype
    out_device = device if device is not None else ref.device
    pieces = [(p.to(dtype=out_dtype, device=out_device)
                if p is not None else torch.zeros((), dtype=out_dtype, device=out_device))
                for p in pieces]
    return torch.stack(pieces)


def coeffs_from_named(named: dict, default: float = 0.0, n: int = 15,
                      dtype=torch.float64, device="cpu") -> torch.Tensor:
    """Build a p_coeffs[15] tensor from a v2-named dict.

    Recognised keys are :data:`P_COEF_NAMES`.  For backward compat,
    keys ``p0``..``p14`` are also accepted via the v1 → v2 mapping
    (so an old paramstest dict still works).
    """
    p = torch.full((n,), default, dtype=dtype, device=device)
    # Map v1 p-index → v2 slot (None means "intentionally drop"; should not occur).
    v1_to_v2 = {
        0: 5, 1: 9, 2: 0, 3: 10, 4: 2, 5: 1, 6: 6, 7: 3, 8: 4,
        9: 7, 10: 8, 11: 11, 12: 12, 13: 13, 14: 14,
    }
    for k, v in named.items():
        if k in P_COEF_NAMES:
            i = P_COEF_NAMES.index(k)
        elif k.startswith("p") and k[1:].isdigit():
            v1_idx = int(k[1:])
            if 0 <= v1_idx < 15:
                i = v1_to_v2[v1_idx]
            else:
                continue
        else:
            continue
        if isinstance(v, torch.Tensor):
            p[i] = v.to(dtype=dtype, device=device)
        else:
            p[i] = float(v)
    return p


__all__ = [
    "HarmonicTerm",
    "P_COEF_NAMES", "PHASE_NAMES", "ISO_NAMES", "AMP_NAMES",
    "v1_term_layout", "v2_term_layout",
    "extended_term_layout", "extended_p_coef_names",
    "distortion_factor", "apply_distortion",
    "build_p_coeffs", "coeffs_from_named",
]
