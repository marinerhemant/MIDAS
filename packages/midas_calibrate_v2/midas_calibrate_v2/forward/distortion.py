"""Analytical radial distortion basis for calibrate-v2.

The distortion *model* (layout tables, the v1↔v2 coefficient mapping, and the
backend-agnostic kernel) now lives in the shared :mod:`midas_distortion` leaf so
that calibration (here), :mod:`midas_peakfit` and :mod:`midas_transforms` all
evaluate one definition. This module re-exports those canonical symbols and adds
the two calibrate-v2-specific helpers (:func:`build_p_coeffs`,
:func:`coeffs_from_named`) that assemble a torch coefficient vector from an
unpacked-parameter dict.

See :mod:`midas_distortion.core` for the model description.
"""
from __future__ import annotations

from typing import List

import torch

# Single source of truth — the distortion layout + kernel + v1↔v2 maps.
from midas_distortion import (  # noqa: F401  (re-exported)
    HarmonicTerm,
    P_COEF_NAMES,
    PHASE_NAMES,
    ISO_NAMES,
    AMP_NAMES,
    v1_term_layout,
    v2_term_layout,
    extended_term_layout,
    extended_p_coef_names,
    V1_TO_V2_DISTORTION,
    V2_TO_V1_DISTORTION,
    v1_to_v2_coeffs,
    v2_to_v1_coeffs,
    distortion_factor,
    apply_distortion,
)

#: Named distortion blocks, cumulative, cheapest first.
#:
#: The 15 coefficients are not interchangeable and should not be refined as one
#: undifferentiated lump.  The radial (isotropic) block is azimuth-independent
#: and needs only radial coverage; every ``a_k``/``phi_k`` pair is a k-fold
#: azimuthal harmonic and needs azimuth to be identifiable.  On a detector that
#: sees one narrow wedge of each ring, the harmonics are degenerate with the
#: beam centre (1-fold) and the tilts (2-fold) and simply rail at their bounds —
#: which is why :func:`azimuth_coverage_gate` tells you to refine ``"radial"``.
#: Before this existed there was no way to say that: refinement was all fifteen
#: or none.
DISTORTION_BLOCKS = {
    "none":            (),
    "radial":          ("iso_R2", "iso_R4", "iso_R6"),
    "radial+2fold":    ("iso_R2", "iso_R4", "iso_R6", "a2", "phi2"),
    "radial+1fold":    ("iso_R2", "iso_R4", "iso_R6", "a2", "phi2",
                        "a1", "phi1"),
    "radial+3fold":    ("iso_R2", "iso_R4", "iso_R6", "a2", "phi2",
                        "a1", "phi1", "a3", "phi3"),
    "radial+4fold":    ("iso_R2", "iso_R4", "iso_R6", "a2", "phi2",
                        "a1", "phi1", "a3", "phi3", "a4", "phi4"),
    "full":            tuple(P_COEF_NAMES),
}


def resolve_distortion_block(spec) -> tuple:
    """Normalise a distortion selector to a tuple of v2 coefficient names.

    Accepts ``True`` (= ``"full"``), ``False``/``None`` (= ``"none"``), a block
    name from :data:`DISTORTION_BLOCKS`, or an explicit sequence of v2 names
    (``"iso_R2"``, ``"a3"``, ``"phi3"``, …).
    """
    if spec is True:
        return DISTORTION_BLOCKS["full"]
    if spec is False or spec is None:
        return ()
    if isinstance(spec, str):
        try:
            return DISTORTION_BLOCKS[spec]
        except KeyError:
            raise ValueError(
                f"unknown distortion block {spec!r}; "
                f"choose from {list(DISTORTION_BLOCKS)} or pass an explicit "
                f"list of coefficient names") from None
    names = tuple(spec)
    unknown = [n for n in names if n not in P_COEF_NAMES]
    if unknown:
        raise ValueError(
            f"unknown distortion coefficient(s) {unknown}; "
            f"valid names: {list(P_COEF_NAMES)}")
    # An amplitude without its phase is not a meaningful degree of freedom, and
    # a phase without its amplitude has zero gradient — catch both here rather
    # than letting the LM sit on a flat direction.
    for a, phi in (("a1", "phi1"), ("a2", "phi2"), ("a3", "phi3"),
                   ("a4", "phi4"), ("a5", "phi5"), ("a6", "phi6")):
        if (a in names) != (phi in names):
            raise ValueError(
                f"{a} and {phi} must be refined together — an amplitude with a "
                f"fixed phase is a constrained direction, and a phase with a "
                f"zero amplitude has no gradient at all")
    return names


# v2 slot index for each v1 p-index (name → position in P_COEF_NAMES).
_V1_IDX_TO_V2_SLOT = {
    v1_idx: P_COEF_NAMES.index(name)
    for v1_idx, name in V1_TO_V2_DISTORTION.items()
}


# ----------------------------------------------------------- helpers

def build_p_coeffs(unpacked: dict, *, dtype=None, device=None) -> torch.Tensor:
    """Stack the 15 v2-named distortion params from an unpacked dict.

    Centralises the v2 name list — call sites no longer need to know the
    order.  Missing names default to 0.0 (e.g. when a spec fixes a coefficient
    at zero by omission).
    """
    pieces: List = []
    ref = None
    for nm in P_COEF_NAMES:
        v = unpacked.get(nm)
        if v is None:
            pieces.append(None)
            continue
        if not isinstance(v, torch.Tensor):
            v = torch.as_tensor(
                v, dtype=dtype if dtype is not None else torch.float64,
                device=device if device is not None else "cpu")
        if ref is None:
            ref = v
        pieces.append(v)
    if ref is None:
        # Nothing supplied — return a 15-zero vector.
        return torch.zeros(
            15, dtype=dtype if dtype is not None else torch.float64,
            device=device if device is not None else "cpu")
    out_dtype = dtype if dtype is not None else ref.dtype
    out_device = device if device is not None else ref.device
    pieces = [
        (p.to(dtype=out_dtype, device=out_device)
         if p is not None else torch.zeros((), dtype=out_dtype, device=out_device))
        for p in pieces
    ]
    return torch.stack(pieces)


def coeffs_from_named(named: dict, default: float = 0.0, n: int = 15,
                      dtype=torch.float64, device="cpu") -> torch.Tensor:
    """Build a p_coeffs[15] tensor from a v2-named dict.

    Recognised keys are :data:`P_COEF_NAMES`.  For backward compat, keys
    ``p0``..``p14`` are also accepted via the v1 → v2 mapping (so an old
    paramstest dict still works).
    """
    p = torch.full((n,), default, dtype=dtype, device=device)
    for k, v in named.items():
        if k in P_COEF_NAMES:
            i = P_COEF_NAMES.index(k)
        elif k.startswith("p") and k[1:].isdigit():
            v1_idx = int(k[1:])
            if 0 <= v1_idx < 15:
                i = _V1_IDX_TO_V2_SLOT[v1_idx]
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
    "DISTORTION_BLOCKS", "resolve_distortion_block",
    "HarmonicTerm",
    "P_COEF_NAMES", "PHASE_NAMES", "ISO_NAMES", "AMP_NAMES",
    "v1_term_layout", "v2_term_layout",
    "extended_term_layout", "extended_p_coef_names",
    "V1_TO_V2_DISTORTION", "V2_TO_V1_DISTORTION",
    "v1_to_v2_coeffs", "v2_to_v1_coeffs",
    "distortion_factor", "apply_distortion",
    "build_p_coeffs", "coeffs_from_named",
]
