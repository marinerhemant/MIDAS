"""Swappable energy model: monochromatic / pink / white.

Thin adapter over :class:`midas_pink.spectrum.ParameterisedSpectrum` so the
forward model has a single code path regardless of bandwidth.  A monochromatic
beam is just a ``delta`` spectrum (one energy, weight 1).

Returns ``(energies_keV, lambdas_A, weights)`` as torch tensors -- the same
triple the pink-beam workflow already consumes.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    import torch

__all__ = ["make_spectrum", "monochromatic"]


def make_spectrum(E0_keV, *, half_bw=0.05, n_samples=51, kind="boxcar",
                  rel_bw=None, fixed=True, dtype=None):
    """Build a (fixed by default) spectrum.

    Parameters
    ----------
    E0_keV : float
        Centre energy.
    half_bw : float
        Fractional half-width of the energy support.
    n_samples : int
        Number of grid samples (odd preferred so E0 is sampled exactly).
    kind : {"boxcar", "gaussian", "delta"}
        Initial spectral shape.  ``"delta"`` = monochromatic.
    rel_bw : float, optional
        Bandwidth of the initialiser (required for boxcar/gaussian).
    fixed : bool
        If True the weights are buffers (not fit); set False to refine S(E).
    """
    import torch
    from midas_pink.spectrum import ParameterisedSpectrum

    if rel_bw is None and kind in {"boxcar", "gaussian"}:
        rel_bw = half_bw  # sensible default: fill the support
    return ParameterisedSpectrum(
        E0_keV=float(E0_keV),
        half_bw=float(half_bw),
        n_samples=int(n_samples),
        init_kind=kind,
        init_rel_bw=rel_bw,
        fixed=fixed,
        dtype=dtype or torch.float64,
    )


def monochromatic(E0_keV, *, dtype=None):
    """A single-energy (delta) spectrum -- the monochromatic limit.

    Uses a 3-point grid so the centre energy E0 is sampled *exactly* (the
    delta init puts all weight on the centre sample).
    """
    return make_spectrum(E0_keV, half_bw=1e-3, n_samples=3, kind="delta",
                         fixed=True, dtype=dtype)
