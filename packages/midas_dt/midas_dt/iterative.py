"""Iterative reconstruction: SIRT and TV-regularised least squares.

``recon.reconstruct`` uses ``midas-tomo``'s gridrec, which is filtered back
projection: one pass, fast, and the right default. It has one failure mode that
matters for XRD-CT, where angle counts are often low because every projection
costs a full detector frame at every translation -- **streaking**. FBP assumes
the angular sampling satisfies Nyquist; below that it does not degrade
gracefully, it puts structured artefacts through the image that a per-voxel
peak fit will happily fit.

SIRT and TV trade compute for artefact suppression. Neither is a free
improvement:

* they are **iterative**, so the answer depends on where you stop
* they are **biased** -- SIRT toward smoothness, TV toward piecewise-constant
  images -- and TV in particular will manufacture flat regions with sharp
  edges out of a noisy continuum if the weight is too high
* their noise propagation is not analytic, so per-voxel sigma has to come from
  Monte Carlo (``recon.reconstruct(variance_samples=K)`` on the gridrec path
  is cheaper and better understood)

Use gridrec unless streaking is visibly limiting, then compare.

Measured, and read the caveat before quoting it. Two uniform discs, one
off-centre, reconstructed from the **closed-form** Radon transform so that no
method inverts its own discretisation, correlation against the truth over the
central region:

    n_angles      gridrec       SIRT         TV
           8       0.8595     0.9516     0.9520
          16       0.8699     0.9658     0.9386
          32       0.8722     0.9736     0.9398
          90       0.8728     0.9795     0.9404

gridrec plateaus because of a ~1 px centring offset in that harness, not
because of angular sampling: scanning ``shift`` lifts it to 0.9016 at 90
angles. So the fair comparison at 90 angles is **0.90 gridrec vs 0.98 SIRT.**

**The phantom flatters the iterative methods.** Two uniform discs is
piecewise-constant and non-negative, which is exactly what SIRT's clamp and
TV's prior assume. This is NOT a general accuracy ranking, and it is not
evidence for either method on a continuous or textured sample. It supports one
narrow claim: on sparse angular sampling of a blocky object, the iterative
reconstructors recover more of it than FBP does.

Two harness traps, both hit while producing that table:

* A first version generated the sinogram with this package's own projector --
  the operator SIRT and TV invert. gridrec scored 0.74 even at 90 angles, and
  the reason was that the other two were solving the discretisation that made
  the data. Any comparison against gridrec needs an independently generated
  sinogram.
* ``sirt`` and ``tv_reconstruct`` default to ``size = n_translations`` while
  gridrec pads to a power of two, so their outputs are different grids (32 vs
  64 above) at the same pixel scale. Crop to a common centred region before
  comparing anything.

**Why these are implemented here rather than imported.** The plan pointed at
``midas_dct_tt.recon``, which has exactly these algorithms. That package is
unpublished (404 on PyPI), and ``midas-dt`` is public -- an extra pointing at
it would be uninstallable for every user. Its API is also DCT-shaped: 3-D
isotropic TV, ``clamp=(0, 1)`` for an occupancy field, a logit
parameterisation. XRD-CT wants 2-D slices, N channels, and non-negative
intensity with no upper bound. What IS carried over is its best idea, the
adjoint by autograd (see :func:`backproject`).
"""

from __future__ import annotations

import logging

import numpy as np

from .conventions import RECON_SIGN
from .recon import Reconstruction
from .sinogram import SinogramStack

__all__ = ["backproject", "sirt", "tv_reconstruct"]

log = logging.getLogger(__name__)


def _torch():
    try:
        import torch
    except ImportError as exc:                       # pragma: no cover
        raise ImportError(
            "iterative reconstruction needs midas-invert. Install with "
            "`pip install midas-dt[direct]`, or use recon.reconstruct, which "
            "is the gridrec default and needs only midas-tomo."
        ) from exc
    return torch


def backproject(A, y):
    """``A^T y``, obtained by differentiating ``<A x, y>`` with respect to x.

    For a linear operator that IS the transpose, exactly, by construction --
    not a separately-written routine that has to be kept in step with the
    projector. Hand-written adjoints that quietly stop matching their forward
    operator are a recurring bug in this family of code: the iteration still
    converges, to the wrong image, and nothing errors.

    ``tests/test_iterative.py`` checks it against the explicit sparse transpose
    anyway, because "exact by construction" is a claim about the code and not a
    substitute for measuring it.
    """
    torch = _torch()
    return torch.sparse.mm(A.t(), y.T).T


def _prepare(stack: SinogramStack, size, dtype, device):
    """Common setup: the operator, the data, and the geometry it implies."""
    from .direct import projection_matrix

    torch = _torch()
    dtype = dtype or torch.float64
    n_bins, n_omega, n_trans = stack.intensity.shape
    size = int(size or n_trans)
    A = projection_matrix(size, stack.omega_deg, n_trans,
                          dtype=dtype, device=device)
    b = torch.as_tensor(np.asarray(stack.intensity, dtype=np.float64),
                        dtype=dtype, device=device).reshape(n_bins, -1)
    return A, b, size, n_bins, dtype


def sirt(
    stack: SinogramStack,
    *,
    size: int | None = None,
    n_iter: int = 50,
    relaxation: float = 1.0,
    non_negative: bool = True,
    apply_sign: bool = True,
    device=None,
    dtype=None,
    callback=None,
) -> Reconstruction:
    """SIRT over every bin of *stack* simultaneously.

    ``x <- x + lambda * C * A^T (R * (b - A x))``, with ``R`` the inverse row
    sums and ``C`` the inverse column sums -- the standard simultaneous
    weighting. Both are computed from ``A`` itself, so they cannot fall out of
    step with the operator.

    All ``n_bins`` channels are iterated together as columns of one matrix, so
    the cost is one sparse mat-mul per iteration regardless of channel count,
    not one per channel.

    Parameters
    ----------
    n_iter : int
        SIRT has no stopping rule; this IS the regularisation. Too few and the
        image is over-smooth, too many and it converges toward the noisy
        least-squares solution. Compare against gridrec rather than tuning
        until it looks nice.
    non_negative : bool
        Clamp at zero each iteration. Intensity cannot be negative, and the
        constraint is most of where SIRT's artefact suppression comes from.
        Note this is a clamp at 0 only -- unlike the occupancy problems this
        algorithm is often used for, there is no upper bound.
    apply_sign : bool
        Multiply by :data:`~midas_dt.conventions.RECON_SIGN` so the output has
        the same sign convention as the gridrec path. Leave it on unless you
        want to compare raw iterates.
    """
    torch = _torch()
    A, b, size, n_bins, dtype = _prepare(stack, size, dtype, device)
    n_vox = size * size

    ones_vox = torch.ones((1, n_vox), dtype=dtype, device=device)
    row_sum = torch.sparse.mm(A, ones_vox.T).T                      # A 1
    col_sum = backproject(A, torch.ones((1, b.shape[1]), dtype=dtype,
                                        device=device))             # A^T 1
    R = torch.where(row_sum > 0, 1.0 / row_sum, torch.zeros_like(row_sum))
    C = torch.where(col_sum > 0, 1.0 / col_sum, torch.zeros_like(col_sum))

    x = torch.zeros((n_bins, n_vox), dtype=dtype, device=device)
    for it in range(n_iter):
        residual = b - torch.sparse.mm(A, x.T).T
        x = x + relaxation * C * backproject(A, R * residual)
        if non_negative:
            x = torch.clamp(x, min=0.0)
        if callback is not None:
            callback(it, float(torch.linalg.matrix_norm(residual)))
    resid = float(torch.linalg.matrix_norm(b - torch.sparse.mm(A, x.T).T)
                  / torch.clamp(torch.linalg.matrix_norm(b), min=1e-30))
    log.info("SIRT: %d iterations, relative residual %.4g", n_iter, resid)

    img = x.detach().cpu().numpy().reshape(n_bins, size, size)
    if apply_sign:
        img = img * RECON_SIGN
    return Reconstruction(
        intensity=img, variance=None, bin_shape=stack.bin_shape,
        channel=stack.channel, limits=stack.limits,
        sign_applied=RECON_SIGN if apply_sign else 1.0,
    )


def tv_reconstruct(
    stack: SinogramStack,
    *,
    size: int | None = None,
    tv_weight: float = 1e-3,
    steps: int = 300,
    lr: float = 0.05,
    optimizer: str = "adam",
    non_negative: bool = True,
    apply_sign: bool = True,
    weighted: bool = True,
    device=None,
    dtype=None,
) -> Reconstruction:
    """Least squares plus an anisotropic total-variation penalty.

    Minimises ``|| A x - b ||^2_w + tv_weight * TV(x)`` over all bins at once,
    with ``midas_invert.fit`` doing the optimisation.

    TV is the edge-preserving prior: it penalises the integral of the gradient
    magnitude, so it suppresses streaks and noise while tolerating genuine
    sharp boundaries in a way an L2 smoothness penalty does not.

    **``tv_weight`` is not a free parameter.** Too high and TV does what it is
    designed to do -- produce a piecewise-constant image -- turning a noisy
    continuum into flat regions with invented edges, which a downstream peak
    fit will then report as real structure with small scatter. There is no
    default that is right for every dataset. Start at zero, confirm you
    reproduce gridrec, and raise it only while the streaks fall faster than the
    features do.

    Normalisation: TV is divided by the number of voxels and the data term by
    the number of measurements, so ``tv_weight`` means roughly the same thing
    across grid sizes and channel counts. It is still dataset-dependent.
    """
    torch = _torch()
    from midas_invert import fit

    A, b, size, n_bins, dtype = _prepare(stack, size, dtype, device)
    n_vox = size * size

    if weighted:
        var = torch.as_tensor(np.asarray(stack.variance, dtype=np.float64),
                              dtype=dtype, device=device).reshape(n_bins, -1)
        # Floor at a fraction of the median rather than at ~0: an unfloored
        # 1/variance gives near-empty bins astronomically large weights and the
        # fit chases the background instead of the peak.
        pos = var[var > 0]
        floor = (1e-3 * torch.median(pos)) if pos.numel() else torch.tensor(
            1.0, dtype=dtype, device=device)
        w = 1.0 / torch.clamp(var, min=float(floor))
        w = w / w.mean()
    else:
        w = torch.ones_like(b)

    # Parameterise the image itself, softplus-gated when non-negative. A raw
    # clamp has zero gradient once it binds, so a voxel pushed negative early
    # in the optimisation could never come back.
    scale = float(torch.clamp(b.abs().max(), min=1e-12)) / max(size, 1)
    raw = torch.zeros((n_bins, n_vox), dtype=dtype, device=device,
                      requires_grad=True)

    def image():
        return (scale * torch.nn.functional.softplus(raw) if non_negative
                else scale * raw)

    def loss_fn():
        x = image()
        resid = torch.sparse.mm(A, x.T).T - b
        data = torch.mean(w * resid * resid)
        if tv_weight <= 0:
            return data
        v = x.reshape(n_bins, size, size)
        tv = ((v[:, 1:, :] - v[:, :-1, :]).abs().sum()
              + (v[:, :, 1:] - v[:, :, :-1]).abs().sum())
        return data + tv_weight * tv / (n_bins * n_vox)

    info = fit([raw], loss_fn, steps=steps, lr=lr, optimizer=optimizer)

    with torch.no_grad():
        x = image()
        resid = float(
            torch.linalg.matrix_norm(torch.sparse.mm(A, x.T).T - b)
            / torch.clamp(torch.linalg.matrix_norm(b), min=1e-30))
    log.info("TV: %d steps, weight %.3g, loss %.4g, relative residual %.4g",
             steps, tv_weight, float(info["loss"]), resid)

    img = x.detach().cpu().numpy().reshape(n_bins, size, size)
    if apply_sign:
        img = img * RECON_SIGN
    return Reconstruction(
        intensity=img, variance=None, bin_shape=stack.bin_shape,
        channel=stack.channel, limits=stack.limits,
        sign_applied=RECON_SIGN if apply_sign else 1.0,
    )
