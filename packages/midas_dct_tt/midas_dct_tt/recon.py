"""Shape reconstruction from a TT sinogram: classical baseline + differentiable.

Phase 2 of ``implementation_plan.md``.

Two arms, deliberately:

* :func:`sirt` -- the classical algebraic baseline (SIRT), which is what the
  field actually uses. It is here to be *beaten fairly*, or not beaten, and
  either way to be reported.
* :func:`reconstruct_differentiable` -- gradient recovery of ``chi`` through the
  Phase-1 forward model, optimised with :func:`midas_invert.fit`.

The adjoint
-----------
Iterative tomography lives or dies on ``A`` and ``A^T`` being an exact
transpose pair. A hand-written back-projector that does not exactly adjoint the
forward projector is the classic silent bug: the iteration still converges, to
the wrong thing. So :func:`backproject` does not implement a back-projector at
all -- it differentiates ``<A x, y>`` with respect to ``x``, which *is* ``A^T y``
by definition for a linear ``A``. Correct by construction, and
``tests/test_recon.py`` still runs the dot-product test on it.

Geometry note
-------------
The reconstruction grid lives in the **sample frame**, where the tomographic axis
is the grain's own ``G`` direction (the alignment maps ``G_sample`` onto the lab
axis, so its preimage is fixed). The missing cone is therefore a cone about
``G_sample``, and that is the direction along which a reconstructed sphere
elongates -- see :func:`axial_elongation`.
"""
from __future__ import annotations

import torch
from midas_invert import fit

from .project import project_rays_to_plane

__all__ = [
    "axial_elongation",
    "backproject",
    "dice",
    "forward_operator",
    "iou",
    "reconstruct_differentiable",
    "sirt",
    "total_variation",
]


def forward_operator(positions, alignment, psi_deg, detector, *, voxel_volume_um3):
    """Build the linear projector ``A: chi (N,) -> sinogram (S, n_u, n_v)``.

    Rotations are precomputed once for the whole scan. The returned closure is
    linear in ``chi`` (no sigmoid, no acceptance) -- that linearity is what SIRT
    and the adjoint rely on, so the deformation-aware path in
    :mod:`midas_dct_tt.forward` is deliberately *not* used here.
    """
    normal = alignment.beam_direction()
    angles = psi_deg if isinstance(psi_deg, torch.Tensor) else torch.as_tensor(psi_deg)
    rots = [alignment.sample_rotation(float(a)) for a in angles]
    kw = dict(
        normal=normal,
        distance_um=detector.distance_um,
        voxel_volume_um3=voxel_volume_um3,
        pixel_um=detector.pixel_um,
        detector_shape=detector.shape,
        center_px=detector.center_px,
    )

    def A(chi):
        return torch.stack(
            [
                project_rays_to_plane(
                    positions @ R.to(device=chi.device, dtype=chi.dtype).transpose(-1, -2),
                    chi, normal, **kw,
                )
                for R in rots
            ],
            dim=0,
        )

    return A


def backproject(A, y, n_voxels, *, device=None, dtype=torch.float64):
    """``A^T y``, obtained by differentiating ``<A x, y>`` -- exact by construction.

    For a linear operator this is the transpose, not an approximation to it. See
    the module docstring for why it is done this way.
    """
    x = torch.zeros(n_voxels, device=device, dtype=dtype, requires_grad=True)
    inner = (A(x) * y.to(device=x.device, dtype=x.dtype)).sum()
    (g,) = torch.autograd.grad(inner, x)
    return g


def sirt(
    sinogram,
    A,
    n_voxels,
    *,
    n_iter: int = 50,
    relaxation: float = 1.0,
    clamp=(0.0, 1.0),
    device=None,
    dtype=torch.float64,
    callback=None,
):
    """SIRT: the classical algebraic baseline.

    ``chi <- clamp(chi + lambda * C * A^T (R * (b - A chi)))`` with ``C`` the
    inverse column sums and ``R`` the inverse row sums, the standard
    simultaneous-iteration weighting. Row/column sums are computed once from
    ``A`` itself, so they cannot disagree with the operator.

    Returns the reconstructed ``chi`` ``(N,)``. Run with a fair iteration budget
    before comparing against :func:`reconstruct_differentiable` -- an
    under-relaxed, under-iterated baseline is not a baseline.
    """
    b = sinogram.to(device=device, dtype=dtype)
    ones_vox = torch.ones(n_voxels, device=device, dtype=dtype)
    with torch.no_grad():
        row_sum = A(ones_vox)                                  # A 1
    col_sum = backproject(A, torch.ones_like(b), n_voxels, device=device, dtype=dtype)

    R = torch.where(row_sum > 0, 1.0 / row_sum, torch.zeros_like(row_sum))
    C = torch.where(col_sum > 0, 1.0 / col_sum, torch.zeros_like(col_sum))

    chi = torch.zeros(n_voxels, device=device, dtype=dtype)
    for it in range(n_iter):
        with torch.no_grad():
            residual = b - A(chi)
        update = backproject(A, R * residual, n_voxels, device=device, dtype=dtype)
        chi = chi + relaxation * C * update
        if clamp is not None:
            chi = chi.clamp(clamp[0], clamp[1])
        if callback is not None:
            callback(it, float(torch.linalg.vector_norm(residual)))
    return chi


def total_variation(chi, shape):
    """Isotropic total variation of ``chi`` on a regular ``(nx, ny, nz)`` grid.

    The edge-preserving prior for shape recovery: penalises area, not curvature,
    so it tolerates facets while suppressing the streaks a missing cone produces.
    """
    v = chi.reshape(shape)
    dx = v[1:, :, :] - v[:-1, :, :]
    dy = v[:, 1:, :] - v[:, :-1, :]
    dz = v[:, :, 1:] - v[:, :, :-1]
    return dx.abs().sum() + dy.abs().sum() + dz.abs().sum()


def reconstruct_differentiable(
    sinogram,
    A,
    n_voxels,
    *,
    shape=None,
    tv_weight: float = 0.0,
    steps: int = 300,
    lr: float = 0.2,
    optimizer: str = "adam",
    init_logit: float = -2.0,
    device=None,
    dtype=torch.float64,
):
    """Recover ``chi`` by gradient descent through the forward model.

    ``chi = sigmoid(logits)`` keeps the result in ``[0, 1]`` under any optimiser
    with no projection step, and the optional TV prior acts on ``chi``.
    Optimisation is delegated to :func:`midas_invert.fit`.

    Returns ``(chi, info)`` where ``info`` is that function's result dict.

    ``fit`` defaults to a cosine ``lr`` schedule because at a fixed rate Adam can
    end a run worse than a point it already visited.  **That does not happen
    here** and the schedule is switched off: measured on a planted sphere,
    ``info["final_over_min"]`` is 0.0 at 200 and 400 steps, with and without a
    TV prior -- this fit is *truncated*, not oscillating (``tail_improvement``
    0.24-0.38, i.e. still descending steeply when the budget runs out).
    Annealing on top of that only halves the effective rate, and it costs
    real accuracy: dice 0.973 -> 0.916 at ``steps=200, lr=0.4``.  ``return_best``
    is left on -- it is free, and bit-identical here for exactly the reason
    above.  Raise ``steps`` before reaching for a schedule; ``info["settled"]``
    is False for this fit and is telling the truth.
    """
    b = sinogram.to(device=device, dtype=dtype)
    logits = torch.full((n_voxels,), float(init_logit),
                        device=device, dtype=dtype, requires_grad=True)

    def loss_fn():
        chi = torch.sigmoid(logits)
        resid = A(chi) - b
        loss = (resid ** 2).sum()
        if tv_weight > 0.0:
            if shape is None:
                raise ValueError("tv_weight > 0 needs the grid shape for finite differences")
            loss = loss + tv_weight * total_variation(chi, shape)
        return loss

    info = fit([logits], loss_fn, steps=steps, lr=lr, optimizer=optimizer,
               lr_schedule="none", log_every=1)
    return torch.sigmoid(logits.detach()), info


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------
def dice(a, b, *, threshold: float = 0.5) -> float:
    """Dice coefficient of two occupancy fields thresholded at ``threshold``.

    ``2|A n B| / (|A| + |B|)``. 1 is perfect. Thresholded rather than continuous
    because "did we recover the grain's shape" is a question about the segmented
    volume, and a continuous overlap would flatter a blurry reconstruction.
    """
    x = (a.detach() > threshold)
    y = (b.detach() > threshold)
    denom = float(x.sum() + y.sum())
    if denom == 0.0:
        return 1.0
    return float(2.0 * (x & y).sum()) / denom


def iou(a, b, *, threshold: float = 0.5) -> float:
    """Intersection over union of two thresholded occupancy fields."""
    x = (a.detach() > threshold)
    y = (b.detach() > threshold)
    union = float((x | y).sum())
    if union == 0.0:
        return 1.0
    return float((x & y).sum()) / union


def axial_elongation(chi, positions, axis) -> float:
    """RMS extent along ``axis`` divided by RMS extent transverse to it.

    The missing-cone observable. A planted sphere has 1.0 by symmetry, so any
    excess in a reconstruction is the cone's doing -- which makes this a
    falsifiable prediction rather than a qualitative "it looks smeared".

    ``axis`` should be the tomographic axis expressed in the **sample** frame,
    i.e. the grain's ``G`` direction (see the module docstring).
    """
    w = chi.detach()
    total = float(w.sum())
    if total <= 0.0:
        raise ValueError("empty reconstruction: no mass to measure")
    p = positions.to(dtype=w.dtype, device=w.device)
    a = torch.as_tensor(axis, dtype=w.dtype, device=w.device)
    a = a / torch.linalg.vector_norm(a)

    centroid = (w.unsqueeze(-1) * p).sum(dim=0) / total
    d = p - centroid
    along = d @ a
    perp2 = (d * d).sum(dim=-1) - along ** 2

    rms_along = float((w * along ** 2).sum() / total) ** 0.5
    rms_perp = float((w * perp2).sum() / total / 2.0) ** 0.5    # 2 transverse dof
    return rms_along / rms_perp
