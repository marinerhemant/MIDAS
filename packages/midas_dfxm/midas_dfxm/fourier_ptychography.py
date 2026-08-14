"""Differentiable Fourier-ptychographic DFXM (FP-DFXM) reconstruction.

Recovers a **complex** object (amplitude + PHASE) beyond the single-image objective NA from a
stack of aperture-shifted DFXM images, by gradient descent through the FP forward -- one autodiff
graph over the object (and, optionally, the pupil aberrations). This is the phase-retrieval
*inverse* that midas otherwise lacks; it complements the intensity forward
(:func:`midas_dfxm.wave_imaging.dfxm_image_dynamical`), which is the *forward* microscope and is
NOT re-implemented here. FP gives quantitative phase (non-elastic domain walls, dislocation g.b
phase vortices) and super-resolution.

Forward (objective pupil = a low-pass in the diffracted-beam Fourier plane; aperture at ``s_j``):

    I_j(x) = | IFFT( P(k) * shift(FFT(O), s_j) ) |^2 .

The overlapping apertures synthesise a wider effective NA -> resolution past ``lambda/(2 NA)`` and
recovered phase. The gradient-based solve jointly fits the object (and pupil) with a Poisson/amplitude
loss and gives uncertainty via the Hessian -- beyond the iterative ePIE/HIO projection algorithms.
"""
from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple

import torch


def _cdtype(dtype: torch.dtype) -> torch.dtype:
    return torch.complex128 if dtype in (torch.float64,) else torch.complex64


def disk_pupil(grid_size: int, k_cutoff: float, *, device=None,
               dtype: torch.dtype = torch.float64) -> torch.Tensor:
    """Circular objective pupil (low-pass) of radius ``k_cutoff`` [cycles/pixel] in the
    Fourier plane, as a complex ``(grid_size, grid_size)`` mask (unshifted / fftfreq order)."""
    k = torch.fft.fftfreq(grid_size, device=device).to(dtype)
    KX, KY = torch.meshgrid(k, k, indexing="xy")
    return (torch.hypot(KX, KY) <= k_cutoff).to(_cdtype(dtype))


def aperture_grid(half: int, step_px: int) -> list[Tuple[int, int]]:
    """``(2*half+1)**2`` aperture shifts (Fourier-plane pixels) on a square grid with pitch
    ``step_px``. Overlapping apertures (``step_px`` < pupil diameter) are required for FP."""
    return [(i * step_px, j * step_px) for i in range(-half, half + 1)
            for j in range(-half, half + 1)]


def fp_forward(obj: torch.Tensor, pupil: torch.Tensor,
               shifts: Sequence[Tuple[int, int]]) -> torch.Tensor:
    """FP image stack ``(M, H, W)``: ``I_j = |IFFT(pupil * shift(FFT(obj), s_j))|**2``.

    Differentiable in ``obj`` and ``pupil``. ``shifts`` are ``(dx, dy)`` in Fourier-plane pixels.
    """
    Ohat = torch.fft.fft2(obj)
    imgs = []
    for (dx, dy) in shifts:
        sh = torch.roll(Ohat, shifts=(dy, dx), dims=(-2, -1))
        imgs.append(torch.fft.ifft2(pupil * sh).abs() ** 2)
    return torch.stack(imgs, 0)


def reconstruct_fp(images: torch.Tensor, pupil: torch.Tensor,
                   shifts: Sequence[Tuple[int, int]], *, iters: int = 400, lr: float = 0.05,
                   init: Optional[torch.Tensor] = None,
                   optimize_pupil_phase: bool = False, restarts: int = 1,
                   seed: Optional[int] = None,
                   return_loss: bool = False):
    """Recover the complex object from an FP image stack by Adam on the amplitude loss
    ``mean (sqrt(I_model) - sqrt(I_meas))^2``.

    Returns the recovered complex object (detached); with ``return_loss`` returns
    ``(object, final_loss)``. With ``optimize_pupil_phase`` a per-pixel pupil phase is
    co-fitted (aberration self-calibration) -- the joint inverse the differentiable
    formulation enables. Init defaults to the central-aperture image amplitude, zero phase.

    ``restarts`` -- escaping the AMPLITUDE stall (NOT a vortex-charge cure). The zero-phase
    init can stall in a poor local minimum (high loss) when aperture overlap is weak.
    ``restarts > 1`` runs extra solves from RANDOM-phase inits and keeps the lowest-loss
    one, which reliably reaches the amplitude global minimum (noise floor ~1e-5..1e-4 vs a
    stall ~1e-3). Restart 0 always uses the deterministic default/``init`` so ``restarts=1``
    reproduces the single-start result exactly; ``seed`` makes the random restarts reproducible.

    WHAT ``restarts`` DOES NOT FIX -- vortex-charge recovery (a /verify finding, 2026-08).
    Reaching the loss floor does NOT certify a recovered dislocation g.b winding. In the
    weak-overlap / under-resolved regime the floor is winding-DEGENERATE: independent restarts
    reach the same ~3.9e-5 loss with core windings scattered over {-1, 0, +1, +2}, and the
    reconstruction is littered with ~100 spurious noise-induced phase singularities, so no
    loss gate and no loop radius reliably returns the true charge (measured ~55-67% success
    across noise seeds). Vortex charge is trustworthy ONLY when aperture overlap is AMPLE /
    the problem is well conditioned -- there the single-start solve already yields a CLEAN
    reconstruction (exactly one singularity, correct winding, robust to noise). Recovery is
    governed by conditioning (overlap), not by the number of restarts. ``return_loss`` gates
    amplitude convergence, not charge correctness; to trust a winding, check the reconstruction
    is clean (e.g. a single phase singularity), not merely that the loss reached the floor.
    """
    sqrtI = images.clamp_min(0).sqrt()
    cdt = _cdtype(images.dtype)
    base = (sqrtI[len(shifts) // 2] if init is None else init).to(cdt)
    gen: Optional[torch.Generator] = None
    if seed is not None:
        gen = torch.Generator(device=base.device)
        gen.manual_seed(int(seed))

    def _solve(obj_init: torch.Tensor):
        obj = obj_init.clone().requires_grad_(True)
        params = [obj]
        ph = None
        if optimize_pupil_phase:
            ph = torch.zeros_like(pupil.real, requires_grad=True)
            params.append(ph)
        opt = torch.optim.Adam(params, lr=lr)

        def _pupil():
            return pupil * torch.exp(1j * ph.to(pupil.dtype)) if optimize_pupil_phase else pupil

        for _ in range(iters):
            opt.zero_grad()
            loss = ((fp_forward(obj, _pupil(), shifts).clamp_min(0).sqrt() - sqrtI) ** 2).mean()
            loss.backward()
            opt.step()
        with torch.no_grad():
            final = float(((fp_forward(obj, _pupil(), shifts).clamp_min(0).sqrt()
                            - sqrtI) ** 2).mean())
        return obj.detach(), final

    best_obj, best_loss = None, None
    for r in range(max(1, int(restarts))):
        if r == 0:
            obj_init = base                                   # deterministic, backward-compatible
        else:
            rnd = torch.rand(base.shape, generator=gen, device=base.device,
                             dtype=pupil.real.dtype)
            obj_init = (base.abs() * torch.exp(1j * (2 * math.pi * rnd).to(cdt))).to(cdt)
        obj, final = _solve(obj_init)
        if best_loss is None or final < best_loss:
            best_obj, best_loss = obj, final
    return (best_obj, best_loss) if return_loss else best_obj


def align_global_phase(rec: torch.Tensor, ref: torch.Tensor,
                       mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Remove the global phase ambiguity of an FP reconstruction relative to ``ref``
    (FP recovers the object only up to a constant phase)."""
    a, b = (rec, ref) if mask is None else (rec[mask], ref[mask])
    return rec * torch.exp(-1j * torch.angle((a * b.conj()).sum()))
