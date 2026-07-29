"""Physics-forward orientation + mosaicity fitting for DFXM mosaicity scans.

Moment analysis (per-pixel centre of mass) and phenomenological Gaussian fits both
report the *measured* orientation spread, which is the intrinsic sample mosaicity
convolved with the instrument resolution. This module instead fits the physical
forward model: a local orientation with an intrinsic mosaic covariance, convolved with
the (known/calibrated) instrument resolution, evaluated on the motor grid. Fitting it

  1. **deconvolves the instrument resolution** -> the recovered mosaic is the intrinsic
     sample property, not the instrument-broadened one (moments cannot do this);
  2. supports **multiple orientation components** (sub-grains) per pixel;
  3. is **spatially regularizable** -- all pixels can be fit jointly with a smoothness
     prior on the orientation field, turning an ill-posed per-pixel fit into a
     well-posed global inversion (moments are strictly per-pixel);
  4. is fully **differentiable** and provides per-pixel uncertainty.

For Gaussian intrinsic mosaic and Gaussian resolution the convolution is Gaussian with
covariance ``Sigma_total = Sigma_mosaic + Sigma_res``; the fit recovers
``Sigma_mosaic`` by construction. The peak *position* gives the local orientation.

Everything is torch-differentiable and device-portable.
"""
from __future__ import annotations

import torch

from midas_invert.optimize import fit as _invert_fit  # noqa: F401 (available for callers)


def _spd_from_cholesky(l11, l21, l22):
    """Lower-Cholesky (l11,l21,l22) -> SPD 2x2 covariance (batched)."""
    L11 = torch.nn.functional.softplus(l11)
    L22 = torch.nn.functional.softplus(l22)
    s00 = L11 ** 2
    s01 = L11 * l21
    s11 = l21 ** 2 + L22 ** 2
    return s00, s01, s11  # entries of Sigma = L L^T


def _inv2(a, b, d):
    """Inverse of SPD 2x2 [[a,b],[b,d]] -> (ia,ib,id_), det."""
    det = a * d - b * b + 1e-30
    return d / det, -b / det, a / det, det


def moment_orientation(data, chi, phi):
    """Per-pixel first-moment (centre-of-mass) orientation -- the baseline estimator.

    ``data`` ``(P, M)`` non-negative intensities, ``chi``/``phi`` ``(M,)`` motor coords.
    Returns ``(P, 2)``.
    """
    w = data.clamp_min(0)
    wsum = w.sum(-1, keepdim=True) + 1e-30
    return torch.stack([(w * chi).sum(-1) / wsum.squeeze(-1),
                        (w * phi).sum(-1) / wsum.squeeze(-1)], dim=-1)


def fit_orientation_mosaicity(
    data: torch.Tensor,
    chi: torch.Tensor,
    phi: torch.Tensor,
    res_cov,
    *,
    n_components: int = 1,
    lambda_smooth: float = 0.0,
    shape=None,
    steps: int = 500,
    lr: float = 0.02,
    max_offset: float = 1.5,
) -> dict:
    """Fit local orientation + intrinsic mosaic from a DFXM mosaicity scan.

    Parameters
    ----------
    data : (P, M) tensor
        Per-pixel intensities over ``M`` motor settings (normalise externally or not;
        each pixel is peak-normalised internally).
    chi, phi : (M,) tensor
        Motor coordinates of the two rocking axes.
    res_cov : (2, 2) array
        Instrument resolution covariance (calibrated, e.g. from a near-perfect crystal),
        which is deconvolved from the fitted total width.
    n_components : int
        Number of orientation components per pixel (>=2 resolves sub-grains).
    lambda_smooth : float
        Spatial curvature-smoothness weight on the (main-component) orientation field;
        requires ``shape=(ny,nx)``. ``0`` -> independent per-pixel fits.
    max_offset : float
        Bound (deg) on secondary components relative to the main -- prevents runaway.

    Returns
    -------
    dict with ``orientation`` (P, K, 2), ``mosaic_cov`` (P, K, 2, 2) *intrinsic*
    (resolution-deconvolved), ``amplitude`` (P, K), ``background`` (P,), ``loss``,
    and ``orientation_std`` (P, 2) a per-pixel uncertainty on the main orientation.
    """
    device, dtype = data.device, data.dtype
    P, M = data.shape
    K = n_components
    D = data / (data.amax(-1, keepdim=True) + 1e-30)
    R = torch.as_tensor(res_cov, device=device, dtype=dtype)          # (2,2)
    chi = chi.to(device, dtype); phi = phi.to(device, dtype)

    # init main component at the per-pixel argmax mode; moment as reference
    am = D.argmax(-1)
    c0 = chi[am].clone(); p0 = phi[am].clone()

    # parameters (batched over pixels, components)
    cmain = c0.clone().requires_grad_(True)
    pmain = p0.clone().requires_grad_(True)
    off_c = torch.zeros(P, K, device=device, dtype=dtype, requires_grad=True)  # secondary offsets
    off_p = torch.zeros(P, K, device=device, dtype=dtype, requires_grad=True)
    amp = torch.zeros(P, K, device=device, dtype=dtype, requires_grad=True)
    amp.data[:, 0] = 2.0                                              # main dominant
    if K > 1:
        amp.data[:, 1:] = -1.5
    l11 = torch.full((P, K), -1.5, device=device, dtype=dtype, requires_grad=True)
    l21 = torch.zeros(P, K, device=device, dtype=dtype, requires_grad=True)
    l22 = torch.full((P, K), -1.5, device=device, dtype=dtype, requires_grad=True)
    bg = torch.full((P,), -2.0, device=device, dtype=dtype, requires_grad=True)
    params = [cmain, pmain, off_c, off_p, amp, l11, l21, l22, bg]
    opt = torch.optim.Adam(params, lr=lr)

    def component_centers():
        cc = cmain[:, None] + max_offset * torch.tanh(off_c)         # (P,K)
        pp = pmain[:, None] + max_offset * torch.tanh(off_p)
        cc = torch.cat([cmain[:, None], cc[:, 1:]], dim=1) if K > 1 else cmain[:, None]
        pp = torch.cat([pmain[:, None], pp[:, 1:]], dim=1) if K > 1 else pmain[:, None]
        return cc, pp

    ny_nx = shape
    for _ in range(steps):
        opt.zero_grad()
        cc, pp = component_centers()
        A = torch.sigmoid(amp)                                        # (P,K)
        s00, s01, s11 = _spd_from_cholesky(l11, l21, l22)             # intrinsic mosaic (P,K)
        # total = mosaic + resolution
        t00 = s00 + R[0, 0]; t01 = s01 + R[0, 1]; t11 = s11 + R[1, 1]
        ia, ib, idd, _ = _inv2(t00, t01, t11)                        # (P,K)
        dc = chi[None, None, :] - cc[:, :, None]                     # (P,K,M)
        dp = phi[None, None, :] - pp[:, :, None]
        q = ia[:, :, None] * dc**2 + 2 * ib[:, :, None] * dc * dp + idd[:, :, None] * dp**2
        model = (A[:, :, None] * torch.exp(-0.5 * q)).sum(1) + torch.sigmoid(bg)[:, None] * 0.1
        data_loss = ((model - D) ** 2).mean()
        loss = data_loss
        if lambda_smooth > 0 and ny_nx is not None:
            ny, nx = ny_nx
            cf = cmain.reshape(ny, nx); pf = pmain.reshape(ny, nx)
            def curv(f):
                c = torch.zeros((), device=device, dtype=dtype)
                if nx > 2:
                    c = c + (f[:, 2:] - 2 * f[:, 1:-1] + f[:, :-2]).pow(2).mean()
                if ny > 2:
                    c = c + (f[2:] - 2 * f[1:-1] + f[:-2]).pow(2).mean()
                return c
            loss = loss + lambda_smooth * (curv(cf) + curv(pf))
        loss.backward(); opt.step()

    with torch.no_grad():
        cc, pp = component_centers()
        A = torch.sigmoid(amp)
        s00, s01, s11 = _spd_from_cholesky(l11, l21, l22)
        orientation = torch.stack([cc, pp], dim=-1)                  # (P,K,2)
        mosaic_cov = torch.stack([torch.stack([s00, s01], -1),
                                  torch.stack([s01, s11], -1)], -2)  # (P,K,2,2) intrinsic
        # crude per-pixel orientation uncertainty ~ total width / sqrt(SNR-ish signal)
        tot_w = torch.sqrt(s00 + R[0, 0] + s11 + R[1, 1])[:, 0]
        signal = D.sum(-1).clamp_min(1.0)
        ostd = (tot_w / torch.sqrt(signal))[:, None].expand(P, 2)
    return {
        "orientation": orientation, "mosaic_cov": mosaic_cov,
        "amplitude": A, "background": torch.sigmoid(bg),
        "loss": float(loss.detach()), "orientation_std": ostd,
        "moment": moment_orientation(data, chi, phi),
    }
