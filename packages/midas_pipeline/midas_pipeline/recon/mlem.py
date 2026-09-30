"""MLEM / OS-EM sinogram reconstruction for pf-HEDM.

``forward_project`` / ``back_project`` are the projectors ported from
``utils/mlem_recon.py`` (numpy and torch, dispatched on input type).

``mlem_recon`` / ``osem_recon`` are NOT the legacy recipe any more: they use
measured zeros, a padded reconstruction grid (interior tomography) and an
exact adjoint projector pair. See the block comment above ``_ray_weights``
for what changed and the measurements behind it. They run in torch on the
input's device; ndarray input runs on CPU and returns an ndarray.
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np

try:  # torch is an optional runtime dependency of the package
    import torch
    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - torch is in pyproject deps
    torch = None  # type: ignore
    _TORCH_AVAILABLE = False


ArrayLike = Union[np.ndarray, "torch.Tensor"]


# ---------------------------------------------------------------------------
# NumPy projectors (lifted verbatim from utils/mlem_recon.py)
# ---------------------------------------------------------------------------


def _rotation_matrix(angle_rad: float) -> np.ndarray:
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([[c, -s], [s, c]])


def _forward_project_np(image: np.ndarray, angles_deg: np.ndarray) -> np.ndarray:
    N = image.shape[0]
    assert image.shape[1] == N, "Image must be square"
    n_thetas = len(angles_deg)
    sino = np.zeros((n_thetas, N), dtype=np.float64)

    center = (N - 1) / 2.0
    x = np.arange(N) - center

    for i, angle in enumerate(angles_deg):
        angle_rad = np.deg2rad(angle)
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        for j, t in enumerate(x):
            s_vals = x
            ray_x = t * cos_a - s_vals * sin_a + center
            ray_y = t * sin_a + s_vals * cos_a + center
            ix = np.floor(ray_x).astype(int)
            iy = np.floor(ray_y).astype(int)
            fx = ray_x - ix
            fy = ray_y - iy
            valid = (ix >= 0) & (ix < N - 1) & (iy >= 0) & (iy < N - 1)
            ix = np.clip(ix, 0, N - 2)
            iy = np.clip(iy, 0, N - 2)
            vals = (
                image[iy, ix] * (1 - fx) * (1 - fy)
                + image[iy, ix + 1] * fx * (1 - fy)
                + image[iy + 1, ix] * (1 - fx) * fy
                + image[iy + 1, ix + 1] * fx * fy
            )
            sino[i, j] = np.sum(vals * valid)
    return sino


def _back_project_np(sinogram: np.ndarray, angles_deg: np.ndarray, N: int) -> np.ndarray:
    M = sinogram.shape[1]
    image = np.zeros((N, N), dtype=np.float64)
    center_img = (N - 1) / 2.0
    center_det = (M - 1) / 2.0
    yy, xx = np.mgrid[0:N, 0:N]
    xx = xx.astype(np.float64) - center_img
    yy = yy.astype(np.float64) - center_img
    for i, angle in enumerate(angles_deg):
        angle_rad = np.deg2rad(angle)
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        t = xx * cos_a + yy * sin_a + center_det
        it = np.floor(t).astype(int)
        ft = t - it
        valid = (it >= 0) & (it < M - 1)
        it_safe = np.clip(it, 0, M - 2)
        vals = sinogram[i, it_safe] * (1 - ft) + sinogram[i, it_safe + 1] * ft
        image += vals * valid
    return image


# ---------------------------------------------------------------------------
# Torch projectors (vectorized, differentiable, device-portable)
# ---------------------------------------------------------------------------


def _forward_project_torch(image: "torch.Tensor", angles_deg: "torch.Tensor") -> "torch.Tensor":
    """Torch radon transform.

    Vectorized over angles & detector pixels. Operates on the image
    parameter's dtype/device. Bilinear interpolation matches the
    numpy path to within float precision; autograd-safe because we
    never index in-place.
    """
    if image.ndim != 2 or image.shape[0] != image.shape[1]:
        raise ValueError("Image must be 2-D and square")
    dtype = image.dtype
    device = image.device
    N = image.shape[0]
    angles_deg = angles_deg.to(device=device, dtype=dtype)
    n_thetas = angles_deg.shape[0]

    center = (N - 1) / 2.0
    coord = torch.arange(N, dtype=dtype, device=device) - center  # (N,)
    cos_a = torch.cos(torch.deg2rad(angles_deg))  # (T,)
    sin_a = torch.sin(torch.deg2rad(angles_deg))

    # Ray coords: (T, N_det, N_int) → world (ray_x, ray_y)
    t = coord.view(1, -1, 1)          # (1, N, 1)
    s = coord.view(1, 1, -1)          # (1, 1, N)
    ca = cos_a.view(-1, 1, 1)         # (T, 1, 1)
    sa = sin_a.view(-1, 1, 1)
    ray_x = t * ca - s * sa + center  # (T, N, N)
    ray_y = t * sa + s * ca + center

    # Bilinear interpolation
    ix0 = torch.floor(ray_x).long()
    iy0 = torch.floor(ray_y).long()
    fx = ray_x - ix0.to(dtype)
    fy = ray_y - iy0.to(dtype)

    in_bounds = (ix0 >= 0) & (ix0 < N - 1) & (iy0 >= 0) & (iy0 < N - 1)
    ix_safe = ix0.clamp(0, N - 2)
    iy_safe = iy0.clamp(0, N - 2)

    # Linear-index into a 1-D view of the image so we never index image with
    # out-of-bounds tensors (would error or produce nondeterministic backward).
    flat = image.reshape(-1)
    i00 = (iy_safe * N + ix_safe).reshape(-1)
    i01 = (iy_safe * N + (ix_safe + 1)).reshape(-1)
    i10 = ((iy_safe + 1) * N + ix_safe).reshape(-1)
    i11 = ((iy_safe + 1) * N + (ix_safe + 1)).reshape(-1)
    v00 = flat[i00].reshape(ix_safe.shape)
    v01 = flat[i01].reshape(ix_safe.shape)
    v10 = flat[i10].reshape(ix_safe.shape)
    v11 = flat[i11].reshape(ix_safe.shape)

    sample = (
        v00 * (1 - fx) * (1 - fy)
        + v01 * fx * (1 - fy)
        + v10 * (1 - fx) * fy
        + v11 * fx * fy
    )
    sample = sample * in_bounds.to(dtype)
    sino = sample.sum(dim=-1)  # (T, N)
    return sino


def _back_project_torch(sinogram: "torch.Tensor", angles_deg: "torch.Tensor", N: int) -> "torch.Tensor":
    dtype = sinogram.dtype
    device = sinogram.device
    M = sinogram.shape[1]
    angles_deg = angles_deg.to(device=device, dtype=dtype)
    center_img = (N - 1) / 2.0
    center_det = (M - 1) / 2.0

    grid = torch.arange(N, dtype=dtype, device=device)
    yy = grid.view(-1, 1).expand(N, N) - center_img
    xx = grid.view(1, -1).expand(N, N) - center_img

    cos_a = torch.cos(torch.deg2rad(angles_deg))  # (T,)
    sin_a = torch.sin(torch.deg2rad(angles_deg))

    image = torch.zeros((N, N), dtype=dtype, device=device)
    for i in range(angles_deg.shape[0]):
        t = xx * cos_a[i] + yy * sin_a[i] + center_det
        it0 = torch.floor(t).long()
        ft = t - it0.to(dtype)
        in_bounds = (it0 >= 0) & (it0 < M - 1)
        it_safe = it0.clamp(0, M - 2)
        row = sinogram[i]
        v_lo = row[it_safe]
        v_hi = row[(it_safe + 1).clamp(0, M - 1)]
        vals = v_lo * (1 - ft) + v_hi * ft
        image = image + vals * in_bounds.to(dtype)
    return image


# ---------------------------------------------------------------------------
# Public API: forward_project / back_project / mlem / osem (dispatch)
# ---------------------------------------------------------------------------


def forward_project(image: ArrayLike, angles_deg: ArrayLike) -> ArrayLike:
    """Radon transform.

    Tensor input → tensor output on input's device/dtype, autograd
    intact. ndarray input → ndarray output via the legacy NumPy path.
    """
    if _TORCH_AVAILABLE and isinstance(image, torch.Tensor):
        angles_t = (
            angles_deg
            if isinstance(angles_deg, torch.Tensor)
            else torch.as_tensor(angles_deg, dtype=image.dtype, device=image.device)
        )
        return _forward_project_torch(image, angles_t)
    return _forward_project_np(np.asarray(image), np.asarray(angles_deg))


def back_project(sinogram: ArrayLike, angles_deg: ArrayLike, N: int) -> ArrayLike:
    if _TORCH_AVAILABLE and isinstance(sinogram, torch.Tensor):
        angles_t = (
            angles_deg
            if isinstance(angles_deg, torch.Tensor)
            else torch.as_tensor(angles_deg, dtype=sinogram.dtype, device=sinogram.device)
        )
        return _back_project_torch(sinogram, angles_t, N)
    return _back_project_np(np.asarray(sinogram), np.asarray(angles_deg), N)


# ----- MLEM / OS-EM ---------------------------------------------------------
#
# What changed from the utils/mlem_recon.py port, and why (ESRF ma5608, 2026-09):
#
# 1. Measured zeros are data. The port built its sensitivity and its update
#    from the NON-ZERO cells only, so "this grain is not on this ray" never
#    constrained the image: 0.150 / 0.053 accuracy on a Voronoi phantom where
#    FBP scored 0.946 / 0.788. Rows with no signal at all are still dropped:
#    those are reflections not recorded at any scan position (a gap, a missed
#    peak), i.e. unmeasured, not measured-zero.
# 2. The image is reconstructed on a grid wider than the scanned field and
#    cropped back. Scanning 3DXRD is interior tomography: material outside the
#    scanned field still crosses the beam at some omega, so the sinogram holds
#    mass an image the width of the scan cannot represent. With the unpadded
#    grid, standard MLEM pumped that mass into the edge pixels and diverged to
#    inf on sinograms made by an independent projector (skimage radon); the
#    port's [0.1, 10] update clip only hid it.
# 3. The back-projector is the exact adjoint of the forward projector
#    (scatter-add of the same bilinear weights), which MLEM's fixed point
#    assumes. The public back_project() is a pixel-driven approximation.
# 4. The support threshold is relative to the peak sensitivity, not 1e-10.
#
# Which is better on real data DEPENDS ON THE DATA. On ESRF ma5608 alumina (dense,
# 180 deg, 0.3 um beam; Amendment 28, run once) phantoms favoured this MLEM over FBP
# (0.961 vs 0.936; 0.898 vs 0.734, three seeds each), but on the real layer its
# half-split was 0.789 vs FBP 0.864 and 17/204 grains kept most of their mass on
# the image border (FBP: 0); a per-row background term changed neither. On 20-ID-E
# Fe9Cr (sparse, 360 deg with a 16-deg missing wedge, 10 um beam; 2026-09-28) it is
# the reverse: half-split on the sample 0.771 vs FBP 0.559 and 0.731 vs 0.570,
# agreement with the per-voxel map 0.766 vs 0.669 and 0.762 vs 0.692. Run
# reconstruct with method="all" and read Recons/ReconQuality.json; see
# manuals/pf-hedm/phase-6-reconstruction.md.


def _ray_weights(N: int, M: int, angles_deg: "torch.Tensor"):
    """Bilinear ray weights for an N x N image and M detector bins.

    Same geometry as ``forward_project`` (which is the N == M case): detector
    bin t and ray sample s map to image column ``t cos - s sin`` and row
    ``t sin + s cos`` about the image centre. Returns 4 flat index tensors and
    4 weight tensors, each (T, M, N), with out-of-image samples weighted 0.
    """
    dtype, device = angles_deg.dtype, angles_deg.device
    c_img, c_det = (N - 1) / 2.0, (M - 1) / 2.0
    t = (torch.arange(M, dtype=dtype, device=device) - c_det).view(1, -1, 1)
    s = (torch.arange(N, dtype=dtype, device=device) - c_img).view(1, 1, -1)
    a = torch.deg2rad(angles_deg)
    ca, sa = torch.cos(a).view(-1, 1, 1), torch.sin(a).view(-1, 1, 1)
    rx = t * ca - s * sa + c_img
    ry = t * sa + s * ca + c_img
    ix0, iy0 = torch.floor(rx).long(), torch.floor(ry).long()
    fx, fy = rx - ix0.to(dtype), ry - iy0.to(dtype)
    inb = ((ix0 >= 0) & (ix0 < N - 1) & (iy0 >= 0) & (iy0 < N - 1)).to(dtype)
    ix, iy = ix0.clamp(0, N - 2), iy0.clamp(0, N - 2)
    idx = (iy * N + ix, iy * N + ix + 1, (iy + 1) * N + ix, (iy + 1) * N + ix + 1)
    w = ((1 - fx) * (1 - fy) * inb, fx * (1 - fy) * inb,
         (1 - fx) * fy * inb, fx * fy * inb)
    return idx, w


class _Projector:
    """Matched forward / adjoint pair for one angle set, N x N image, M bins.
    Weights are built once per chunk of angles and reused every iteration."""

    def __init__(self, angles_deg, N, M, chunk_elems=2e7):
        self.N, self.M = N, M
        per_angle = M * N
        step = max(1, int(chunk_elems // per_angle))
        self.chunks = []
        for i in range(0, angles_deg.shape[0], step):
            self.chunks.append((i, _ray_weights(N, M, angles_deg[i:i + step])))

    def forward(self, x):
        flat = x.reshape(-1)
        out = []
        for _, (idx, w) in self.chunks:
            out.append(sum(wk * flat[ik] for ik, wk in zip(idx, w)).sum(-1))
        return torch.cat(out, 0)

    def adjoint(self, r):
        img = torch.zeros(self.N * self.N, dtype=r.dtype, device=r.device)
        for i0, (idx, w) in self.chunks:
            rr = r[i0:i0 + idx[0].shape[0]].unsqueeze(-1)
            for ik, wk in zip(idx, w):
                img = img.index_add(0, ik.reshape(-1), (wk * rr).reshape(-1))
        return img.reshape(self.N, self.N)


def _em(sinogram, angles_deg, n_iter, n_subsets, init, pad, support_rel):
    """Shared MLEM (n_subsets == 1) / OS-EM core, torch in and out."""
    dtype, device = sinogram.dtype, sinogram.device
    T, M = sinogram.shape
    angles_deg = angles_deg.to(device=device, dtype=dtype)
    p = int(round(max(pad, 0.0) * M / 2.0))
    N = M + 2 * p
    measured = torch.nonzero((sinogram > 0).any(dim=1), as_tuple=False).reshape(-1)
    if measured.numel() == 0:
        return torch.zeros((M, M), dtype=dtype, device=device)
    y = sinogram.index_select(0, measured)
    ang = angles_deg.index_select(0, measured)
    subsets = [torch.arange(i, y.shape[0], n_subsets, device=device)
               for i in range(min(n_subsets, y.shape[0]))]
    projs = [_Projector(ang.index_select(0, s), N, M) for s in subsets]
    sens = [P.adjoint(torch.ones((s.numel(), M), dtype=dtype, device=device))
            for P, s in zip(projs, subsets)]
    total = sum(sens)
    support = total > support_rel * total.max()
    sens = [torch.where(support, sk.clamp_min(1e-12 * float(total.max())),
                        torch.ones_like(sk)) for sk in sens]
    if init is None:
        x = torch.ones((N, N), dtype=dtype, device=device)
    else:
        x = torch.zeros((N, N), dtype=dtype, device=device)
        x = x.index_put((torch.arange(p, p + M, device=device).view(-1, 1),
                         torch.arange(p, p + M, device=device).view(1, -1)),
                        init.to(device=device, dtype=dtype))
        x = torch.where(x > 0, x, torch.full_like(x, 1e-6))
    x = torch.where(support, x, torch.zeros_like(x))
    floor = 1e-9 * float(y.max())
    ys = [y.index_select(0, s) for s in subsets]
    for _ in range(n_iter):
        for P, yk, sk in zip(projs, ys, sens):
            ratio = yk / P.forward(x).clamp_min(floor)
            x = torch.where(support, x * P.adjoint(ratio) / sk, torch.zeros_like(x))
    x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return x[p:p + M, p:p + M]


def _run_em(sinogram, angles_deg, n_iter, n_subsets, init, pad, support_rel):
    if _TORCH_AVAILABLE and isinstance(sinogram, torch.Tensor):
        a = (angles_deg if isinstance(angles_deg, torch.Tensor)
             else torch.as_tensor(angles_deg, dtype=sinogram.dtype, device=sinogram.device))
        i = (init if init is None or isinstance(init, torch.Tensor)
             else torch.as_tensor(init, dtype=sinogram.dtype, device=sinogram.device))
        return _em(sinogram, a, n_iter, n_subsets, i, pad, support_rel)
    s = torch.as_tensor(np.asarray(sinogram, dtype=np.float64))
    a = torch.as_tensor(np.asarray(angles_deg, dtype=np.float64))
    i = None if init is None else torch.as_tensor(np.asarray(init, dtype=np.float64))
    return _em(s, a, n_iter, n_subsets, i, pad, support_rel).numpy()


def mlem_recon(
    sinogram: ArrayLike,
    angles_deg: ArrayLike,
    *,
    n_iter: int = 50,
    init: Optional[ArrayLike] = None,
    mask: Optional[np.ndarray] = None,
    eps: float = 1e-10,
    pad: float = 0.5,
    support_rel: float = 1e-3,
) -> ArrayLike:
    """MLEM reconstruction of one grain's sinogram.

    Parameters
    ----------
    sinogram : ndarray or torch.Tensor (T, M)
        Rows are reflections (omega), columns scan positions. A row that is
        zero everywhere is treated as unmeasured and dropped; zeros inside a
        measured row are data.
    angles_deg : (T,) omega per row.
    n_iter : iterations (early stopping is the only regulariser).
    init : optional (M, M) start image, same type as ``sinogram``.
    mask, eps : accepted for backward compatibility and ignored.
    pad : the image is reconstructed on an (M + 2p) grid, p = round(pad*M/2),
        and cropped to the central M x M. 0 disables padding (and reintroduces
        the interior-tomography divergence when material lies outside the
        scanned field).
    support_rel : pixels with sensitivity below this fraction of the peak are
        held at 0.

    Returns
    -------
    (M, M), same array type (and, for tensors, device) as ``sinogram``.
    Tensor inputs stay differentiable.
    """
    return _run_em(sinogram, angles_deg, n_iter, 1, init, pad, support_rel)


def osem_recon(
    sinogram: ArrayLike,
    angles_deg: ArrayLike,
    *,
    n_iter: int = 10,
    n_subsets: int = 4,
    init: Optional[ArrayLike] = None,
    eps: float = 1e-10,
    pad: float = 0.5,
    support_rel: float = 1e-3,
) -> ArrayLike:
    """Ordered-subsets EM: MLEM with the rows split into ``n_subsets``
    interleaved subsets, one multiplicative update per subset. Same
    conventions and parameters as :func:`mlem_recon`."""
    return _run_em(sinogram, angles_deg, n_iter, max(1, int(n_subsets)), init, pad,
                   support_rel)


# Back-compat aliases (mirrors legacy ``mlem_recon.mlem`` / ``.osem``)
mlem = mlem_recon
osem = osem_recon
