"""K×K subpixel-oversampled hard binning.

Bridges :class:`HardBinGeometry` (single sample per pixel) and v1's
exact polygon-area kernel (continuous).  Each pixel is split into
``K × K`` subpixels; each subpixel contributes ``1/K²`` of the pixel's
intensity to its single nearest bin.  As ``K`` grows the result
approaches v1's ``floor`` mode integration evaluated at full subpixel
resolution.

Trade-offs:

- ``K = 1`` ⇔ :class:`HardBinGeometry` (one sample, one bin).
- ``K = 2`` (4 subpixels per pixel) closes ~75% of the bin-edge
  quantisation error of K=1 at 4× the integrate cost.
- ``K = 4`` (16 subpixels) closes ~94% at 16× the cost.
- v1 polygon-area kernel: exact (within numerical precision) at
  ~50× the cost of K=1.

When to use:

- ``K=1`` (HardBinGeometry): max throughput, fixed geometry.
- ``K=2..4``: better than hard-bin, still pure torch, no numba.
- v1 ``build_map`` + ``integrate(mode='floor')``: exact, but routes
  through numba.

Not differentiable in geometry — same caveat as :class:`HardBinGeometry`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

import numpy as np

from ..forward import pixel_to_REta_from_spec
from ..spec import IntegrationSpec
from .mask import normalise_mask
from .trans_opt import apply_trans_opt_forward, needs_trans_opt


def _rect_subpixel_offsets(K: int, *, dtype, device):
    """Return K² (dy, dz) offset tensors on a regular K×K grid inside
    the unit-square pixel cell.

    Reproduces the historical SubpixelBinGeometry sampling pattern
    (centred grid in ±0.5).
    """
    if K == 1:
        z = torch.zeros((), dtype=dtype, device=device)
        return [(z, z)]
    step = 1.0 / K
    offs = (torch.arange(K, dtype=dtype, device=device) + 0.5) * step - 0.5
    out = []
    for dy in offs:
        for dz in offs:
            out.append((dy, dz))
    return out


def _hex_cell_subpixel_offsets(K: int, *, dtype, device):
    """Return K² (dy, dz) offset tensors distributed inside the hex
    unit cell in pixel-index space.

    Hex cell vertices: (±1/2, ±1/3) and (0, ±2/3).  Cell area = 1
    (matches the cartesian unit square so the 1/K² intensity weighting
    in :func:`integrate_subpixel` carries over unchanged).  Offsets are
    laid out on a stratified raster of the bounding box and the first
    K² inside-hex points are kept; layout is deterministic so the
    geometry is reproducible.
    """
    if K == 1:
        z = torch.zeros((), dtype=dtype, device=device)
        return [(z, z)]
    target = K * K
    M = max(K, int(np.ceil(K * np.sqrt(4.0 / 3.0))))
    accepted: list = []
    while len(accepted) < target:
        accepted = []
        ys = np.linspace(-0.5, 0.5, M + 2)[1:-1]            # interior centres
        zs = np.linspace(-2.0 / 3.0, 2.0 / 3.0, M + 2)[1:-1]
        for z in zs:
            for y in ys:
                if abs(y) <= 0.5 and abs(z) <= 2.0 / 3.0 - (2.0 / 3.0) * abs(y):
                    accepted.append((float(y), float(z)))
        M += 1
    accepted = accepted[:target]
    out = []
    for (y, z) in accepted:
        out.append((
            torch.tensor(y, dtype=dtype, device=device),
            torch.tensor(z, dtype=dtype, device=device),
        ))
    return out


@dataclass
class SubpixelBinGeometry:
    """K×K-oversampled hard-bin geometry.

    Fields:
        flat_bin   : (K*K, n_pix) long — bin index for each subpixel of each pixel
        valid      : (K*K, n_pix) bool — subpixels that landed inside a bin
        K          : oversampling factor (1, 2, 3, …)
        n_r, n_eta : ints
        n_pixels_y, n_pixels_z : detector dimensions
        trans_opt  : list of ImTransOpt op codes
    """
    flat_bin: torch.Tensor
    valid: torch.Tensor
    K: int
    n_r: int
    n_eta: int
    n_pixels_y: int
    n_pixels_z: int
    trans_opt: list = None        # type: ignore[assignment]

    @classmethod
    def from_spec(cls, spec: IntegrationSpec, K: int = 2,
                   *, mask: Optional[np.ndarray] = None,
                   pixel_shape: str = "rect") -> "SubpixelBinGeometry":
        """Build the geometry by sampling each pixel at K×K subpixel
        offsets centred on a regular sub-grid (offsets in [-0.5+0.5/K,
        0.5-0.5/K]).

        Parameters
        ----------
        spec :
            v2 :class:`IntegrationSpec`. The lattice mapping
            (``cartesian`` / ``hex_offset_y``) is read from the spec.
        K :
            Oversampling factor (1, 2, …). Number of subpixel samples per
            pixel is ``K²`` for ``pixel_shape='rect'`` (regular grid in
            index space) and **same** for ``pixel_shape='hexagon'`` (the
            ``K²`` samples are arranged on a quasi-regular grid clipped
            to the hex cell — opt-in shape-aware splatting).
        mask :
            Optional 2D mask marking pixels to skip.
        pixel_shape :
            ``'rect'`` (default) — samples on a regular K×K grid in
            pixel-index space (±0.5 range). Bit-identical to prior
            behaviour for cartesian lattices.
            ``'hexagon'`` — opt-in, only valid when
            ``spec.lattice='hex_offset_y'``. Each pixel's K² samples are
            arranged on the K×K grid then rejected/reflected to lie
            inside the regular hex cell of apothem ``a``. The samples
            still travel through :func:`pixel_to_REta_from_spec` (which
            applies the hex centroid mapping); the difference is that
            ``hexagon`` confines the samples to the *physical* hex
            footprint instead of the index-unit square.
        """
        if K < 1:
            raise ValueError(f"K must be >= 1, got {K}")
        if pixel_shape not in ("rect", "hexagon"):
            raise ValueError(
                f"pixel_shape must be 'rect' or 'hexagon', got {pixel_shape!r}"
            )
        spec.validate()
        lattice = getattr(spec, "lattice", "cartesian")
        if pixel_shape == "hexagon" and lattice != "hex_offset_y":
            raise ValueError(
                f"pixel_shape='hexagon' requires spec.lattice='hex_offset_y'; "
                f"got lattice={lattice!r}"
            )
        NY, NZ = spec.NrPixelsY, spec.NrPixelsZ
        dt, dev = spec.dtype(), spec.device()
        mask_np = normalise_mask(mask, NrPixelsY=NY, NrPixelsZ=NZ)
        mask_flat_t = (None if mask_np is None
                       else torch.from_numpy(mask_np.reshape(-1)))

        if pixel_shape == "hexagon":
            offsets = _hex_cell_subpixel_offsets(K, dtype=dt, device=dev)
        else:
            offsets = _rect_subpixel_offsets(K, dtype=dt, device=dev)

        # Pixel-grid coords
        ys = torch.arange(NY, dtype=dt, device=dev)
        zs = torch.arange(NZ, dtype=dt, device=dev)
        Z, Y = torch.meshgrid(zs, ys, indexing="ij")

        n_r, n_eta = spec.n_r_bins, spec.n_eta_bins
        with torch.no_grad():
            flat_bins = []
            valids = []
            for dy, dz in offsets:
                out = pixel_to_REta_from_spec(Y + dy, Z + dz, spec)
                R_flat = out.R_px.reshape(-1)
                Eta_flat = out.eta_deg.reshape(-1)
                rb = ((R_flat - spec.RMin) / spec.RBinSize).floor().long()
                eb = ((Eta_flat - spec.EtaMin) / spec.EtaBinSize).floor().long()
                valid = (rb >= 0) & (rb < n_r) & (eb >= 0) & (eb < n_eta)
                if mask_flat_t is not None:
                    valid = valid & ~mask_flat_t
                rb_c = rb.clamp(0, n_r - 1)
                eb_c = eb.clamp(0, n_eta - 1)
                flat_bins.append(eb_c * n_r + rb_c)
                valids.append(valid)
        flat_bin = torch.stack(flat_bins)            # (n_samples, n_pix)
        valid_t  = torch.stack(valids)                # (n_samples, n_pix)

        return cls(
            flat_bin=flat_bin, valid=valid_t,
            K=int(K),
            n_r=n_r, n_eta=n_eta,
            n_pixels_y=NY, n_pixels_z=NZ,
            trans_opt=list(spec.TransOpt),
        )

    @property
    def device(self) -> torch.device:
        return self.flat_bin.device

    @property
    def n_subpixels(self) -> int:
        return self.K * self.K


def integrate_subpixel(
    image: torch.Tensor,
    geom: SubpixelBinGeometry,
    *,
    apply_trans_opt: bool = True,
    normalize: bool = True,
) -> torch.Tensor:
    """K×K-oversampled hard-bin integrate. Returns ``(n_eta, n_r)``.

    Each pixel contributes ``image[pix] / K²`` to each of its K²
    subpixels' bins (when ``normalize=False``); when ``normalize=True``
    divides by per-bin pixel-equivalent count so the output is mean
    intensity per bin (matches v1's ``Normalize=1``).
    """
    if image.shape != (geom.n_pixels_z, geom.n_pixels_y):
        raise ValueError(
            f"image shape {tuple(image.shape)} does not match "
            f"geometry ({geom.n_pixels_z}, {geom.n_pixels_y})"
        )
    if apply_trans_opt and geom.trans_opt and needs_trans_opt(geom.trans_opt):
        image = apply_trans_opt_forward(
            image, geom.trans_opt,
            NrPixelsY=geom.n_pixels_y, NrPixelsZ=geom.n_pixels_z,
        )
    img_flat = image.reshape(-1).to(torch.float64)
    n_bins = geom.n_eta * geom.n_r
    inv_K2 = 1.0 / (geom.K * geom.K)

    sums = torch.zeros(n_bins, dtype=img_flat.dtype, device=img_flat.device)
    counts = torch.zeros(n_bins, dtype=img_flat.dtype, device=img_flat.device)

    contrib = img_flat * inv_K2                              # per subpixel
    one_contrib = torch.full_like(img_flat, inv_K2)
    for k in range(geom.K * geom.K):
        v = geom.valid[k]
        idx = geom.flat_bin[k][v]
        sums = sums.index_add(0, idx, contrib[v])
        counts = counts.index_add(0, idx, one_contrib[v])

    if not normalize:
        return sums.reshape(geom.n_eta, geom.n_r)
    out = sums / counts.clamp(min=1e-12)
    return out.reshape(geom.n_eta, geom.n_r)


def integrate_subpixel_batch(
    images: torch.Tensor,
    geom: SubpixelBinGeometry,
    *,
    apply_trans_opt: bool = True,
    normalize: bool = True,
) -> torch.Tensor:
    """Batched variant. ``(N, NrPixelsZ, NrPixelsY)`` → ``(N, n_eta, n_r)``."""
    if images.ndim != 3 or images.shape[1:] != (geom.n_pixels_z,
                                                  geom.n_pixels_y):
        raise ValueError(
            f"images shape {tuple(images.shape)} does not match "
            f"(N, {geom.n_pixels_z}, {geom.n_pixels_y})"
        )
    if apply_trans_opt and geom.trans_opt and needs_trans_opt(geom.trans_opt):
        images = torch.stack([
            apply_trans_opt_forward(
                images[i], geom.trans_opt,
                NrPixelsY=geom.n_pixels_y, NrPixelsZ=geom.n_pixels_z,
            )
            for i in range(images.shape[0])
        ])
    n_imgs = images.shape[0]
    n_bins = geom.n_eta * geom.n_r
    img_flat = images.reshape(n_imgs, -1).to(torch.float64)  # (N, n_pix)
    inv_K2 = 1.0 / (geom.K * geom.K)

    sums = torch.zeros(n_imgs, n_bins, dtype=img_flat.dtype,
                        device=img_flat.device)
    counts = torch.zeros(n_bins, dtype=img_flat.dtype, device=img_flat.device)

    for k in range(geom.K * geom.K):
        v = geom.valid[k]
        idx = geom.flat_bin[k][v]
        contrib = img_flat[:, v] * inv_K2
        sums = sums.index_add(1, idx, contrib)
        ones = torch.full((int(v.sum()),), inv_K2,
                           dtype=img_flat.dtype, device=img_flat.device)
        counts = counts.index_add(0, idx, ones)

    if not normalize:
        return sums.reshape(n_imgs, geom.n_eta, geom.n_r)
    out = sums / counts.clamp(min=1e-12).unsqueeze(0)
    return out.reshape(n_imgs, geom.n_eta, geom.n_r)


__all__ = ["SubpixelBinGeometry", "integrate_subpixel",
           "integrate_subpixel_batch"]
