"""Write v2 integration results in the v1 / C binary formats.

The beamline chain downstream of ``IntegratorFitPeaksGPUStream`` reads three
files, and v2's own writers (CSV / xye / esg / HDF5) produce none of them:

``lineout.bin``              interleaved ``(R, I)`` float64 pairs, per frame
``lineout_simple_mean.bin``  same layout, unweighted per-η mean
``Int2D.bin``                ``(n_frames, n_r, n_eta)`` float64 cake

Two conventions have to be bridged, and both are easy to get wrong:

* v2's ``integrate_*`` return **(n_eta, n_r)**; v1 writes **(n_r, n_eta)**.
* v1 weights its 1-D profile by ``Σ areaWeight`` per bin. v2's hard/subpixel
  kernels weight by per-bin pixel-equivalent *count*, which is the same
  quantity up to the constant pixel area for a regular sub-grid, so it is the
  correct analogue — see :func:`bin_weights`.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import torch

__all__ = ["bin_weights", "profile_1d_v1", "r_axis_from_spec",
           "write_lineout_bin", "write_int2d_bin", "write_v1_outputs"]

#: Bins with less weight than this are treated as empty, matching v1's
#: ``AREA_THRESHOLD`` guard in ``integrate``/``profile_1d``.
WEIGHT_THRESHOLD = 1e-10


def _geom_device(geom) -> torch.device:
    """Where the geometry's tensors live. The weights must be built there."""
    for v in vars(geom).values():
        if torch.is_tensor(v):
            return v.device
        if isinstance(v, (list, tuple)) and v and torch.is_tensor(v[0]):
            return v[0].device
    return torch.device("cpu")


def bin_weights(geom, integrate_fn) -> torch.Tensor:
    """Per-bin weight, shape ``(n_eta, n_r)``.

    Integrating an image of ones with ``normalize=False`` yields exactly the
    per-bin accumulator the kernel divides by — the count of pixel-equivalents
    landing in each bin. It depends only on the geometry, never on the data, so
    compute it once per geometry and reuse it for every frame.
    """
    dev = _geom_device(geom)
    ones = torch.ones((geom.n_pixels_z, geom.n_pixels_y), dtype=torch.float64,
                      device=dev)
    # trans_opt is a property of the image, not the geometry; a uniform image
    # is invariant under it, and skipping it avoids a needless copy.
    return integrate_fn(ones, geom, normalize=False, apply_trans_opt=False)


def profile_1d_v1(int2d: torch.Tensor,
                  weights: Optional[torch.Tensor] = None,
                  *, mode: str = "area_weighted") -> np.ndarray:
    """Reduce a v2 ``(n_eta, n_r)`` cake to a ``(n_r,)`` profile.

    ``area_weighted`` reproduces v1's ``Σ(I·A) / Σ(A)``; ``simple_mean``
    reproduces its ``Σ(I where A>0) / count(A>0)``.
    """
    I = int2d.detach().cpu().to(torch.float64)
    if mode == "simple_mean":
        valid = (I != 0)
        n = valid.sum(dim=0).clamp(min=1)
        return (I.sum(dim=0) / n).cpu().numpy()
    if mode != "area_weighted":
        raise ValueError(f"unknown mode {mode!r}")
    if weights is None:
        raise ValueError("area_weighted needs weights; see bin_weights()")
    W = weights.detach().cpu().to(torch.float64)
    valid = W > WEIGHT_THRESHOLD
    num = (I * W * valid).sum(dim=0)
    den = (W * valid).sum(dim=0)
    out = torch.where(den > WEIGHT_THRESHOLD, num / den.clamp(min=WEIGHT_THRESHOLD),
                      torch.zeros_like(num))
    return out.cpu().numpy()


def r_axis_from_spec(spec) -> np.ndarray:
    """R bin centres in pixels, matching v1's ``r_axis``."""
    return spec.RMin + spec.RBinSize * (np.arange(spec.n_r_bins) + 0.5)


def write_lineout_bin(path: Path | str, r_axis: np.ndarray,
                      profiles: Sequence[np.ndarray]) -> Path:
    """Interleaved ``(R, I)`` float64 pairs, one block per frame."""
    path = Path(path)
    n_r = len(r_axis)
    buf = np.empty(len(profiles) * n_r * 2, dtype=np.float64)
    for i, prof in enumerate(profiles):
        if len(prof) != n_r:
            raise ValueError(f"profile {i} has {len(prof)} bins, expected {n_r}")
        block = buf[i * n_r * 2:(i + 1) * n_r * 2]
        block[0::2] = r_axis
        block[1::2] = prof
    path.write_bytes(buf.tobytes())
    return path


def write_int2d_bin(path: Path | str,
                    cakes: Sequence[torch.Tensor | np.ndarray]) -> Path:
    """``(n_frames, n_r, n_eta)`` float64 — transposing v2's (n_eta, n_r)."""
    path = Path(path)
    stack = []
    for c in cakes:
        arr = c.detach().cpu().numpy() if isinstance(c, torch.Tensor) else np.asarray(c)
        stack.append(np.ascontiguousarray(arr.T, dtype=np.float64))
    path.write_bytes(np.stack(stack).tobytes())
    return path


def write_v1_outputs(out_dir: Path | str,
                     cakes: Sequence[torch.Tensor],
                     *,
                     spec,
                     weights: Optional[torch.Tensor] = None,
                     write_2d: bool = True,
                     write_simple_mean: bool = True) -> list[Path]:
    """Write the three v1 files for a run. Returns the paths written."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    r_axis = r_axis_from_spec(spec)

    written: list[Path] = []
    if weights is not None:
        written.append(write_lineout_bin(
            out_dir / "lineout.bin", r_axis,
            [profile_1d_v1(c, weights, mode="area_weighted") for c in cakes]))
    if write_simple_mean:
        written.append(write_lineout_bin(
            out_dir / "lineout_simple_mean.bin", r_axis,
            [profile_1d_v1(c, mode="simple_mean") for c in cakes]))
    if write_2d:
        written.append(write_int2d_bin(out_dir / "Int2D.bin", cakes))
    return written
