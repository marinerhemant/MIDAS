"""Sparse vs dense splatter parity, plus a large-P sanity check.

Sparse splatter must agree with dense to <1% MSE for typical sigma+radius
combinations; values outside the radius are truncated, which is below the
~0.3% Gaussian-tail level for radius_yz=3, radius_f=2.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
_PKG_ROOT = _HERE.parent
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from midas_grain_odf.spot_extract import (  # noqa: E402
    SpotPatchSpec,
    splat_spots_to_patches_dense,
    splat_spots_to_patches_sparse,
)


def _random_inputs(K=20, S=30, F=7, P=31, seed=0):
    g = torch.Generator().manual_seed(seed)
    spot_y = (P / 2) + torch.randn(K, S, generator=g, dtype=torch.float64) * (P / 8)
    spot_z = (P / 2) + torch.randn(K, S, generator=g, dtype=torch.float64) * (P / 8)
    spot_f = (F / 2) + torch.randn(K, S, generator=g, dtype=torch.float64) * (F / 6)
    weights = torch.softmax(
        torch.randn(K, generator=g, dtype=torch.float64), dim=0
    )
    valid = (torch.rand(K, S, generator=g, dtype=torch.float64) > 0.05).double()

    spec = SpotPatchSpec(
        n_spots=S, patch_F=F, patch_P=P,
        sigma_yz=1.0, sigma_f=0.6,
        anchor_y=torch.zeros(S, dtype=torch.float64),
        anchor_z=torch.zeros(S, dtype=torch.float64),
        anchor_f=torch.zeros(S, dtype=torch.float64),
    )
    return spec, spot_y, spot_z, spot_f, weights, valid


def test_sparse_matches_dense_small_P():
    """At P=31 the dense implementation is feasible; sparse should match it."""
    spec, sy, sz, sf, w, v = _random_inputs(K=20, S=30, F=7, P=31)

    dense = splat_spots_to_patches_dense(spec, sy, sz, sf, w, v)
    sparse = splat_spots_to_patches_sparse(spec, sy, sz, sf, w, v,
                                           radius_yz=3, radius_f=2)

    abs_err = (dense - sparse).abs()
    max_dense = dense.abs().max().item()
    print(f"  max |dense| = {max_dense:.4f}")
    print(f"  max |dense - sparse| = {abs_err.max().item():.4e}")
    print(f"  RMS |dense - sparse| = {abs_err.pow(2).mean().sqrt().item():.4e}")
    # Gaussian truncation at radius=3 sigma=1 gives tails ~0.3% per cell.
    # We allow ~3% relative max error.
    assert abs_err.max().item() < 0.03 * max_dense


def test_sparse_gradient_flows():
    """Gradient w.r.t. spot positions must propagate through scatter_add."""
    spec, sy, sz, sf, w, v = _random_inputs(K=4, S=5, F=7, P=15)
    sy.requires_grad_(True)
    sz.requires_grad_(True)
    sf.requires_grad_(True)

    out = splat_spots_to_patches_sparse(spec, sy, sz, sf, w, v)
    out.sum().backward()

    assert sy.grad is not None and sy.grad.abs().sum() > 0
    assert sz.grad is not None and sz.grad.abs().sum() > 0
    assert sf.grad is not None and sf.grad.abs().sum() > 0


def test_sparse_scales_to_large_P():
    """At P=80 (real Ti-7Al spreads) the dense path is too memory-heavy.

    This test exercises the sparse splatter at the production patch size.
    Asserts only that it returns a finite tensor of the right shape;
    correctness has been validated against the dense path at P=31.
    """
    spec, sy, sz, sf, w, v = _random_inputs(K=50, S=100, F=10, P=80)
    t0 = time.time()
    out = splat_spots_to_patches_sparse(spec, sy, sz, sf, w, v)
    elapsed = time.time() - t0
    print(f"  sparse splat at K=50, S=100, F=10, P=80: {elapsed*1000:.1f} ms")
    assert out.shape == (100, 10, 80, 80)
    assert torch.isfinite(out).all()
