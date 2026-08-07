"""lm_batched must refuse single precision, and say why.

Its Jacobian is finite-difference. Measured on 1-ID shade_LSHR (100 grains,
|dposition| vs c-omp): float32 gives 4.34 um at the optimal sqrt(eps) step and
229.96 um at the old fixed 1e-6 default, against float64's 1.53 um -- while
reporting convergence throughout. The speed it buys is ~0-13%.

This is NOT a statement about float32 generally: the gradient paths measured
identical in float32 and float64. It is a statement about difference quotients.
"""
from __future__ import annotations

import pytest
import torch

from midas_fit_grain.solvers.lm_batched import minimize_lm_batched


def _res(pos, euler, lattice):
    return (pos - 1.0).reshape(pos.shape[0], -1)


@pytest.mark.parametrize("dtype", [torch.float32])
def test_lm_batched_refuses_low_precision(dtype):
    kw = dict(pos_scaled=torch.zeros(2, 3, dtype=dtype),
              euler=torch.zeros(2, 3, dtype=dtype),
              lattice=torch.ones(2, 6, dtype=dtype))
    with pytest.raises(ValueError, match="finite differences|FINITE DIFFERENCES"):
        minimize_lm_batched(_res, kw["pos_scaled"], kw["euler"], kw["lattice"],
                            max_iter=1)


def test_lm_batched_accepts_float64():
    out = minimize_lm_batched(_res, torch.zeros(2, 3, dtype=torch.float64),
                              torch.zeros(2, 3, dtype=torch.float64),
                              torch.ones(2, 6, dtype=torch.float64), max_iter=2)
    assert "pos_scaled" in out


def test_lm_batched_escape_hatch():
    out = minimize_lm_batched(_res, torch.zeros(2, 3, dtype=torch.float32),
                              torch.zeros(2, 3, dtype=torch.float32),
                              torch.ones(2, 6, dtype=torch.float32), max_iter=1,
                              allow_low_precision=True)
    assert "pos_scaled" in out
