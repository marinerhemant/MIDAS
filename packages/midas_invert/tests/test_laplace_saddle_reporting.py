"""At a saddle the Laplace approximation is invalid, and must say so.

``pinv`` of an indefinite Hessian gives negative diagonal entries. Clamping
those at zero reported ``std = 0.0`` -- the WORST-conditioned direction dressed
up as the most confident. There is no "true sigma" to restore: at a saddle
there is no Gaussian posterior whose width could be quoted, so the fix is a
contract change (spec_autograd_classB_classC.md, C3), not a formula.
"""

from __future__ import annotations

import math

import pytest
import torch

from midas_invert.uq import laplace_uncertainty


def test_positive_definite_case_is_unchanged():
    """A well-posed problem must keep its numbers."""
    A = torch.tensor([[4.0, 1.0], [1.0, 3.0]], dtype=torch.float64)

    def loss(t):
        return 0.5 * t @ A @ t

    out = laplace_uncertainty(loss, torch.zeros(2, dtype=torch.float64))
    cov = torch.linalg.inv(A)
    assert out["is_positive_definite"] is True
    assert out["n_negative_eigvals"] == 0
    assert torch.allclose(out["std"], torch.sqrt(torch.diag(cov)), atol=1e-9)
    assert torch.isfinite(out["std"]).all()


def test_saddle_reports_nan_not_zero():
    """x^2 - y^2: the y direction has no posterior width at all."""
    def loss(t):
        return t[0] ** 2 - t[1] ** 2

    out = laplace_uncertainty(loss, torch.zeros(2, dtype=torch.float64))
    assert out["is_positive_definite"] is False
    assert out["n_negative_eigvals"] == 1
    assert math.isnan(float(out["std"][1])), (
        "a non-positive variance must read NaN; 0.0 claims certainty in the "
        "worst-conditioned direction"
    )
    assert float(out["std"][1]) != 0.0


def test_sigma_alias_matches_std():
    def loss(t):
        return t[0] ** 2 - t[1] ** 2

    out = laplace_uncertainty(loss, torch.zeros(2, dtype=torch.float64))
    a, b = out["std"], out["sigma"]
    assert torch.equal(torch.isnan(a), torch.isnan(b))


def test_flag_is_readable_without_inspecting_eigenvalues():
    """The eigen-diagnostics always exposed this; the point is that a caller
    reading only ``std`` -- the obvious field, and the one ``sigma`` invites --
    now cannot miss it."""
    def bad(t):
        return t[0] ** 2 - 3.0 * t[1] ** 2

    out = laplace_uncertainty(bad, torch.zeros(2, dtype=torch.float64))
    assert set(["is_positive_definite", "n_negative_eigvals"]).issubset(out)
    assert out["is_positive_definite"] is False


def test_negative_definite_marks_every_component():
    def loss(t):
        return -(t[0] ** 2) - t[1] ** 2

    out = laplace_uncertainty(loss, torch.zeros(2, dtype=torch.float64))
    assert out["n_negative_eigvals"] == 2
    assert bool(torch.isnan(out["std"]).all())
