"""Tests for `midas_defect.lattice`.

Covers all four mandatory categories from implementation_plan.md §1.3:
  1. Synthetic correctness (unit)
  2. Autograd correctness (autograd)
  3. Device portability (device)
  4. Real-data regression (real_data) — pinned to scope-script shell list
"""

from __future__ import annotations

import math
import pytest
import numpy as np
import torch

from midas_defect.lattice import (
    CUAL2_A_DEFAULT,
    CUAL2_C_DEFAULT,
    Shell,
    cual2_crystal,
    q_inv_of_hkl_torch,
    tetragonal_shells,
)


# ---------------------------------------------------------------------------
# 1. Synthetic correctness
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_cual2_crystal_defaults():
    """CuAl2 crystal has the expected I4/mcm space group, tetragonal cell, 2 ASU atoms."""
    cr = cual2_crystal()
    assert cr.space_group.number == 140
    assert cr.lattice.a == pytest.approx(CUAL2_A_DEFAULT)
    assert cr.lattice.c == pytest.approx(CUAL2_C_DEFAULT)
    assert cr.lattice.alpha == 90.0
    assert cr.lattice.beta == 90.0
    assert cr.lattice.gamma == 90.0
    assert len(cr.atoms) == 2
    elements = {a.element for a in cr.atoms}
    assert elements == {"Cu", "Al"}


@pytest.mark.unit
def test_q_magnitudes_for_known_hkls():
    """Closed-form |q| for the lowest CuAl2 reflections matches numpy reference."""
    a, c = CUAL2_A_DEFAULT, CUAL2_C_DEFAULT
    # tetragonal: 1/d² = (h²+k²)/a² + l²/c²; |q| = 2π/d
    cases = [(1, 1, 0), (2, 0, 0), (0, 0, 2), (1, 1, 2), (3, 1, 0)]
    expected = [
        2 * math.pi * math.sqrt((h * h + k * k) / (a * a) + (l * l) / (c * c))
        for (h, k, l) in cases
    ]
    hkl_t = torch.tensor(cases, dtype=torch.float64)
    q = q_inv_of_hkl_torch(hkl_t, a, c)
    for q_got, q_exp in zip(q.tolist(), expected):
        assert q_got == pytest.approx(q_exp, rel=1e-12)


@pytest.mark.unit
def test_tetragonal_shells_basic_properties():
    """Shells are sorted by |q|, every |q| ≤ q_max, no duplicate hkl, no (000)."""
    cr = cual2_crystal()
    qmax = 6.0
    shells = tetragonal_shells(cr, q_max_inv_A=qmax)
    assert len(shells) > 0
    qs = [s.q_inv_A for s in shells]
    assert qs == sorted(qs)
    assert all(s.q_inv_A <= qmax + 1e-6 for s in shells)
    # no duplicate hkl across shells
    all_hkls: list = []
    for s in shells:
        all_hkls.extend(s.hkls)
        assert (0, 0, 0) not in s.hkls
    assert len(all_hkls) == len(set(all_hkls)), "duplicate hkl across shells"


@pytest.mark.unit
def test_shells_match_q_inv_of_hkl():
    """Each shell's representative hkl is consistent with the differentiable q-getter.

    Shells are aggregated on a q-tolerance grid (rounded to `q_tol_inv_A`),
    so the comparison tolerance has to be at least that wide.
    """
    cr = cual2_crystal()
    q_tol = 1e-5
    shells = tetragonal_shells(cr, q_max_inv_A=5.0, q_tol_inv_A=q_tol)
    for s in shells[:5]:
        h, k, l = s.hkls[0]
        q_diff = q_inv_of_hkl_torch([[h, k, l]], cr.lattice.a, cr.lattice.c)
        assert float(q_diff[0]) == pytest.approx(s.q_inv_A, abs=q_tol)


# ---------------------------------------------------------------------------
# 2. Autograd correctness
# ---------------------------------------------------------------------------

@pytest.mark.autograd
def test_q_inv_gradient_wrt_a_and_c():
    """gradcheck-style: dq/da and dq/dc match finite-difference."""
    hkl = torch.tensor([[1, 1, 0], [2, 0, 0], [0, 0, 2]], dtype=torch.float64)

    def fn(a_c):
        return q_inv_of_hkl_torch(hkl, a_c[0], a_c[1]).sum()

    a_c = torch.tensor([CUAL2_A_DEFAULT, CUAL2_C_DEFAULT],
                       dtype=torch.float64, requires_grad=True)
    grad_auto = torch.autograd.grad(fn(a_c), a_c)[0]

    eps = 1e-6
    grad_fd = torch.zeros_like(a_c)
    for i in range(2):
        plus = a_c.detach().clone()
        plus[i] += eps
        minus = a_c.detach().clone()
        minus[i] -= eps
        grad_fd[i] = (fn(plus) - fn(minus)) / (2 * eps)

    assert torch.allclose(grad_auto, grad_fd, atol=1e-4, rtol=1e-4)


@pytest.mark.autograd
def test_q_inv_supports_gradcheck():
    """Formal gradcheck on a small set of reflections in float64."""
    hkl = torch.tensor([[1, 1, 0], [2, 1, 2]], dtype=torch.float64)
    a_c = torch.tensor([6.066, 4.874], dtype=torch.float64, requires_grad=True)

    def fn(a_c_):
        return q_inv_of_hkl_torch(hkl, a_c_[0], a_c_[1])

    # gradcheck wants a fn returning a tensor; eps must be small but not below
    # float64 precision floor.
    assert torch.autograd.gradcheck(
        fn, (a_c,), eps=1e-6, atol=1e-5, rtol=1e-3,
        nondet_tol=0.0,
    )


# ---------------------------------------------------------------------------
# 3. Device portability
# ---------------------------------------------------------------------------

@pytest.mark.device
def test_q_inv_device_portable(_device_param, _dtype_param):
    """Same |q| values on CPU vs CUDA/MPS, within float-precision tolerance.

    MPS does not support float64 in PyTorch; that combo is skipped.
    """
    if _device_param.type == "mps" and _dtype_param == torch.float64:
        pytest.skip("MPS backend does not support float64 in PyTorch")
    hkl = torch.tensor([[1, 1, 0], [2, 0, 0], [0, 0, 2], [1, 1, 2]],
                       dtype=_dtype_param)
    a = torch.tensor(CUAL2_A_DEFAULT, dtype=_dtype_param)
    c = torch.tensor(CUAL2_C_DEFAULT, dtype=_dtype_param)
    q_cpu = q_inv_of_hkl_torch(hkl, a, c, device=torch.device("cpu"))
    q_dev = q_inv_of_hkl_torch(
        hkl.to(_device_param), a.to(_device_param), c.to(_device_param),
        device=_device_param, dtype=_dtype_param,
    )
    tol = 1e-5 if _dtype_param == torch.float32 else 1e-10
    assert torch.allclose(q_cpu, q_dev.cpu(), atol=tol, rtol=tol)


@pytest.mark.device
def test_tetragonal_shells_deterministic(_device_param):
    """`tetragonal_shells` is device-independent (runs on CPU regardless)."""
    cr = cual2_crystal()
    shells_a = tetragonal_shells(cr, q_max_inv_A=5.0)
    shells_b = tetragonal_shells(cr, q_max_inv_A=5.0)
    assert len(shells_a) == len(shells_b)
    for sa, sb in zip(shells_a, shells_b):
        assert sa == sb


# ---------------------------------------------------------------------------
# 4. Real-data regression — pinned to scope-script behavior
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_shells_against_scope_overcount():
    """Cross-check: the interactive scope script (demk_combined_html.py) only
    applied I-centering (h+k+l=2n) and reported 126 shells up to q=10.15.
    That was an over-count — I4/mcm has an additional reflection condition
    (0kl: l=2n, the c-glide ⊥ a) which forbids (1,0,1), (3,0,1), (1,0,3),
    ..., 13 reflections in this q-range. `midas_hkls` applies the full
    space-group rules, so the correct count is 113.

    This test pins both numbers, so if upstream changes the space-group
    tables we notice.
    """
    cr = cual2_crystal()
    shells = tetragonal_shells(cr, q_max_inv_A=10.15, q_tol_inv_A=1e-5)
    assert len(shells) == 113, (
        f"got {len(shells)} shells; expected 113 (I4/mcm full rules); "
        f"scope-script over-count was 126."
    )
    # The forbidden-by-c-glide reflections that scope wrongly included must
    # NOT appear in any shell.
    forbidden = {(1, 0, 1), (3, 0, 1), (1, 0, 3), (3, 0, 3), (1, 0, 5)}
    all_hkls = {hkl for s in shells for hkl in s.hkls}
    assert forbidden.isdisjoint(all_hkls), (
        f"forbidden reflections leaked through: "
        f"{sorted(forbidden & all_hkls)}"
    )

    # Loosening the merge tolerance can only reduce (or preserve) shell count.
    shells_loose = tetragonal_shells(cr, q_max_inv_A=10.15, q_tol_inv_A=1e-3)
    assert len(shells_loose) <= len(shells)
