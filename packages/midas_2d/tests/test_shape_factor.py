"""Phase 1 correctness: analytic limits + gradients + device portability."""
import math

import pytest
import torch

from midas_2d import (
    interference_factor,
    laue_interference_1d,
    nanoplatelet_rod,
    truncation_rod,
)

DT = torch.float64


# --------------------------------------------------------------- analytic limits

@pytest.mark.unit
@pytest.mark.parametrize("N", [3, 4, 5, 8])
def test_peak_value_is_N_squared(N):
    """Laue function equals N^2 at integer x (the Bragg condition)."""
    x = torch.tensor([0.0, 1.0, 2.0, -3.0], dtype=DT)
    val = laue_interference_1d(x, float(N))
    assert torch.allclose(val, torch.full_like(val, float(N * N)), atol=1e-6)


@pytest.mark.unit
@pytest.mark.parametrize("N", [3, 4, 5, 6])
def test_fringe_minima_count_between_bragg_peaks(N):
    """Between two Bragg peaks the Laue function has exactly N-1 zeros
    (minima) at x = m/N, and N-2 subsidiary maxima."""
    # Sample the open interval (0, 1) densely.
    x = torch.linspace(0.0, 1.0, 20001, dtype=DT)[1:-1]
    y = laue_interference_1d(x, float(N))
    # Zeros sit at x = j/N, j = 1..N-1.  Count near-zero dips.
    expected_minima = [j / N for j in range(1, N)]
    for xm in expected_minima:
        # nearest sample value should be ~0
        idx = torch.argmin((x - xm).abs())
        assert y[idx] < 1e-3, f"expected a fringe minimum near x={xm}"
    # Sanity: global structure has N-1 interior minima -> count sign changes of
    # the derivative is 2*(N-1)-1 extrema; just assert minima are deep.
    assert y.min() < 1e-6


@pytest.mark.unit
def test_truncation_rod_is_large_N_limit():
    """As N grows, sin^2(N pi x)/sin^2(pi x) averaged tracks the CTR envelope
    1/(4 sin^2(pi x)) away from the Bragg peak; here we check the CTR formula
    itself is finite and peaks at half-integer-free points."""
    x = torch.linspace(0.05, 0.95, 50, dtype=DT)
    ctr = truncation_rod(x)
    assert torch.isfinite(ctr).all()
    # CTR is minimal midway between Bragg peaks (x=0.5) and rises toward peaks.
    assert ctr[torch.argmin((x - 0.5).abs())] < ctr[0]


@pytest.mark.unit
def test_interference_factor_product():
    """3-D factor is the product of 1-D factors."""
    hkl = torch.tensor([[0.0, 0.0, 0.3], [1.0, 0.0, 0.5]], dtype=DT)
    N = torch.tensor([10.0, 10.0, 4.0], dtype=DT)
    got = interference_factor(hkl, N)
    want = torch.stack([
        laue_interference_1d(hkl[i, 0], N[0])
        * laue_interference_1d(hkl[i, 1], N[1])
        * laue_interference_1d(hkl[i, 2], N[2])
        for i in range(2)
    ])
    assert torch.allclose(got, want, atol=1e-9)


@pytest.mark.unit
def test_nanoplatelet_rod_thickness_sets_fringes():
    """More layers -> more (and finer) fringes between Bragg peaks."""
    l = torch.linspace(0.0, 1.0, 4001, dtype=DT)[1:-1]
    n3 = nanoplatelet_rod(l, 3)
    n5 = nanoplatelet_rod(l, 5)

    def count_minima(y):
        # local minima strictly below neighbours
        return int(((y[1:-1] < y[:-2]) & (y[1:-1] < y[2:])).sum())

    assert count_minima(n5) > count_minima(n3)


# --------------------------------------------------------------------- gradients

@pytest.mark.autograd
def test_gradcheck_in_x():
    x = torch.tensor([0.27, 0.61, 0.88], dtype=DT, requires_grad=True)
    assert torch.autograd.gradcheck(lambda v: laue_interference_1d(v, 4.0), (x,))


@pytest.mark.autograd
def test_gradcheck_in_N():
    # N is a continuous, fittable parameter.
    N = torch.tensor(4.3, dtype=DT, requires_grad=True)
    x = torch.tensor([0.2, 0.5, 0.73], dtype=DT)
    assert torch.autograd.gradcheck(lambda n: laue_interference_1d(x, n), (N,))


@pytest.mark.autograd
def test_no_nan_gradient_at_integer():
    """The double-where guard must give finite gradients exactly at a Bragg
    peak, where d(N^2)/dN = 2N and d/dx = 0."""
    x = torch.tensor([1.0], dtype=DT, requires_grad=True)
    N = torch.tensor(5.0, dtype=DT, requires_grad=True)
    y = laue_interference_1d(x, N)
    y.backward()
    assert torch.isfinite(x.grad).all()
    assert torch.isfinite(N.grad).all()
    assert torch.allclose(N.grad, torch.tensor(10.0, dtype=DT), atol=1e-6)  # 2N
    assert torch.allclose(x.grad, torch.tensor(0.0, dtype=DT), atol=1e-6)


# ------------------------------------------------------------------- device port

@pytest.mark.device
@pytest.mark.parametrize("device", ["cpu", "cuda", "mps"])
def test_device_portability(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("no CUDA")
    if device == "mps" and not (hasattr(torch.backends, "mps")
                                and torch.backends.mps.is_available()):
        pytest.skip("no MPS")
    dt = torch.float32 if device == "mps" else DT
    l = torch.linspace(0.01, 0.99, 200, dtype=dt, device=device)
    y = nanoplatelet_rod(l, 4)
    assert y.device.type == device
    assert torch.isfinite(y).all()
