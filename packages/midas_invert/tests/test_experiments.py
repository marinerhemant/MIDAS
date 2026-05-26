"""WS-4: D-optimal reflection selection for the strain tensor."""
import math

import pytest
import torch

from midas_invert.experiments import (
    design_logdet,
    plan_strain_measurements,
    strain_forward,
    strain_normal_coeffs,
)

DT = torch.float64


def _fibonacci_sphere(n):
    import torch
    i = torch.arange(n, dtype=DT) + 0.5
    phi = torch.acos(1 - 2 * i / n)
    theta = math.pi * (1 + 5 ** 0.5) * i
    return torch.stack([torch.sin(phi) * torch.cos(theta),
                        torch.sin(phi) * torch.sin(theta),
                        torch.cos(phi)], dim=1)


@pytest.mark.unit
def test_isotropic_strain_reads_same_on_all_reflections():
    g = _fibonacci_sphere(20)
    C = strain_normal_coeffs(g)
    eps_iso = torch.tensor([1e-3, 1e-3, 1e-3, 0, 0, 0], dtype=DT)
    eps_n = C @ eps_iso
    assert torch.allclose(eps_n, torch.full_like(eps_n, 1e-3), atol=1e-9)


@pytest.mark.unit
def test_uniaxial_cos2_dependence():
    g = torch.tensor([[1., 0., 0.], [0., 1., 0.], [1., 1., 0.]], dtype=DT)
    C = strain_normal_coeffs(g)
    eps = torch.tensor([1.0, 0, 0, 0, 0, 0], dtype=DT)   # e11 only
    eps_n = C @ eps
    assert abs(float(eps_n[0]) - 1.0) < 1e-9     # g||x -> e11
    assert abs(float(eps_n[1]) - 0.0) < 1e-9     # g||y -> 0
    assert abs(float(eps_n[2]) - 0.5) < 1e-9     # 45 deg -> g1^2 = 0.5


@pytest.mark.unit
def test_diverse_set_is_full_rank_clustered_is_not():
    diverse = _fibonacci_sphere(30)
    # clustered: all near +z
    torch.manual_seed(0)
    clustered = torch.tensor([0., 0., 1.], dtype=DT) + 0.03 * torch.randn(30, 3, dtype=DT)
    ld_div = design_logdet(diverse, prior_precision=0.0)
    ld_clu = design_logdet(clustered, prior_precision=1e-9)   # tiny reg to avoid -inf
    assert ld_div > ld_clu + 10            # diverse set is far better conditioned


@pytest.mark.unit
def test_greedy_selection_beats_clustered_subset():
    cand = _fibonacci_sphere(60)
    chosen = plan_strain_measurements(cand, k=10)
    idx = [i for i, _ in chosen]
    assert len(set(idx)) == 10
    ld_opt = design_logdet(cand[idx], prior_precision=1e-9)
    # vs the 10 most mutually-clustered candidates (smallest spread): take a cap
    z = cand[:, 2]
    clustered_idx = torch.argsort(z)[:10]
    ld_clu = design_logdet(cand[clustered_idx], prior_precision=1e-9)
    assert ld_opt > ld_clu


@pytest.mark.unit
def test_greedy_picks_spread_directions():
    """The first few greedy picks should not be collinear (they span axes)."""
    cand = _fibonacci_sphere(60)
    chosen = plan_strain_measurements(cand, k=3)
    idx = [i for i, _ in chosen]
    g = cand[idx]
    g = g / torch.linalg.vector_norm(g, dim=1, keepdim=True)
    # pairwise |dot| should be well below 1 (not collinear)
    dots = (g @ g.T).abs()
    offdiag = dots[~torch.eye(3, dtype=torch.bool)]
    assert float(offdiag.max()) < 0.95
