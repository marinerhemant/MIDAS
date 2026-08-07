"""CPFEM<->DFXM interchange + JAX-bridge parity tests."""
import numpy as np
import pytest
import torch

from midas_dfxm.cpfem import (
    cpfem_true_strain,
    load_cpfem_field,
    save_cpfem_field,
    validate_dfxm_on_cpfem,
)
from midas_dfxm.generators import field_from_strain

DT = torch.float64
FULL_SET = [(2, 0, 0), (0, 2, 0), (0, 0, 2), (2, 2, 0), (0, 2, 2), (2, 0, 2), (2, 2, 2)]


def _cp_like_field(n=16):
    # A smooth heterogeneous strain field standing in for a CPFEM solution.
    xs = torch.linspace(-1, 1, n, dtype=DT)
    pts = torch.stack([xs, torch.zeros(n, dtype=DT), torch.zeros(n, dtype=DT)], dim=-1)
    e = torch.zeros(n, 6, dtype=DT)
    e[:, 0] = 1e-3 * torch.sin(2 * xs)
    e[:, 1] = -4e-4 * xs
    e[:, 5] = 3e-4 * torch.cos(xs)
    return field_from_strain(e, pts)


@pytest.mark.unit
def test_cpfem_npz_roundtrip(tmp_path):
    field = _cp_like_field()
    p = str(tmp_path / "cp.npz")
    save_cpfem_field(p, field.F.numpy(), field.positions.numpy(),
                     lattice_params=tuple(field.lattice_params.tolist()))
    loaded = load_cpfem_field(p)
    assert torch.allclose(loaded.F, field.F)
    assert torch.allclose(loaded.positions, field.positions)


@pytest.mark.unit
def test_cpfem_true_strain_matches_planted():
    n = 12
    xs = torch.linspace(-1, 1, n, dtype=DT)
    pts = torch.stack([xs, torch.zeros(n, dtype=DT), torch.zeros(n, dtype=DT)], dim=-1)
    e = torch.zeros(n, 6, dtype=DT)
    e[:, 0] = 7e-4
    field = field_from_strain(e, pts)
    eps = cpfem_true_strain(field)
    assert torch.allclose(eps[:, 0], torch.full((n,), 7e-4, dtype=DT), atol=1e-9)


@pytest.mark.unit
def test_dfxm_recovers_cpfem_strain_clean():
    field = _cp_like_field()
    out = validate_dfxm_on_cpfem(field, FULL_SET, noise_std=0.0)
    # Clean, rank-6 reflection set -> DFXM inverse reproduces the CP strain field.
    assert out["rms_error"] < 1e-9
    assert torch.allclose(out["recovered"], out["truth"], atol=1e-8)


# --------------------------------------------------------------------------
# JAX bridge parity (end-to-end CPFEM<->DFXM seed)
# --------------------------------------------------------------------------
@pytest.mark.unit
def test_jax_bridge_matches_torch():
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    from midas_dfxm.examples.jax_dfxm_bridge import normal_strain_jax

    from midas_dfxm.inverse import normal_strain

    field = _cp_like_field(10)
    ns_torch = normal_strain(field, (1, 1, 1)).numpy()
    ns_jax = np.asarray(normal_strain_jax(
        jnp.asarray(field.F.numpy()), jnp.eye(3),
        jnp.asarray(field.lattice_params.numpy()), (1, 1, 1)))
    assert np.max(np.abs(ns_torch - ns_jax)) < 1e-12
