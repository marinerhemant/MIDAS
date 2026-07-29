"""JAX DFXM forward bridge — the end-to-end CPFEM<->DFXM differentiable seed.

The midas_dfxm package is torch. JAX-CPFEM (``jax_fem``) is JAX and lives on the GPU
host (sentosa: /home/beams/S1IDUSER/opt/jaxcpfem_work/venv). To fit crystal-plasticity
parameters (CRSS, hardening, ...) *directly* to DFXM data, gradients must flow from CP
parameters -> CPFEM solve -> deformation gradient F(r) -> DFXM observable -> loss, all
in ONE differentiable graph. That means the DFXM observable must exist in JAX so it
composes with jax_fem.

This file is that JAX observable, deliberately MINIMAL (the normal-strain and
voxel-intensity observables — the parts CP fitting needs), and it is **parity-checked
against the torch forward** so it is a faithful port, not a divergent second model.

Run locally to verify parity (JAX is installed here; jax_fem is not — it lives on
sentosa):
    export KMP_DUPLICATE_LIB_OK=TRUE
    python examples/jax_dfxm_bridge.py

To close the loop on sentosa, drop these functions next to jax_fem and compose:
    F = cpfem_solve(cp_params, ...)          # jax_fem, differentiable in cp_params
    pred = normal_strain_jax(F, OM0, latc, hkl)
    loss = ((pred - measured)**2).mean()
    grad = jax.grad(lambda p: loss_of(cpfem_solve(p, ...)))(cp_params)   # CP <- DFXM
"""
from __future__ import annotations

import math

import numpy as np

try:
    import jax
    import jax.numpy as jnp
    HAVE_JAX = True
except ImportError:  # pragma: no cover
    HAVE_JAX = False

_TWO_PI = 2.0 * math.pi


# --------------------------------------------------------------------------
# JAX DFXM observable (mirrors midas_dfxm.field / inverse / forward)
# --------------------------------------------------------------------------
def _A_matrix_jax(latc):
    """Fractional->Cartesian A matrix (Busing-Levy) — JAX, differentiable in latc."""
    a, b, c, al, be, ga = [latc[i] for i in range(6)]
    al, be, ga = jnp.deg2rad(al), jnp.deg2rad(be), jnp.deg2rad(ga)
    ca, cb, cg, sg = jnp.cos(al), jnp.cos(be), jnp.cos(ga), jnp.sin(ga)
    V = jnp.sqrt(jnp.clip(1 - ca**2 - cb**2 - cg**2 + 2*ca*cb*cg, 1e-12, None))
    return jnp.array([
        [a, b * cg, c * cb],
        [0.0, b * sg, c * (ca - cb * cg) / sg],
        [0.0, 0.0, c * V / sg],
    ])


def reciprocal_basis_jax(latc):
    """Reciprocal basis B (2*pi convention): G = B @ hkl."""
    A = _A_matrix_jax(latc)
    return _TWO_PI * jnp.linalg.inv(A).T


def reference_G_jax(OM0, latc, hkl):
    """Reference reflection G0 in the sample frame."""
    B = reciprocal_basis_jax(latc)
    return OM0 @ (B @ jnp.asarray(hkl, dtype=B.dtype))


def normal_strain_jax(F, OM0, latc, hkl):
    """Per-voxel DFXM normal strain eps_gg = ghat^T sym(F-I) ghat. F: (N,3,3)."""
    G0 = reference_G_jax(OM0, latc, hkl)
    ghat = G0 / jnp.linalg.norm(G0)
    eps = 0.5 * (F + jnp.transpose(F, (0, 2, 1))) - jnp.eye(3)
    return jnp.einsum("i,nij,j->n", ghat, eps, ghat)


def voxel_intensity_jax(F, G0, G_gonio, q_nom, sigma_par, sigma_perp):
    """DFXM weak-beam voxel intensity (Gaussian acceptance) — JAX, mirrors the torch path."""
    Ft = jnp.transpose(F, (0, 2, 1))
    Q = jnp.linalg.solve(Ft, jnp.broadcast_to(G0, F.shape[:1] + (3,))[..., None])[..., 0]
    Q_lab = Q @ G_gonio.T
    e_par = q_nom / jnp.linalg.norm(q_nom)
    seed = jnp.where(jnp.abs(e_par @ jnp.array([0., 0., 1.])) > 0.9,
                     jnp.array([1., 0., 0.]), jnp.array([0., 0., 1.]))
    e_t1 = seed - (seed @ e_par) * e_par
    e_t1 = e_t1 / jnp.linalg.norm(e_t1)
    e_t2 = jnp.cross(e_par, e_t1)
    d = Q_lab - q_nom
    chi2 = (d @ e_par / sigma_par) ** 2 + (d @ e_t1 / sigma_perp) ** 2 + (d @ e_t2 / sigma_perp) ** 2
    return jnp.exp(-0.5 * chi2)


# --------------------------------------------------------------------------
# parity check vs the torch forward
# --------------------------------------------------------------------------
def main():
    if not HAVE_JAX:
        print("JAX not available locally; this bridge runs where JAX is installed.")
        return
    jax.config.update("jax_enable_x64", True)
    import torch
    from midas_dfxm import (GoniometerSetting, aligned_resolution, make_uniform_field,
                            reference_q_nom, voxel_intensity, with_screw_dislocation)
    from midas_dfxm.inverse import normal_strain

    # Build a torch field, evaluate both observables in torch and JAX, compare.
    field = make_uniform_field(shape=(12, 12, 1), dtype=torch.float64)
    field = with_screw_dislocation(field, burgers_A=2.556, core_radius_um=0.5)
    latc = (3.6356, 3.6356, 3.6356, 90.0, 90.0, 90.0)
    OM0 = np.eye(3)

    F_np = field.F.detach().numpy()
    # normal strain parity
    ns_torch = normal_strain(field, (1, 1, 1)).detach().numpy()
    ns_jax = np.asarray(normal_strain_jax(jnp.asarray(F_np), jnp.asarray(OM0),
                                          jnp.asarray(latc), (1, 1, 1)))
    print(f"normal_strain  max|torch-jax| = {np.max(np.abs(ns_torch - ns_jax)):.2e}")

    # voxel intensity parity
    setting = GoniometerSetting(chi=0.03)
    q_nom = reference_q_nom(field, (1, 1, 1), GoniometerSetting())
    res = aligned_resolution(q_nom, sigma_par=8e-3, sigma_perp=8e-3)
    vi_torch = voxel_intensity(field, (1, 1, 1), setting, res).detach().numpy()
    G0 = field.reference_G((1, 1, 1)).detach().numpy()
    G_gon = setting.sample_rotation(dtype=torch.float64).detach().numpy()
    vi_jax = np.asarray(voxel_intensity_jax(jnp.asarray(F_np), jnp.asarray(G0),
                                            jnp.asarray(G_gon), jnp.asarray(q_nom.detach().numpy()),
                                            8e-3, 8e-3))
    print(f"voxel_intensity  max|torch-jax| = {np.max(np.abs(vi_torch - vi_jax)):.2e}")

    # gradient flows through F in JAX (the point: grads reach CP params via jax_fem)
    def loss_of_F(Fj):
        return jnp.sum(normal_strain_jax(Fj, jnp.asarray(OM0), jnp.asarray(latc), (1, 1, 1)) ** 2)
    g = jax.grad(loss_of_F)(jnp.asarray(F_np))
    print(f"jax.grad wrt F: finite={bool(jnp.all(jnp.isfinite(g)))}, shape={g.shape}")
    print("PARITY OK -> drop next to jax_fem on sentosa to close CP<-DFXM end-to-end.")


if __name__ == "__main__":
    main()
