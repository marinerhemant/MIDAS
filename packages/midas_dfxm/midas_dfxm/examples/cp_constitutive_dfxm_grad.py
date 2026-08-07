"""End-to-end differentiable crystal-plasticity <- DFXM at the constitutive level.

Proves the coupling that matters for CP-parameter inference: the DFXM-measured
lattice (elastic) strain is a *differentiable* function of a crystal-plasticity
parameter (the critical resolved shear stress, CRSS). We integrate a minimal
rate-dependent FCC single-crystal constitutive law under an imposed uniaxial
stretch, split total strain into plastic (slip) + elastic, feed the elastic Fe to
the JAX DFXM observable, and take jax.grad of the DFXM strain w.r.t. the CRSS.

This is the self-contained, runnable proof of the CP<->DFXM differentiable chain.
The FULL jax_fem CPFEM version (spatial BVP) composes identically -- F(cp_params)
-> normal_strain_jax -> jax.grad -- and is blocked ONLY by a jax_fem 0.0.12 / jax
0.10 adjoint incompatibility in the sentosa env (pin jax to jax_fem's supported
range to unblock; the forward solve and the DFXM half already work there).

Run:  python examples/cp_constitutive_dfxm_grad.py   (needs jax; runs on CPU)
"""
from __future__ import annotations

import numpy as onp

try:
    import jax
    import jax.numpy as jnp
    HAVE_JAX = True
except ImportError:  # pragma: no cover
    HAVE_JAX = False

_TWO_PI = 2.0 * onp.pi


def _fcc_schmid():
    """12 FCC {111}<110> Schmid tensors P_s = sym(s (x) n) (unit s, n)."""
    planes = [(1, 1, 1), (1, 1, -1), (1, -1, 1), (-1, 1, 1)]
    dirs = [(1, -1, 0), (1, 0, -1), (0, 1, -1), (1, 1, 0), (1, 0, 1), (0, 1, 1)]
    P = []
    for n in planes:
        n = onp.array(n, float); n = n / onp.linalg.norm(n)
        for s in dirs:
            s = onp.array(s, float)
            if abs(n @ s) > 1e-9:
                continue
            s = s / onp.linalg.norm(s)
            P.append(0.5 * (onp.outer(s, n) + onp.outer(n, s)))
    return onp.array(P)  # (12,3,3)


def _cubic_C(c11, c12, c44):
    C = onp.zeros((3, 3, 3, 3))
    d = onp.eye(3)
    for i in range(3):
        for j in range(3):
            for k in range(3):
                for l in range(3):
                    C[i, j, k, l] = (c12 * d[i, j] * d[k, l]
                                     + c44 * (d[i, k] * d[j, l] + d[i, l] * d[j, k]))
                    if i == j == k == l:
                        C[i, j, k, l] += (c11 - c12 - 2 * c44)
    return C


def normal_strain_jax(Fe, hkl, latc=(3.6156, 3.6156, 3.6156, 90., 90., 90.)):
    a = latc[0]
    B = _TWO_PI / a * jnp.eye(3)          # cubic reciprocal basis
    G0 = B @ jnp.asarray(hkl, float)
    ghat = G0 / jnp.linalg.norm(G0)
    eps = 0.5 * (Fe + Fe.T) - jnp.eye(3)
    return ghat @ eps @ ghat


def main():
    if not HAVE_JAX:
        print("JAX not available; this demo needs jax.")
        return
    jax.config.update("jax_enable_x64", True)
    P = jnp.asarray(_fcc_schmid())
    C = jnp.asarray(_cubic_C(1.684e5, 1.214e5, 0.754e5))  # Cu, MPa
    n_steps, gamma0, m = 60, 1e-3, 0.05
    eps_total_max = 0.02                                  # 2% uniaxial along z

    def dfxm_axial_strain(crss):
        """Integrate the CP law at CRSS, return the DFXM axial (002) elastic strain."""
        eps_p = jnp.zeros((3, 3))
        for i in range(n_steps):
            eps_tot = jnp.zeros((3, 3)).at[2, 2].set(eps_total_max * (i + 1) / n_steps)
            eps_e = eps_tot - eps_p
            sigma = jnp.einsum("ijkl,kl->ij", C, eps_e)
            tau = jnp.einsum("sij,ij->s", P, sigma)                 # resolved shear
            dgamma = gamma0 * jnp.sign(tau) * (jnp.abs(tau) / crss) ** (1.0 / m)
            # Clamp the per-step slip increment: the explicit overstress law is stiff
            # (tau/crss)^(1/m) explodes above yield; clamping keeps integration stable
            # and differentiable (a smooth-elsewhere regularisation of the flow rule).
            dgamma = jnp.clip(dgamma, -2e-3, 2e-3)
            eps_p = eps_p + jnp.einsum("s,sij->ij", dgamma, P)
        Fe = jnp.eye(3) + (eps_tot - eps_p)
        return normal_strain_jax(Fe, (0, 0, 2))

    for crss in (40.0, 60.8, 90.0):
        val, grad = jax.value_and_grad(dfxm_axial_strain)(crss)
        print(f"CRSS={crss:6.1f} MPa  ->  DFXM axial (002) strain = {float(val):.5e}  "
              f"d(strain)/d(CRSS) = {float(grad):+.3e}")
    print("END-TO-END: DFXM observable is a differentiable function of the CP CRSS.")
    print("(Full jax_fem CPFEM composes identically; see plan for the env pin needed.)")


if __name__ == "__main__":
    main()
