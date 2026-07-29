"""Bayesian posterior for the small-box structure refinement.

Companion to :func:`midas_pdf.structure.refine_structure`.  The Laplace
posterior baked into that function assumes the χ² surface is well-modelled
by a Gaussian at the MAP; this module lets you either

  * (SVI) fit a variational Gaussian posterior — cheap; a strict
    generalisation of Laplace, and validates the Laplace assumption if
    the two agree;
  * (NUTS) draw HMC samples from the exact posterior — the ground truth
    the other two are approximating.

Both routes are driven by the SAME forward-model callable as
:func:`refine_structure`, so switching between MAP + Laplace, SVI, and
NUTS is a one-line change in analysis code.

Requires ``pyro-ppl`` (install with the ``[bayes]`` extra).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Optional

import numpy as np
import torch

__all__ = [
    "BayesianRefineResult",
    "bayesian_refine_svi",
    "bayesian_refine_nuts",
]


# ---------------------------------------------------------------------------
# Result container (mirrors _RefineResult but for posterior sampling)
# ---------------------------------------------------------------------------

@dataclass
class BayesianRefineResult:
    posterior_samples: Dict[str, torch.Tensor]        # {param_name: (n_samples,)}
    G_samples: torch.Tensor                            # (n_samples, n_r)
    G_mean: torch.Tensor                               # (n_r,)
    G_std: torch.Tensor                                # (n_r,)
    method: str                                        # "SVI" | "NUTS"
    diagnostic: dict = field(default_factory=dict)

    def summary(self) -> Dict[str, Dict[str, float]]:
        """Per-parameter posterior mean / std / q05 / q95."""
        out = {}
        for name, s in self.posterior_samples.items():
            s = s.cpu()
            out[name] = {
                "mean": float(s.mean()),
                "std":  float(s.std()),
                "q05":  float(torch.quantile(s, 0.05)),
                "q95":  float(torch.quantile(s, 0.95)),
            }
        return out


# ---------------------------------------------------------------------------
# Shared: wrap the structure-refinement model as a Pyro model
# ---------------------------------------------------------------------------

def _build_pyro_model(
    crystal_tensor,
    r: torch.Tensor,
    G_obs: torch.Tensor,
    pairs,
    sigma_obs: torch.Tensor,
    bg_order: Optional[int] = None,
    prior_widths: Optional[Dict[str, float]] = None,
    map_init: Optional[Dict[str, float]] = None,
):
    """Build a Pyro model + parameter-name list for the refinement.

    Parameters have wide-Gaussian priors centred at ``map_init`` (defaults
    to sensible defaults):

        a       ~ Normal(map_a,       prior_width_a)          # lattice (Å)
        u_iso   ~ HalfNormal(prior_width_u)                    # positive
        scale   ~ HalfNormal(prior_width_scale)                # positive
        bg_j    ~ Normal(0, prior_width_bg)   for j = 0..bg_order

    Likelihood: G_obs ~ Normal(G_calc, sigma_obs).
    """
    import pyro
    import pyro.distributions as dist
    from .structure import pdffit_gr

    # Sensible defaults
    init = {"a": 3.524, "u_iso": 0.006, "scale": 1.0}
    if map_init:
        for k, v in map_init.items():
            if v is None:
                continue
            # only accept scalar-like values; skip list entries (e.g. bg_coef)
            try:
                init[k] = float(v)
            except (TypeError, ValueError):
                pass
    priors = {"a": 0.02, "u_iso": 0.005, "scale": 0.5, "bg": 5.0}
    if prior_widths: priors.update(prior_widths)

    lat0 = crystal_tensor.lattice_params.detach()
    angles = lat0[3:]
    r_scale = float(r.max())
    n_bg = 0 if bg_order is None else bg_order + 1

    param_names = ["a", "u_iso", "scale"] + [f"bg_{j}" for j in range(n_bg)]

    def model():
        # smooth positive priors (LogNormal) avoid the HalfNormal boundary
        # instability seen in NUTS trajectories near u_iso = 0 / scale = 0
        a       = pyro.sample("a",     dist.Normal(init["a"], priors["a"]))
        u_iso   = pyro.sample("u_iso", dist.LogNormal(
                    torch.tensor(float(np.log(max(init["u_iso"], 1e-4)))),
                    torch.tensor(1.0)))            # broad on log-scale
        scale   = pyro.sample("scale", dist.LogNormal(
                    torch.tensor(float(np.log(max(init["scale"], 1e-4)))),
                    torch.tensor(0.5)))
        bg_coeffs = torch.zeros(n_bg, dtype=torch.float64) if n_bg == 0 else \
            torch.stack([pyro.sample(f"bg_{j}", dist.Normal(0.0, priors["bg"]))
                         for j in range(n_bg)])
        lp = torch.cat([a.reshape(1).expand(3), angles])
        G_calc = pdffit_gr(crystal_tensor, r, pairs,
                            scale=scale, u_iso=u_iso, lattice_params=lp)
        if n_bg:
            powers = torch.stack([(r / r_scale) ** j for j in range(n_bg)], dim=1)
            G_calc = G_calc + powers @ bg_coeffs
        with pyro.plate("r_points", G_obs.shape[0]):
            pyro.sample("G_obs", dist.Normal(G_calc, sigma_obs), obs=G_obs)

    return model, param_names, n_bg


# ---------------------------------------------------------------------------
# 1) SVI — fit a variational Gaussian posterior
# ---------------------------------------------------------------------------

def bayesian_refine_svi(
    crystal_tensor,
    r,
    G_obs,
    pairs,
    *,
    sigma_obs,
    map_init: Optional[Dict[str, float]] = None,
    prior_widths: Optional[Dict[str, float]] = None,
    bg_order: Optional[int] = None,
    n_steps: int = 2000,
    lr: float = 5e-3,
    n_posterior_samples: int = 500,
    verbose: bool = False,
) -> BayesianRefineResult:
    """Variational (Gaussian) posterior via Pyro SVI.

    Fits a mean-field / diagonal-Gaussian guide by SVI, then samples
    ``n_posterior_samples`` parameter draws + evaluates the model at each
    to build a G(r) posterior band.

    Strictly generalises Laplace: if the true posterior is Gaussian, SVI
    converges to the same point. Where they disagree, SVI's answer is
    the correct one at the level of the model.
    """
    import pyro
    from pyro.infer import SVI, Trace_ELBO
    from pyro.infer.autoguide import AutoNormal, init_to_value
    from pyro.optim import Adam

    r_t   = torch.as_tensor(r, dtype=torch.float64)
    G_t   = torch.as_tensor(G_obs, dtype=torch.float64)
    sig_t = torch.as_tensor(sigma_obs, dtype=torch.float64).clamp(min=1e-12)

    model, param_names, n_bg = _build_pyro_model(
        crystal_tensor, r_t, G_t, pairs, sig_t, bg_order=bg_order,
        prior_widths=prior_widths, map_init=map_init,
    )
    pyro.clear_param_store()
    # Seed the variational guide at the MAP so it converges to the mode
    init_vals = {"a": torch.tensor(float((map_init or {}).get("a", 3.524)),
                                    dtype=torch.float64),
                 "u_iso": torch.tensor(float((map_init or {}).get("u_iso", 0.006)),
                                        dtype=torch.float64),
                 "scale": torch.tensor(float((map_init or {}).get("scale", 1.0)),
                                        dtype=torch.float64)}
    for j in range(n_bg):
        init_vals[f"bg_{j}"] = torch.tensor(
            float((map_init or {}).get(f"bg_{j}", 0.0)), dtype=torch.float64)
    guide = AutoNormal(model, init_loc_fn=init_to_value(values=init_vals),
                       init_scale=0.001)
    optim = Adam({"lr": lr})
    svi = SVI(model, guide, optim, loss=Trace_ELBO())
    losses = []
    for step in range(n_steps):
        losses.append(svi.step())
        if verbose and step % 200 == 0:
            print(f"  SVI step {step:5d}: ELBO = {losses[-1]:.2f}")

    # Draw posterior samples from the guide
    with torch.no_grad():
        posterior_samples: Dict[str, list] = {n: [] for n in param_names}
        G_samples = []
        from .structure import pdffit_gr
        lat0 = crystal_tensor.lattice_params.detach()
        angles = lat0[3:]
        r_scale = float(r_t.max())
        for _ in range(n_posterior_samples):
            guide_trace = pyro.poutine.trace(guide).get_trace()
            a_s     = guide_trace.nodes["a"]["value"]
            u_s     = guide_trace.nodes["u_iso"]["value"]
            scale_s = guide_trace.nodes["scale"]["value"]
            posterior_samples["a"].append(a_s)
            posterior_samples["u_iso"].append(u_s)
            posterior_samples["scale"].append(scale_s)
            bg_coeffs = torch.zeros(n_bg, dtype=torch.float64)
            for j in range(n_bg):
                b = guide_trace.nodes[f"bg_{j}"]["value"]
                bg_coeffs[j] = b
                posterior_samples[f"bg_{j}"].append(b)
            lp = torch.cat([a_s.reshape(1).expand(3), angles])
            G_calc = pdffit_gr(crystal_tensor, r_t, pairs,
                                scale=scale_s, u_iso=u_s, lattice_params=lp)
            if n_bg:
                powers = torch.stack([(r_t / r_scale) ** j
                                     for j in range(n_bg)], dim=1)
                G_calc = G_calc + powers @ bg_coeffs
            G_samples.append(G_calc)

    posterior_samples = {n: torch.stack(v) for n, v in posterior_samples.items()
                          if v}
    G_samples = torch.stack(G_samples)                        # (N, R)
    return BayesianRefineResult(
        posterior_samples=posterior_samples,
        G_samples=G_samples,
        G_mean=G_samples.mean(dim=0),
        G_std=G_samples.std(dim=0),
        method="SVI",
        diagnostic={"final_elbo": float(losses[-1]),
                     "n_svi_steps": n_steps,
                     "elbo_history_first_last": (losses[0], losses[-1])},
    )


# ---------------------------------------------------------------------------
# 2) NUTS — exact HMC posterior samples
# ---------------------------------------------------------------------------

def bayesian_refine_nuts(
    crystal_tensor,
    r,
    G_obs,
    pairs,
    *,
    sigma_obs,
    map_init: Optional[Dict[str, float]] = None,
    prior_widths: Optional[Dict[str, float]] = None,
    bg_order: Optional[int] = None,
    n_warmup: int = 200,
    n_samples: int = 500,
    verbose: bool = False,
) -> BayesianRefineResult:
    """HMC (NUTS) samples from the exact posterior.

    Slower than SVI but samples the true posterior without a
    Gaussian-guide assumption. Use this as ground truth for validating
    SVI / Laplace on new problem classes.
    """
    import pyro
    from pyro.infer import MCMC, NUTS
    from .structure import pdffit_gr

    r_t   = torch.as_tensor(r, dtype=torch.float64)
    G_t   = torch.as_tensor(G_obs, dtype=torch.float64)
    sig_t = torch.as_tensor(sigma_obs, dtype=torch.float64).clamp(min=1e-12)

    model, param_names, n_bg = _build_pyro_model(
        crystal_tensor, r_t, G_t, pairs, sig_t, bg_order=bg_order,
        prior_widths=prior_widths, map_init=map_init,
    )
    pyro.clear_param_store()
    kernel = NUTS(model, jit_compile=False, adapt_step_size=True)
    # Seed the chain at the MAP so NUTS starts in the right basin — critical
    # for peaked posteriors on nonlinear structure models.
    init_params = None
    if map_init is not None:
        init_params = {}
        for k in ("a", "u_iso", "scale"):
            if k in map_init:
                init_params[k] = torch.tensor(float(map_init[k]), dtype=torch.float64)
        for j in range(n_bg):
            k = f"bg_{j}"
            if k in map_init:
                init_params[k] = torch.tensor(float(map_init[k]), dtype=torch.float64)
    mcmc = MCMC(kernel, num_samples=n_samples, warmup_steps=n_warmup,
                 initial_params=init_params, disable_progbar=(not verbose))
    mcmc.run()
    posterior_samples = mcmc.get_samples()      # {name: tensor}

    # Rebuild G(r) at each sample
    G_samples = []
    lat0 = crystal_tensor.lattice_params.detach()
    angles = lat0[3:]
    r_scale = float(r_t.max())
    with torch.no_grad():
        n = posterior_samples["a"].shape[0]
        for i in range(n):
            a_s     = posterior_samples["a"][i]
            u_s     = posterior_samples["u_iso"][i]
            scale_s = posterior_samples["scale"][i]
            bg_coeffs = torch.zeros(n_bg, dtype=torch.float64)
            for j in range(n_bg):
                bg_coeffs[j] = posterior_samples[f"bg_{j}"][i]
            lp = torch.cat([a_s.reshape(1).expand(3), angles])
            G_calc = pdffit_gr(crystal_tensor, r_t, pairs,
                                scale=scale_s, u_iso=u_s, lattice_params=lp)
            if n_bg:
                powers = torch.stack([(r_t / r_scale) ** j
                                     for j in range(n_bg)], dim=1)
                G_calc = G_calc + powers @ bg_coeffs
            G_samples.append(G_calc)
    G_samples = torch.stack(G_samples)

    return BayesianRefineResult(
        posterior_samples=posterior_samples,
        G_samples=G_samples,
        G_mean=G_samples.mean(dim=0),
        G_std=G_samples.std(dim=0),
        method="NUTS",
        diagnostic={"n_warmup": n_warmup, "n_samples": n_samples,
                     "diagnostics": mcmc.diagnostics()},
    )
