"""Bayesian three-way joint SAXS + SANS + PDF posterior.

Extends the MAP + Laplace :func:`joint_refine_three_way` with Pyro SVI
(AutoNormal guide) and NUTS routes. Returns per-parameter samples plus
per-sample $G(r)$, $I_{\\text{SAXS}}(Q)$, and $I_{\\text{SANS}}(Q)$
so all three posterior bands can be plotted from a single fit.

The identifiability payoff is 2-D on the $(D, U_{\\text{iso}})$
marginal: joint SAXS+PDF already decoupled the two in Rev 8; adding
SANS with an independent contrast channel over-constrains the fit
further. On a synthetic problem the joint 2-D posterior tightens
below the two-way baseline in proportion to the additional channel
information.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch

from .joint import sphere_characteristic_function
from .model import SAXSModel


@dataclass
class ThreeWayBayesianResult:
    posterior_samples: Dict[str, torch.Tensor]
    G_samples: torch.Tensor                      # (n, n_r)
    I_saxs_samples: torch.Tensor                 # (n, n_q_saxs)
    I_sans_samples: torch.Tensor                 # (n, n_q_sans)
    G_mean: torch.Tensor
    G_std: torch.Tensor
    I_saxs_mean: torch.Tensor
    I_saxs_std: torch.Tensor
    I_sans_mean: torch.Tensor
    I_sans_std: torch.Tensor
    method: str
    diagnostic: dict = field(default_factory=dict)

    def summary(self) -> Dict[str, Dict[str, float]]:
        out = {}
        for name, s in self.posterior_samples.items():
            s = s.detach()
            out[name] = {
                "mean": float(s.mean()),
                "std":  float(s.std()),
                "q05":  float(torch.quantile(s, 0.05)),
                "q95":  float(torch.quantile(s, 0.95)),
            }
        return out

    def correlation(self, name_a: str, name_b: str) -> float:
        a = self.posterior_samples[name_a].detach()
        b = self.posterior_samples[name_b].detach()
        return float(torch.corrcoef(torch.stack([a, b]))[0, 1])


def _build_three_way_pyro_model(
    crystal_tensor,
    r_pdf: torch.Tensor,
    G_obs: torch.Tensor,
    pairs,
    sigma_G: torch.Tensor,
    q_saxs: torch.Tensor,
    I_saxs: torch.Tensor,
    sigma_I_saxs: torch.Tensor,
    q_sans: torch.Tensor,
    I_sans: torch.Tensor,
    sigma_I_sans: torch.Tensor,
    saxs_model: SAXSModel,
    sans_model: SAXSModel,
    map_init: Dict[str, float],
    prior_widths: Dict[str, float],
):
    import pyro
    import pyro.distributions as dist
    from midas_pdf.structure import pdffit_gr

    lat0 = crystal_tensor.lattice_params.detach()
    angles = lat0[3:]
    names = ["a", "u_iso", "scale_pdf", "diameter_A",
             "scale_saxs", "background_saxs",
             "scale_sans", "background_sans"]

    def model():
        a = pyro.sample("a", dist.Normal(map_init["a"], prior_widths["a"]))
        u_iso = pyro.sample("u_iso", dist.LogNormal(
            torch.tensor(float(np.log(max(map_init["u_iso"], 1e-4)))),
            torch.tensor(1.0)))
        scale_pdf = pyro.sample("scale_pdf", dist.LogNormal(
            torch.tensor(float(np.log(max(map_init["scale_pdf"], 1e-4)))),
            torch.tensor(0.5)))
        diameter_A = pyro.sample("diameter_A", dist.LogNormal(
            torch.tensor(float(np.log(max(map_init["diameter_A"], 1e-4)))),
            torch.tensor(0.5)))
        scale_saxs = pyro.sample("scale_saxs", dist.LogNormal(
            torch.tensor(float(np.log(max(map_init["scale_saxs"], 1e-4)))),
            torch.tensor(0.5)))
        background_saxs = pyro.sample("background_saxs", dist.Normal(
            torch.tensor(float(map_init["background_saxs"])),
            torch.tensor(float(prior_widths["background_saxs"]))))
        scale_sans = pyro.sample("scale_sans", dist.LogNormal(
            torch.tensor(float(np.log(max(map_init["scale_sans"], 1e-4)))),
            torch.tensor(0.5)))
        background_sans = pyro.sample("background_sans", dist.Normal(
            torch.tensor(float(map_init["background_sans"])),
            torch.tensor(float(prior_widths["background_sans"]))))

        lp = torch.cat([a.reshape(1).expand(3), angles])
        G_bulk = pdffit_gr(crystal_tensor, r_pdf, pairs,
                            scale=scale_pdf, u_iso=u_iso, lattice_params=lp)
        gamma = sphere_characteristic_function(r_pdf, diameter_A)
        G_calc = G_bulk * gamma
        I_saxs_calc = saxs_model.I(q_saxs, D_median=diameter_A / 2.0,
                                     scale=scale_saxs,
                                     background=background_saxs)
        I_sans_calc = sans_model.I(q_sans, D_median=diameter_A / 2.0,
                                     scale=scale_sans,
                                     background=background_sans)

        with pyro.plate("r_points", G_obs.shape[0]):
            pyro.sample("G_obs", dist.Normal(G_calc, sigma_G), obs=G_obs)
        with pyro.plate("q_saxs_points", I_saxs.shape[0]):
            pyro.sample("I_saxs_obs", dist.Normal(I_saxs_calc, sigma_I_saxs),
                        obs=I_saxs)
        with pyro.plate("q_sans_points", I_sans.shape[0]):
            pyro.sample("I_sans_obs", dist.Normal(I_sans_calc, sigma_I_sans),
                        obs=I_sans)

    return model, names


def joint_three_way_refine_svi(
    *,
    crystal_tensor,
    r_pdf: torch.Tensor,
    G_obs: torch.Tensor,
    pairs,
    sigma_G: torch.Tensor,
    q_saxs: torch.Tensor,
    I_saxs: torch.Tensor,
    sigma_I_saxs: torch.Tensor,
    saxs_model: SAXSModel,
    q_sans: torch.Tensor,
    I_sans: torch.Tensor,
    sigma_I_sans: torch.Tensor,
    sans_model: SAXSModel,
    map_init: Dict[str, float],
    prior_widths: Optional[Dict[str, float]] = None,
    n_steps: int = 800,
    lr: float = 5e-3,
    n_posterior_samples: int = 400,
    verbose: bool = False,
) -> ThreeWayBayesianResult:
    """SVI variational posterior over the three-way joint."""
    import pyro
    from pyro.infer import SVI, Trace_ELBO
    from pyro.infer.autoguide import AutoNormal, init_to_value
    from pyro.optim import Adam
    from midas_pdf.structure import pdffit_gr

    if prior_widths is None:
        prior_widths = {"a": 0.02, "background_saxs": 5.0,
                         "background_sans": 5.0}
    else:
        prior_widths = dict(prior_widths)
        prior_widths.setdefault("a", 0.02)
        prior_widths.setdefault("background_saxs", 5.0)
        prior_widths.setdefault("background_sans", 5.0)

    r_pdf_t = torch.as_tensor(r_pdf, dtype=torch.float64)
    G_t = torch.as_tensor(G_obs, dtype=torch.float64)
    sigma_G_t = torch.as_tensor(sigma_G, dtype=torch.float64).clamp(min=1e-12)
    q_saxs_t = torch.as_tensor(q_saxs, dtype=torch.float64)
    I_saxs_t = torch.as_tensor(I_saxs, dtype=torch.float64)
    sigma_I_saxs_t = torch.as_tensor(sigma_I_saxs, dtype=torch.float64).clamp(min=1e-12)
    q_sans_t = torch.as_tensor(q_sans, dtype=torch.float64)
    I_sans_t = torch.as_tensor(I_sans, dtype=torch.float64)
    sigma_I_sans_t = torch.as_tensor(sigma_I_sans, dtype=torch.float64).clamp(min=1e-12)

    model, names = _build_three_way_pyro_model(
        crystal_tensor, r_pdf_t, G_t, pairs, sigma_G_t,
        q_saxs_t, I_saxs_t, sigma_I_saxs_t,
        q_sans_t, I_sans_t, sigma_I_sans_t,
        saxs_model, sans_model, map_init, prior_widths,
    )

    pyro.clear_param_store()
    init_vals = {name: torch.tensor(float(map_init[name]), dtype=torch.float64)
                  for name in names}
    guide = AutoNormal(model,
                        init_loc_fn=init_to_value(values=init_vals),
                        init_scale=1e-3)
    optim = Adam({"lr": lr})
    svi = SVI(model, guide, optim, loss=Trace_ELBO())
    elbo_history = []
    for step in range(n_steps):
        elbo_history.append(svi.step())
        if verbose and step % 200 == 0:
            print(f"  SVI step {step:5d}: ELBO = {elbo_history[-1]:.2f}")

    lat0 = crystal_tensor.lattice_params.detach()
    angles = lat0[3:]
    samples_by_name: Dict[str, List] = {n: [] for n in names}
    G_samples = []
    I_saxs_samples = []
    I_sans_samples = []
    with torch.no_grad():
        for _ in range(n_posterior_samples):
            trace = pyro.poutine.trace(guide).get_trace()
            vals = {n: trace.nodes[n]["value"] for n in names}
            for n in names:
                samples_by_name[n].append(vals[n])
            lp = torch.cat([vals["a"].reshape(1).expand(3), angles])
            G_bulk = pdffit_gr(crystal_tensor, r_pdf_t, pairs,
                                scale=vals["scale_pdf"], u_iso=vals["u_iso"],
                                lattice_params=lp)
            G_calc = G_bulk * sphere_characteristic_function(
                r_pdf_t, vals["diameter_A"])
            I_saxs_calc = saxs_model.I(q_saxs_t,
                                         D_median=vals["diameter_A"] / 2.0,
                                         scale=vals["scale_saxs"],
                                         background=vals["background_saxs"])
            I_sans_calc = sans_model.I(q_sans_t,
                                         D_median=vals["diameter_A"] / 2.0,
                                         scale=vals["scale_sans"],
                                         background=vals["background_sans"])
            G_samples.append(G_calc)
            I_saxs_samples.append(I_saxs_calc)
            I_sans_samples.append(I_sans_calc)

    posterior_samples = {n: torch.stack(v).squeeze()
                          for n, v in samples_by_name.items()}
    G_samples_t = torch.stack(G_samples)
    I_saxs_samples_t = torch.stack(I_saxs_samples)
    I_sans_samples_t = torch.stack(I_sans_samples)

    return ThreeWayBayesianResult(
        posterior_samples=posterior_samples,
        G_samples=G_samples_t,
        I_saxs_samples=I_saxs_samples_t,
        I_sans_samples=I_sans_samples_t,
        G_mean=G_samples_t.mean(dim=0),
        G_std=G_samples_t.std(dim=0),
        I_saxs_mean=I_saxs_samples_t.mean(dim=0),
        I_saxs_std=I_saxs_samples_t.std(dim=0),
        I_sans_mean=I_sans_samples_t.mean(dim=0),
        I_sans_std=I_sans_samples_t.std(dim=0),
        method="SVI",
        diagnostic={"final_elbo": float(elbo_history[-1]),
                     "n_svi_steps": n_steps},
    )


def joint_three_way_refine_nuts(
    *,
    crystal_tensor,
    r_pdf: torch.Tensor,
    G_obs: torch.Tensor,
    pairs,
    sigma_G: torch.Tensor,
    q_saxs: torch.Tensor,
    I_saxs: torch.Tensor,
    sigma_I_saxs: torch.Tensor,
    saxs_model: SAXSModel,
    q_sans: torch.Tensor,
    I_sans: torch.Tensor,
    sigma_I_sans: torch.Tensor,
    sans_model: SAXSModel,
    map_init: Dict[str, float],
    prior_widths: Optional[Dict[str, float]] = None,
    n_warmup: int = 100,
    n_samples: int = 200,
    verbose: bool = False,
) -> ThreeWayBayesianResult:
    """HMC (NUTS) samples from the exact three-way joint posterior.

    Slower than :func:`joint_three_way_refine_svi` but samples the
    exact posterior without a Gaussian-guide assumption.  Use SVI for
    routine work; reach for NUTS when you need to validate SVI or
    report identifiability rigorously.  Requires per-problem step-size
    and mass-matrix tuning for reliable sampling on stiff or
    multi-modal joint models --- caller responsible.
    """
    import pyro
    from pyro.infer import MCMC, NUTS
    from midas_pdf.structure import pdffit_gr

    if prior_widths is None:
        prior_widths = {"a": 0.02, "background_saxs": 5.0,
                         "background_sans": 5.0}
    else:
        prior_widths = dict(prior_widths)
        prior_widths.setdefault("a", 0.02)
        prior_widths.setdefault("background_saxs", 5.0)
        prior_widths.setdefault("background_sans", 5.0)

    r_pdf_t = torch.as_tensor(r_pdf, dtype=torch.float64)
    G_t = torch.as_tensor(G_obs, dtype=torch.float64)
    sigma_G_t = torch.as_tensor(sigma_G, dtype=torch.float64).clamp(min=1e-12)
    q_saxs_t = torch.as_tensor(q_saxs, dtype=torch.float64)
    I_saxs_t = torch.as_tensor(I_saxs, dtype=torch.float64)
    sigma_I_saxs_t = torch.as_tensor(sigma_I_saxs, dtype=torch.float64).clamp(min=1e-12)
    q_sans_t = torch.as_tensor(q_sans, dtype=torch.float64)
    I_sans_t = torch.as_tensor(I_sans, dtype=torch.float64)
    sigma_I_sans_t = torch.as_tensor(sigma_I_sans, dtype=torch.float64).clamp(min=1e-12)

    model, names = _build_three_way_pyro_model(
        crystal_tensor, r_pdf_t, G_t, pairs, sigma_G_t,
        q_saxs_t, I_saxs_t, sigma_I_saxs_t,
        q_sans_t, I_sans_t, sigma_I_sans_t,
        saxs_model, sans_model, map_init, prior_widths,
    )
    pyro.clear_param_store()
    init_params = {n: torch.tensor(float(map_init[n]), dtype=torch.float64)
                    for n in names}
    kernel = NUTS(model, jit_compile=False, adapt_step_size=True)
    mcmc = MCMC(kernel, num_samples=n_samples, warmup_steps=n_warmup,
                 initial_params=init_params, disable_progbar=(not verbose))
    mcmc.run()
    posterior_samples = mcmc.get_samples()

    # Build channel forward samples
    lat0 = crystal_tensor.lattice_params.detach()
    angles = lat0[3:]
    G_samples = []
    I_saxs_samples = []
    I_sans_samples = []
    with torch.no_grad():
        n = posterior_samples["a"].shape[0]
        for i in range(n):
            vals = {name: posterior_samples[name][i] for name in names}
            lp = torch.cat([vals["a"].reshape(1).expand(3), angles])
            G_bulk = pdffit_gr(crystal_tensor, r_pdf_t, pairs,
                                scale=vals["scale_pdf"], u_iso=vals["u_iso"],
                                lattice_params=lp)
            G_calc = G_bulk * sphere_characteristic_function(
                r_pdf_t, vals["diameter_A"])
            I_saxs_calc = saxs_model.I(q_saxs_t,
                                         D_median=vals["diameter_A"] / 2.0,
                                         scale=vals["scale_saxs"],
                                         background=vals["background_saxs"])
            I_sans_calc = sans_model.I(q_sans_t,
                                         D_median=vals["diameter_A"] / 2.0,
                                         scale=vals["scale_sans"],
                                         background=vals["background_sans"])
            G_samples.append(G_calc)
            I_saxs_samples.append(I_saxs_calc)
            I_sans_samples.append(I_sans_calc)
    G_samples_t = torch.stack(G_samples)
    I_saxs_samples_t = torch.stack(I_saxs_samples)
    I_sans_samples_t = torch.stack(I_sans_samples)

    return ThreeWayBayesianResult(
        posterior_samples=posterior_samples,
        G_samples=G_samples_t,
        I_saxs_samples=I_saxs_samples_t,
        I_sans_samples=I_sans_samples_t,
        G_mean=G_samples_t.mean(dim=0),
        G_std=G_samples_t.std(dim=0),
        I_saxs_mean=I_saxs_samples_t.mean(dim=0),
        I_saxs_std=I_saxs_samples_t.std(dim=0),
        I_sans_mean=I_sans_samples_t.mean(dim=0),
        I_sans_std=I_sans_samples_t.std(dim=0),
        method="NUTS",
        diagnostic={"n_warmup": n_warmup, "n_samples": n_samples},
    )


__all__ = ["ThreeWayBayesianResult",
            "joint_three_way_refine_svi",
            "joint_three_way_refine_nuts"]
