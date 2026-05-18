"""Mean-field Gaussian variational inference via pyro.

Approximates the posterior as N(μ, diag σ²) per refined dimension; trains
``μ, log σ`` via SVI.  Cheaper than HMC, gives a proper posterior, validates
the Laplace approximation.

Optional dependency: ``pip install pyro-ppl``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import torch

try:
    import pyro
    import pyro.distributions as dist
    from pyro.infer import SVI, Trace_ELBO
    from pyro.infer.autoguide import AutoDiagonalNormal
    _HAS_PYRO = True
except ImportError:
    _HAS_PYRO = False

from ..parameters.pack import (
    PackInfo, refined_indices, refined_bounds,
    write_refined_back, unpack_spec, pack_spec,
)
from ..parameters.spec import CalibrationSpec


@dataclass
class VIConfig:
    n_steps: int = 2000
    lr: float = 1e-2
    log_every: int = 200
    fallback_span: float = 1.0


def vi_run(
    spec: CalibrationSpec,
    log_likelihood_fn: Callable[[Dict[str, torch.Tensor]], torch.Tensor],
    *,
    config: VIConfig = VIConfig(),
    dtype=torch.float64, device="cpu",
    verbose: bool = False,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, Tuple[float, float]], List[float]]:
    """Run mean-field VI on the refined parameters.

    Parameters
    ----------
    log_likelihood_fn : callable
        ``unpacked -> scalar log-likelihood``.  Combined with the prior on
        each refined parameter to form the model log-density.

    Returns
    -------
    map_unpacked : dict of MAP/mean estimates (the variational posterior mean).
    margins : dict {name: (mean, std)} over refined scalars.
    elbo_trace : list of ELBO values per logged step.
    """
    if not _HAS_PYRO:
        raise RuntimeError(
            "pyro-ppl is required for VI; install with `pip install pyro-ppl`."
        )

    pyro.clear_param_store()

    x_full, info = pack_spec(spec, dtype=dtype, device=device)
    lo, hi = refined_bounds(spec, info, fallback_span=config.fallback_span,
                             dtype=dtype, device=device)
    refined_idx = refined_indices(info).to(device)
    refined_param_names: List[str] = []
    for n, r in zip(info.names, info.refined):
        if r:
            refined_param_names.append(n)

    def model():
        # Sample each refined parameter from its prior (uniform over bounds if
        # no prior was set).  Build the unpacked dict and add the data
        # log-likelihood as a scalar factor.
        samples: Dict[str, torch.Tensor] = {}
        for name, p in spec.parameters.items():
            if not p.refined:
                samples[name] = p.init_tensor(dtype=dtype, device=device)
                continue
            sl = info.slice(name)
            lo_n = lo[sl.start - sum(0 if not info.refined[i] else 0 for i in range(0))]
            # Simpler: pull bounds for this param from spec.
            blo, bhi = p.make_logit_bounds(fallback_span=config.fallback_span)
            shape = p.shape if p.shape else ()
            d = dist.Uniform(blo, bhi).expand(shape).to_event(len(shape))
            v = pyro.sample(name, d)
            samples[name] = v.to(dtype=dtype)
        ll = log_likelihood_fn(samples)
        pyro.factor("data_likelihood", ll)

    guide = AutoDiagonalNormal(model)
    optim = pyro.optim.Adam({"lr": config.lr})
    svi = SVI(model, guide, optim, loss=Trace_ELBO())

    elbo_trace: List[float] = []
    for step in range(config.n_steps):
        loss = svi.step()
        if step % config.log_every == 0:
            elbo_trace.append(-float(loss))
            if verbose:
                print(f"[vi {step:5d}] ELBO ≈ {-loss:.6e}")

    posterior_mean = guide.median()
    quantiles = guide.quantiles([0.16, 0.84])
    margins: Dict[str, Tuple[float, float]] = {}
    for name in refined_param_names:
        if name in posterior_mean:
            mean = float(posterior_mean[name].detach().reshape(-1)[0])
            q16 = float(quantiles[name][0].detach().reshape(-1)[0])
            q84 = float(quantiles[name][1].detach().reshape(-1)[0])
            margins[name] = (mean, 0.5 * (q84 - q16))

    map_unpacked: Dict[str, torch.Tensor] = {}
    for name, p in spec.parameters.items():
        if name in posterior_mean:
            map_unpacked[name] = posterior_mean[name].detach()
        else:
            map_unpacked[name] = p.init_tensor(dtype=dtype, device=device)

    return map_unpacked, margins, elbo_trace


__all__ = ["VIConfig", "vi_run"]
