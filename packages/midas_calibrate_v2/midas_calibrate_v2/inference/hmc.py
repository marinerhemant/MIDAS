"""NUTS via pyro — full posterior sampling.

Slow on ~25-dim with image-scale data; gated behind ``--bayesian-full``.
Optional dependency: ``pip install pyro-ppl``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import torch

try:
    import pyro
    import pyro.distributions as dist
    from pyro.infer import MCMC, NUTS
    _HAS_PYRO = True
except ImportError:
    _HAS_PYRO = False

from ..parameters.pack import pack_spec, refined_bounds
from ..parameters.spec import CalibrationSpec


@dataclass
class HMCConfig:
    n_warmup: int = 200
    n_samples: int = 500
    step_size: float = 0.01
    target_accept_prob: float = 0.8


def hmc_run(
    spec: CalibrationSpec,
    log_likelihood_fn: Callable[[Dict[str, torch.Tensor]], torch.Tensor],
    *,
    config: HMCConfig = HMCConfig(),
    fallback_span: float = 1.0,
    dtype=torch.float64, device="cpu",
) -> Dict[str, torch.Tensor]:
    """Sample the posterior with NUTS.

    Returns a dict mapping parameter name to a [n_samples, ...] tensor of
    posterior samples.  Marginal statistics, credible intervals, etc. are
    derived from these.
    """
    if not _HAS_PYRO:
        raise RuntimeError(
            "pyro-ppl is required for HMC; install with `pip install pyro-ppl`."
        )

    pyro.clear_param_store()
    x_full, info = pack_spec(spec, dtype=dtype, device=device)
    lo, hi = refined_bounds(spec, info, fallback_span=fallback_span,
                             dtype=dtype, device=device)

    def model():
        samples: Dict[str, torch.Tensor] = {}
        for name, p in spec.parameters.items():
            if not p.refined:
                samples[name] = p.init_tensor(dtype=dtype, device=device)
                continue
            blo, bhi = p.make_logit_bounds(fallback_span=fallback_span)
            shape = p.shape if p.shape else ()
            d = dist.Uniform(blo, bhi).expand(shape).to_event(len(shape))
            v = pyro.sample(name, d)
            samples[name] = v.to(dtype=dtype)
        ll = log_likelihood_fn(samples)
        pyro.factor("data_likelihood", ll)

    nuts = NUTS(model, step_size=config.step_size,
                 target_accept_prob=config.target_accept_prob)
    mcmc = MCMC(nuts, num_samples=config.n_samples, warmup_steps=config.n_warmup)
    mcmc.run()
    return {k: v.detach() for k, v in mcmc.get_samples().items()}


__all__ = ["HMCConfig", "hmc_run"]
