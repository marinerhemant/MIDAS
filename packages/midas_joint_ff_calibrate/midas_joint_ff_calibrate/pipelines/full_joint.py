"""Full-joint driver: refine every refined parameter at once.

Use after the alternating driver provides a good init, OR when Gaussian
priors on grain strain make the joint problem well-conditioned (paper-4
§4.4 option 3).  Reports MAP plus Laplace covariance — the headline
"σ on each per-panel parameter" number that paper-3 §9 derives but
cannot recover from a single image alone.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Optional

import torch

from midas_peakfit import (
    GenericLMConfig,
    LaplaceResult,
    ParameterSpec,
    fisher_at_map,
    lm_minimise,
)


@dataclass
class FullJointResult:
    unpacked: Dict[str, torch.Tensor]
    cost: float
    rc: int
    laplace: Optional[LaplaceResult] = None


@dataclass
class FullJointDriver:
    """Full-joint refinement of every refined parameter in the spec.

    Parameters
    ----------
    spec
        The joint :class:`ParameterSpec`.  The user is responsible for
        having set ``refined=True`` on every parameter to refine, and
        attached :class:`midas_peakfit.GaussianPrior` to any nuisance
        block (typically ``grain_lattice``) that needs regularisation.
    residual_fn
        ``unpacked -> [M]`` joint residual closure (powder + HEDM +
        gauge + prior, all wired through
        :func:`midas_joint_ff_calibrate.loss.joint_residual`).
    lm_config
        LM hyperparameters.  Default: 200 iters, ftol_rel 1e-10.
    fallback_span
        Spec-bound fallback span for unbounded parameters.
    sigma_r
        Per-row residual scale used for Fisher covariance.  Float or
        per-row tensor.  Default 1.0 (residual already scaled by user
        weights so this is interpretable as a unit-residual model).
    compute_laplace
        If True, run :func:`fisher_at_map` at the converged MAP and return
        Laplace covariance / σ for every refined dimension.
    """
    spec: ParameterSpec
    residual_fn: Callable[[Dict[str, torch.Tensor]], torch.Tensor]
    lm_config: GenericLMConfig = field(default_factory=lambda: GenericLMConfig(max_iter=200, ftol_rel=1e-10))
    fallback_span: float = 1.0
    sigma_r: float = 1.0
    compute_laplace: bool = True

    def run(self) -> FullJointResult:
        unpacked, cost, rc = lm_minimise(
            self.spec, self.residual_fn,
            config=self.lm_config,
            fallback_span=self.fallback_span,
        )
        # Hot-write MAP back to the spec's init values so a subsequent
        # alternating refinement or post-hoc analysis sees the converged
        # state.
        for n, v in unpacked.items():
            self.spec.set_init(n, v.detach())

        laplace: Optional[LaplaceResult] = None
        if self.compute_laplace:
            laplace = fisher_at_map(
                self.spec, self.residual_fn, unpacked,
                sigma_r=self.sigma_r,
                fallback_span=self.fallback_span,
            )
        return FullJointResult(unpacked=unpacked, cost=cost, rc=rc, laplace=laplace)


__all__ = ["FullJointDriver", "FullJointResult"]
