"""Alternating refinement driver — paper-4 recommended default (§4.4 option 2).

Outer loop:

    Pass A   freeze grain_lattice; refine geometry + grain_euler + grain_pos
    Pass B   freeze geometry + grain_euler + grain_pos; refine grain_lattice

Iterate to fixed point.  Pass A has the joint-loss multi-modality story;
Pass B is N_g independent 6-DoF problems and is cheap.

Why this works:
- Pass A's HEDM residual is well-conditioned because grain strain is
  frozen at the prior estimate; residual systematic absorption goes into
  ``Wavelength`` (anchored by the powder ring radii).
- Pass B's per-grain strain refinement sees a well-determined geometry
  and solves a well-conditioned 6-DoF least-squares per grain.
- Convergence is monotonic in the joint loss (each pass strictly decreases
  it) and typically reaches fixed point in ≤5 outer iterations for
  realistic data.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import torch

from midas_peakfit import (
    GenericLMConfig,
    ParameterSpec,
    lm_minimise,
)


@dataclass
class AlternatingResult:
    unpacked: Dict[str, torch.Tensor]
    cost_history: List[float] = field(default_factory=list)
    pass_a_costs: List[float] = field(default_factory=list)
    pass_b_costs: List[float] = field(default_factory=list)
    n_outer: int = 0
    converged: bool = False


@dataclass
class AlternatingDriver:
    """Alternating refinement of a joint spec.

    Parameters
    ----------
    spec
        Joint :class:`ParameterSpec` containing geometry blocks (Lsd,
        BC_y, BC_z, ...), optional per-panel blocks, and the three HEDM
        nuisance blocks (``grain_euler``, ``grain_pos``, ``grain_lattice``).
    residual_fn
        ``unpacked -> [M]``.  Typically built via
        :func:`midas_joint_ff_calibrate.loss.joint_residual`.
    pass_a_thaw, pass_b_thaw
        Lists of parameter names to thaw on each pass.  All other refined
        parameters in the spec are frozen for the duration of that pass.
        Default reflects the §4.4 option-2 split.
    lm_config_a, lm_config_b
        LM configs per pass (max_iter, tolerances, ...).
    n_outer_max
        Hard cap on outer iterations.
    rel_cost_tol
        Stop when the relative drop in joint loss across one full A→B
        outer iteration is below this.  Default 1e-4.
    fallback_span
        Spec-bound fallback span for parameters without explicit bounds.
    """
    spec: ParameterSpec
    residual_fn: Callable[[Dict[str, torch.Tensor]], torch.Tensor]
    pass_a_thaw: List[str] = field(default_factory=lambda: [
        # Geometry block — paper-3 standard set, plus multi-panel.
        "Lsd", "BC_y", "BC_z", "tx", "ty", "tz", "Wedge", "Wavelength",
        "panel_delta_yz", "panel_delta_theta",
        "panel_delta_lsd", "panel_delta_p2",
        # HEDM orientations + positions — strain frozen.
        "grain_euler", "grain_pos",
    ])
    pass_b_thaw: List[str] = field(default_factory=lambda: ["grain_lattice"])
    lm_config_a: GenericLMConfig = field(default_factory=lambda: GenericLMConfig(max_iter=80, ftol_rel=1e-9))
    lm_config_b: GenericLMConfig = field(default_factory=lambda: GenericLMConfig(max_iter=40, ftol_rel=1e-9))
    n_outer_max: int = 8
    rel_cost_tol: float = 1e-4
    fallback_span: float = 1.0

    def _set_refined_subset(self, names_to_thaw: List[str]) -> None:
        """Freeze all currently-refined parameters except those in
        ``names_to_thaw``.  Idempotent — restores no state from earlier
        calls.  Caller is responsible for re-freezing the appropriate
        block before the next pass.
        """
        for n, p in self.spec.parameters.items():
            if n in names_to_thaw:
                p.refined = True
            else:
                p.refined = False

    def run(self, verbose: bool = False) -> AlternatingResult:
        prior_total: Optional[float] = None
        result = AlternatingResult(unpacked={}, n_outer=0, converged=False)

        # Snapshot the originally-refined names so we can restore the spec
        # at exit (don't leave the user's spec mutated).
        original_refined = list(self.spec.refined_names())

        try:
            for k in range(self.n_outer_max):
                # ----- Pass A: geometry + grain_euler + grain_pos
                self._set_refined_subset(self.pass_a_thaw)
                unpacked_a, cost_a, rc_a = lm_minimise(
                    self.spec, self.residual_fn,
                    config=self.lm_config_a,
                    fallback_span=self.fallback_span,
                )
                result.pass_a_costs.append(cost_a)
                # Hot-restart pass B from pass A's MAP.
                for n, v in unpacked_a.items():
                    self.spec.set_init(n, v.detach())

                # ----- Pass B: grain_lattice only
                self._set_refined_subset(self.pass_b_thaw)
                unpacked_b, cost_b, rc_b = lm_minimise(
                    self.spec, self.residual_fn,
                    config=self.lm_config_b,
                    fallback_span=self.fallback_span,
                )
                result.pass_b_costs.append(cost_b)
                for n, v in unpacked_b.items():
                    self.spec.set_init(n, v.detach())

                total = cost_b   # pass B is the latest evaluated joint loss
                result.cost_history.append(total)
                if verbose:
                    print(f"[alt iter {k}] passA cost={cost_a:.6e}  "
                          f"passB cost={cost_b:.6e}")

                if prior_total is not None:
                    rel = abs(prior_total - total) / max(abs(prior_total), 1e-30)
                    if rel < self.rel_cost_tol:
                        result.converged = True
                        result.n_outer = k + 1
                        result.unpacked = unpacked_b
                        break
                prior_total = total
            else:
                # Loop exhausted without break — record the final state.
                result.unpacked = unpacked_b
                result.n_outer = self.n_outer_max
        finally:
            # Restore the original refined-name set on the user's spec.
            for n, p in self.spec.parameters.items():
                p.refined = (n in original_refined)

        return result


__all__ = ["AlternatingDriver", "AlternatingResult"]
