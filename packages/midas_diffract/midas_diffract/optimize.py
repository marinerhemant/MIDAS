"""Single-grain orientation + lattice-parameter recovery via gradient descent.

Three-phase L-BFGS schedule (orientation -> lattice -> joint), with
nearest-neighbour spot association at every step. Designed for the
companion-paper FF-HEDM single-grain demo, but works for any geometry
the underlying ``HEDMForwardModel`` supports.

Quick start
-----------
    import midas_diffract as md
    result = md.optimize_single_grain(
        model,
        observed_spots=obs_angular,       # (N, 3): (2theta, eta, omega) in rad
        init_euler=init_euler_rad,        # (3,) Bunge angles in rad
        init_lattice=init_latc,           # (6,) [a, b, c, alpha, beta, gamma]
        position=torch.zeros(3),          # grain centroid in lab frame (um)
        loss=md.SpotMatchingLoss(metric="l2"),
    )
    print(result["misori_deg"], result["lattice_errors"])

The function does not assume any particular weighting; pass a
:class:`midas_diffract.SpotMatchingLoss` configured with whatever
``weights=`` you need (typically derived from measurement resolution).
"""
from __future__ import annotations

import math
from typing import Any, Callable, Dict, Optional

import torch

from .forward import HEDMForwardModel
from .losses import SpotMatchingLoss

DEG2RAD = math.pi / 180.0
RAD2DEG = 180.0 / math.pi


class NoMatchError(RuntimeError):
    """Too few observed spots matched for the optimisation to be defined.

    Raised instead of returning a sentinel loss: with no matched spots there is
    no residual, hence no gradient and no information, so any number returned
    would be a fiction that a caller could compare against a real fit.
    """

    def __init__(self, message: str, *, n_matched: int):
        super().__init__(message)
        self.n_matched = n_matched


def _associate(pred_valid: torch.Tensor, observed: torch.Tensor,
               max_dist: float) -> "tuple[torch.Tensor, torch.Tensor]":
    """Nearest-neighbour observed->predicted association.

    Returns ``(pred_matched, obs_matched)``. Both have leading dim equal to
    the number of observed spots whose nearest predicted neighbour is
    within ``max_dist``.
    """
    dists = torch.cdist(observed, pred_valid)
    min_dists, nn_idx = dists.min(dim=1)
    keep = min_dists < max_dist
    return pred_valid[nn_idx[keep]], observed[keep]


def optimize_single_grain(
    model: HEDMForwardModel,
    observed_spots: torch.Tensor,
    init_euler: torch.Tensor,
    init_lattice: torch.Tensor,
    position: Optional[torch.Tensor] = None,
    *,
    loss: Optional[SpotMatchingLoss] = None,
    max_match_distance: float = 0.5,
    min_matches: int = 5,
    phase1_steps: int = 15,
    phase2_steps: int = 15,
    phase3_steps: int = 10,
    lbfgs_max_iter: int = 20,
    convergence_misori_deg: float = 1e-3,
    convergence_lattice_err: float = 1e-5,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Recover Bunge Euler angles and lattice parameters for a single grain.

    Three-phase L-BFGS schedule:
      1. Orientation only (Euler angles) -- eta/omega-sensitive.
      2. Lattice parameters only -- 2theta-sensitive.
      3. Joint refinement.

    Parameters
    ----------
    model : HEDMForwardModel
        Pre-built forward model. Its ``hkls`` and ``thetas`` set the
        reflection list against which the grain is fit.
    observed_spots : Tensor (N, 3)
        Angular coordinates of observed spots in radians:
        ``(2theta, eta, omega)``. Use the ``angular`` output of
        :meth:`HEDMForwardModel.predict_spot_coords` for synthetic data.
    init_euler : Tensor (3,)
        Initial Bunge Euler angles in radians.
    init_lattice : Tensor (6,)
        Initial lattice parameters ``[a, b, c, alpha, beta, gamma]``,
        in Angstroms / degrees.
    position : Tensor (3,), optional
        Grain centroid in the lab frame (microns). Defaults to the origin.
    loss : SpotMatchingLoss, optional
        Loss object. Defaults to ``SpotMatchingLoss(metric="l2")``.
    max_match_distance : float
        Discard observed spots whose nearest predicted neighbour is
        farther than this in the angular metric (radians).
    min_matches : int
        If fewer than this many spots are matched at any iteration, the fit is
        not defined and the function returns a record with ``success=False``
        and ``loss_history=[inf]`` rather than a sentinel value that could be
        mistaken for a converged fit.
    phase1_steps, phase2_steps, phase3_steps : int
        Outer-loop step counts per phase. Each step is one L-BFGS call
        with up to ``lbfgs_max_iter`` inner iterations.
    convergence_misori_deg, convergence_lattice_err : float
        Early-exit thresholds for phases 1, 2, and 3.
    verbose : bool
        If True, print a per-step progress table.

    Returns
    -------
    dict with keys:
        ``euler_rad`` -- (3,) recovered Euler angles in radians.
        ``euler_deg`` -- (3,) same in degrees.
        ``lattice``   -- (6,) recovered lattice parameters.
        ``misori_deg``-- final misorientation against ``init_euler`` (deg)
                         when no ground truth is supplied; see
                         :func:`evaluate_recovery` for ground-truth eval.
        ``loss_history`` -- list of phase-final loss values.
        ``success``   -- False if too few spots ever matched; the parameters
                         are then the caller's own seed, ``loss_history`` is
                         ``[inf]``, and ``failure_reason`` says why.
        ``n_matched`` -- spots matched at the last evaluated state.
    """
    if loss is None:
        loss = SpotMatchingLoss(metric="l2")
    if position is None:
        position = torch.zeros(3, dtype=init_euler.dtype, device=init_euler.device)

    pos = position.unsqueeze(0)
    R_init = HEDMForwardModel.euler2mat(init_euler).detach()

    opt_euler = init_euler.clone().requires_grad_(True)
    opt_latc = init_lattice.clone().requires_grad_(False)
    loss_history: list = []
    _last_n_matched = [0]

    def make_closure(params):
        def closure():
            for p in params:
                if p.grad is not None:
                    p.grad.zero_()
            spots = model(opt_euler.unsqueeze(0), pos, lattice_params=opt_latc)
            coords, valid = HEDMForwardModel.predict_spot_coords(
                spots, space="angular"
            )
            pred_flat = coords.squeeze().reshape(-1, 3)
            valid_flat = valid.squeeze().reshape(-1)
            pred_valid = pred_flat[valid_flat > 0.5]
            # These two early-outs used to return torch.tensor(1e6,
            # requires_grad=True) -- a FRESH LEAF, disconnected from opt_euler,
            # on which .backward() was never called. So opt_euler.grad stayed
            # None, L-BFGS took no step, and optimize_single_grain returned
            # loss_history = [1e6, 1e6, 1e6] with the parameters moved exactly
            # 0.0 and no error raised. A constant is the worst possible signal
            # here: it looks like a fit that plateaued.
            #
            # There is nothing to restore -- with no matched spots there are no
            # observations and hence no gradient. Raise instead, and let the
            # caller record the failure.
            if pred_valid.shape[0] == 0:
                raise NoMatchError(
                    "no valid predicted spots at the current state", n_matched=0
                )
            pred_match, obs_match = _associate(
                pred_valid, observed_spots, max_match_distance
            )
            if pred_match.shape[0] < min_matches:
                raise NoMatchError(
                    f"only {pred_match.shape[0]} spots matched, need "
                    f"{min_matches}",
                    n_matched=int(pred_match.shape[0]),
                )
            _last_n_matched[0] = int(pred_match.shape[0])
            l = loss(pred_match, obs_match)
            l.backward()
            return l
        return closure

    def current_misori_deg() -> float:
        with torch.no_grad():
            R_cur = HEDMForwardModel.euler2mat(opt_euler)
            trace = torch.trace(R_init.T @ R_cur)
            return torch.acos(torch.clamp((trace - 1) / 2, -1, 1)).item() * RAD2DEG

    def log(step: int, l: torch.Tensor) -> None:
        if not verbose:
            return
        misori = current_misori_deg()
        lat_err = (opt_latc[:3] - init_lattice[:3]).abs().max().item()
        print(f"{step:5d}  {l.item():12.6e}  {misori:12.6f}  {lat_err:10.6f}")

    def _failed(exc: "NoMatchError") -> Dict[str, Any]:
        """Report a fit that never had enough data to be defined.

        ``loss_history`` is inf rather than a finite constant so a caller
        cannot compare it against a real fit and prefer it, and ``success`` is
        False so "the parameters did not move" is distinguishable from "the fit
        converged".
        """
        return {
            "euler_rad": opt_euler.detach().clone(),
            "euler_deg": opt_euler.detach().clone() * RAD2DEG,
            "lattice": opt_latc.detach().clone(),
            "misori_deg": current_misori_deg(),
            "loss_history": [float("inf")],
            "success": False,
            "n_matched": exc.n_matched,
            "failure_reason": str(exc),
        }

    if verbose:
        print(f"{'Step':>5}  {'Loss':>12}  {'dMisori(deg)':>12}  {'dLat':>10}")
        print("-" * 55)
        print("--- Phase 1: Orientation ---")

    optimizer = torch.optim.LBFGS(
        [opt_euler], lr=1.0, max_iter=lbfgs_max_iter,
        line_search_fn="strong_wolfe",
    )
    try:
        for step in range(phase1_steps):
            l = optimizer.step(make_closure([opt_euler]))
            log(step, l)
            if current_misori_deg() < convergence_misori_deg and step > 0:
                break
    except NoMatchError as exc:
        return _failed(exc)
    loss_history.append(float(l.detach()))

    if verbose:
        print("--- Phase 2: Lattice parameters ---")
    opt_euler.requires_grad_(False)
    opt_latc.requires_grad_(True)
    optimizer = torch.optim.LBFGS(
        [opt_latc], lr=1.0, max_iter=lbfgs_max_iter,
        line_search_fn="strong_wolfe",
    )
    try:
        for step in range(phase2_steps):
            l = optimizer.step(make_closure([opt_latc]))
            log(step + phase1_steps, l)
            if (opt_latc[:3] - init_lattice[:3]).abs().max().item() > 0 \
                    and abs(float(l.detach()) - loss_history[-1]) < convergence_lattice_err:
                break
    except NoMatchError as exc:
        return _failed(exc)
    loss_history.append(float(l.detach()))

    if verbose:
        print("--- Phase 3: Joint refinement ---")
    opt_euler.requires_grad_(True)
    opt_latc.requires_grad_(True)
    optimizer = torch.optim.LBFGS(
        [opt_euler, opt_latc], lr=0.5, max_iter=lbfgs_max_iter,
        line_search_fn="strong_wolfe",
    )
    try:
        for step in range(phase3_steps):
            l = optimizer.step(make_closure([opt_euler, opt_latc]))
            log(step + phase1_steps + phase2_steps, l)
    except NoMatchError as exc:
        return _failed(exc)
    loss_history.append(float(l.detach()))

    return {
        "euler_rad": opt_euler.detach().clone(),
        "euler_deg": opt_euler.detach().clone() * RAD2DEG,
        "lattice": opt_latc.detach().clone(),
        "misori_deg": current_misori_deg(),
        "loss_history": loss_history,
        "success": True,
        "n_matched": _last_n_matched[0],
    }


def evaluate_recovery(
    result: Dict[str, Any],
    gt_euler: torch.Tensor,
    gt_lattice: torch.Tensor,
) -> Dict[str, float]:
    """Evaluate a recovery against ground truth.

    Returns misorientation (deg) and per-element lattice errors.
    Useful in unit tests and the demo notebooks.
    """
    R_gt = HEDMForwardModel.euler2mat(gt_euler)
    R_rec = HEDMForwardModel.euler2mat(result["euler_rad"])
    trace = torch.trace(R_gt.T @ R_rec)
    misori = torch.acos(torch.clamp((trace - 1) / 2, -1, 1)).item() * RAD2DEG
    lat_err = (result["lattice"] - gt_lattice).abs()
    return {
        "misori_deg": misori,
        "lattice_max_err": lat_err.max().item(),
        "lattice_errors": lat_err.detach().clone(),
    }
