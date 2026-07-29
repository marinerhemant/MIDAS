"""Δ-PDF (difference pair distribution function) for time-resolved / operando.

For two measured states a, b (e.g. two time points, two loads, two temperatures)
the difference PDF is simply

    ΔG(r) = G_b(r) - G_a(r)

The reason this belongs in a *error-propagating* pipeline: if the two
measurements are independent, the variance of the difference is

    σ²(ΔG) = σ²(G_a) + σ²(G_b)

so each ΔG(r) feature can be tested against noise — a real structural change
versus a fluctuation. That significance test is exactly what time-resolved and
operando studies need, and it is impossible without propagated uncertainty.

(Note: "Δ-PDF" here is the *difference* PDF of total-scattering powder/limited-
detector data. It is distinct from 3D-ΔPDF of single-crystal diffuse scattering
— a different measurement and reconstruction; see dev/PLAN.md.)
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch

__all__ = [
    "delta_pdf",
    "significant_mask",
    "sequence_delta_pdf",
    "significant_features",
    "cluster_significant_regions",
]


def delta_pdf(
    G_a: torch.Tensor,
    G_b: torch.Tensor,
    *,
    sigma_a: Optional[torch.Tensor] = None,
    sigma_b: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Difference PDF ``ΔG = G_b - G_a`` with propagated σ.

    Both ``G_a`` and ``G_b`` must be sampled on the same r-grid. Returns
    ``(delta_G, sigma_delta)``; ``sigma_delta`` is all-zeros if neither input
    σ is given. Assumes the two states are statistically independent
    (σ² adds); pass only the σ you have (a missing one is treated as 0).
    """
    if G_a.shape != G_b.shape:
        raise ValueError(
            f"G_a shape {tuple(G_a.shape)} != G_b shape {tuple(G_b.shape)}"
        )
    dG = G_b - G_a
    var = torch.zeros_like(dG)
    if sigma_a is not None:
        sa = torch.as_tensor(sigma_a, dtype=dG.dtype)
        var = var + sa * sa
    if sigma_b is not None:
        sb = torch.as_tensor(sigma_b, dtype=dG.dtype)
        var = var + sb * sb
    sigma_dG = torch.sqrt(var.clamp(min=0.0))
    return dG, sigma_dG


def significant_mask(
    delta_G: torch.Tensor,
    sigma_delta: torch.Tensor,
    *,
    n_sigma: float = 3.0,
) -> torch.Tensor:
    """Boolean mask where ``|ΔG| > n_sigma · σ(ΔG)`` — statistically real change.

    Points with ``σ = 0`` (no uncertainty supplied) are reported as not
    significant rather than dividing by zero.
    """
    s = torch.as_tensor(sigma_delta, dtype=delta_G.dtype)
    return (s > 0) & (delta_G.abs() > n_sigma * s)


# ---------------------------------------------------------------------------
# Rev 15: sequence Δ-PDF for time-resolved / operando series
# ---------------------------------------------------------------------------


def sequence_delta_pdf(
    G_stack: torch.Tensor,
    *,
    sigma_stack: Optional[torch.Tensor] = None,
    baseline: str | int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-frame ΔG(t, r) = G(t, r) − G_baseline(r) with propagated σ.

    Parameters
    ----------
    G_stack : (T, R) tensor
        Stack of PDFs, one row per time / load / temperature step.
    sigma_stack : (T, R) tensor, optional
        Per-frame σ(G).  If ``None``, only ΔG is returned with zeros for σ.
    baseline : ``int`` or ``"mean"``
        Which row of ``G_stack`` to use as reference, or ``"mean"`` to
        subtract the frame-averaged G.  Default ``0`` = first frame.

    Returns
    -------
    delta_G : (T, R) tensor
    sigma_delta : (T, R) tensor
        For a single-row baseline: ``σ²(ΔG_t) = σ_t² + σ_ref²`` (frames
        independent).  For ``baseline="mean"`` the frame ``t`` is itself
        inside the mean, so the correct variance (independent frames of
        equal σ) is ``σ_t² · (T-1) / T`` — the code uses the general
        formula
        ``(1-1/T)² σ_t² + Σ_{k≠t} σ_k² / T²``
        which reduces to ``σ_t² (T-1) / T`` when every σ_k is equal.
    """
    if G_stack.ndim != 2:
        raise ValueError(f"G_stack must be (T, R), got shape {tuple(G_stack.shape)}")
    T = G_stack.shape[0]

    if baseline == "mean":
        G_ref = G_stack.mean(dim=0, keepdim=True)                # (1, R)
    else:
        idx = int(baseline)
        if not (0 <= idx < T):
            raise IndexError(f"baseline row {idx} out of range for T={T}")
        G_ref = G_stack[idx:idx + 1]                             # (1, R)

    delta_G = G_stack - G_ref                                    # (T, R)

    if sigma_stack is None:
        return delta_G, torch.zeros_like(delta_G)

    if sigma_stack.shape != G_stack.shape:
        raise ValueError("sigma_stack shape must match G_stack")
    var_t = sigma_stack * sigma_stack                            # (T, R)
    if baseline == "mean":
        # ΔG_t = G_t - (1/T)Σ_k G_k = (1 - 1/T) G_t - (1/T)Σ_{k≠t} G_k
        # ⇒ Var(ΔG_t) = (1 - 1/T)² σ_t² + (1/T²) Σ_{k≠t} σ_k²
        # (frames independent; frame t and the mean are correlated because
        # frame t is in the mean, so Var(σ_t²) has a (1 - 1/T)² prefactor,
        # not 1.  Reduces to σ² (T-1)/T for equal σ_k.)
        var_total = var_t.sum(dim=0, keepdim=True)               # (1, R)
        w0 = (1.0 - 1.0 / T)
        var_delta = w0 * w0 * var_t + (var_total - var_t) / (T * T)
    else:
        var_ref = var_t[idx:idx + 1]                             # (1, R)
        var_delta = var_t + var_ref
        # Row t=idx is exactly ΔG=0 (self-diff) so σ must be 0 there — the
        # naive σ²=2σ_t² would be a spurious residual noise floor at the
        # baseline frame that a domain expert would flag.
        zero_row = torch.zeros_like(var_t)
        frame_index = torch.arange(T, device=var_t.device).unsqueeze(-1)
        var_delta = torch.where(frame_index == idx, zero_row, var_delta)
    sigma_delta = torch.sqrt(var_delta.clamp(min=0.0))
    return delta_G, sigma_delta


def significant_features(
    delta_G: torch.Tensor,
    sigma_delta: torch.Tensor,
    *,
    n_sigma: float = 3.0,
) -> torch.Tensor:
    """Per-frame per-r significance mask for a sequence.

    Parameters
    ----------
    delta_G : (T, R) tensor
    sigma_delta : (T, R) tensor
    n_sigma : float

    Returns
    -------
    mask : (T, R) bool tensor
    """
    if delta_G.shape != sigma_delta.shape:
        raise ValueError("delta_G and sigma_delta shapes must match")
    s = torch.as_tensor(sigma_delta, dtype=delta_G.dtype)
    return (s > 0) & (delta_G.abs() > n_sigma * s)


def cluster_significant_regions(
    mask: torch.Tensor,
    r: torch.Tensor,
    *,
    min_width_points: int = 2,
) -> list[list[Tuple[float, float]]]:
    """Extract contiguous-r intervals of significance, per frame.

    Returns a length-T list where element t is a list of
    ``(r_lo, r_hi)`` intervals (Å) where ``mask[t]`` is True for at least
    ``min_width_points`` consecutive r points.  Single-point islands are
    dropped by default — genuine PDF features span at least 2 r bins.
    """
    if mask.ndim == 1:
        mask = mask.unsqueeze(0)
    r_np = r.detach().cpu().numpy() if isinstance(r, torch.Tensor) else r
    out: list[list[Tuple[float, float]]] = []
    for row in mask:
        row_np = row.detach().cpu().numpy().astype(bool)
        intervals: list[Tuple[float, float]] = []
        i = 0
        n = row_np.size
        while i < n:
            if row_np[i]:
                j = i
                while j < n and row_np[j]:
                    j += 1
                if (j - i) >= min_width_points:
                    intervals.append((float(r_np[i]), float(r_np[j - 1])))
                i = j
            else:
                i += 1
        out.append(intervals)
    return out
