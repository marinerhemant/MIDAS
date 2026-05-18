"""Calibration diagnostics — auto-detection of the three known
failure modes from B1 (basis incompleteness), B6 (basin escape),
and v1's silent strain blow-up.

Three lightweight gates:

  1. ``cross_validation_gate(fits, unpacked, n_train_rings)`` — splits
     the fits by ring and asks whether the held-out residual is
     statistically distinguishable from the training residual.  A
     clean pass means the analytical model generalises.

  2. ``strain_cap_check(history, threshold_uE)`` — flags runs whose
     converged strain exceeds a calibrant threshold.  Catches every
     basin escape in the B6 sweep (strain ≥ 800 μϵ in failure cases).

  3. ``basin_check(v1_init, unpacked)`` — measures the Lsd / BC drift
     between seed and converged values.  Drift outside the B6 basin
     (±0.3 % Lsd, ±1.5 px BC) is suspicious — either the seed was
     way off, or LM walked into a side basin.

Each gate returns a :class:`DiagnosticResult` with a ``severity`` of
``"ok"``, ``"warn"``, or ``"fail"`` plus a one-line explanation.  Use
:func:`run_all_gates` to evaluate all three in one call.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, List

import torch

from ..forward.panels import PanelLayout
from ..loss.pseudo_strain import pseudo_strain_residual


@dataclass
class DiagnosticResult:
    name: str
    severity: str            # "ok" | "warn" | "fail"
    message: str
    metrics: Dict[str, float]


def cross_validation_gate(
    fits,                                 # FittedDataset with ring_idx
    unpacked: Dict[str, torch.Tensor],
    *,
    n_train_rings: Optional[int] = None,  # default: split at floor(n_rings * 2/3)
    panel_layout: Optional[PanelLayout] = None,
    fail_med_ratio: float = 1.5,         # test/train median > this → fail
    warn_med_ratio: float = 1.2,         # test/train median > this → warn
    ks_p_threshold: float = 0.01,        # KS p-value below this is suspicious
) -> DiagnosticResult:
    """Held-out-ring cross-validation gate.

    Calibrate on rings 0..N_train-1 implicit in ``unpacked`` (the caller
    should have done this); evaluate the residual on *all* fits at the
    converged geometry, then split by ring index and compare.

    Note: this gate does NOT re-fit.  It assumes ``unpacked`` is the
    converged MAP and the fits include held-out rings that were
    *not* used in the LM.  To use it as a true CV gate, call the
    calibration with ``spec.max_ring_number = n_train_rings`` (so the
    cake produces only train rings), then call this function with
    a separate fits dataset that includes the test rings.

    For an in-line check on a single calibration that used all rings,
    this gate degenerates to comparing the upper-third of rings vs the
    lower two-thirds — still informative, less rigorous.
    """
    rho_d = fits.rho_d
    with torch.no_grad():
        r_all = pseudo_strain_residual(
            fits.Y_pix, fits.Z_pix, fits.ring_two_theta_deg, unpacked,
            rho_d=rho_d, panel_layout=panel_layout,
            panel_idx=getattr(fits, "panel_idx", None),
        )
    r_uE = (r_all.abs() * 1e6).cpu().numpy()
    ring_arr = fits.ring_idx.cpu().numpy()
    rings_present = sorted(set(ring_arr))
    if len(rings_present) < 4:
        return DiagnosticResult(
            name="cross_validation",
            severity="warn",
            message=f"too few rings ({len(rings_present)}) to run CV gate",
            metrics={"n_rings": float(len(rings_present))},
        )

    if n_train_rings is None:
        n_train_rings = max(1, int(len(rings_present) * 2 / 3))
    train_mask = ring_arr < n_train_rings
    test_mask = ~train_mask
    if test_mask.sum() < 5:
        return DiagnosticResult(
            name="cross_validation",
            severity="warn",
            message=f"only {int(test_mask.sum())} test fits — split too narrow",
            metrics={"n_test": float(test_mask.sum())},
        )

    train_uE = r_uE[train_mask]
    test_uE = r_uE[test_mask]
    med_train = float(_np_median(train_uE))
    med_test = float(_np_median(test_uE))
    ratio = med_test / max(med_train, 1e-3)

    ks_p = _ks_2samp_p(train_uE, test_uE)

    if ratio > fail_med_ratio and ks_p < ks_p_threshold:
        sev = "fail"
        msg = (f"held-out median {med_test:.2f} μϵ vs train {med_train:.2f} μϵ "
               f"({ratio:+.2f}× , KS p={ks_p:.2e}) — analytical basis is "
               f"incomplete on rings ≥{n_train_rings}")
    elif ratio > warn_med_ratio and ks_p < ks_p_threshold:
        sev = "warn"
        msg = (f"held-out residual {ratio:.2f}× train, KS p={ks_p:.2e} — "
               f"borderline systematic on rings ≥{n_train_rings}")
    else:
        sev = "ok"
        msg = (f"held-out residual {ratio:.2f}× train, KS p={ks_p:.2e} — "
               f"model generalises across rings")

    return DiagnosticResult(
        name="cross_validation",
        severity=sev,
        message=msg,
        metrics={
            "med_train_uE": med_train,
            "med_test_uE": med_test,
            "ratio": ratio,
            "ks_pvalue": ks_p,
            "n_train": float(train_mask.sum()),
            "n_test": float(test_mask.sum()),
            "n_train_rings": float(n_train_rings),
        },
    )


def strain_cap_check(
    history,                              # list of IterRecord with mean_strain_uE
    *,
    threshold_uE: float = 100.0,
    warn_uE: float = 50.0,
) -> DiagnosticResult:
    """Strain-cap rejection.  B6 showed that all 24 basin-escape failures
    have converged strain ≥ 800 μϵ.  A 100 μϵ cap rejects every escape
    while never tripping on a real calibrant (all four reference
    datasets converge at < 35 μϵ).
    """
    if not history:
        return DiagnosticResult(
            name="strain_cap",
            severity="warn",
            message="no iterations recorded",
            metrics={"strain_uE": float("nan")},
        )
    final = history[-1]
    strain = float(getattr(final, "mean_strain_uE", float("nan")))
    if strain != strain:                   # NaN
        return DiagnosticResult(
            name="strain_cap",
            severity="fail",
            message="converged strain is NaN — LM diverged",
            metrics={"strain_uE": float("nan")},
        )
    if strain > threshold_uE:
        return DiagnosticResult(
            name="strain_cap",
            severity="fail",
            message=(f"converged strain {strain:.1f} μϵ exceeds calibrant cap "
                     f"{threshold_uE:.0f} μϵ — likely basin escape (B6 failure mode)"),
            metrics={"strain_uE": strain, "threshold_uE": threshold_uE},
        )
    if strain > warn_uE:
        return DiagnosticResult(
            name="strain_cap",
            severity="warn",
            message=(f"converged strain {strain:.1f} μϵ above warn level "
                     f"{warn_uE:.0f} μϵ — review residual distribution"),
            metrics={"strain_uE": strain, "warn_uE": warn_uE},
        )
    return DiagnosticResult(
        name="strain_cap",
        severity="ok",
        message=f"converged strain {strain:.2f} μϵ — within calibrant range",
        metrics={"strain_uE": strain},
    )


def basin_check(
    v1_init,                              # V1Params at the seed (before LM)
    unpacked: Dict[str, torch.Tensor],
    *,
    fail_lsd_pct: float = 5.0,           # |ΔLsd|/Lsd > this → fail
    warn_lsd_pct: float = 1.0,           # > this → warn (B6 basin edge)
    fail_bc_px: float = 50.0,            # |ΔBC| > this → fail
    warn_bc_px: float = 5.0,             # > this → warn (B6 basin edge)
) -> DiagnosticResult:
    """Compare seed (v1_init) vs converged geometry.  B6 found that
    LM always converges (100 % rate across the 48-trial sweep) but
    walks into a side basin once the seed perturbation exceeds
    ±1 % Lsd or ±5 px BC.  This gate flags runs where the converged
    geometry sits far from the seed — usually a sign that the v1
    seed was stale or LM jumped basins.
    """
    Lsd_seed = float(v1_init.Lsd)
    Lsd_final = float(unpacked["Lsd"])
    dLsd = Lsd_final - Lsd_seed
    pct = 100.0 * abs(dLsd) / max(abs(Lsd_seed), 1.0)

    BCy_seed = float(v1_init.BC_y)
    BCz_seed = float(v1_init.BC_z)
    BCy_final = float(unpacked["BC_y"])
    BCz_final = float(unpacked["BC_z"])
    dBC = ((BCy_final - BCy_seed) ** 2 + (BCz_final - BCz_seed) ** 2) ** 0.5

    metrics = {
        "Lsd_seed": Lsd_seed, "Lsd_final": Lsd_final,
        "delta_Lsd_um": dLsd, "delta_Lsd_pct": pct,
        "BC_seed_y": BCy_seed, "BC_seed_z": BCz_seed,
        "BC_final_y": BCy_final, "BC_final_z": BCz_final,
        "delta_BC_px": dBC,
    }
    if pct > fail_lsd_pct or dBC > fail_bc_px:
        return DiagnosticResult(
            name="basin_check",
            severity="fail",
            message=(f"converged geometry walked far from seed: "
                     f"ΔLsd={dLsd:+.0f} μm ({pct:+.2f} %), ΔBC={dBC:.1f} px — "
                     f"likely basin escape (B6 ≥ 1× failure regime)"),
            metrics=metrics,
        )
    if pct > warn_lsd_pct or dBC > warn_bc_px:
        return DiagnosticResult(
            name="basin_check",
            severity="warn",
            message=(f"seed-to-MAP drift: ΔLsd={dLsd:+.0f} μm ({pct:+.2f} %), "
                     f"ΔBC={dBC:.1f} px — outside the safe basin "
                     f"(±{warn_lsd_pct:.1f} % / ±{warn_bc_px:.0f} px), "
                     f"verify before downstream use"),
            metrics=metrics,
        )
    return DiagnosticResult(
        name="basin_check",
        severity="ok",
        message=(f"seed-to-MAP drift: ΔLsd={dLsd:+.0f} μm ({pct:+.2f} %), "
                 f"ΔBC={dBC:.1f} px — within safe basin"),
        metrics=metrics,
    )


def run_all_gates(
    *,
    v1_init,
    unpacked: Dict[str, torch.Tensor],
    history,
    fits=None,
    panel_layout: Optional[PanelLayout] = None,
    n_train_rings: Optional[int] = None,
    strain_threshold_uE: float = 100.0,
    strain_warn_uE: float = 50.0,
) -> List[DiagnosticResult]:
    """Run every gate, returning a list of DiagnosticResult.  The CV
    gate is skipped if ``fits`` is None.
    """
    out: List[DiagnosticResult] = []
    out.append(strain_cap_check(history,
                                  threshold_uE=strain_threshold_uE,
                                  warn_uE=strain_warn_uE))
    out.append(basin_check(v1_init, unpacked))
    if fits is not None:
        out.append(cross_validation_gate(fits, unpacked,
                                           n_train_rings=n_train_rings,
                                           panel_layout=panel_layout))
    return out


def summarise(diagnostics: List[DiagnosticResult]) -> str:
    """Compact human-readable summary of a gate-result list."""
    icon = {"ok": "✓", "warn": "⚠", "fail": "✗"}
    lines = ["Calibration diagnostics:"]
    for d in diagnostics:
        lines.append(f"  {icon.get(d.severity, '?')} [{d.name}] {d.message}")
    return "\n".join(lines)


def worst_severity(diagnostics: List[DiagnosticResult]) -> str:
    """Return ``"fail"`` if any gate failed, else ``"warn"`` if any
    warned, else ``"ok"``."""
    sev_rank = {"ok": 0, "warn": 1, "fail": 2}
    rank_sev = {0: "ok", 1: "warn", 2: "fail"}
    worst = max((sev_rank.get(d.severity, 0) for d in diagnostics), default=0)
    return rank_sev[worst]


# ------------------------------------------------------------------- helpers


def _np_median(arr) -> float:
    import numpy as np
    return float(np.median(arr))


def _ks_2samp_p(a, b) -> float:
    """Two-sample KS test p-value.  Uses scipy if available; falls
    back to Kolmogorov asymptotic p-value computed by hand (stdlib
    only) so this module has no hard scipy dep."""
    try:
        from scipy.stats import ks_2samp
        return float(ks_2samp(a, b).pvalue)
    except ImportError:
        pass
    import math
    import numpy as np
    a = np.sort(np.asarray(a, dtype=float))
    b = np.sort(np.asarray(b, dtype=float))
    na, nb = len(a), len(b)
    if na == 0 or nb == 0:
        return 1.0
    pooled = np.sort(np.concatenate([a, b]))
    cdf_a = np.searchsorted(a, pooled, side="right") / na
    cdf_b = np.searchsorted(b, pooled, side="right") / nb
    D = float(np.max(np.abs(cdf_a - cdf_b)))
    en = math.sqrt(na * nb / (na + nb))
    # Kolmogorov asymptotic CDF.
    lam = (en + 0.12 + 0.11 / en) * D
    p = 2.0 * sum((-1) ** (k - 1) * math.exp(-2 * (lam ** 2) * (k ** 2))
                  for k in range(1, 50))
    return max(min(p, 1.0), 0.0)


__all__ = [
    "DiagnosticResult",
    "cross_validation_gate",
    "strain_cap_check",
    "basin_check",
    "run_all_gates",
    "summarise",
    "worst_severity",
]
