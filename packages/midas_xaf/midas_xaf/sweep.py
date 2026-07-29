"""Configuration sweeps + plots for XAF-HEDM design.

Vary one knob (opening angle, energy, distance, material, ...) and tabulate the
design metrics, so we can read off the best configuration -- and, for the v2-cell
decision, exactly how strain determinability improves from a 15 deg to a 20 deg
opening.

The headline plot is **strain sensitivity (s_min) vs opening angle**, for a
single mounting vs the merged cross-axis measurement.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .config import XAFConfig
from .forward import XAFForwardModel
from .sample import make_sample
from . import geometry, metrics


def evaluate_config(cfg: XAFConfig, *, with_gain: bool = True) -> Dict[str, Any]:
    """Compute the full metric row for one configuration."""
    fwd = XAFForwardModel(cfg)
    grains = make_sample(cfg)
    sim = fwd.simulate(grains)
    spg = metrics.spots_per_grain(sim)
    gsum = geometry.geometry_summary(cfg)

    row: Dict[str, Any] = {
        "opening_full_deg": cfg.opening_full_deg,
        "energy_keV": cfg.energy_keV,
        "material": cfg.material,
        "Lsd_mm": gsum["Lsd_mm"],
        "tth_max_deg": gsum["tth_max_deg"],
        "detector_limited": gsum["detector_limited"],
        "median_spots_per_grain": float(np.median(spg)),
        "min_spots_per_grain": int(spg.min()),
        "friedel_completeness": metrics.friedel_completeness(sim),
        "n_accessible_spots": len(sim.table),
    }
    if with_gain:
        gain = metrics.cross_axis_gain(fwd, grains)
        row.update(gain)
    return row


def sweep(
    base: XAFConfig,
    field: str,
    values: Sequence[Any],
    *,
    with_gain: bool = True,
    verbose: bool = True,
) -> List[Dict[str, Any]]:
    """Sweep ``field`` over ``values``, returning one metric row per value."""
    rows = []
    for v in values:
        cfg = replace(base, **{field: v})
        row = evaluate_config(cfg, with_gain=with_gain)
        row["_swept_field"] = field
        row["_swept_value"] = v
        rows.append(row)
        if verbose:
            g = (f" smin {row.get('median_s_min_single', float('nan')):.3f}->"
                 f"{row.get('median_s_min_merged', float('nan')):.3f} "
                 f"(x{row.get('s_min_gain', float('nan')):.2f})") if with_gain else ""
            print(f"[{field}={v}] Lsd={row['Lsd_mm']:.0f}mm "
                  f"spots/gr={row['median_spots_per_grain']:.0f} "
                  f"Friedel={row['friedel_completeness']:.2f}{g}")
    return rows


def sweep_opening(
    base: Optional[XAFConfig] = None,
    openings_deg: Sequence[float] = (10.0, 12.0, 15.0, 18.0, 20.0, 25.0, 30.0),
    **kw,
) -> List[Dict[str, Any]]:
    """Headline sweep: strain determinability vs face-opening full-cone angle."""
    base = base or XAFConfig(material="zirconia_monoclinic", n_grains=25)
    return sweep(base, "opening_full_deg", openings_deg, **kw)


# --------------------------------------------------------------------------- #
#  Plotting (optional matplotlib dependency)                                  #
# --------------------------------------------------------------------------- #
def plot_opening_sweep(rows: List[Dict[str, Any]], out_path: str) -> str:
    """Plot strain sensitivity + coverage vs opening angle; save to ``out_path``."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = [r["_swept_value"] for r in rows]
    prec_s = [r.get("strain_precision_ue_single", np.nan) for r in rows]
    prec_m = [r.get("strain_precision_ue_merged", np.nan) for r in rows]
    spg = [r["median_spots_per_grain"] for r in rows]
    fr = [r["friedel_completeness"] for r in rows]

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11, 4.2))
    ax0.plot(x, prec_s, "o-", label="single mounting", color="tab:gray")
    ax0.plot(x, prec_m, "s-", label="merged (cross-axis)", color="tab:red")
    ax0.axvline(15.0, ls=":", color="k", alpha=0.5)
    ax0.axvline(23.0, ls=":", color="tab:blue", alpha=0.6)
    ax0.set_yscale("log")
    ax0.set_xlabel("face opening (full cone, deg)")
    ax0.set_ylabel(r"worst-direction strain precision  $1\sigma$  ($\mu\varepsilon$)")
    ax0.set_title("Strain precision (CRLB) vs opening  [lower = better]")
    ax0.legend(); ax0.grid(alpha=0.3, which="both")

    ax1b = ax1.twinx()
    ax1.plot(x, spg, "^-", color="tab:green", label="spots/grain")
    ax1b.plot(x, fr, "d-", color="tab:purple", label="Friedel completeness")
    ax1.set_xlabel("face opening (full cone, deg)")
    ax1.set_ylabel("median spots per grain", color="tab:green")
    ax1b.set_ylabel("Friedel completeness", color="tab:purple")
    ax1b.set_ylim(0, 1.02)
    ax1.set_title("Coverage vs opening")
    ax1.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path
