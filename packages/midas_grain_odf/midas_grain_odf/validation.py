"""Fit-quality validation for per-grain ODF recovery.

After fitting, users can check whether each grain's recovered ODF
actually reproduces the measured patch shape and intensity.

Two complementary fidelity metrics are reported per grain:

  - shape_r2 (RAW): cell-wise R^2 between predicted and measured.
                    Sensitive to absolute intensity calibration AND
                    sub-pixel positional offsets.
  - median_scaled_shape_r2: per-spot R^2 AFTER fitting an optimal
                    intensity scale c_s* = <pred·meas>/<pred·pred>
                    (closed form; profile likelihood). This isolates
                    shape fidelity from absolute intensity calibration
                    and is the relevant number when comparing to
                    real-data spots whose absolute counts depend on
                    structure factor, beam, exposure, etc.

This module provides:

  - validate_fit_quality(...) -> dict
        Quantitative per-grain metrics:
            - shape_r2:       coefficient of determination of pred vs meas
                              over all (frame, y, z) cells
            - integral_r2:    same but for per-spot integrated intensity
            - mean_residual:  mean | meas - pred | (in patch units)
            - per_grain_loss: mean MSE per grain (matches fit objective)
        And, when ground-truth plants are supplied, per-particle
        capture flags at strict and loose radii.

  - plot_fit_validation(...)
        Multi-panel PNG: per-grain shape comparison (measured / predicted /
        residual) + scatter (per-spot integrated intensity, meas vs pred)
        + ODF capture summary.

Usage::

    from midas_grain_odf.validation import (
        validate_fit_quality, plot_fit_validation,
    )
    metrics = validate_fit_quality(specs, model, position, patch_F, patch_P)
    plot_fit_validation(specs, model, position, patch_F, patch_P,
                         "fit_summary.png")

Pass `plants=...` if you have planted ground truth (synthetic studies);
the validation will additionally report per-particle capture.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch

DEG = math.pi / 180.0


def _predict_per_grain(model, odf, anchors, position, patch_F, patch_P,
                        sigma_yz: float, sigma_f: float) -> torch.Tensor:
    """Forward the recovered ODF through the splatter at the grain's
    anchors. Returns (S, F, P, P)."""
    from midas_grain_odf.forward_helpers import forward_orientations
    from midas_grain_odf.spot_extract import (
        SpotPatchSpec, splat_spots_to_patches,
    )
    R_recov, w_recov = odf.sample()
    K = R_recov.shape[0]
    spots = forward_orientations(model, R_recov, position)
    sy = spots.y_pixel.reshape(K, -1)
    sz = spots.z_pixel.reshape(K, -1)
    sf = spots.frame_nr.reshape(K, -1)
    sv = spots.valid.reshape(K, -1)
    sy_sel = sy[:, anchors["spot_indexer"]]
    sz_sel = sz[:, anchors["spot_indexer"]]
    sf_sel = sf[:, anchors["spot_indexer"]]
    sv_sel = sv[:, anchors["spot_indexer"]]
    spec = SpotPatchSpec(
        n_spots=int(anchors["spot_indexer"].numel()),
        patch_F=patch_F, patch_P=patch_P,
        sigma_yz=sigma_yz, sigma_f=sigma_f,
        anchor_y=anchors["measured_y"].clone(),
        anchor_z=anchors["measured_z"].clone(),
        anchor_f=anchors["measured_f"].clone(),
    )
    return splat_spots_to_patches(spec, sy_sel, sz_sel, sf_sel,
                                    w_recov, sv_sel)


def _r2(pred: torch.Tensor, meas: torch.Tensor) -> float:
    """Coefficient of determination on raw values (no scaling)."""
    p = pred.flatten().to(torch.float64)
    m = meas.flatten().to(torch.float64)
    ss_res = ((p - m) ** 2).sum()
    ss_tot = ((m - m.mean()) ** 2).sum().clamp(min=1e-30)
    return float(1.0 - ss_res / ss_tot)


def _r2_scaled(pred: torch.Tensor, meas: torch.Tensor) -> tuple[float, float]:
    """Profile out a single optimal scale c* = <p·m> / <p·p>, then
    report the residual fit on (c* · pred, meas). Returns (R², c*).

    Profile-likelihood: this is the best-case fit when the absolute
    intensity calibration is unknown. It isolates *shape* fidelity
    from *intensity* fidelity.
    """
    p = pred.flatten().to(torch.float64)
    m = meas.flatten().to(torch.float64)
    pm = (p * m).sum()
    pp = (p * p).sum().clamp(min=1e-30)
    c = float(pm / pp)
    res = c * p - m
    ss_res = (res ** 2).sum()
    ss_tot = ((m - m.mean()) ** 2).sum().clamp(min=1e-30)
    return float(1.0 - ss_res / ss_tot), c


def validate_fit_quality(
    specs: Sequence[Any],
    model,
    position: torch.Tensor,
    *,
    patch_F: int,
    patch_P: int,
    sigma_yz: float = 1.0,
    sigma_f: float = 0.6,
    plants: Optional[Sequence[Dict[str, torch.Tensor]]] = None,
    capture_radius_strict_deg: float = 0.10,
) -> Dict[str, Any]:
    """Compute per-grain fit-quality metrics.

    Parameters
    ----------
    specs : list of MultiGrainSpec
        After fitting, with ``spec.odf`` containing the recovered ODF.
    plants : optional list of dicts
        Per-grain ground truth, each dict with keys
        ``R_avg`` (3, 3), ``R_planted`` (P, 3, 3), ``w`` (P,),
        ``aa`` (P, 3) -- the planted axis-angle vectors. When provided
        the report adds per-particle capture flags at strict and loose
        radii.

    Returns
    -------
    dict with keys
        ``per_grain``: list of dicts (one per grain) with shape_r2,
            integral_r2, mean_residual, mean_meas, captured (if plants).
        ``cluster``: aggregate metrics over all grains/spots.
    """
    from midas_grain_odf.odf import matrix_to_axis_angle
    G = len(specs)
    per_grain = []
    all_meas_int = []
    all_pred_int = []
    for gi, spec in enumerate(specs):
        anchors = dict(
            spot_indexer=spec.spot_indexer,
            measured_y=spec.measured_y,
            measured_z=spec.measured_z,
            measured_f=spec.measured_f,
        )
        with torch.no_grad():
            pred = _predict_per_grain(model, spec.odf, anchors, position,
                                       patch_F, patch_P, sigma_yz, sigma_f)
        meas = spec.measured_patches
        # cell-wise R^2 (raw and after fitting optimal per-spot intensity
        # scale c*; the latter isolates shape fidelity from absolute
        # intensity calibration, which matters for real-data deployment).
        shape_r2 = _r2(pred, meas)
        # per-spot integrated intensity
        meas_int = meas.flatten(1).sum(dim=1)
        pred_int = pred.flatten(1).sum(dim=1)
        integral_r2 = _r2(pred_int, meas_int)
        # Per-spot scale-free shape R^2: closed-form optimal c_s per spot.
        # c_s* = <pred·meas>_spot / <pred·pred>_spot
        p_flat = pred.flatten(1).to(torch.float64)
        m_flat = meas.flatten(1).to(torch.float64)
        pm = (p_flat * m_flat).sum(dim=1)
        pp = (p_flat * p_flat).sum(dim=1).clamp(min=1e-30)
        c_per_spot = pm / pp
        scaled_pred = (c_per_spot.unsqueeze(1) * p_flat)
        ss_res = ((scaled_pred - m_flat) ** 2).sum(dim=1)
        ss_tot = ((m_flat - m_flat.mean(dim=1, keepdim=True)) ** 2).sum(
            dim=1).clamp(min=1e-30)
        per_spot_shape_r2 = (1.0 - ss_res / ss_tot).cpu().numpy()
        median_scaled_shape_r2 = float(np.median(per_spot_shape_r2))
        median_scale_c = float(np.median(c_per_spot.cpu().numpy()))
        mean_residual = float((pred - meas).abs().mean())
        mean_meas = float(meas.mean())
        all_meas_int.append(meas_int.cpu())
        all_pred_int.append(pred_int.cpu())
        entry = dict(
            grain=gi,
            shape_r2=shape_r2,
            integral_r2=integral_r2,
            median_scaled_shape_r2=median_scaled_shape_r2,
            median_scale_c=median_scale_c,
            mean_residual=mean_residual,
            mean_meas=mean_meas,
            n_spots=int(spec.spot_indexer.numel()),
        )
        if plants is not None:
            plant = plants[gi]
            R_planted = plant["R_planted"]
            R_avg = plant["R_avg"]
            w_planted = plant["w"]
            aa_planted_axis = plant["aa"]
            with torch.no_grad():
                R_recov, w_recov = spec.odf.sample()
            R_inv = R_avg.transpose(-1, -2)
            aa_p = matrix_to_axis_angle(R_inv.unsqueeze(0) @ R_planted)
            aa_r = matrix_to_axis_angle(R_inv.unsqueeze(0) @ R_recov)
            radius_strict = capture_radius_strict_deg * DEG
            radius_loose = max(0.10,
                aa_planted_axis.norm(dim=-1).max().item() * 0.5 / DEG) * DEG
            captured_strict = []
            captured_loose = []
            for p in range(aa_p.shape[0]):
                d = (aa_r - aa_p[p:p+1]).norm(dim=-1)
                captured_strict.append(bool((d < radius_strict).any()))
                captured_loose.append(bool((d < radius_loose).any()))
            cap_mass_strict = sum(float(w_planted[p]) for p in range(
                aa_p.shape[0]) if captured_strict[p])
            cap_mass_loose = sum(float(w_planted[p]) for p in range(
                aa_p.shape[0]) if captured_loose[p])
            entry["captured_mass_strict"] = cap_mass_strict
            entry["captured_mass_loose"] = cap_mass_loose
            entry["per_particle"] = [
                dict(weight=float(w_planted[p]),
                     axis_angle_deg=float(aa_planted_axis[p].norm() / DEG),
                     captured_strict=captured_strict[p],
                     captured_loose=captured_loose[p])
                for p in range(aa_p.shape[0])
            ]
        per_grain.append(entry)

    cluster = dict(
        n_grains=G,
        mean_shape_r2=float(sum(g["shape_r2"] for g in per_grain) / G),
        mean_integral_r2=float(sum(g["integral_r2"] for g in per_grain) / G),
        mean_scaled_shape_r2=float(
            sum(g["median_scaled_shape_r2"] for g in per_grain) / G),
        mean_residual=float(sum(g["mean_residual"] for g in per_grain) / G),
    )
    if plants is not None:
        cluster["mean_captured_mass_loose"] = float(
            sum(g["captured_mass_loose"] for g in per_grain) / G)
        cluster["min_captured_mass_loose"] = float(
            min(g["captured_mass_loose"] for g in per_grain))
    return dict(per_grain=per_grain, cluster=cluster)


def plot_fit_validation(
    specs: Sequence[Any],
    model,
    position: torch.Tensor,
    *,
    patch_F: int,
    patch_P: int,
    sigma_yz: float = 1.0,
    sigma_f: float = 0.6,
    plants: Optional[Sequence[Dict[str, torch.Tensor]]] = None,
    output_path: str = "fit_validation.png",
    title: Optional[str] = None,
) -> Dict[str, Any]:
    """Render a per-grain validation figure and return the metrics dict.

    Layout (rows = grains, columns = panels):
      0 | brightest-spot measured patch (frame-summed)
      1 | brightest-spot predicted patch
      2 | residual
      3 | per-spot integrated intensity scatter (measured vs predicted)
      4 | per-particle capture summary (text panel; only when plants given)
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    metrics = validate_fit_quality(
        specs, model, position,
        patch_F=patch_F, patch_P=patch_P,
        sigma_yz=sigma_yz, sigma_f=sigma_f,
        plants=plants,
    )
    G = len(specs)
    n_cols = 5 if plants is not None else 4
    fig, axes = plt.subplots(G, n_cols,
                              figsize=(3.0 * n_cols, 3.0 * G),
                              squeeze=False)

    for gi, (spec, m_g) in enumerate(zip(specs, metrics["per_grain"])):
        anchors = dict(
            spot_indexer=spec.spot_indexer,
            measured_y=spec.measured_y,
            measured_z=spec.measured_z,
            measured_f=spec.measured_f,
        )
        with torch.no_grad():
            pred = _predict_per_grain(model, spec.odf, anchors, position,
                                       patch_F, patch_P, sigma_yz, sigma_f)
        meas = spec.measured_patches
        spot_brightness = meas.flatten(1).sum(dim=1)
        s_top = int(spot_brightness.argmax().item())
        m = meas[s_top].sum(dim=0).cpu().numpy()
        p = pred[s_top].sum(dim=0).detach().cpu().numpy()
        d = m - p
        vmax = max(float(m.max()), float(p.max()), 1e-30)

        # Apply optimal per-spot scale c* to predicted (profile likelihood).
        c_star = m_g["median_scale_c"]
        p_scaled = p * c_star
        d_scaled = m - p_scaled
        vmax = max(float(m.max()), float(p_scaled.max()), 1e-30)

        ax = axes[gi, 0]
        ax.imshow(m, vmin=0, vmax=vmax, cmap="viridis")
        ax.set_title(f"grain {gi}: measured\n"
                     f"raw shape R²={m_g['shape_r2']:.3f}", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])

        ax = axes[gi, 1]
        ax.imshow(p_scaled, vmin=0, vmax=vmax, cmap="viridis")
        ax.set_title(f"grain {gi}: predicted × c*\n"
                     f"scaled shape R²={m_g['median_scaled_shape_r2']:.3f}\n"
                     f"c* (median scale) = {c_star:.2f}",
                     fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])

        ax = axes[gi, 2]
        ax.imshow(d_scaled, vmin=-vmax * 0.5, vmax=vmax * 0.5,
                   cmap="seismic")
        ax.set_title(f"grain {gi}: residual (scaled)\n"
                     f"integral R²={m_g['integral_r2']:.3f}",
                     fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])

        # scatter: per-spot integrated intensity
        ax = axes[gi, 3]
        meas_int = meas.flatten(1).sum(dim=1).detach().cpu().numpy()
        pred_int = pred.flatten(1).sum(dim=1).detach().cpu().numpy()
        ax.scatter(meas_int, pred_int, s=15, alpha=0.6)
        lim = float(max(meas_int.max(), pred_int.max())) * 1.05
        ax.plot([0, lim], [0, lim], "k--", lw=1, label="y=x")
        ax.set_xlim(0, lim); ax.set_ylim(0, lim)
        ax.set_xlabel("measured integral", fontsize=9)
        ax.set_ylabel("predicted integral", fontsize=9)
        ax.set_title(
            f"grain {gi}: per-spot integral\nR²={m_g['integral_r2']:.3f}, "
            f"n_spots={m_g['n_spots']}", fontsize=9)
        ax.legend(fontsize=8)

        # particle-capture summary (if plants)
        if plants is not None:
            ax = axes[gi, 4]
            ax.axis("off")
            lines = [
                f"captured (loose): {m_g['captured_mass_loose']:.3f}",
                f"captured (strict): {m_g['captured_mass_strict']:.3f}",
                "",
                "per-planted-particle:",
            ]
            for pi, p_info in enumerate(m_g["per_particle"]):
                tag = ("✓" if p_info["captured_loose"] else "✗")
                lines.append(
                    f"  p{pi}: w={p_info['weight']:.3f}, "
                    f"|aa|={p_info['axis_angle_deg']:.2f}°  {tag}"
                )
            ax.text(0.0, 0.5, "\n".join(lines), fontsize=10,
                     family="monospace", verticalalignment="center",
                     transform=ax.transAxes)

    if title is None:
        title = (f"Fit validation: G={G}, "
                  f"mean shape R²={metrics['cluster']['mean_shape_r2']:.3f}, "
                  f"mean integral R²={metrics['cluster']['mean_integral_r2']:.3f}")
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(output_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return metrics
