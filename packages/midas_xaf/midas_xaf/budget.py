"""Beam-time budget model.

Estimates total measurement time for an XAF-HEDM configuration and exposes the
key planning trade: box beam is fast but position-poor; point/line scanning
localises grains and beats overlap but multiplies the time by the number of beam
positions. Also relates strain precision to the number of grains averaged for the
micromechanics goal.
"""
from __future__ import annotations

import math
from typing import Dict

import numpy as np

from .config import XAFConfig


def beamtime_estimate(
    cfg: XAFConfig,
    *,
    exposure_s: float = 1.0,
    readout_s: float = 0.005,
    overhead_per_scan_s: float = 30.0,
    remount_overhead_min: float = 30.0,
) -> Dict[str, float]:
    """Total frames and wall-clock time for a configuration."""
    wedge_width = 2.0 * cfg.wedge_half_deg
    frames_per_wedge = wedge_width / abs(cfg.omega_step_deg)
    frames_per_mounting_per_scan = frames_per_wedge * len(cfg.wedge_centers_deg)

    if cfg.beam_mode == "box":
        scans_per_mounting = 1
    else:
        step = max(cfg.beam_size_um, 1e-6)
        scans_per_mounting = int(np.ceil(2.0 * cfg.sample_radius_um / step)) + 1

    frames_total = (frames_per_mounting_per_scan * scans_per_mounting
                    * cfg.n_mountings)
    exposure_time = frames_total * (exposure_s + readout_s)
    scan_overhead = scans_per_mounting * cfg.n_mountings * overhead_per_scan_s
    remount_overhead = (cfg.n_mountings - 1) * remount_overhead_min * 60.0
    total_s = exposure_time + scan_overhead + remount_overhead
    return {
        "beam_mode": cfg.beam_mode,
        "n_mountings": cfg.n_mountings,
        "scans_per_mounting": scans_per_mounting,
        "frames_total": int(frames_total),
        "exposure_hours": exposure_time / 3600.0,
        "overhead_hours": (scan_overhead + remount_overhead) / 3600.0,
        "total_hours": total_s / 3600.0,
    }


def grains_needed_for_stress(
    per_grain_strain_ue: float,
    target_macro_stress_MPa: float,
    E_GPa: float = 400.0,
) -> int:
    """Grains to average so the macro-stress error meets a target.

    Macro-stress error scales as ``E * (per_grain_strain / sqrt(N))``; invert
    for the required N.
    """
    E_MPa = E_GPa * 1e3
    per_grain_stress = E_MPa * per_grain_strain_ue * 1e-6
    if target_macro_stress_MPa <= 0:
        return 0
    return int(math.ceil((per_grain_stress / target_macro_stress_MPa) ** 2))
