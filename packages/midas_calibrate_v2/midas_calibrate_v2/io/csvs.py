"""Diagnostic CSV writers — calibrant-screen map and iteration trace.

Mirrors v1's ``calibrant_screen_out.csv`` and ``ci_profiles.csv`` so
existing analysis pipelines / paper figures can read v2's output.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np


def write_calibrant_screen_csv(
    path: str | Path,
    *,
    ring_idx: np.ndarray,
    eta_deg: np.ndarray,
    R_obs_px: np.ndarray,
    R_pred_px: np.ndarray,
    panel_idx: Optional[np.ndarray] = None,
    snr: Optional[np.ndarray] = None,
) -> None:
    """Write a per-fit calibrant-screen CSV.

    Columns: ring, η_deg, R_obs_px, R_pred_px, strain_uE, panel,
    snr (last two written only when supplied).
    """
    R_obs = np.asarray(R_obs_px, dtype=np.float64)
    R_pred = np.asarray(R_pred_px, dtype=np.float64)
    rings = np.asarray(ring_idx, dtype=np.int64)
    eta = np.asarray(eta_deg, dtype=np.float64)
    strain = (1.0 - R_obs / np.maximum(R_pred, 1e-12)) * 1e6

    cols = ["ring", "eta_deg", "R_obs_px", "R_pred_px", "strain_uE"]
    cols_data = [rings, eta, R_obs, R_pred, strain]
    if panel_idx is not None:
        cols.append("panel")
        cols_data.append(np.asarray(panel_idx, dtype=np.int64))
    if snr is not None:
        cols.append("snr")
        cols_data.append(np.asarray(snr, dtype=np.float64))

    with open(path, "w") as f:
        f.write(",".join(cols) + "\n")
        for i in range(len(rings)):
            row = []
            for c, name in zip(cols_data, cols):
                v = c[i]
                if name in ("ring", "panel"):
                    row.append(str(int(v)))
                else:
                    row.append(f"{float(v):.6e}")
            f.write(",".join(row) + "\n")


def write_iteration_trace_csv(
    path: str | Path,
    history: Sequence,
    *,
    extra_columns: Optional[dict] = None,
) -> None:
    """Write a per-iteration trace CSV from a list of ``IterRecord``-like
    dataclasses.

    Reads attributes by name; missing attributes get an empty string.
    Optionally pass ``extra_columns`` (column-name → list of values) to
    append.
    """
    if not history:
        return
    base_cols = ["iteration", "n_fitted", "cost", "rc",
                 "mean_strain_uE", "Lsd", "BC_y", "BC_z", "ty", "tz"]
    cols = list(base_cols)
    if extra_columns:
        cols += list(extra_columns.keys())
    with open(path, "w") as f:
        f.write(",".join(cols) + "\n")
        for i, rec in enumerate(history):
            row = []
            for c in base_cols:
                v = getattr(rec, c, None)
                if v is None:
                    row.append("")
                elif isinstance(v, (int, np.integer)):
                    row.append(str(int(v)))
                else:
                    row.append(f"{float(v):.6e}")
            if extra_columns:
                for c in extra_columns:
                    v = extra_columns[c][i] if i < len(extra_columns[c]) else ""
                    if isinstance(v, (int, np.integer)):
                        row.append(str(int(v)))
                    elif v == "":
                        row.append("")
                    else:
                        row.append(f"{float(v):.6e}")
            f.write(",".join(row) + "\n")


__all__ = ["write_calibrant_screen_csv", "write_iteration_trace_csv"]
