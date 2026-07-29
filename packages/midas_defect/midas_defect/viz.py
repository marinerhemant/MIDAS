"""Plotly HTML renderers for midas_defect.

* `render_rod_overlay_html(cloud, rods, *, shells, html_path, ...)` —
  3-panel HTML: (1) detector × ω cloud, (2) q-space cloud with detected
  rods overlaid as colored line segments + (optional) CuAl₂ shell halo,
  (3) per-rod 1-D intensity profile.

Imports plotly lazily so the package can be installed without it.
"""

from __future__ import annotations

from collections import defaultdict
from typing import List, Optional, Sequence

import math
import numpy as np

from .data_io import VoxelCloud
from .lattice import Shell, cual2_crystal, tetragonal_shells
from .rod_detect import QRod


__all__ = ["render_rod_overlay_html"]


def _sphere_uniform(R: float, n: int, seed: int) -> tuple:
    rng = np.random.default_rng(seed)
    p = rng.standard_normal((n, 3))
    p /= np.linalg.norm(p, axis=1, keepdims=True)
    return R * p[:, 0], R * p[:, 1], R * p[:, 2]


def _stratified_subsample(intensity: np.ndarray, *, n_bright: int, n_haze: int
                          ) -> np.ndarray:
    """Top-N-brightest + random-N-faint indices."""
    n_total = len(intensity)
    n_bright = min(n_bright, n_total)
    order = np.argsort(intensity)[::-1]
    bright_idx = order[:n_bright]
    rest = order[n_bright:]
    n_haze = min(n_haze, len(rest))
    rng = np.random.default_rng(0)
    haze_idx = rng.choice(rest, size=n_haze, replace=False) if n_haze > 0 else np.array([], int)
    return np.concatenate([bright_idx, haze_idx]), n_bright


def render_rod_overlay_html(
    cloud: VoxelCloud,
    rods: List[QRod],
    *,
    crystal=None,
    html_path: str,
    n_bright: int = 80_000,
    n_haze: int = 200_000,
    shell_qmax_pct: float = 99.5,
    title: Optional[str] = None,
) -> str:
    """Render a 3-panel HTML showing voxel cloud + detected rods.

    Returns the path written.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    if crystal is None:
        crystal = cual2_crystal()

    # subsample for plotly
    sel, n_bright_kept = _stratified_subsample(cloud.intensity,
                                               n_bright=n_bright, n_haze=n_haze)
    qx_s = cloud.qx[sel]; qy_s = cloud.qy[sel]; qz_s = cloud.qz[sel]
    I_s  = cloud.intensity[sel]
    dy_s = cloud.det_y_px[sel] * cloud.geometry.px_um / 1000.0   # mm
    dz_s = cloud.det_z_px[sel] * cloud.geometry.px_um / 1000.0
    om_s = cloud.omega_deg[sel]
    is_bright = np.zeros(len(sel), bool)
    is_bright[:n_bright_kept] = True
    color = np.log10(I_s + 1.0)

    # shell overlay
    q_inv_max = float(np.percentile(np.sqrt(qx_s ** 2 + qy_s ** 2 + qz_s ** 2),
                                    shell_qmax_pct))
    shells = tetragonal_shells(crystal, q_max_inv_A=q_inv_max)
    sx, sy, sz, sq, slab = [], [], [], [], []
    for i, s in enumerate(shells):
        n_per = int(np.clip(60 + 12 * s.q_inv_A, 60, 180))
        x, y, z = _sphere_uniform(s.q_inv_A, n_per, seed=int(s.q_inv_A * 1e6) + i)
        sx.append(x); sy.append(y); sz.append(z)
        sq.append(np.full(n_per, s.q_inv_A))
        head = ", ".join(f"({h}{k}{l})" for (h, k, l) in s.hkls[:4])
        if len(s.hkls) > 4:
            head += f" … ({len(s.hkls)})"
        slab.extend([f"|q|={s.q_inv_A:.3f} 1/Å<br>{head}"] * n_per)
    sx = np.concatenate(sx); sy = np.concatenate(sy); sz = np.concatenate(sz)
    sq = np.concatenate(sq)

    # build figure
    fig = make_subplots(
        rows=2, cols=2,
        specs=[[{"type": "scene"}, {"type": "scene"}],
               [{"type": "xy", "colspan": 2}, None]],
        row_heights=[0.6, 0.4],
        subplot_titles=(
            "Detector × ω voxels",
            f"Q-space voxels + {len(rods)} detected rod(s) "
            f"(CuAl₂ shells overlay)",
            "Per-rod 1-D intensity profile",
        ),
        horizontal_spacing=0.04, vertical_spacing=0.10,
    )

    # 0: det-omega haze + 1: det-omega bright
    fig.add_trace(go.Scatter3d(
        x=dy_s[~is_bright], y=dz_s[~is_bright], z=om_s[~is_bright],
        mode="markers",
        marker=dict(size=1.1, color=color[~is_bright], colorscale="Hot",
                    opacity=0.18, showscale=False),
        name="det-ω haze", showlegend=False, hoverinfo="skip",
    ), row=1, col=1)
    fig.add_trace(go.Scatter3d(
        x=dy_s[is_bright], y=dz_s[is_bright], z=om_s[is_bright],
        mode="markers",
        marker=dict(size=2.0, color=color[is_bright], colorscale="Hot",
                    opacity=0.85, showscale=False),
        name="det-ω bright", showlegend=False,
        hovertemplate="detY=%{x:.1f}mm<br>detZ=%{y:.1f}mm<br>ω=%{z:.2f}°<extra></extra>",
    ), row=1, col=1)

    # 2: q-space haze
    fig.add_trace(go.Scatter3d(
        x=qx_s[~is_bright], y=qy_s[~is_bright], z=qz_s[~is_bright],
        mode="markers",
        marker=dict(size=1.1, color=color[~is_bright], colorscale="Viridis",
                    opacity=0.18, showscale=False),
        name="q-space haze", showlegend=False, hoverinfo="skip",
    ), row=1, col=2)
    # 3: q-space bright
    fig.add_trace(go.Scatter3d(
        x=qx_s[is_bright], y=qy_s[is_bright], z=qz_s[is_bright],
        mode="markers",
        marker=dict(size=2.0, color=color[is_bright], colorscale="Viridis",
                    opacity=0.85, colorbar=dict(title="log10(I)", x=1.02, y=0.78, len=0.5)),
        name="q-space bright", showlegend=False,
        hovertemplate="qx=%{x:.2f}<br>qy=%{y:.2f}<br>qz=%{z:.2f}<extra></extra>",
    ), row=1, col=2)
    # 4: CuAl₂ shell halo
    fig.add_trace(go.Scatter3d(
        x=sx, y=sy, z=sz, mode="markers",
        marker=dict(size=1.2, color=sq, colorscale="Greys",
                    opacity=0.18, showscale=False),
        hovertext=slab, hoverinfo="text",
        name=f"CuAl₂ shells ({len(shells)})",
        showlegend=True,
    ), row=1, col=2)
    # 5..(5+R-1): rod line segments (one trace each, colored by rank)
    palette = ["#ff3333", "#33aaff", "#33ff33", "#ffaa33", "#aa33ff",
               "#ff33aa", "#33ffaa", "#aaff33"]
    for ri, r in enumerate(rods[:len(palette)]):
        # two endpoints of the rod segment
        p0 = r.pivot + r.t_min * r.direction
        p1 = r.pivot + r.t_max * r.direction
        col = palette[ri % len(palette)]
        # label: defect_normal_hkl if available, else direction
        hkl = r.defect_normal_hkl
        name = (f"rod {ri+1}: ({hkl[0]}{hkl[1]}{hkl[2]}), "
                f"len={r.length:.2f}, ΣI={r.integrated_intensity:.0f}"
                if hkl is not None
                else f"rod {ri+1}: dir=({r.direction[0]:.2f},"
                     f"{r.direction[1]:.2f},{r.direction[2]:.2f}), "
                     f"len={r.length:.2f}, ΣI={r.integrated_intensity:.0f}")
        fig.add_trace(go.Scatter3d(
            x=[p0[0], p1[0]], y=[p0[1], p1[1]], z=[p0[2], p1[2]],
            mode="lines",
            line=dict(color=col, width=6),
            name=name, showlegend=True,
        ), row=1, col=2)

    # 2-D rod profile panel (bottom)
    for ri, r in enumerate(rods[:len(palette)]):
        col = palette[ri % len(palette)]
        if r.profile_t is not None and r.profile_I is not None:
            fig.add_trace(go.Scatter(
                x=r.profile_t, y=r.profile_I,
                mode="markers", marker=dict(size=4, color=col, opacity=0.6),
                name=f"rod {ri+1} profile", showlegend=False,
            ), row=2, col=1)

    title_str = title or (
        f"Demk CuAl₂ layer file {cloud.layer_start_filenr}  —  "
        f"{cloud.n_voxels()} voxels  —  threshold {cloud.threshold:.0f}  —  "
        f"{len(rods)} rods detected"
    )

    fig.update_layout(
        height=1250, width=1500, title=title_str,
        margin=dict(l=10, r=10, t=80, b=40),
    )
    fig.update_scenes(xaxis_title="det Y (mm)", yaxis_title="det Z (mm)",
                      zaxis_title="ω (deg)", aspectmode="cube", row=1, col=1)
    fig.update_scenes(xaxis_title="q_x (1/Å)", yaxis_title="q_y (1/Å)",
                      zaxis_title="q_z (1/Å)", aspectmode="data", row=1, col=2)
    fig.update_xaxes(title="t along rod (1/Å)", row=2, col=1)
    fig.update_yaxes(title="intensity", row=2, col=1)

    fig.write_html(html_path, include_plotlyjs="cdn")
    return html_path
