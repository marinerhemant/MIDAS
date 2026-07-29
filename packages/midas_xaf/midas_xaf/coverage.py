"""Reciprocal-space coverage maps for XAF-HEDM.

The signature figure of the technique: which scattering-vector **directions** are
accessible, in the common sample frame, per mounting. It makes the two
orthogonal blind cones visible and shows the cross-axis merge filling them.

Accessibility is computed analytically but consistently with the forward model:
a direction at Bragg angle ``theta`` is accessible in a mounting if, as the
sample rotates about the vertical axis, some ω-solution of the diffraction
condition falls inside an accessible wedge **and** the diffracted beam clears a
face opening (the same ω/η exit-cone gate the forward model uses).

Two views:
* :func:`direction_coverage` — a dense, material-independent grid of directions
  (pure geometry: the blind cones and their union).
* :func:`material_reflection_coverage` — a material's actual reflections plotted
  at their own Bragg angles (shows spot richness + which mounting captures each).
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from .config import XAFConfig
from . import geometry as geo

_FACE_NORMALS = np.array(
    [[1., 0, 0], [-1., 0, 0], [0, 1., 0], [0, -1., 0], [0, 0, 1.], [0, 0, -1.]])


def _fibonacci_sphere(n: int) -> np.ndarray:
    """``n`` ~evenly spaced unit vectors on the sphere."""
    i = np.arange(n) + 0.5
    phi = np.arccos(1.0 - 2.0 * i / n)
    golden = math.pi * (1.0 + 5 ** 0.5)
    theta = golden * i
    return np.stack([np.sin(phi) * np.cos(theta),
                     np.sin(phi) * np.sin(theta),
                     np.cos(phi)], axis=1)


def _accessible_at_theta(cfg: XAFConfig, dirs: np.ndarray, theta: float,
                         Rmount: np.ndarray) -> np.ndarray:
    """Bool (N,): sample-frame directions accessible in one mounting at ``theta``.

    Rotation is about lab +z; ``Rmount`` maps the sample frame into the lab at
    ω=0 for this mounting.  ``theta`` is the Bragg angle (rad).
    """
    v0 = dirs @ Rmount.T                       # lab directions at ω=0, (N,3)
    vx, vy, vz = v0[:, 0], v0[:, 1], v0[:, 2]
    rho = np.hypot(vx, vy)
    sth = math.sin(theta)
    with np.errstate(invalid="ignore", divide="ignore"):
        cosval = -sth / rho
    solvable = (rho > 1e-12) & (np.abs(cosval) <= 1.0)
    phi = np.arctan2(vy, vx)
    base = np.arccos(np.clip(cosval, -1.0, 1.0))

    half = math.radians(cfg.wedge_half_deg)
    centers = np.radians(np.asarray(cfg.wedge_centers_deg))
    cos_open = math.cos(math.radians(cfg.opening_half_deg))
    Rn = (Rmount @ _FACE_NORMALS.T).T          # face normals after remount, (6,3)
    Qm = 2.0 * sth                             # |Q| in units of 1/lambda

    acc = np.zeros(len(dirs), dtype=bool)
    for sign in (1.0, -1.0):
        omega = -phi + sign * base
        # incident wedge gate
        dwin = np.min(np.abs(((omega[:, None] - centers + math.pi)
                              % (2 * math.pi)) - math.pi), axis=1)
        in_wedge = solvable & (dwin <= half)
        # exit-cone gate: diffracted beam must clear a (rotated) face opening
        co, so = np.cos(omega), np.sin(omega)
        qx = vx * co - vy * so
        qy = vx * so + vy * co
        qz = vz
        kx, ky, kz = 1.0 + Qm * qx, Qm * qy, Qm * qz
        kn = np.sqrt(kx * kx + ky * ky + kz * kz)
        sx, sy, sz = kx / kn, ky / kn, kz / kn
        best = np.full(len(dirs), -2.0)
        for f in range(6):
            nlx = Rn[f, 0] * co - Rn[f, 1] * so
            nly = Rn[f, 0] * so + Rn[f, 1] * co
            cosang = sx * nlx + sy * nly + sz * Rn[f, 2]
            best = np.maximum(best, cosang)
        acc |= in_wedge & (best >= cos_open)
    return acc


def _coverage_labels(cfg: XAFConfig, dirs: np.ndarray,
                     thetas: np.ndarray) -> np.ndarray:
    """Bitmask per direction: bit m set if accessible in mounting m (union over
    the supplied per-direction Bragg angles)."""
    labels = np.zeros(len(dirs), dtype=int)
    uniq = np.unique(thetas)
    for m in range(cfg.n_mountings):
        Rm = np.asarray(geo.mounting_matrix(cfg, m), float)
        acc_m = np.zeros(len(dirs), dtype=bool)
        if thetas.size == len(dirs):           # per-direction theta (reflections)
            for th in uniq:
                sel = thetas == th
                acc_m[sel] |= _accessible_at_theta(cfg, dirs[sel], float(th), Rm)
        else:                                  # shared theta grid (continuous)
            for th in thetas:
                acc_m |= _accessible_at_theta(cfg, dirs, float(th), Rm)
        labels |= (acc_m.astype(int) << m)
    return labels


def direction_coverage(cfg: XAFConfig, *, n_dirs: int = 8000,
                       n_shells: int = 6) -> Tuple[np.ndarray, np.ndarray]:
    """Material-independent coverage: (dirs (N,3), bitmask labels (N,))."""
    dirs = _fibonacci_sphere(n_dirs)
    thetas = np.radians(np.linspace(1.0, cfg.tth_max_deg, n_shells) / 2.0)
    return dirs, _coverage_labels(cfg, dirs, thetas)


def material_reflection_coverage(cfg: XAFConfig
                                 ) -> Tuple[np.ndarray, np.ndarray]:
    """A material's reflection directions (crystal=sample frame) + coverage
    bitmask at each reflection's own Bragg angle."""
    from .crystal import build_reflections
    hkls, thetas, _ = build_reflections(cfg.material, cfg.wavelength_A,
                                        cfg.tth_max_deg)
    G = hkls.cpu().numpy()
    dirs = G / np.linalg.norm(G, axis=1, keepdims=True)
    return dirs, _coverage_labels(cfg, dirs, thetas.cpu().numpy())


# --------------------------------------------------------------------------- #
#  Plotting                                                                    #
# --------------------------------------------------------------------------- #
def _stereographic(dirs: np.ndarray) -> np.ndarray:
    """Upper-hemisphere stereographic projection (fold antipodes, Friedel)."""
    d = dirs.copy()
    d[d[:, 2] < 0] *= -1.0                      # fold: G and -G are equivalent
    denom = 1.0 + d[:, 2]
    return np.stack([d[:, 0] / denom, d[:, 1] / denom], axis=1)


_LABEL_STYLE = {
    0: ("neither", "#dddddd", 4),
    1: ("mounting 1 only", "tab:blue", 8),
    2: ("mounting 2 only", "tab:orange", 8),
    3: ("both", "tab:green", 8),
}


def plot_coverage(dirs: np.ndarray, labels: np.ndarray, out_path: str, *,
                  title: str = "reciprocal-space coverage",
                  show_neither: bool = True) -> str:
    """Stereographic scatter coloured by mounting coverage; saved to disk."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xy = _stereographic(dirs)
    fig, ax = plt.subplots(figsize=(6.0, 6.0))
    for lab, (name, color, size) in _LABEL_STYLE.items():
        if lab == 0 and not show_neither:
            continue
        sel = labels == lab
        if sel.any():
            ax.scatter(xy[sel, 0], xy[sel, 1], s=size, c=color,
                       label=f"{name} ({sel.sum()})", edgecolors="none")
    circ = plt.Circle((0, 0), 1.0, fill=False, color="k", lw=1.0)
    ax.add_patch(circ)
    ax.set_aspect("equal"); ax.set_xlim(-1.1, 1.1); ax.set_ylim(-1.1, 1.1)
    ax.axis("off")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    fig.tight_layout(); fig.savefig(out_path, dpi=140); plt.close(fig)
    return out_path


def plot_opening_coverage_panel(base_cfg: XAFConfig, openings_deg: Sequence[float],
                                out_path: str, *, n_dirs: int = 6000,
                                n_shells: int = 5) -> str:
    """Row of stereographic coverage maps, one per opening -- visualises the
    v2-cell gain (larger opening -> more of reciprocal space reached)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from dataclasses import replace

    n = len(openings_deg)
    fig, axes = plt.subplots(1, n, figsize=(3.7 * n, 4.2))
    axes = np.atleast_1d(axes)
    for ax, op in zip(axes, openings_deg):
        cfg = replace(base_cfg, opening_full_deg=op)
        dirs, lab = direction_coverage(cfg, n_dirs=n_dirs, n_shells=n_shells)
        fr = coverage_fraction(lab)
        xy = _stereographic(dirs)
        for l, (_n, color, size) in _LABEL_STYLE.items():
            if l == 0:
                continue
            sel = lab == l
            if sel.any():
                ax.scatter(xy[sel, 0], xy[sel, 1], s=size, c=color, edgecolors="none")
        ax.add_patch(plt.Circle((0, 0), 1.0, fill=False, color="k", lw=1.0))
        ax.set_aspect("equal"); ax.set_xlim(-1.1, 1.1); ax.set_ylim(-1.1, 1.1)
        ax.axis("off")
        ax.set_title(f"{op:.0f}°  (single {fr['single_mounting']*100:.0f}% → "
                     f"merged {fr['merged']*100:.0f}%)", fontsize=10)
    handles = [plt.Line2D([0], [0], marker="o", ls="", color=_LABEL_STYLE[k][1],
                          label=_LABEL_STYLE[k][0]) for k in (1, 2, 3)]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=9)
    fig.suptitle("XAF-HEDM reciprocal-space coverage vs face opening", y=0.99)
    fig.tight_layout(rect=[0, 0.05, 1, 0.95])
    fig.savefig(out_path, dpi=140); plt.close(fig)
    return out_path


def plot_reflection_coverage_panel(cfgs: Sequence[XAFConfig],
                                   labels_txt: Sequence[str], out_path: str) -> str:
    """Row of per-config reflection-coverage maps (e.g. materials or energies)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(cfgs)
    fig, axes = plt.subplots(1, n, figsize=(3.7 * n, 4.3))
    axes = np.atleast_1d(axes)
    for ax, cfg, title in zip(axes, cfgs, labels_txt):
        dirs, lab = material_reflection_coverage(cfg)
        xy = _stereographic(dirs)
        for l, (_n, color, size) in _LABEL_STYLE.items():
            sel = lab == l
            if sel.any():
                ax.scatter(xy[sel, 0], xy[sel, 1], s=(size if l else 3),
                           c=color, edgecolors="none")
        ax.add_patch(plt.Circle((0, 0), 1.0, fill=False, color="k", lw=1.0))
        ax.set_aspect("equal"); ax.set_xlim(-1.1, 1.1); ax.set_ylim(-1.1, 1.1)
        ax.axis("off")
        acc = int((lab > 0).sum())
        ax.set_title(f"{title}\n{len(dirs)} refl · {acc} accessible", fontsize=9)
    handles = [plt.Line2D([0], [0], marker="o", ls="", color=_LABEL_STYLE[k][1],
                          label=_LABEL_STYLE[k][0]) for k in (1, 2, 3)]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=9)
    fig.tight_layout(rect=[0, 0.05, 1, 0.97])
    fig.savefig(out_path, dpi=140); plt.close(fig)
    return out_path


def plot_mounting_progression(cfg: XAFConfig, out_path: str, *,
                              n_dirs: int = 7000, n_shells: int = 5) -> str:
    """One-vs-two-vs-three-mounting coverage, side by side.

    ``cfg`` should have ``n_mountings=3`` and orthogonal ``remount_specs``; each
    panel shows the directions reachable using the first k mountings."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dirs, labels = direction_coverage(cfg, n_dirs=n_dirs, n_shells=n_shells)
    xy = _stereographic(dirs)
    nm = cfg.n_mountings
    fig, axes = plt.subplots(1, nm, figsize=(3.7 * nm, 4.3))
    axes = np.atleast_1d(axes)
    for k, ax in enumerate(axes, start=1):
        covered = (labels & ((1 << k) - 1)) > 0     # any of first k mountings
        ax.scatter(xy[~covered, 0], xy[~covered, 1], s=4, c="#dddddd", edgecolors="none")
        ax.scatter(xy[covered, 0], xy[covered, 1], s=8, c="tab:green", edgecolors="none")
        ax.add_patch(plt.Circle((0, 0), 1.0, fill=False, color="k", lw=1.0))
        ax.set_aspect("equal"); ax.set_xlim(-1.1, 1.1); ax.set_ylim(-1.1, 1.1)
        ax.axis("off")
        ax.set_title(f"{k} mounting{'s' if k > 1 else ''}  "
                     f"({covered.mean()*100:.0f}%)", fontsize=11)
    fig.suptitle("XAF-HEDM reciprocal-space coverage: 1 vs 2 vs 3 mountings", y=0.99)
    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    fig.savefig(out_path, dpi=140); plt.close(fig)
    return out_path


def coverage_fraction(labels: np.ndarray) -> Dict[str, float]:
    """Summary: fraction of directions covered by single vs merged."""
    n = len(labels)
    single = np.mean((labels & 1) > 0)
    merged = np.mean(labels > 0)
    both = np.mean(labels == 3)
    return {"single_mounting": float(single), "merged": float(merged),
            "both": float(both), "gain": float(merged / single) if single else float("inf")}
