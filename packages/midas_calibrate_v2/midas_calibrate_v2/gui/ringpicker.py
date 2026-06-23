"""Manual ring-picker GUI for midas-calibrate-v2.

A lightweight Tkinter + matplotlib tool that produces a sub-pixel
``(BC_y, BC_z, Lsd)`` seed by clicking on visible calibrant arcs.

When to use
-----------
The fully-automated auto-seed in :func:`first_time_calibrate` works on a
well-centred beam with full rings.  This GUI is the fallback for cases
where the auto-seed cannot work:

  * **Off-detector beam** — the auto-seed's chord-bisector assumes BC
    lies inside the image.
  * **Limited arcs only** — Pilatus-class panel-gap geometry or large
    beamstop arms occlude most of each ring.
  * **Sparse / noisy rings** where multi-hypothesis Lsd matching latches
    onto the wrong sim-ring index.

Workflow
--------
  1. Hover to read pixel coordinates.
  2. Switch to *Arc-fit BC*, click 5–12 points along ONE visible ring,
     then press *Fit BC from arc*.  Kåsa → geometric LM gives a sub-px
     centre and the ring radius, from which one *Lsd* estimate is
     derived using the selected hkl's 2θ.
  3. Repeat for additional rings in *Pick ring* mode (any number).
  4. Press *Joint fit* to lock the centre across every picked ring and
     solve for a single (BC, Lsd) that explains all radii together.
     This is the result you save.
  5. *Save seed JSON* — a one-shot seed file consumable verbatim by
     :func:`first_time_calibrate` with ``auto_seed=False``.

Usage (CLI)
-----------
::

   midas-calibrate-v2-pick path/to/CeO2.h5 \\
       --calibrant CeO2 --wavelength 0.1839 --pixel-um 150 \\
       --data-loc exchange/data --dark path/to/dark.h5 \\
       --output seed.json

   midas-calibrate-v2-pick CeO2.tif \\
       --space-group 225 --lattice 5.41165 \\
       --wavelength 0.1839 --pixel-um 200 --output seed.json

Calibrant presets
-----------------
``CeO2`` (SRM 674b), ``LaB6`` (SRM 660c), ``Si`` (SRM 640), ``Ni``,
``Al2O3``, ``Cr2O3``.  Any other calibrant can be given as
``--space-group N --lattice a [b c alpha beta gamma]``.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from itertools import cycle
from pathlib import Path
from typing import List, Optional, Tuple

# Tk + matplotlib OpenMP collision is the recurring "OMP error 15" surface.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np

from ..io.readers import read_image, read_dark
from ._circle_fit import (
    kasa_circle_fit, geometric_lm_refine, joint_bc_lsd_fit,
)


# ---------------------------------------------------------------------------
# Calibrant presets
# ---------------------------------------------------------------------------

@dataclass
class CalibrantSpec:
    name: str
    space_group: int
    # cell parameters (a, b, c, alpha, beta, gamma)
    a: float; b: float; c: float
    alpha: float = 90.0; beta: float = 90.0; gamma: float = 90.0


CALIBRANT_PRESETS: dict[str, CalibrantSpec] = {
    "CeO2":   CalibrantSpec("CeO2 SRM 674b", 225, 5.41165, 5.41165, 5.41165),
    "LaB6":   CalibrantSpec("LaB6 SRM 660c", 221, 4.15689, 4.15689, 4.15689),
    "Si":     CalibrantSpec("Si SRM 640",    227, 5.43094, 5.43094, 5.43094),
    "Ni":     CalibrantSpec("Ni",            225, 3.52400, 3.52400, 3.52400),
    "Al2O3":  CalibrantSpec("Al2O3 (corundum)", 167,
                            4.75870, 4.75870, 12.99290, 90.0, 90.0, 120.0),
    "Cr2O3":  CalibrantSpec("Cr2O3", 167,
                            4.95870, 4.95870, 13.60170, 90.0, 90.0, 120.0),
}


def _hkls_for_calibrant(
    cal: CalibrantSpec,
    *,
    wavelength_A: float,
    two_theta_max_deg: float,
) -> List[dict]:
    """Generate allowed hkls + 2θ for ``cal`` via midas-hkls.

    Returns a list of ``{"hkl": (h,k,l), "two_theta_deg": float,
    "d_spacing": float, "ring_nr": int}`` sorted by 2θ.  Same-ring_nr
    duplicates (symmetry-equivalent reflections at the same radius)
    are collapsed to a single entry.
    """
    from midas_hkls import SpaceGroup, Lattice, generate_hkls
    lat = Lattice(a=cal.a, b=cal.b, c=cal.c,
                  alpha=cal.alpha, beta=cal.beta, gamma=cal.gamma)
    refs = generate_hkls(SpaceGroup.from_number(cal.space_group), lat,
                         wavelength_A=wavelength_A,
                         two_theta_max_deg=two_theta_max_deg)
    seen_ring: dict[int, dict] = {}
    for r in refs:
        if r.ring_nr in seen_ring:
            continue
        seen_ring[r.ring_nr] = {
            "hkl": (int(r.h), int(r.k), int(r.l)),
            "two_theta_deg": float(r.two_theta_deg),
            "d_spacing": float(r.d_spacing),
            "ring_nr": int(r.ring_nr),
        }
    return sorted(seen_ring.values(), key=lambda d: d["two_theta_deg"])


# ---------------------------------------------------------------------------
# GUI
# ---------------------------------------------------------------------------

@dataclass
class _RingPick:
    """A single ring's clicked points + the assumed hkl."""
    hkl: Tuple[int, int, int]
    two_theta_deg: float
    xs: list = field(default_factory=list)
    ys: list = field(default_factory=list)


class RingPicker:
    """Tkinter+matplotlib ring-picker.  See module docstring for usage."""

    def __init__(
        self,
        *,
        image: np.ndarray,
        image_label: str,
        calibrant: CalibrantSpec,
        wavelength_A: float,
        pixel_size_um: float,
        output_path: Path,
        two_theta_max_deg: float = 20.0,
    ):
        # ---- delayed Tk imports so the module is importable headlessly
        import matplotlib
        matplotlib.use("TkAgg")
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_tkagg import (
            FigureCanvasTkAgg, NavigationToolbar2Tk,
        )
        import tkinter as tk
        from tkinter import ttk

        self._tk = tk; self._ttk = ttk
        self._plt = plt
        self._FigureCanvasTkAgg = FigureCanvasTkAgg
        self._NavigationToolbar2Tk = NavigationToolbar2Tk

        # ---- inputs
        self.image = image
        self.image_label = image_label
        self.calibrant = calibrant
        self.wavelength_A = float(wavelength_A)
        self.pixel_size_um = float(pixel_size_um)
        self.output_path = Path(output_path)
        self.n_pixels_z, self.n_pixels_y = int(image.shape[0]), int(image.shape[1])
        self.two_theta_max_deg = float(two_theta_max_deg)

        # reflection list for the active calibrant
        self.refs = _hkls_for_calibrant(
            calibrant,
            wavelength_A=self.wavelength_A,
            two_theta_max_deg=self.two_theta_max_deg,
        )
        if not self.refs:
            raise RuntimeError(
                "No reflections within two_theta_max_deg — check wavelength / lattice."
            )

        # ---- state
        self.BC: Optional[Tuple[float, float]] = None     # (BC_y, BC_z)
        self.arc_points: List[Tuple[float, float]] = []   # buffer for arc-fit BC
        self.arc_fit_radius: Optional[float] = None
        self.arc_fit_rms: Optional[float] = None
        self.ring_picks: dict[Tuple[int, int, int], _RingPick] = {}
        self.lsd_arc: dict[Tuple[int, int, int], float] = {}
        self.lsd_pick: dict[Tuple[int, int, int], float] = {}
        self.lsd_combined: Optional[float] = None
        self.joint_fit_summary: Optional[dict] = None     # populated by joint fit

        # ---- build window
        self.root = tk.Tk()
        self.root.title(
            f"midas-calibrate-v2 ring picker — {image_label} "
            f"({calibrant.name}, λ={wavelength_A:.4f} Å)"
        )
        self.root.option_add("*Font", ("Helvetica", 13))

        self._build_controls()
        self._build_canvas()
        self._update_status()

    # ----------------------------------------------------------------- build
    def _build_controls(self) -> None:
        tk, ttk = self._tk, self._ttk
        ctrl = ttk.Frame(self.root); ctrl.pack(side=tk.TOP, fill=tk.X)

        # geometry seed (read-only summary)
        geo = ttk.LabelFrame(ctrl, text="Geometry seed")
        geo.pack(side=tk.LEFT, padx=6, pady=6)
        ttk.Label(geo, text=f"λ = {self.wavelength_A} Å").grid(row=0, column=0, sticky="w", padx=6)
        ttk.Label(geo, text=f"pixel = {self.pixel_size_um} µm").grid(row=1, column=0, sticky="w", padx=6)
        ttk.Label(geo, text=f"{self.calibrant.name}").grid(row=2, column=0, sticky="w", padx=6)
        ttk.Label(geo, text=f"SG {self.calibrant.space_group}, "
                            f"a={self.calibrant.a} Å").grid(
            row=3, column=0, sticky="w", padx=6)
        ttk.Label(geo, text=f"img {self.n_pixels_y}×{self.n_pixels_z} px").grid(
            row=4, column=0, sticky="w", padx=6)

        # mode
        self.mode = tk.StringVar(value="Arc-fit BC")
        mode_frame = ttk.LabelFrame(ctrl, text="Click mode")
        mode_frame.pack(side=tk.LEFT, padx=6, pady=6)
        for m in ("Hover only", "Arc-fit BC", "Set BC", "Pick ring"):
            ttk.Radiobutton(mode_frame, text=m, variable=self.mode, value=m
                            ).pack(anchor="w", padx=6)

        # hkl dropdown driven by the live reflection list
        hkl_frame = ttk.LabelFrame(ctrl, text="Active ring")
        hkl_frame.pack(side=tk.LEFT, padx=6, pady=6)
        self.hkl_var = tk.StringVar()
        self.hkl_combo = ttk.Combobox(
            hkl_frame, textvariable=self.hkl_var,
            values=[self._hkl_label(r) for r in self.refs],
            width=24, state="readonly",
        )
        self.hkl_combo.current(0)
        self.hkl_combo.pack(padx=6, pady=6)
        self.hkl_combo.bind("<<ComboboxSelected>>", lambda _e: self._redraw())
        ttk.Label(hkl_frame, text="(label = hkl @ 2θ°)").pack(padx=6)

        # actions
        act = ttk.LabelFrame(ctrl, text="Actions")
        act.pack(side=tk.LEFT, padx=6, pady=6)
        ttk.Button(act, text="Fit BC from arc", command=self._fit_bc_from_arc
                   ).grid(row=0, column=0, columnspan=2, padx=4, pady=2, sticky="ew")
        ttk.Button(act, text="Clear arc pts", command=self._clear_arc_points
                   ).grid(row=1, column=0, padx=4, pady=2, sticky="ew")
        ttk.Button(act, text="Clear BC", command=self._clear_bc
                   ).grid(row=1, column=1, padx=4, pady=2, sticky="ew")
        ttk.Button(act, text="Clear last pick", command=self._clear_last_pick
                   ).grid(row=2, column=0, padx=4, pady=2, sticky="ew")
        ttk.Button(act, text="Clear all picks", command=self._clear_all_picks
                   ).grid(row=2, column=1, padx=4, pady=2, sticky="ew")
        ttk.Button(act, text="Compute per-ring Lsd", command=self._compute_per_ring_lsd
                   ).grid(row=3, column=0, columnspan=2, padx=4, pady=2, sticky="ew")
        ttk.Button(act, text="Joint (BC, Lsd) fit", command=self._joint_fit
                   ).grid(row=4, column=0, columnspan=2, padx=4, pady=2, sticky="ew")
        self.show_pred = tk.BooleanVar(value=False)
        ttk.Checkbutton(act, text="Show predicted rings",
                        variable=self.show_pred, command=self._redraw
                        ).grid(row=5, column=0, columnspan=2, sticky="w", padx=4)
        ttk.Button(act, text="Save seed JSON", command=self._save_seed
                   ).grid(row=6, column=0, columnspan=2, padx=4, pady=4, sticky="ew")

        # status
        self.status = ttk.LabelFrame(ctrl, text="Status")
        self.status.pack(side=tk.LEFT, padx=6, pady=6, fill=tk.X, expand=True)
        self.lbl_mouse = ttk.Label(self.status, text="mouse: —")
        self.lbl_mouse.pack(anchor="w", padx=6)
        self.lbl_arc = ttk.Label(self.status, text="arc pts: 0")
        self.lbl_arc.pack(anchor="w", padx=6)
        self.lbl_bc = ttk.Label(self.status, text="BC: not set")
        self.lbl_bc.pack(anchor="w", padx=6)
        self.lbl_pick = ttk.Label(self.status, text="ring picks: 0")
        self.lbl_pick.pack(anchor="w", padx=6)
        self.lbl_lsd = ttk.Label(self.status, text="Lsd: —")
        self.lbl_lsd.pack(anchor="w", padx=6)
        self.lbl_joint = ttk.Label(self.status, text="joint fit: not run")
        self.lbl_joint.pack(anchor="w", padx=6)

    def _build_canvas(self) -> None:
        plt = self._plt
        self.fig = plt.Figure(figsize=(8.5, 8.5))
        self.ax = self.fig.add_subplot(111)
        lo, hi = np.percentile(self.image, 45), np.percentile(self.image, 99)
        self.ax.imshow(
            np.log10(np.clip(self.image, max(1, lo), None)),
            cmap="gray", vmin=np.log10(max(1, lo)), vmax=np.log10(hi),
            origin="lower",
        )
        # generous off-detector margin so BC outside the image is reachable
        margin = max(self.n_pixels_y, self.n_pixels_z) // 8
        self.ax.set_xlim(-margin, self.n_pixels_y + margin)
        self.ax.set_ylim(-margin, self.n_pixels_z + margin)
        self.ax.set_aspect("equal")
        self.ax.set_title(
            f"{self.image_label}  (log scale, clip [{lo:.0f}, {hi:.0f}])"
        )

        # persistent artists
        self.bc_artist, = self.ax.plot([], [], "r+", ms=18, mew=2.5, clip_on=False)
        self.pick_artist, = self.ax.plot([], [], "y.", ms=7, clip_on=False)
        self.arc_pt_artist, = self.ax.plot(
            [], [], "o", ms=8, mfc="none", mec="magenta", mew=1.6, clip_on=False)
        self.arc_fit_artist, = self.ax.plot(
            [], [], color="magenta", lw=1.2, alpha=0.85, clip_on=False)
        self.pred_artists: list = []

        self.canvas = self._FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(
            side=self._tk.TOP, fill=self._tk.BOTH, expand=True)
        self._NavigationToolbar2Tk(self.canvas, self.root).update()
        self.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self.canvas.mpl_connect("button_press_event", self._on_click)

    # ----------------------------------------------------------------- helpers
    def _hkl_label(self, r: dict) -> str:
        h, k, l = r["hkl"]
        return f"({h}{k}{l})  2θ={r['two_theta_deg']:.3f}°"

    def _active_ref(self) -> dict:
        return self.refs[self.hkl_combo.current()]

    def _on_motion(self, evt) -> None:
        if evt.inaxes is not self.ax:
            return
        self.lbl_mouse.config(text=f"mouse: ({evt.xdata:.1f}, {evt.ydata:.1f}) px")

    def _on_click(self, evt) -> None:
        if evt.inaxes is not self.ax:
            return
        x, y = float(evt.xdata), float(evt.ydata)
        mode = self.mode.get()
        if mode == "Set BC":
            self.BC = (x, y)
        elif mode == "Arc-fit BC":
            self.arc_points.append((x, y))
        elif mode == "Pick ring":
            ref = self._active_ref()
            key = ref["hkl"]
            ring = self.ring_picks.setdefault(
                key, _RingPick(hkl=key, two_theta_deg=ref["two_theta_deg"]))
            ring.xs.append(x); ring.ys.append(y)
        self._update_status()
        self._redraw()

    def _update_status(self) -> None:
        # arc-fit pts
        self.lbl_arc.config(text=f"arc pts: {len(self.arc_points)}")
        # BC
        if self.BC is None:
            self.lbl_bc.config(text="BC: not set")
        else:
            extra = (f" (arc-fit r={self.arc_fit_radius:.1f} px, "
                     f"RMS={self.arc_fit_rms:.2f} px)"
                     if self.arc_fit_radius is not None else "")
            self.lbl_bc.config(
                text=f"BC: ({self.BC[0]:.2f}, {self.BC[1]:.2f}) px{extra}")
        # ring picks
        n_total = sum(len(r.xs) for r in self.ring_picks.values())
        n_rings = sum(1 for r in self.ring_picks.values() if len(r.xs) > 0)
        self.lbl_pick.config(text=f"ring picks: {n_total} across {n_rings} ring(s)")
        # Lsd
        all_lsd = list(self.lsd_arc.values()) + list(self.lsd_pick.values())
        if not all_lsd:
            self.lbl_lsd.config(text="Lsd: —")
            self.lsd_combined = None
        else:
            self.lsd_combined = float(np.median(all_lsd))
            self.lbl_lsd.config(
                text=f"Lsd (median over {len(all_lsd)} est.) "
                     f"= {self.lsd_combined/1e3:.3f} mm")
        # joint fit
        if self.joint_fit_summary is None:
            self.lbl_joint.config(text="joint fit: not run")
        else:
            j = self.joint_fit_summary
            self.lbl_joint.config(
                text=f"joint: BC=({j['cx']:.3f}, {j['cy']:.3f}) "
                     f"Lsd={j['lsd_um']/1e3:.3f} mm  "
                     f"RMS={j['rms_total_px']:.2f} px "
                     f"(N={j['n_total']}, rings={j['n_rings']})")

    # ----------------------------------------------------------------- actions
    def _fit_bc_from_arc(self) -> None:
        if len(self.arc_points) < 3:
            self.lbl_arc.config(text=f"arc pts: {len(self.arc_points)} — need ≥3")
            return
        xs = [p[0] for p in self.arc_points]
        ys = [p[1] for p in self.arc_points]
        try:
            cx, cy, R, rms_k = kasa_circle_fit(xs, ys)
            cx, cy, R, rms = geometric_lm_refine(xs, ys, cx, cy, R)
        except Exception as e:
            self.lbl_arc.config(text=f"fit failed: {e}")
            return
        self.BC = (cx, cy)
        self.arc_fit_radius = R
        self.arc_fit_rms = rms
        ref = self._active_ref()
        tth = np.radians(ref["two_theta_deg"])
        self.lsd_arc[ref["hkl"]] = R * self.pixel_size_um / np.tan(tth)
        self._update_status()
        self._redraw()

    def _clear_arc_points(self) -> None:
        self.arc_points.clear()
        self._update_status()
        self._redraw()

    def _clear_bc(self) -> None:
        self.BC = None
        self.arc_fit_radius = None
        self.arc_fit_rms = None
        self.joint_fit_summary = None
        self._update_status()
        self._redraw()

    def _clear_last_pick(self) -> None:
        # pop from whichever ring was last appended to — search ring with most picks
        if not self.ring_picks:
            return
        ref = self._active_ref()
        key = ref["hkl"]
        if key in self.ring_picks and self.ring_picks[key].xs:
            self.ring_picks[key].xs.pop()
            self.ring_picks[key].ys.pop()
        self._update_status()
        self._redraw()

    def _clear_all_picks(self) -> None:
        self.ring_picks.clear()
        self.lsd_pick.clear()
        self.joint_fit_summary = None
        self._update_status()
        self._redraw()

    def _compute_per_ring_lsd(self) -> None:
        if self.BC is None:
            self.lbl_lsd.config(text="Lsd: set BC first"); return
        self.lsd_pick.clear()
        for hkl, ring in self.ring_picks.items():
            if len(ring.xs) < 1:
                continue
            xs = np.asarray(ring.xs); ys = np.asarray(ring.ys)
            R = float(np.mean(np.hypot(xs - self.BC[0], ys - self.BC[1])))
            tth = np.radians(ring.two_theta_deg)
            self.lsd_pick[hkl] = R * self.pixel_size_um / np.tan(tth)
        self._update_status()
        self._redraw()

    def _joint_fit(self) -> None:
        rings_for_joint: List[Tuple[list, list, float]] = []
        # include the arc-fit points as an extra ring if available
        if self.arc_points and self.arc_fit_radius is not None:
            ref = self._active_ref()
            xs = [p[0] for p in self.arc_points]
            ys = [p[1] for p in self.arc_points]
            rings_for_joint.append((xs, ys, np.radians(ref["two_theta_deg"])))
        # plus every ring with ≥3 picks
        for hkl, ring in self.ring_picks.items():
            if len(ring.xs) >= 3:
                rings_for_joint.append(
                    (list(ring.xs), list(ring.ys),
                     np.radians(ring.two_theta_deg)))
        if not rings_for_joint:
            self.lbl_joint.config(
                text="joint fit: need either an arc-fit OR ≥1 ring "
                     "with ≥3 picks")
            return
        try:
            res = joint_bc_lsd_fit(rings_for_joint, pixel_size_um=self.pixel_size_um)
        except Exception as e:
            self.lbl_joint.config(text=f"joint fit failed: {e}")
            return
        self.joint_fit_summary = res
        # adopt joint fit as the working geometry
        self.BC = (res["cx"], res["cy"])
        self.lsd_combined = res["lsd_um"]
        self._update_status()
        self._redraw()

    def _redraw(self) -> None:
        # BC
        if self.BC is not None:
            self.bc_artist.set_data([self.BC[0]], [self.BC[1]])
        else:
            self.bc_artist.set_data([], [])
        # arc-fit point buffer + fitted circle
        if self.arc_points:
            xs = [p[0] for p in self.arc_points]
            ys = [p[1] for p in self.arc_points]
            self.arc_pt_artist.set_data(xs, ys)
        else:
            self.arc_pt_artist.set_data([], [])
        if self.BC is not None and self.arc_fit_radius is not None:
            theta = np.linspace(0, 2 * np.pi, 1200)
            self.arc_fit_artist.set_data(
                self.BC[0] + self.arc_fit_radius * np.cos(theta),
                self.BC[1] + self.arc_fit_radius * np.sin(theta))
        else:
            self.arc_fit_artist.set_data([], [])
        # ring picks
        all_x, all_y = [], []
        for ring in self.ring_picks.values():
            all_x.extend(ring.xs); all_y.extend(ring.ys)
        self.pick_artist.set_data(all_x, all_y)
        # predicted rings overlay
        for a in self.pred_artists:
            a.remove()
        self.pred_artists.clear()
        if self.show_pred.get() and self.BC is not None and self.lsd_combined is not None:
            theta = np.linspace(0, 2 * np.pi, 1200)
            cmap_pts = self._plt.cm.cool(np.linspace(0.05, 0.95, len(self.refs)))
            colors = cycle(cmap_pts)
            for ref in self.refs:
                tth = np.radians(ref["two_theta_deg"])
                R = self.lsd_combined / self.pixel_size_um * np.tan(tth)
                (line,) = self.ax.plot(
                    self.BC[0] + R * np.cos(theta),
                    self.BC[1] + R * np.sin(theta),
                    color=next(colors), lw=0.6, alpha=0.8)
                self.pred_artists.append(line)
        self.canvas.draw_idle()

    def _save_seed(self) -> None:
        seed = self._build_seed_dict()
        self.output_path.write_text(json.dumps(seed, indent=2))
        msg = f"saved seed → {self.output_path}"
        print(msg)
        self.lbl_joint.config(text=msg)

    def _build_seed_dict(self) -> dict:
        per_ring = {}
        for hkl, ring in self.ring_picks.items():
            per_ring[str(list(hkl))] = {
                "two_theta_deg": ring.two_theta_deg,
                "n_picks": len(ring.xs),
                "xs": list(ring.xs),
                "ys": list(ring.ys),
                "lsd_um": self.lsd_pick.get(hkl),
            }
        return {
            "image_label": self.image_label,
            "image_shape_yz": [self.n_pixels_y, self.n_pixels_z],
            "wavelength_A": self.wavelength_A,
            "pixel_size_um": self.pixel_size_um,
            "calibrant": {
                "name": self.calibrant.name,
                "space_group": self.calibrant.space_group,
                "lattice": [self.calibrant.a, self.calibrant.b, self.calibrant.c,
                            self.calibrant.alpha, self.calibrant.beta,
                            self.calibrant.gamma],
            },
            "BC_y_px": None if self.BC is None else self.BC[0],
            "BC_z_px": None if self.BC is None else self.BC[1],
            "Lsd_um": self.lsd_combined,
            "joint_fit": self.joint_fit_summary,
            "arc_fit": {
                "n_points": len(self.arc_points),
                "radius_px": self.arc_fit_radius,
                "rms_residual_px": self.arc_fit_rms,
                "xs": [p[0] for p in self.arc_points],
                "ys": [p[1] for p in self.arc_points],
            },
            "per_ring_picks": per_ring,
            "lsd_per_ring_um_arc": {
                str(list(k)): v for k, v in self.lsd_arc.items()},
            "lsd_per_ring_um_pick": {
                str(list(k)): v for k, v in self.lsd_pick.items()},
        }

    def mainloop(self) -> None:
        self.root.mainloop()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="midas-calibrate-v2-pick",
        description=(
            "Manual ring-picker for midas-calibrate-v2.  Produces a "
            "sub-pixel (BC, Lsd) seed from clicked arcs on any image."),
    )
    p.add_argument("image", type=Path,
                   help="image file (TIFF/HDF5/CBF/GE — auto-detected)")
    p.add_argument("--dark", type=Path, default=None,
                   help="optional dark image to subtract")
    p.add_argument("--data-loc", type=str, default="exchange/data",
                   help="HDF5 dataset path (for .h5/.hdf5 only; default exchange/data)")
    p.add_argument("--data-type", type=int, default=1,
                   help="GE binary numeric type (1=u16, 2=f64, 3=f32, 4=u32, 5=i32)")
    p.add_argument("--skip-frame", type=int, default=0,
                   help="leading-frame skip for multi-frame files")
    p.add_argument("--im-trans", type=int, nargs="*", default=(),
                   help="MIDAS ImTransOpt codes: 1 flip-Y, 2 flip-Z, 3 transpose")
    p.add_argument("--calibrant", type=str, default=None,
                   help=f"calibrant preset, one of {list(CALIBRANT_PRESETS)}")
    p.add_argument("--space-group", type=int, default=None,
                   help="space-group number (for custom calibrants)")
    p.add_argument("--lattice", type=float, nargs="+", default=None,
                   help="lattice: 'a' (cubic) OR 'a b c alpha beta gamma'")
    p.add_argument("--name", type=str, default="custom",
                   help="calibrant label (custom calibrants)")
    p.add_argument("--wavelength", type=float, required=True,
                   help="X-ray wavelength in Å")
    p.add_argument("--pixel-um", type=float, required=True,
                   help="detector pixel size in µm")
    p.add_argument("--two-theta-max", type=float, default=20.0,
                   help="hkl generation cutoff in degrees (default 20°)")
    p.add_argument("--output", type=Path, default=None,
                   help="seed JSON path (default <image>.seed.json)")
    return p.parse_args(argv)


def _resolve_calibrant(args: argparse.Namespace) -> CalibrantSpec:
    if args.calibrant is not None:
        if args.calibrant not in CALIBRANT_PRESETS:
            raise SystemExit(
                f"unknown calibrant preset '{args.calibrant}'. "
                f"available: {list(CALIBRANT_PRESETS)}")
        return CALIBRANT_PRESETS[args.calibrant]
    if args.space_group is None or args.lattice is None:
        raise SystemExit(
            "either --calibrant <preset>, or both --space-group and --lattice "
            "are required")
    L = args.lattice
    if len(L) == 1:
        a = b = c = float(L[0]); alpha = beta = gamma = 90.0
    elif len(L) == 6:
        a, b, c, alpha, beta, gamma = (float(x) for x in L)
    else:
        raise SystemExit("--lattice expects either 1 value (cubic) or 6 values")
    return CalibrantSpec(args.name, args.space_group, a, b, c, alpha, beta, gamma)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    img = read_image(
        args.image,
        data_loc=args.data_loc,
        skip_frame=args.skip_frame,
        im_trans=tuple(args.im_trans),
        data_type=args.data_type,
    )
    dark = read_dark(
        args.dark,
        data_loc=args.data_loc,
        skip_frame=args.skip_frame,
        im_trans=tuple(args.im_trans),
        data_type=args.data_type,
    ) if args.dark else None
    if dark is not None:
        if dark.shape != img.shape:
            raise SystemExit(
                f"dark shape {dark.shape} != image shape {img.shape}")
        img = img - dark

    calibrant = _resolve_calibrant(args)
    output = args.output or args.image.with_suffix(args.image.suffix + ".seed.json")
    print(f"image {img.shape}, range [{img.min():.0f}, {img.max():.0f}]")
    print(f"calibrant: {calibrant.name}  (SG {calibrant.space_group}, "
          f"a={calibrant.a} Å)")
    print(f"λ = {args.wavelength} Å, pixel = {args.pixel_um} µm")
    print(f"seed will write to: {output}")

    picker = RingPicker(
        image=img,
        image_label=args.image.name,
        calibrant=calibrant,
        wavelength_A=args.wavelength,
        pixel_size_um=args.pixel_um,
        output_path=output,
        two_theta_max_deg=args.two_theta_max,
    )
    picker.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
