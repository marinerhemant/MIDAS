"""2θ ring overlay helpers for the detector image view.

Given a ParamFile + a list of HKL ring radii (in pixels) draws circles centred
on the beam centre.
"""

from __future__ import annotations
from typing import Iterable, List

import numpy as np
import pyqtgraph as pg


def draw_rings(image_view, bcy: float, bcz: float,
               radii_px: Iterable[float],
               pen=None) -> List[pg.PlotDataItem]:
    """Draw ring circles centered at (bcy, bcz). Returns the added items."""
    if pen is None:
        pen = pg.mkPen('y', width=1)
    theta = np.linspace(0, 2 * np.pi, 360)
    items = []
    for r in radii_px:
        x = bcy + r * np.cos(theta)
        y = bcz + r * np.sin(theta)
        item = pg.PlotDataItem(x, y, pen=pen)
        image_view.add_overlay(item)
        items.append(item)
    return items


def two_theta_to_radius_px(two_theta_deg: Iterable[float], lsd: float, px: float) -> List[float]:
    """R[px] = (Lsd / px) * tan(2θ).  Lsd, px in same length unit (μm)."""
    r = []
    for t in two_theta_deg:
        rad = np.deg2rad(t)
        r.append((lsd / px) * np.tan(rad))
    return r
