"""Shared synthetic test scaffolding."""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np
import torch

from midas_diffract.forward import HEDMForwardModel, HEDMGeometry


DEG2RAD = math.pi / 180.0


def make_fcc_hkls(a_A: float = 3.61, wavelength_A: float = 0.1729,
                  d_min_A: float = 0.6, h_max: int = 4
                  ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Enumerate Bragg-allowed FCC reflections.

    Returns
    -------
    G : Tensor (M, 3)        Cartesian G-vectors (1/Å)
    th : Tensor (M,)         Bragg angles (radians)
    hkls_int : Tensor (M, 3) Integer Miller indices (float64 for fwd-model API)
    """
    hkls = []
    ints = []
    for h in range(-h_max, h_max + 1):
        for k in range(-h_max, h_max + 1):
            for l in range(-h_max, h_max + 1):
                if (h, k, l) == (0, 0, 0):
                    continue
                # FCC: h,k,l all even or all odd
                parity = (h % 2, k % 2, l % 2)
                if parity not in {(0, 0, 0), (1, 1, 1)}:
                    continue
                G = np.array([h, k, l], dtype=np.float64) / a_A
                Gn = np.linalg.norm(G)
                d = 1.0 / Gn
                if d < d_min_A:
                    continue
                arg = wavelength_A * Gn / 2.0
                if abs(arg) >= 1.0:
                    continue
                theta = np.arcsin(arg)
                hkls.append((G, theta))
                ints.append([h, k, l])
    G = torch.tensor(np.array([h[0] for h in hkls]), dtype=torch.float64)
    th = torch.tensor(np.array([h[1] for h in hkls]), dtype=torch.float64)
    hkls_int = torch.tensor(np.array(ints), dtype=torch.float64)
    return G, th, hkls_int


def standard_ff_geometry() -> HEDMGeometry:
    """Nygren-style FF geometry: Dexela 2923 at 884 mm, ~71.7 keV."""
    return HEDMGeometry(
        Lsd=884_000.0,                # 884 mm in micrometers
        y_BC=1944.0, z_BC=1536.0,     # detector center
        px=74.8,
        omega_start=-180.0, omega_step=0.25,
        n_frames=1440,
        n_pixels_y=3888, n_pixels_z=3072,
        min_eta=6.0,
        wavelength=0.1729,
        flip_y=True,
    )


def make_model(device: torch.device = torch.device("cpu"),
               compile: "bool | str" = False) -> HEDMForwardModel:
    """Build the FCC reference model used by tests and benchmarks.

    ``compile`` is forwarded to ``HEDMForwardModel(compile=...)`` so the
    paper's perf benchmark can flip on torch.compile without touching the
    bench script call sites.
    """
    G, th, hkls_int = make_fcc_hkls()
    geom = standard_ff_geometry()
    model = HEDMForwardModel(
        hkls=G, thetas=th, geometry=geom,
        hkls_int=hkls_int, device=device, compile=compile,
    )
    model = model.to(torch.float64)
    return model


def random_orientation(seed: int = 0) -> torch.Tensor:
    """Return a random rotation matrix (3, 3) float64."""
    g = torch.Generator().manual_seed(seed)
    # Sample a random quaternion uniformly on S^3.
    q = torch.randn(4, generator=g, dtype=torch.float64)
    q = q / q.norm()
    w, x, y, z = q
    R = torch.tensor([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=torch.float64)
    return R
