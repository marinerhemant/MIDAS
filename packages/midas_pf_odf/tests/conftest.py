"""Shared scaffolding for midas-pf-odf synthetic tests.

Adapts the ``midas_grain_odf`` test fixtures to the pf-HEDM regime:
  - Single FF detector distance, ~Nygren geometry
  - ScanConfig with a small Σ for fast unit tests
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np
import torch

from midas_diffract.forward import HEDMForwardModel, HEDMGeometry, ScanConfig


def make_fcc_hkls(a_A: float = 3.61, wavelength_A: float = 0.1729,
                  d_min_A: float = 0.7, h_max: int = 3
                  ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Smaller FCC HKL set than the parent package (faster tests)."""
    hkls, ints = [], []
    for h in range(-h_max, h_max + 1):
        for k in range(-h_max, h_max + 1):
            for l in range(-h_max, h_max + 1):
                if (h, k, l) == (0, 0, 0):
                    continue
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


def standard_pf_geometry() -> HEDMGeometry:
    """FF-style geometry adequate for pf — Nygren-flavored."""
    return HEDMGeometry(
        Lsd=884_000.0,
        y_BC=1944.0, z_BC=1536.0,
        px=74.8,
        omega_start=-180.0, omega_step=0.25,
        n_frames=1440,
        n_pixels_y=3888, n_pixels_z=3072,
        min_eta=6.0,
        wavelength=0.1729,
        flip_y=True,
    )


def small_scan_config(
    sample_size_um: float = 30.0,
    n_scans: int = 11,
    beam_size_um: float = 5.0,
    dtype: torch.dtype = torch.float64,
) -> ScanConfig:
    """Σ scan positions covering ``±sample_size_um/2`` plus a margin.

    Σ should be odd so a center scan exists; default 11 = a fast unit-test
    setting that gives multi-scan coverage without blowing up the patch
    tensor.
    """
    half = sample_size_um / 2.0 + beam_size_um
    positions = torch.linspace(-half, half, n_scans, dtype=dtype)
    return ScanConfig(beam_positions=positions, beam_size=float(beam_size_um))


def build_model(scan_config: ScanConfig,
                hkls_int: torch.Tensor,
                G_cart: torch.Tensor,
                thetas: torch.Tensor,
                device: str = "cpu",
                dtype: torch.dtype = torch.float64,
                compile: "bool | str" = False) -> HEDMForwardModel:
    """Build the pf-HEDM reference model.

    ``compile`` is forwarded to ``HEDMForwardModel(compile=...)`` so the
    paper's perf benchmark can flip on torch.compile without modifying
    bench-script call sites.
    """
    geom = standard_pf_geometry()
    model = HEDMForwardModel(
        hkls=G_cart, thetas=thetas, geometry=geom,
        hkls_int=hkls_int, scan_config=scan_config, device=device,
        compile=compile,
    )
    return model.to(dtype)
