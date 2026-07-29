"""XAF-HEDM forward orchestration over the two orthogonal-axis mountings.

Thin layer on :class:`midas_diffract.HEDMForwardModel`: it simulates the grain
population once per mounting (rotating orientations by the remount transform for
mounting 2), applies the XAF access gates (omega wedges + exit cone), and
assembles a merged spot table plus the per-grain / per-mounting
:class:`~midas_diffract.SpotDescriptors` that the metrics need for autograd
sensitivity analysis.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch

from .config import XAFConfig
from .crystal import build_reflections, get_material
from . import geometry as geo
from .sample import GrainPopulation


@dataclass
class SpotTable:
    """Flat, merged table of accessible spots across both mountings."""
    mounting_id: torch.Tensor   # (S,) int
    grain_id: torch.Tensor      # (S,) int
    hkl: torch.Tensor           # (S, 3) int
    omega: torch.Tensor         # (S,) rad
    two_theta: torch.Tensor     # (S,) rad
    eta: torch.Tensor           # (S,) rad
    y_pixel: torch.Tensor       # (S,)
    z_pixel: torch.Tensor       # (S,)

    def __len__(self) -> int:
        return int(self.mounting_id.shape[0])


@dataclass
class XAFSimulation:
    """Per-mounting descriptors + masks, and the merged accessible spot table."""
    cfg: XAFConfig
    spot_desc: List["object"]        # SpotDescriptors per mounting, (N,K,M)
    access_mask: List[torch.Tensor]  # bool per mounting, (N,K,M)
    mounting_euler: List[torch.Tensor]
    table: SpotTable


class XAFForwardModel:
    """Build once, simulate many grain populations under the same geometry."""

    def __init__(self, cfg: XAFConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        dtype = torch.float64 if cfg.dtype_double else torch.float32
        # MPS has no float64 support; the forward model computes in float32
        # internally regardless, so drop to float32 on that backend.
        if self.device.type == "mps":
            dtype = torch.float32
        self.dtype = dtype

        # Reflections up to the exit-cone cap (nothing beyond is accessible).
        hkls, thetas, hkls_int = build_reflections(
            cfg.material, cfg.wavelength_A, cfg.tth_max_deg, dtype=dtype)
        self.hkls_int = hkls_int

        import midas_diffract as md
        self.geometry = geo.build_hedm_geometry(cfg)
        self.scan_config = self._build_scan_config(cfg, md)
        self.model = md.HEDMForwardModel(
            hkls, thetas, self.geometry, hkls_int=hkls_int,
            scan_config=self.scan_config, device=self.device)

        # Nominal reference lattice (needed whenever strain is applied).
        mat = get_material(cfg.material)
        self._latc0 = torch.tensor(mat.lattice, dtype=dtype, device=self.device)

        # Remount matrices: numpy (for orientation composition) + torch (for
        # differentiable position transforms).  The grain COM rotates with the
        # sample, so position in mounting m is R_mount @ position.
        self._Rmounts = [geo.mounting_matrix(cfg, m) for m in range(cfg.n_mountings)]
        self._Rmounts_t = [torch.tensor(R, dtype=dtype, device=self.device)
                           for R in self._Rmounts]

    # -- beam mode --------------------------------------------------------- #
    def _build_scan_config(self, cfg: XAFConfig, md):
        """Build a pencil/line-beam ScanConfig for line/point modes (None=box).

        The beam is translated along Y in steps of the beam size across the
        sample; a grain diffracts at a given (scan position, omega) only when
        its omega-rotated Y falls within the beam -- directly localising it.
        """
        if cfg.beam_mode == "box":
            return None
        if cfg.beam_mode not in ("line", "point"):
            raise ValueError(f"unknown beam_mode {cfg.beam_mode!r}")
        if cfg.scan_positions_um is not None:
            positions = list(cfg.scan_positions_um)
        else:
            R = cfg.sample_radius_um
            step = max(cfg.beam_size_um, 1e-6)
            n = int(np.ceil(2 * R / step)) + 1
            positions = list(np.linspace(-R, R, n))
        beam_pos = torch.tensor(positions, dtype=self.dtype, device=self.device)
        return md.ScanConfig(beam_positions=beam_pos, beam_size=cfg.beam_size_um)

    # -- orientation bookkeeping ------------------------------------------- #
    def mounting_euler(self, euler_base: torch.Tensor, mounting: int) -> torch.Tensor:
        """Rotate base Euler angles by the remount transform for a mounting."""
        from midas_stress.orientation import (
            euler_to_orient_mat_batch, orient_mat_to_euler)
        if mounting == 0:
            return euler_base
        eb = euler_base.detach().cpu().numpy()
        oms = euler_to_orient_mat_batch(eb).reshape(-1, 3, 3)
        R = self._Rmounts[mounting]
        rotated = np.einsum("ij,njk->nik", R, oms)
        out = np.stack([orient_mat_to_euler(m) for m in rotated], axis=0)
        return torch.as_tensor(out, dtype=euler_base.dtype, device=euler_base.device)

    def mounting_position(self, pos_base: torch.Tensor, mounting: int) -> torch.Tensor:
        """Rotate grain positions by the remount transform for a mounting.

        The grain COM is rigid in the sample, so it rotates with the cell:
        ``pos_mounting = R_mount @ pos_base``.  Differentiable (used by the
        position-determinability metric)."""
        if mounting == 0:
            return pos_base
        R = self._Rmounts_t[mounting].to(pos_base.dtype)
        return pos_base @ R.T

    # -- forward ----------------------------------------------------------- #
    @staticmethod
    def _to_per_grain_shape(sd, N):
        """Reshape a ``forward_per_grain`` output ``(2N, M)`` -> ``(N, 2, M)``.

        ``forward_per_grain`` stacks the two omega branches along the grain
        axis (rows ``i`` and ``i+N`` for grain ``i``), so reshape ``(2, N, M)``
        then swap the first two axes to recover the ``(N, K=2, M)`` layout the
        rest of the pipeline expects.
        """
        from midas_diffract import SpotDescriptors

        def r(x):
            if x is None:
                return None
            M = x.shape[-1]
            return x.reshape(2, N, M).permute(1, 0, 2).contiguous()

        return SpotDescriptors(
            omega=r(sd.omega), eta=r(sd.eta), two_theta=r(sd.two_theta),
            y_pixel=r(sd.y_pixel), z_pixel=r(sd.z_pixel),
            frame_nr=r(sd.frame_nr), valid=r(sd.valid))

    def _run_mounting(self, euler_m, pos_m, strain):
        """One mounting: element-wise per-grain forward + XAF access mask.

        Uses ``forward_per_grain`` (O(N*M)) to avoid the orientation x strain
        cross-product that ``forward`` forms for a per-grain polycrystal.  In
        line/point modes the accessibility also requires the grain to be
        illuminated by the scanned beam at some position.
        """
        N = euler_m.shape[0]
        latc = self._latc0.unsqueeze(0).expand(N, -1)
        sd_flat = self.model.forward_per_grain(
            euler_m, pos_m, lattice_params=latc, strain=strain)
        sd = self._to_per_grain_shape(sd_flat, N)
        mask = geo.accessible_mask(sd, self.cfg)
        if getattr(sd_flat, "scan_mask", None) is not None:
            # scan_mask (S, 2N, M): grain is seen if lit at >=1 beam position.
            illum_flat = sd_flat.scan_mask.sum(dim=0) > 0            # (2N, M)
            illum = illum_flat.reshape(2, N, sd_flat.omega.shape[-1]
                                       ).permute(1, 0, 2)
            mask = mask & illum
        return sd, mask

    def simulate(self, grains: GrainPopulation) -> XAFSimulation:
        """Simulate all mountings and merge the accessible spots."""
        cfg = self.cfg
        pos = grains.position.to(self.device)
        strain = grains.strain.to(self.device)

        sd_list, mask_list, eul_list = [], [], []
        cols = {k: [] for k in
                ("mounting_id", "grain_id", "hkl", "omega",
                 "two_theta", "eta", "y_pixel", "z_pixel")}

        hkls_int = self.hkls_int.to(self.device)  # (M, 3)
        for m in range(cfg.n_mountings):
            euler_m = self.mounting_euler(grains.euler.to(self.device), m)
            pos_m = self.mounting_position(pos, m)
            sd, mask = self._run_mounting(euler_m, pos_m, strain)
            sd_list.append(sd)
            mask_list.append(mask)
            eul_list.append(euler_m)

            # Flatten accessible spots for the merged table.
            idx = torch.nonzero(mask, as_tuple=False)  # (S, 3): (grain, k, hkl)
            if idx.numel() == 0:
                continue
            g, k, h = idx[:, 0], idx[:, 1], idx[:, 2]
            cols["mounting_id"].append(torch.full((g.shape[0],), m,
                                                  dtype=torch.long, device=self.device))
            cols["grain_id"].append(g)
            cols["hkl"].append(hkls_int[h].round().long())
            cols["omega"].append(sd.omega[g, k, h])
            cols["two_theta"].append(sd.two_theta[g, k, h])
            cols["eta"].append(sd.eta[g, k, h])
            yp = sd.y_pixel[g, k, h] if sd.y_pixel.dim() == 3 else sd.y_pixel[..., g, k, h]
            zp = sd.z_pixel[g, k, h] if sd.z_pixel.dim() == 3 else sd.z_pixel[..., g, k, h]
            cols["y_pixel"].append(yp)
            cols["z_pixel"].append(zp)

        def cat(key, empty_cols=1, int_=False):
            if cols[key]:
                return torch.cat(cols[key], dim=0)
            shape = (0, empty_cols) if empty_cols > 1 else (0,)
            return torch.zeros(shape, dtype=torch.long if int_ else torch.float32,
                               device=self.device)

        table = SpotTable(
            mounting_id=cat("mounting_id", int_=True),
            grain_id=cat("grain_id", int_=True),
            hkl=cat("hkl", empty_cols=3, int_=True),
            omega=cat("omega"), two_theta=cat("two_theta"), eta=cat("eta"),
            y_pixel=cat("y_pixel"), z_pixel=cat("z_pixel"),
        )
        return XAFSimulation(cfg=cfg, spot_desc=sd_list, access_mask=mask_list,
                             mounting_euler=eul_list, table=table)
