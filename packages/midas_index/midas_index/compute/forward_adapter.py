"""Adapter between midas-index and midas_diffract.HEDMForwardModel.

Architecture (revised from dev/implementation_plan.md §1.5):

`HEDMForwardModel.forward()` runs `calc_bragg_geometry` followed by
`project_to_detector` — the latter discretizes onto a pixel grid, which the
indexer does NOT need (it works in continuous lab-frame um). The adapter
therefore delegates only the **Bragg geometry solve** (HKL -> omega, eta,
two_theta) to midas-diffract via `model.calc_bragg_geometry(R, hkls, thetas)`,
and computes `(yl, zl)` + sample-position COM displacement + ring/box gating
in midas-index using simple torch ops.

Output: 14-column TheorSpots tensor that `compare_spots` consumes.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from ..params import IndexerParams

DEG2RAD = math.pi / 180.0
RAD2DEG = 180.0 / math.pi


class IndexerForwardAdapter:
    """Bridge to `midas_diffract.HEDMForwardModel.calc_bragg_geometry`."""

    def __init__(
        self,
        params: "IndexerParams",
        hkls_real: torch.Tensor,           # (M, 7) [g1,g2,g3,ring,d,theta_rad,radius]
        hkls_int: torch.Tensor,            # (M, 4) [h, k, l, ring]
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        from midas_diffract import HEDMForwardModel, HEDMGeometry

        self.params = params
        self.device = device
        self.dtype = dtype
        self.hkls_real = hkls_real.to(device=device, dtype=dtype)
        self.hkls_int = hkls_int.to(device=device, dtype=torch.long)

        # HEDMGeometry is required by HEDMForwardModel even though the
        # indexer doesn't use the pixel-discretization fields. Use the
        # documented FF defaults; pixel layout is irrelevant since we never
        # call project_to_detector.
        self.geom = HEDMGeometry(
            Lsd=float(params.Distance),
            y_BC=0.0, z_BC=0.0,
            px=float(params.px) if params.px > 0 else 1.0,
            omega_start=-180.0,
            omega_step=1.0,
            n_frames=360,
            n_pixels_y=2048,
            n_pixels_z=2048,
            min_eta=float(params.ExcludePoleAngle),
            wavelength=float(params.Wavelength),
        )

        hkls_cart = self.hkls_real[:, :3]              # (M, 3)
        thetas = self.hkls_real[:, 5]                  # (M,) radians
        hkls_int_3 = self.hkls_int[:, :3]              # (M, 3)

        self._model = HEDMForwardModel(
            hkls=hkls_cart,
            thetas=thetas,
            geometry=self.geom,
            hkls_int=hkls_int_3,
        ).to(device=device, dtype=dtype)

        # Per-HKL ring nr (M,) and a sparse ring-radius LUT for (yl, zl).
        self.ring_nr_per_hkl = self.hkls_int[:, 3].to(dtype=dtype)   # (M,)
        max_ring = max(params.RingRadii.keys(), default=0) + 2
        self.ring_radius_lut = torch.zeros(max_ring, device=device, dtype=dtype)
        for rn, r in params.RingRadii.items():
            self.ring_radius_lut[rn] = float(r)

        # Multi-detector pinwheel: per-(ring, eta-bin) coverage mask.
        # Built from EtaCoverage_DetN rows in paramstest. Shape:
        # ``(max_ring + 1, n_eta_bins)`` bool. For ``not is_multi_detector``
        # the mask is all-True (no per-panel constraints).
        self._has_panel_coverage = bool(params.EtaCoverage)
        if self._has_panel_coverage:
            n_eta_bins = 3600                     # 0.1° resolution over [-180°, 180°)
            cov = torch.zeros(
                (max_ring, n_eta_bins),
                device=device, dtype=torch.bool,
            )
            for det_id, arcs in params.EtaCoverage.items():
                for ring_d, lo, hi in arcs:
                    if ring_d < 0 or ring_d >= max_ring:
                        continue
                    # Convert eta in [-180, 180) to bin index in [0, n_eta_bins).
                    lo_b = int(((lo + 180.0) / 360.0) * n_eta_bins)
                    hi_b = int(((hi + 180.0) / 360.0) * n_eta_bins)
                    lo_b = max(0, min(n_eta_bins - 1, lo_b))
                    hi_b = max(0, min(n_eta_bins - 1, hi_b))
                    if lo_b <= hi_b:
                        cov[ring_d, lo_b:hi_b + 1] = True
                    else:                          # wrapped arc; defensive
                        cov[ring_d, lo_b:] = True
                        cov[ring_d, :hi_b + 1] = True
            self.coverage_mask = cov               # (max_ring, n_eta_bins) bool
            self.cov_n_eta_bins = n_eta_bins
        else:
            self.coverage_mask = None
            self.cov_n_eta_bins = 0

        # OmegaRanges + BoxSizes for gating
        if params.OmegaRanges:
            self.omega_ranges = torch.tensor(params.OmegaRanges, device=device, dtype=dtype)
            self.box_sizes = torch.tensor(params.BoxSizes, device=device, dtype=dtype)
        else:
            self.omega_ranges = None
            self.box_sizes = None

        # RingsToReject
        self.rings_to_reject = (
            torch.tensor(params.RingsToReject, device=device, dtype=torch.long)
            if params.RingsToReject else
            torch.empty(0, device=device, dtype=torch.long)
        )

    # -----------------------------------------------------------------
    # simulate(R, pos, lattice)
    # -----------------------------------------------------------------

    def simulate(
        self,
        R: torch.Tensor,                  # (N, 3, 3)
        pos: torch.Tensor,                # (N, 3)  (ga, gb, gc)
        lattice: torch.Tensor | None = None,  # (6,) or None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the forward simulation and pack into [N_total, M_eff, 14] TheorSpots.

        Returns
        -------
        theor : (N, K*M, 14) torch.Tensor — K = 2 (two omega solutions/HKL).
                              Columns documented in dev/implementation_plan.md §1.5.1.
        valid : (N, K*M) torch.bool — True for spots that passed all gating.
        """
        # CPU fast path: numba kernel implements the full simulate per-cell.
        # The torch path stays for GPU + multi-detector panel-coverage cases.
        if R.device.type != "cuda" and not self._has_panel_coverage:
            try:
                from .forward_numba import simulate_numba, _NUMBA_AVAILABLE
                if _NUMBA_AVAILABLE:
                    return simulate_numba(self, R, pos, lattice=lattice)
            except ImportError:
                pass

        device = self.device
        dtype = self.dtype
        N = R.shape[0]

        # If lattice deviates from nominal, derive strained hkls. Otherwise
        # use the model's nominal hkls/thetas.
        hkls_cart = None
        thetas = None
        if lattice is not None:
            hkls_cart, thetas = self._model.correct_hkls_latc(lattice)

        # Bragg geometry: (omega_rad, eta_rad, two_theta_rad, valid_float)
        #   omega.shape = (N, 2*M)  — two omega solutions per HKL stacked
        #   M = number of HKLs
        omega_rad, eta_rad, two_theta_rad, valid_f = self._model.calc_bragg_geometry(
            R, hkls_cart, thetas,
        )
        # `calc_bragg_geometry` doubles the HKL axis: outputs are (..., 2N_eq, M)
        # where it actually concatenated the two omega branches along the
        # second-to-last (formerly N) axis. With our N positions feeding R,
        # the layout is (N, M) per branch, then cat along N -> (2N, M).
        # We unscramble: omega_p, omega_n = omega[:N], omega[N:].
        # Because R has leading shape (N, 3, 3), the model treats N as the
        # position count. Its concat doubles to 2N along that axis.
        # Reshape to (N, 2, M) then flatten to (N, 2*M).
        if omega_rad.dim() == 2:
            two_n, m = omega_rad.shape
            assert two_n == 2 * N, f"unexpected 2N axis size {two_n} (expected {2*N})"
            omega_rad = omega_rad.view(2, N, m).permute(1, 0, 2).reshape(N, 2 * m)
            eta_rad = eta_rad.view(2, N, m).permute(1, 0, 2).reshape(N, 2 * m)
            two_theta_rad = two_theta_rad.view(2, N, m).permute(1, 0, 2).reshape(N, 2 * m)
            valid_f = valid_f.view(2, N, m).permute(1, 0, 2).reshape(N, 2 * m)
        else:  # batched (..., 2N, M)
            raise NotImplementedError("Batched leading dims not yet supported")

        valid = valid_f > 0.5

        omega_deg = omega_rad * RAD2DEG
        eta_deg = eta_rad * RAD2DEG
        theta_deg = (two_theta_rad * 0.5) * RAD2DEG

        # Per-HKL ring lookup, doubled to match 2*M axis.
        # ring_nr_2m[k] for k in [0, 2M) → ring_nr_per_hkl[k % M]
        m = self.ring_nr_per_hkl.shape[0]
        # Two-branch layout means index 2k corresponds to HKL k for branch +,
        # and 2k+1 for branch -. After our reshape, the layout is
        # [hkl0_p, hkl1_p, ..., hkl(M-1)_p, hkl0_n, ..., hkl(M-1)_n] per N.
        ring_nr_per_col = self.ring_nr_per_hkl.repeat(2)              # (2M,)
        ring_nr_b = ring_nr_per_col.view(1, 2 * m).expand(N, 2 * m)    # (N, 2M)
        ring_radius_b = self.ring_radius_lut[ring_nr_b.long()]         # (N, 2M)

        # CalcSpotPosition: yl = -sin(eta)*R, zl = cos(eta)*R
        yl_no_disp = -torch.sin(eta_rad) * ring_radius_b
        zl_no_disp = torch.cos(eta_rad) * ring_radius_b

        # Multi-detector η coverage mask: a predicted spot is "expected
        # to be observable" only if its (ring, η) pair lies inside at
        # least one panel's coverage. Spots that miss every panel are
        # masked out of ``valid`` so they don't pull down completeness.
        # This is the per-panel equivalent of the C OmegaRange + BoxSize
        # gating below (which is geometry-agnostic).
        if self._has_panel_coverage and self.coverage_mask is not None:
            n_eta_bins = self.cov_n_eta_bins
            # Eta in [-180, 180) → bin index.
            eta_b = ((eta_deg + 180.0) / 360.0 * n_eta_bins).long()
            eta_b = eta_b.clamp(0, n_eta_bins - 1)
            ring_long = ring_nr_b.long().clamp_min(0)
            ring_long = torch.minimum(
                ring_long,
                torch.tensor(self.coverage_mask.shape[0] - 1,
                             device=device, dtype=torch.long),
            )
            covered = self.coverage_mask[ring_long, eta_b]   # (N, 2M) bool
            valid = valid & covered

        # OmegaRange + BoxSize gating (matches CalcDiffrSpots_Furnace logic).
        if self.omega_ranges is not None:
            n_ranges = self.omega_ranges.shape[0]
            omeg_min = self.omega_ranges[:, 0].view(1, 1, n_ranges)
            omeg_max = self.omega_ranges[:, 1].view(1, 1, n_ranges)
            box_y_min = self.box_sizes[:, 0].view(1, 1, n_ranges)
            box_y_max = self.box_sizes[:, 1].view(1, 1, n_ranges)
            box_z_min = self.box_sizes[:, 2].view(1, 1, n_ranges)
            box_z_max = self.box_sizes[:, 3].view(1, 1, n_ranges)

            o = omega_deg.unsqueeze(-1)
            y = yl_no_disp.unsqueeze(-1)
            z = zl_no_disp.unsqueeze(-1)
            range_ok = (
                (o > omeg_min) & (o < omeg_max)
                & (y > box_y_min) & (y < box_y_max)
                & (z > box_z_min) & (z < box_z_max)
            ).any(dim=-1)
            valid = valid & range_ok

        # COM displacement: per-tuple (ga, gb, gc) shifts every spot.
        ga = pos[:, 0:1]   # (N, 1)
        gb = pos[:, 1:2]
        gc = pos[:, 2:3]
        # `displacement_spot_needed_COM` formula (FF_HEDM/src/IndexerOMP.c:710):
        #   xi, yi, zi = (Lsd, yl, zl) / |...|
        #   t = (a*cos(omega) - b*sin(omega)) / xi
        #   Δy = a*sin(omega) + b*cos(omega) - t*yi
        #   Δz = c - t*zi
        Lsd = float(self.params.Distance)
        L = torch.sqrt(
            torch.tensor(Lsd, device=device, dtype=dtype) ** 2
            + yl_no_disp ** 2 + zl_no_disp ** 2
        )
        xi_n = Lsd / L
        yi_n = yl_no_disp / L
        zi_n = zl_no_disp / L
        cos_o = torch.cos(omega_rad)
        sin_o = torch.sin(omega_rad)
        t = (ga * cos_o - gb * sin_o) / xi_n
        dy = (ga * sin_o + gb * cos_o) - t * yi_n
        dz = gc - t * zi_n
        yl_disp = yl_no_disp + dy
        zl_disp = zl_no_disp + dz

        # Distance from origin to the (yl, zl) on the detector at Lsd
        distance = torch.sqrt(yl_no_disp ** 2 + zl_no_disp ** 2 + Lsd ** 2)

        # Eta_deg and rad_diff post-displacement (cols 12, 13)
        eta_deg_post = torch.atan2(-yl_disp, zl_disp) * RAD2DEG
        rad_diff = torch.sqrt(yl_disp ** 2 + zl_disp ** 2) - ring_radius_b

        # Assemble 14-column TheorSpots layout
        spotnr = torch.arange(N * 2 * m, device=device, dtype=dtype).reshape(N, 2 * m)
        hkl_idx = (
            torch.arange(m, device=device, dtype=dtype)
            .unsqueeze(0).unsqueeze(0).expand(1, 2, m).reshape(1, 2 * m)
            .expand(N, 2 * m)
        )
        zero = torch.zeros_like(omega_deg)

        theor = torch.stack(
            [
                zero,                  # 0
                spotnr,                # 1
                hkl_idx,               # 2
                distance,              # 3
                yl_no_disp,            # 4
                zl_no_disp,            # 5
                omega_deg,             # 6
                eta_deg,               # 7
                theta_deg,             # 8
                ring_nr_b.to(dtype),   # 9
                yl_disp,               # 10
                zl_disp,               # 11
                eta_deg_post,          # 12
                rad_diff,              # 13
            ],
            dim=-1,
        )
        return theor, valid
