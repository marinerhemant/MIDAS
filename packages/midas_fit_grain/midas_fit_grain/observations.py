"""Per-grain observed-spot batch built from ExtraInfo.bin.

The C refiner operates on a list of "matched" observed spots per grain, sourced
from indexer output that filters ExtraInfo by spot ID. We mirror that contract
here: ``ObservedSpots`` is the N-spot view of one grain, all on one device,
all converted to radians / micrometers up-front so the residual layer never
re-converts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import math
import numpy as np
import torch

from .io_binary import EXTRA_INFO_NCOLS

DEG2RAD = math.pi / 180.0
RAD2DEG = 180.0 / math.pi


def fit_frame_columns(block: np.ndarray, wedge_deg: float = 0.0):
    """ExtraInfo columns the torch refiner fits against, for one Wedge.

    Returns ``(y_um, z_um, omega_deg, eta_deg, two_theta_deg)``.

    * ``Wedge == 0``: cols 0, 1, 2, 6, 7 (YLab, ZLab, Omega, Eta, Ttheta),
      exactly as before -- the wedge-corrected and raw columns coincide.
    * ``Wedge != 0``: the RAW (pre-wedge, detector-corrected) observation,
      cols 9, 10, 8 (YOrig, ZOrig, OmegaIni) with eta recomputed from them.
      The model is then built with the same Wedge (``driver._build_model``),
      so the fit compares a physical wedge forward
      (``midas_diffract.forward`` "Wedge convention") with the raw spot. The
      wedge-corrected columns are exact only for a grain at the rotation
      axis: the grain-position displacement they still contain was never
      rotated about the tilted axis, so a wedge-free model fitted to them
      carries a position error that grows with W and with the grain's
      distance from the axis. 2theta is wedge-invariant (col 7).
    """
    if float(wedge_deg) == 0.0:
        return block[:, 0], block[:, 1], block[:, 2], block[:, 6], block[:, 7]
    y = block[:, 9]
    z = block[:, 10]
    r = np.sqrt(y * y + z * z)
    with np.errstate(invalid="ignore", divide="ignore"):
        eta = RAD2DEG * np.arccos(np.clip(z / np.where(r > 0, r, 1.0), -1.0, 1.0))
    eta = np.where(y > 0, -eta, eta)      # C CalcEtaAngle sign convention
    return y, z, block[:, 8], eta, block[:, 7]


@dataclass
class ObservedSpots:
    """Matched observed spots for one grain (or one batch entry).

    All angular tensors are in **radians**, all spatial tensors in
    **micrometers**. The leading dim is the spot index.
    """
    spot_id: torch.Tensor          # (S,) int64
    ring_nr: torch.Tensor          # (S,) int64
    y_lab: torch.Tensor            # (S,) um, lab Y in the fit frame
                                   # (fit_frame_columns: wedge-corrected at
                                   # W = 0, raw YOrig when W != 0)
    z_lab: torch.Tensor            # (S,) um
    omega: torch.Tensor            # (S,) rad
    eta: torch.Tensor              # (S,) rad
    two_theta: torch.Tensor        # (S,) rad
    grain_radius: torch.Tensor     # (S,) ring-derived radius (um), col 3
    fit_rmse: torch.Tensor         # (S,) per-spot peak-fit RMSE (col 15)

    # Raw (un-wedge-corrected) lab coords for the C-parity DisplacementInTheSpot
    # path; kept around so the iterative refiner can recompute observed pixels
    # at every position update without re-reading the binary.
    y_orig: torch.Tensor           # (S,) col 9
    z_orig: torch.Tensor           # (S,) col 10
    omega_ini: torch.Tensor        # (S,) col 8 (deg in C; converted to rad here)
    mask_touched: torch.Tensor     # (S,) col 14 (0/1 weighting flag)

    @property
    def n_spots(self) -> int:
        return int(self.spot_id.shape[0])

    @classmethod
    def from_extra_info(
        cls,
        extra_info: np.ndarray,                   # (nSpots, 16)
        spot_ids: np.ndarray | torch.Tensor,      # (S,) int — subset to take
        *,
        device: torch.device,
        dtype: torch.dtype,
        wedge_deg: float = 0.0,
    ) -> "ObservedSpots":
        """Pull the rows whose SpotID is in ``spot_ids`` and pack into tensors.

        ``spot_ids`` order is preserved.  Missing IDs raise ``KeyError``.
        ``wedge_deg`` selects the fit frame (:func:`fit_frame_columns`); pass
        the same Wedge the forward model was built with.
        """
        if extra_info.ndim != 2 or extra_info.shape[1] != EXTRA_INFO_NCOLS:
            raise ValueError(
                f"extra_info must be (nSpots, {EXTRA_INFO_NCOLS}), got "
                f"{extra_info.shape}"
            )
        if isinstance(spot_ids, torch.Tensor):
            ids_np = spot_ids.detach().cpu().numpy().astype(np.int64)
        else:
            ids_np = np.asarray(spot_ids, dtype=np.int64)

        all_ids = extra_info[:, 4].astype(np.int64)
        # Build an O(1) lookup: SpotID -> row index.
        id_to_row = {int(sid): row for row, sid in enumerate(all_ids)}

        rows = np.empty(ids_np.shape[0], dtype=np.int64)
        for i, sid in enumerate(ids_np):
            try:
                rows[i] = id_to_row[int(sid)]
            except KeyError as e:
                raise KeyError(
                    f"SpotID {int(sid)} not found in ExtraInfo.bin"
                ) from e

        block = extra_info[rows]  # (S, 16)

        def _t(arr, dt=dtype):
            return torch.as_tensor(arr, dtype=dt, device=device)

        y_fit, z_fit, ome_fit, eta_fit, tth_fit = fit_frame_columns(
            block, wedge_deg)
        return cls(
            spot_id=_t(block[:, 4], torch.int64),
            ring_nr=_t(block[:, 5], torch.int64),
            y_lab=_t(y_fit),
            z_lab=_t(z_fit),
            omega=_t(ome_fit * DEG2RAD),
            eta=_t(eta_fit * DEG2RAD),
            two_theta=_t(tth_fit * DEG2RAD),
            grain_radius=_t(block[:, 3]),
            fit_rmse=_t(block[:, 15]),
            y_orig=_t(block[:, 9]),
            z_orig=_t(block[:, 10]),
            omega_ini=_t(block[:, 8] * DEG2RAD),
            mask_touched=_t(block[:, 14]),
        )

    def to(self, *, device: Optional[torch.device] = None,
           dtype: Optional[torch.dtype] = None) -> "ObservedSpots":
        kw_f = {"dtype": dtype, "device": device} if dtype is not None else {"device": device}
        return ObservedSpots(
            spot_id=self.spot_id.to(device=device) if device else self.spot_id,
            ring_nr=self.ring_nr.to(device=device) if device else self.ring_nr,
            y_lab=self.y_lab.to(**kw_f),
            z_lab=self.z_lab.to(**kw_f),
            omega=self.omega.to(**kw_f),
            eta=self.eta.to(**kw_f),
            two_theta=self.two_theta.to(**kw_f),
            grain_radius=self.grain_radius.to(**kw_f),
            fit_rmse=self.fit_rmse.to(**kw_f),
            y_orig=self.y_orig.to(**kw_f),
            z_orig=self.z_orig.to(**kw_f),
            omega_ini=self.omega_ini.to(**kw_f),
            mask_touched=self.mask_touched.to(**kw_f),
        )

    def g_unit_lab(self) -> torch.Tensor:
        """Unit observed G-vectors in the lab frame.

        Built from ``(η, 2θ, ω)`` per spot. The lab-frame g vector for a
        spot at angles (2θ, η) before omega rotation is

            g_omega = [-sin θ, cos θ sin η, cos θ cos η]

        and the omega-rotated version (taking the convention that the
        crystal is at omega=0 when the spot satisfies Bragg) is

            g = R_z(-ω) · g_omega

        Returns a unit vector ``(S, 3)``.
        """
        theta = self.two_theta * 0.5
        eta = self.eta
        omega = self.omega
        c_th = torch.cos(theta)
        g_om = torch.stack([
            -torch.sin(theta),
            c_th * torch.sin(eta),
            c_th * torch.cos(eta),
        ], dim=-1)
        # Rotate by -omega about z (C convention used by FitOrientationOMP).
        c_w = torch.cos(omega)
        s_w = torch.sin(omega)
        g_lab = torch.stack([
            c_w * g_om[..., 0] + s_w * g_om[..., 1],
            -s_w * g_om[..., 0] + c_w * g_om[..., 1],
            g_om[..., 2],
        ], dim=-1)
        return g_lab / g_lab.norm(dim=-1, keepdim=True).clamp_min(1e-12)
