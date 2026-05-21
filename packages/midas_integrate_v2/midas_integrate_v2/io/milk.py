"""MILK / MAUD pyFAI-substitute adapter.

Drops MIDAS into MILK's integration boundary at
``MILK/integration/integrate.py`` with a pyFAI-shaped API surface.
MILK uses ``pyFAI.multi_geometry.MultiGeometry`` under the hood; this
adapter mirrors the relevant entry points so MILK code can use MIDAS
with minimal-to-no diff:

  - ``__init__(ais, unit, radial_range, azimuth_range, empty, chi_disc)``
  - ``integrate1d(lst_data, npt, ...) -> Result``
  - ``integrate1d_with_sigma(...)`` — MIDAS-native; returns σ.

The adapter forwards pyFAI's ``unit='2th_deg'`` / ``unit='q_A^-1'`` /
``unit='q_nm^-1'`` to MIDAS's R-axis and converts. The resulting Result
carries ``radial`` / ``intensity`` / ``sigma`` as numpy arrays so MAUD's
``maudbatch`` ESG ingest works unchanged.

For Item 46 (APS 1-ID Hydra multi-geometry parity test) we also expose
the MIDAS multi-detector orchestrator under MILK's expected API.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch


@dataclass
class _MILKResult:
    """Mimics pyFAI's IntegrateResult — `.radial`, `.intensity`, `.sigma`."""
    radial: np.ndarray
    intensity: np.ndarray
    sigma: Optional[np.ndarray] = None


def _spec_to_pyfai_unit(spec, R_axis_px, *, unit: str) -> np.ndarray:
    """Convert MIDAS R-axis (pixels) to pyFAI's expected unit."""
    px = float(spec.pxY)
    Lsd = float(spec.Lsd)
    lam_A = float(spec.Wavelength)
    R = np.asarray(R_axis_px, dtype=np.float64)
    two_theta_rad = np.arctan(R * px / Lsd)
    if unit in ("2th_deg", "2theta_deg"):
        return np.degrees(two_theta_rad)
    if unit in ("q_A^-1", "q_invA"):
        return (4.0 * np.pi / lam_A) * np.sin(0.5 * two_theta_rad)
    if unit in ("q_nm^-1",):
        # 1 Å⁻¹ = 10 nm⁻¹
        return 10.0 * (4.0 * np.pi / lam_A) * np.sin(0.5 * two_theta_rad)
    if unit in ("r_mm",):
        return R * px / 1000.0
    raise ValueError(f"unsupported pyFAI unit {unit!r}")


class MILKMultiGeometryAdapter:
    """Drop-in substitute for ``pyFAI.multi_geometry.MultiGeometry``.

    Mirrors the minimal API surface MILK uses in
    ``MILK/integration/integrate.py``: take a list of integrators (one
    per detector / geometry), call ``integrate1d`` per frame, get back
    a Result with ``radial`` and ``intensity`` arrays.

    Parameters
    ----------
    ais :
        List of MIDAS :class:`IntegrationSpec` objects (one per panel /
        geometry).
    unit :
        pyFAI radial unit (``"2th_deg"``, ``"q_A^-1"``, ``"q_nm^-1"``,
        ``"r_mm"``). Recorded for ``integrate1d`` output.
    radial_range :
        Optional ``(low, high)`` cut in the chosen unit.
    azimuth_range :
        Optional ``(eta_low, eta_high)`` cut in degrees.
    empty :
        Default value for non-integrated bins (mimics pyFAI semantics).
    chi_disc :
        Discontinuity in η; we wrap to [-180, 180) by default.
    """

    def __init__(
        self,
        ais: Sequence,
        *,
        unit: str = "2th_deg",
        radial_range: Optional[Tuple[float, float]] = None,
        azimuth_range: Optional[Tuple[float, float]] = None,
        empty: float = 0.0,
        chi_disc: float = 180.0,
    ):
        if not ais:
            raise ValueError("at least one IntegrationSpec required")
        self._specs = list(ais)
        self.unit = unit
        self.radial_range = radial_range
        self.azimuth_range = azimuth_range
        self.empty = empty
        self.chi_disc = chi_disc

    # ----- pyFAI-compatible entrypoints -----

    def integrate1d(
        self,
        lst_data: Sequence[np.ndarray],
        npt: int,
        *,
        correctSolidAngle: bool = True,
        polarization_factor: Optional[float] = None,
        method: str = "polygon",
        normalization_factor: Optional[Sequence[float]] = None,
        sigma_clip: bool = False,
    ) -> _MILKResult:
        """1D integrate aligned frames through each geometry.

        Returns a single combined ``radial`` / ``intensity`` array (and
        ``sigma`` from the polygon kernel — set via ``method='polygon'``).
        """
        if len(lst_data) != len(self._specs):
            raise ValueError(
                f"lst_data length {len(lst_data)} != n_specs {len(self._specs)}"
            )
        from ..binning import (
            HardBinGeometry, PolygonBinGeometry, SubpixelBinGeometry,
            integrate_hard, integrate_polygon, integrate_subpixel,
            integrate_polygon_sums,
        )
        per_radial = []
        # polygon path carries un-normalised accumulators for a pixel-level
        # multi-panel merge; hard/subpixel carry pre-normalised I and sigma.
        per_num, per_varnum, per_area = [], [], []
        per_intensity, per_sigma = [], []
        for k, (img, spec) in enumerate(zip(lst_data, self._specs)):
            img_t = torch.as_tensor(img, dtype=torch.float64)
            if method == "polygon":
                geom = PolygonBinGeometry.from_spec(spec)
                num2d, varnum2d, area2d = integrate_polygon_sums(img_t, geom)
                num1d = num2d.sum(dim=0).numpy()
                varnum1d = varnum2d.sum(dim=0).numpy()
                area1d = area2d.sum(dim=0).numpy()
                R_axis = (
                    spec.RMin
                    + (np.arange(num1d.shape[0]) + 0.5) * spec.RBinSize
                )
                if normalization_factor is not None:
                    nf = float(normalization_factor[k])
                    num1d = num1d / nf
                    varnum1d = varnum1d / (nf * nf)
                per_radial.append(
                    _spec_to_pyfai_unit(spec, R_axis, unit=self.unit)
                )
                per_num.append(num1d)
                per_varnum.append(varnum1d)
                per_area.append(area1d)
                continue
            elif method in ("hard",):
                geom = HardBinGeometry.from_spec(spec)
                int2d = integrate_hard(img_t, geom, normalize=True)
                counts = (int2d > 0).to(int2d.dtype).sum(dim=0).clamp(min=1)
                I = (int2d.sum(dim=0) / counts).numpy()
                R_axis = (
                    spec.RMin
                    + (np.arange(I.shape[0]) + 0.5) * spec.RBinSize
                )
                sig_I = np.sqrt(np.maximum(I, 0.0))
            elif method in ("subpixel", "splitpixel"):
                geom = SubpixelBinGeometry.from_spec(spec, K=2)
                int2d = integrate_subpixel(img_t, geom, normalize=True)
                counts = (int2d > 0).to(int2d.dtype).sum(dim=0).clamp(min=1)
                I = (int2d.sum(dim=0) / counts).numpy()
                R_axis = (
                    spec.RMin
                    + (np.arange(I.shape[0]) + 0.5) * spec.RBinSize
                )
                sig_I = np.sqrt(np.maximum(I, 0.0))
            else:
                raise ValueError(f"unknown method {method!r}")
            radial = _spec_to_pyfai_unit(spec, R_axis, unit=self.unit)
            if normalization_factor is not None:
                I = I / float(normalization_factor[k])
                sig_I = sig_I / float(normalization_factor[k])
            per_radial.append(radial)
            per_intensity.append(I)
            per_sigma.append(sig_I)

        if method == "polygon":
            # --- Pixel-level area-weighted accumulation across panels ---
            # Sum the per-panel numerator (Σ area·I), variance numerator
            # (Σ area²·σ²) and area (Σ area) onto a shared radial grid, THEN
            # divide once:  I = Σnum/Σarea,  σ = sqrt(Σvarnum)/Σarea.  A
            # panel's coverage-edge sliver contributes only its small area,
            # so it cannot dominate the merge — unlike inverse-variance
            # stitching of pre-normalised 1D profiles, where an edge bin's
            # artificially small σ produced a runaway 1/σ² weight and a
            # spurious dropout at every panel-coverage transition.
            radial_min = min(float(r.min()) for r in per_radial)
            radial_max = max(float(r.max()) for r in per_radial)
            if self.radial_range is not None:
                radial_min = max(radial_min, self.radial_range[0])
                radial_max = min(radial_max, self.radial_range[1])
            common_radial = np.linspace(radial_min, radial_max, int(npt))
            num_tot = np.zeros(int(npt), dtype=np.float64)
            varnum_tot = np.zeros_like(num_tot)
            area_tot = np.zeros_like(num_tot)
            for radial, num1d, varnum1d, area1d in zip(
                per_radial, per_num, per_varnum, per_area
            ):
                order = np.argsort(radial)
                rs = radial[order]
                # Interpolate each extensive accumulator onto the shared
                # grid; a panel contributes zero outside its own coverage
                # (left/right=0), so panels are spliced without gaps and
                # without the runaway weight of inverse-variance stitching.
                num_tot += np.interp(
                    common_radial, rs, num1d[order], left=0.0, right=0.0)
                varnum_tot += np.interp(
                    common_radial, rs, varnum1d[order], left=0.0, right=0.0)
                area_tot += np.interp(
                    common_radial, rs, area1d[order], left=0.0, right=0.0)
            valid = area_tot > 0
            safe_area = np.where(valid, area_tot, 1.0)
            I_combined = np.where(valid, num_tot / safe_area, self.empty)
            sigma_combined = np.where(
                valid, np.sqrt(np.maximum(varnum_tot, 0.0)) / safe_area, np.nan
            )
            return _MILKResult(
                radial=common_radial, intensity=I_combined,
                sigma=sigma_combined,
            )

        # Combine onto common radial grid (npt points) — hard / subpixel
        radial_min = max(r[0] for r in per_radial)
        radial_max = min(r[-1] for r in per_radial)
        if self.radial_range is not None:
            radial_min = max(radial_min, self.radial_range[0])
            radial_max = min(radial_max, self.radial_range[1])
        common_radial = np.linspace(radial_min, radial_max, int(npt))
        I_stack = np.zeros((len(self._specs), int(npt)), dtype=np.float64)
        sig_stack = np.zeros_like(I_stack)
        for k, (radial, I, sig) in enumerate(zip(per_radial,
                                                    per_intensity, per_sigma)):
            sort = np.argsort(radial)
            I_stack[k] = np.interp(common_radial,
                                     radial[sort], I[sort])
            sig_stack[k] = np.interp(common_radial,
                                       radial[sort], sig[sort])
        # Inverse-variance combine
        inv_var = np.where(sig_stack > 0, 1.0 / (sig_stack ** 2), 0.0)
        w_sum = inv_var.sum(axis=0)
        I_combined = np.where(
            w_sum > 0,
            (I_stack * inv_var).sum(axis=0) / np.where(w_sum > 0, w_sum, 1),
            self.empty,
        )
        sigma_combined = np.where(
            w_sum > 0,
            1.0 / np.sqrt(np.where(w_sum > 0, w_sum, 1)),
            np.nan,
        )
        return _MILKResult(
            radial=common_radial, intensity=I_combined, sigma=sigma_combined,
        )

    def integrate1d_with_sigma(
        self, lst_data: Sequence[np.ndarray], npt: int, **kw,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """MIDAS-native signature: returns ``(q, I, σ)`` directly."""
        kw.setdefault("method", "polygon")
        res = self.integrate1d(lst_data, npt, **kw)
        return res.radial, res.intensity, res.sigma


__all__ = ["MILKMultiGeometryAdapter", "_MILKResult"]
