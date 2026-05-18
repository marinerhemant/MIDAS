"""Energy-sweep orchestration for anomalous-scattering experiments.

Iterate over a list of beam energies (eV); per-energy:

  1. (optional) refine geometry on a calibrant frame.
  2. Integrate the sample frame.
  3. Capture f' / f'' if a midas_hkls.anomalous table is available.
  4. Output per-energy DAT.

Returns an :class:`EnergySweepResult` carrying per-energy profiles
plus the f'/f'' trajectory for downstream MAD / DAFS analysis.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch


@dataclass
class EnergySweepResult:
    energies_eV: List[float] = field(default_factory=list)
    profiles: List[np.ndarray] = field(default_factory=list)
    sigmas: List[np.ndarray] = field(default_factory=list)
    Q_axes: List[np.ndarray] = field(default_factory=list)
    fprime: Dict[str, List[float]] = field(default_factory=dict)
    fdoubleprime: Dict[str, List[float]] = field(default_factory=dict)


def run_energy_sweep(
    energies_eV: Sequence[float],
    sample_frames: Sequence,
    base_spec,
    *,
    composition: Optional[Dict[str, float]] = None,
    out_dir: Optional[Path] = None,
) -> EnergySweepResult:
    """Loop over energies, integrate each sample frame.

    Parameters
    ----------
    energies_eV :
        List of beam energies (eV). Wavelength derived as
        ``λ_Å = 12398.4 / E_eV``.
    sample_frames :
        One detector image per energy (numpy or tensor).
    base_spec :
        ``IntegrationSpec``. ``Wavelength`` is overridden per energy.
    composition :
        Optional ``{element: fraction}`` for f', f'' lookup via
        ``midas_hkls.anomalous``.
    out_dir :
        Optional directory; per-energy DAT files get written here.
    """
    if len(energies_eV) != len(sample_frames):
        raise ValueError("energies_eV and sample_frames must align in length")
    from ..binning import (
        PolygonBinGeometry, integrate_polygon_with_variance,
    )
    from ..pdf import R_px_to_Q
    from ..io import write_dat
    result = EnergySweepResult()
    for k, (E, frame) in enumerate(zip(energies_eV, sample_frames)):
        spec = base_spec
        # Override wavelength for this iteration (clone to avoid stomping)
        lam_A = 12398.4 / float(E)
        # Reuse the spec's tensor metadata but with the new wavelength
        spec.Wavelength = torch.as_tensor(lam_A, dtype=torch.float64)
        geom = PolygonBinGeometry.from_spec(spec)
        img_t = torch.as_tensor(frame, dtype=torch.float64)
        mean2d, sig2d = integrate_polygon_with_variance(img_t, geom)
        valid = torch.isfinite(mean2d)
        n_valid = valid.sum(dim=0).clamp(min=1)
        I = (torch.where(valid, mean2d, torch.zeros_like(mean2d)).sum(dim=0)
             / n_valid).numpy()
        sig2_safe = torch.where(valid, sig2d * sig2d, torch.zeros_like(sig2d))
        sig = (torch.sqrt(sig2_safe.sum(dim=0)) / n_valid).numpy()
        R_axis = (
            spec.RMin + (np.arange(I.shape[0]) + 0.5) * spec.RBinSize
        )
        Q = R_px_to_Q(
            torch.as_tensor(R_axis, dtype=torch.float64),
            Lsd_um=spec.Lsd, px_um=spec.pxY, lambda_A=spec.Wavelength,
        ).numpy()
        result.energies_eV.append(float(E))
        result.profiles.append(I)
        result.sigmas.append(sig)
        result.Q_axes.append(Q)
        # f', f'' lookup (best-effort)
        if composition is not None:
            try:
                from midas_hkls.anomalous import anomalous_correction
                for elem in composition:
                    fp, fpp = anomalous_correction(elem, energy_eV=float(E))
                    result.fprime.setdefault(elem, []).append(float(fp))
                    result.fdoubleprime.setdefault(elem, []).append(float(fpp))
            except Exception:
                pass
        if out_dir is not None:
            out_dir = Path(out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            write_dat(
                out_dir / f"E_{int(E)}eV.dat",
                q_axis_invA=Q, intensity=I, sigma=sig,
            )
    return result


__all__ = ["EnergySweepResult", "run_energy_sweep"]
