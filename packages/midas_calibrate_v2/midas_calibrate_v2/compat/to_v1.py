"""Write a v2 result back to a v1-compatible paramstest.txt.

Downstream MIDAS HEDM tools consume the v1 format; v2 exports here.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import torch

from midas_calibrate.params import CalibrationParams as V1Params

from ..parameters.spec import CalibrationSpec


# v2 distortion name → v1 p-index (inverse of V1_TO_V2_DISTORTION in from_v1).
_V2_TO_V1_DISTORTION = {
    "iso_R2": "p2", "iso_R4": "p5", "iso_R6": "p4",
    "a1": "p7",  "phi1": "p8",
    "a2": "p0",  "phi2": "p6",
    "a3": "p9",  "phi3": "p10",
    "a4": "p1",  "phi4": "p3",
    "a5": "p11", "phi5": "p12",
    "a6": "p13", "phi6": "p14",
}


def unpacked_to_v1_params(
    unpacked: Dict[str, torch.Tensor],
    template: V1Params,
) -> V1Params:
    """Copy refined values from a v2 unpacked dict back into a v1 params object.

    v2's distortion names (``iso_R2``..``a6/phi6``) are translated back to
    the v1 ``p₀``..``p₁₄`` slots so downstream HEDM tools see the same file.
    """
    out = V1Params(**{k: getattr(template, k) for k in template.__dict__})
    for name, val in unpacked.items():
        if name in ("panel_delta_yz", "panel_delta_theta",
                    "panel_delta_lsd", "panel_delta_p2"):
            continue   # panel data goes to a separate file
        scalar = val.detach().reshape(-1)[0].item() if val.ndim > 0 else val.item()
        # Map v2 distortion names back to v1 p-indices for output.
        target = _V2_TO_V1_DISTORTION.get(name, name)
        if hasattr(out, target):
            cur = getattr(out, target)
            try:
                setattr(out, target, type(cur)(scalar))
            except Exception:
                setattr(out, target, scalar)
    return out


def write_v1_paramstest(
    unpacked: Dict[str, torch.Tensor],
    template: V1Params,
    path: Path | str,
) -> None:
    """Write a v1-compatible paramstest.txt at the given path."""
    out = unpacked_to_v1_params(unpacked, template)
    out.write(path)


def write_panel_shifts_file(
    unpacked: Dict[str, torch.Tensor],
    path: Path | str,
) -> None:
    """Write a v1-compatible PanelShiftsFile (text, six columns).

    Columns: panel_id, δy, δz, δθ, δLsd, δp₂.
    """
    dyz = unpacked.get("panel_delta_yz")
    dth = unpacked.get("panel_delta_theta")
    dl  = unpacked.get("panel_delta_lsd")
    dp2 = unpacked.get("panel_delta_p2")
    if dyz is None or dth is None:
        raise ValueError("panel_delta_yz and panel_delta_theta are required")
    n = dyz.shape[0]
    if dl is None:
        dl = torch.zeros(n)
    if dp2 is None:
        dp2 = torch.zeros(n)
    lines = []
    for k in range(n):
        lines.append(f"{k:5d}  "
                     f"{float(dyz[k, 0]):+.6f}  {float(dyz[k, 1]):+.6f}  "
                     f"{float(dth[k]):+.6e}  "
                     f"{float(dl[k]):+.4f}  {float(dp2[k]):+.6e}")
    Path(path).write_text("\n".join(lines) + "\n")


__all__ = ["unpacked_to_v1_params", "write_v1_paramstest", "write_panel_shifts_file"]
