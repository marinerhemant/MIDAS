"""CalibrationParams — superset of IntegrationParams with refinement knobs.

Round-trip compatible with the C `.txt` files consumed by AutoCalibrateZarr →
CalibrantIntegratorOMP.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

# We reuse midas_integrate's parameter parser as the foundation — calibration
# params are a strict superset of integration params.
from midas_integrate.params import IntegrationParams, parse_params as _parse_int


@dataclass
class CalibrationParams:
    # -------------------------- detector geometry (mirrors IntegrationParams)
    NrPixelsY: int = 0
    NrPixelsZ: int = 0
    pxY: float = 0.0     # μm
    pxZ: float = 0.0
    Lsd: float = 0.0     # μm
    BC_y: float = 0.0
    BC_z: float = 0.0
    tx: float = 0.0      # degrees, fixed (not refined)
    ty: float = 0.0
    tz: float = 0.0
    p0: float = 0.0
    p1: float = 0.0
    p2: float = 0.0
    p3: float = 0.0
    p4: float = 0.0
    p5: float = 0.0
    p6: float = 0.0
    p7: float = 0.0
    p8: float = 0.0
    p9: float = 0.0
    p10: float = 0.0
    p11: float = 0.0
    p12: float = 0.0
    p13: float = 0.0
    p14: float = 0.0
    Parallax: float = 0.0
    RhoD: float = 0.0
    Wavelength: float = 0.0  # Å

    # ----------------------------------------------- crystallography
    SpaceGroup: int = 0
    LatticeConstant: Tuple[float, float, float, float, float, float] = (0, 0, 0, 90, 90, 90)
    MaxRingRad: float = 0.0      # px
    MinRingRad: float = 0.0      # px

    # ----------------------------------------------- E-step binning
    Width: float = 800.0           # μm; ring half-width
    EtaBinSize: float = 5.0        # degrees
    RBinSize: float = 0.25         # px
    AdaptiveEtaBins: bool = True
    DoubletDetection: bool = True
    DoubletSeparation: float = 5.0 # px
    SNRMin: float = 1.0

    # ----------------------------------------------- M-step refinement
    nIterations: int = 5
    Refine: Dict[str, bool] = field(default_factory=lambda: {
        "Lsd": True, "BC": True, "ty": True, "tz": True,
        "Wavelength": False, "Parallax": False,
        **{f"p{i}": True for i in range(15)},
    })
    FixedPanelID: int = 0
    PerPanelLsd: bool = False
    PerPanelDistortion: bool = False

    # ----------------------------------------------- bounds (tolerances)
    tolLsd: float = 15000.0   # μm
    tolBC: float = 20.0       # px
    tolTilts: float = 3.0     # deg
    tolDistortion: float = 0.01
    tolWavelength: float = 0.001    # Å
    tolParallax: float = 50.0       # μm
    tolRotation: float = 0.0        # deg; 0 disables panel rotation refinement

    # ----------------------------------------------- robustness / weighting
    Loss: str = "L2"                   # "L2" | "L1" | "huber"
    HuberDelta: float = 1.0
    TrimmedMeanFraction: float = 1.0   # 1.0 = no trim
    WeightByRadius: bool = False
    WeightBySNR: bool = True
    WeightByRing: bool = True

    RemoveOutliersBetweenIters: bool = True
    OutlierFactor: float = 3.0

    # ----------------------------------------------- compute
    Device: str = "auto"            # "cpu" | "cuda" | "mps" | "auto"
    Dtype: str = "fp64"             # "fp32" | "fp64"

    # ----------------------------------------------- I/O
    DataDirectory: str = "."
    ImagePath: str = ""
    DarkPath: Optional[str] = None
    OutputDirectory: str = "."
    Engine: str = "joint"           # "joint" | "alternating"
    Warmstart: str = "full"         # "none" | "alternating" | "peakfit" | "full"
    Plots: bool = False

    # ----------------------------------------------- pass-through extras
    extra: Dict[str, str] = field(default_factory=dict)

    # ============================================================ I/O helpers
    @classmethod
    def from_file(cls, path: Path | str) -> "CalibrationParams":
        path = Path(path)
        text = path.read_text()
        # Reuse midas_integrate's parser for the geometry block.
        try:
            integ = _parse_int(path)
            params = cls()
            for k, v in integ.__dict__.items():
                if hasattr(params, k):
                    setattr(params, k, v)
        except Exception:
            params = cls()

        # Calibration-specific keys
        for line in text.splitlines():
            # Strip comments and the MIDAS C ';' line terminator (e.g.
            # ``MaxRingRad 312493.037332;``) so values parse as floats.
            line = line.split("#")[0].strip().rstrip(";").strip()
            if not line:
                continue
            parts = line.split(None, 1)
            if len(parts) != 2:
                continue
            key, val = parts[0].strip(), parts[1].strip()

            # Apply known calibration keys
            if key == "SpaceGroup":
                params.SpaceGroup = int(val.split()[0])
            elif key == "LatticeConstant":
                vals = [float(v) for v in val.split()]
                if len(vals) >= 6:
                    params.LatticeConstant = tuple(vals[:6])  # type: ignore
            elif key == "MaxRingRad":
                params.MaxRingRad = float(val.split()[0])
            elif key == "MinRingRad":
                params.MinRingRad = float(val.split()[0])
            elif key == "Width":
                params.Width = float(val.split()[0])
            elif key == "EtaBinSize":
                params.EtaBinSize = float(val.split()[0])
            elif key == "RBinSize":
                params.RBinSize = float(val.split()[0])
            elif key == "nIterations":
                params.nIterations = int(val.split()[0])
            elif key == "tolLsd":
                params.tolLsd = float(val.split()[0])
            elif key == "tolBC":
                params.tolBC = float(val.split()[0])
            elif key == "tolTilts":
                params.tolTilts = float(val.split()[0])
            elif key == "tolDistortion":
                params.tolDistortion = float(val.split()[0])
            elif key == "tolWavelength":
                params.tolWavelength = float(val.split()[0])
            elif key == "tolParallax":
                params.tolParallax = float(val.split()[0])
            elif key == "tolRotation":
                params.tolRotation = float(val.split()[0])
            elif key == "FitWavelength":
                params.Refine["Wavelength"] = bool(int(val.split()[0]))
            elif key == "FitParallax":
                params.Refine["Parallax"] = bool(int(val.split()[0]))
            elif key == "RemoveOutliersBetweenIters":
                params.RemoveOutliersBetweenIters = bool(int(val.split()[0]))
            elif key == "OutlierFactor":
                params.OutlierFactor = float(val.split()[0])
            elif key == "TrimmedMeanFraction":
                params.TrimmedMeanFraction = float(val.split()[0])
            elif key == "Loss":
                params.Loss = val.split()[0]
            elif key == "HuberDelta":
                params.HuberDelta = float(val.split()[0])
            elif key == "ImagePath":
                params.ImagePath = val.split()[0]
            elif key == "DarkPath":
                params.DarkPath = val.split()[0]
            elif key == "DataDirectory":
                params.DataDirectory = val.split()[0]
            elif key == "OutputDirectory":
                params.OutputDirectory = val.split()[0]
            elif key == "Engine":
                params.Engine = val.split()[0]
            elif key == "Warmstart":
                params.Warmstart = val.split()[0]
            elif key == "Wavelength":
                params.Wavelength = float(val.split()[0])
            elif key.startswith("p") and key[1:].isdigit():
                idx = int(key[1:])
                if 0 <= idx <= 14:
                    setattr(params, f"p{idx}", float(val.split()[0]))
            elif key == "BC":
                vs = val.split()
                if len(vs) >= 2:
                    params.BC_y = float(vs[0]); params.BC_z = float(vs[1])
            elif hasattr(params, key) and len(val.split()) == 1:
                # Flat string fields like Lsd, ty, tz, tx
                cur = getattr(params, key)
                try:
                    setattr(params, key, type(cur)(val.split()[0]))
                except (TypeError, ValueError):
                    params.extra[key] = val
            else:
                params.extra[key] = val

        return params

    def to_text(self) -> str:
        """Emit a parameters_refined.txt body (C-format compatible)."""
        lines: List[str] = []
        lines.append(f"Lsd {self.Lsd:.6f}")
        lines.append(f"BC {self.BC_y:.6f} {self.BC_z:.6f}")
        lines.append(f"tx {self.tx:.6f}")
        lines.append(f"ty {self.ty:.6f}")
        lines.append(f"tz {self.tz:.6f}")
        for i in range(15):
            lines.append(f"p{i} {getattr(self, f'p{i}'):.10g}")
        lines.append(f"Parallax {self.Parallax:.6f}")
        lines.append(f"Wavelength {self.Wavelength:.6f}")
        lines.append(f"px {self.pxY:.6f}")
        lines.append(f"NrPixelsY {self.NrPixelsY}")
        lines.append(f"NrPixelsZ {self.NrPixelsZ}")
        lines.append(f"RhoD {self.RhoD:.6f}")
        lines.append(f"SpaceGroup {self.SpaceGroup}")
        lines.append("LatticeConstant " + " ".join(f"{v:.6f}" for v in self.LatticeConstant))
        for k, v in self.extra.items():
            lines.append(f"{k} {v}")
        return "\n".join(lines) + "\n"

    def write(self, path: Path | str) -> None:
        Path(path).write_text(self.to_text())

    # =========================================================== validation
    def validate(self) -> None:
        if self.Lsd <= 0:
            raise ValueError("Lsd must be positive")
        if self.pxY <= 0:
            raise ValueError("pxY (pixel size) must be positive")
        if self.Wavelength <= 0:
            raise ValueError("Wavelength must be positive")
        if self.SpaceGroup < 1 or self.SpaceGroup > 230:
            raise ValueError(f"SpaceGroup must be in [1, 230]; got {self.SpaceGroup}")
        if self.NrPixelsY <= 0 or self.NrPixelsZ <= 0:
            raise ValueError("Detector pixel counts must be positive")
        if self.MaxRingRad <= 0:
            raise ValueError("MaxRingRad must be positive (px)")
