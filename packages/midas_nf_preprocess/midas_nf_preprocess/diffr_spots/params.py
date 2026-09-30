"""Parameters for diffraction-spot prediction.

Mirrors keys parsed by ``MakeDiffrSpots.c`` L306-L405. Multi-distance
``Lsd``, ``OmegaRange``, ``BoxSize``, and ``RingsToUse`` are tuples that
accumulate over repeated lines in the parameter file.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Optional, Union


@dataclass
class DiffrSpotsParams:
    """All parameters consumed by the diffr_spots pipeline."""

    # I/O
    data_directory: str = "."
    output_directory: str = ""
    seed_orientations: str = ""      # path to CSV of quaternions

    # Detector
    distances: list[float] = field(default_factory=list)   # Lsd (um)
    px: float = 0.0                   # px (um)
    max_ring_rad: float = 0.0         # MaxRingRad (um)

    # Crystallography
    space_group: int = 0
    lattice_constant: tuple[float, ...] = (0, 0, 0, 0, 0, 0)  # (a, b, c, alpha, beta, gamma)
    wavelength: float = 0.0           # Angstroms

    # Filtering
    omega_ranges: list[tuple[float, float]] = field(default_factory=list)  # (omega_min, omega_max) per distance
    box_sizes: list[tuple[float, float, float, float]] = field(default_factory=list)  # (yl_min, yl_max, zl_min, zl_max) per distance
    rings_to_use: list[int] = field(default_factory=list)
    exclude_pole_angle: float = 0.0
    nr_orientations: int = 0          # NrOrientations
    wedge: float = 0.0                # Wedge (deg), Parameters-file sign
                                      # (midas_diffract.forward "Wedge
                                      # convention"); same value NF fitting uses

    def __post_init__(self) -> None:
        if not self.output_directory:
            self.output_directory = self.data_directory

    @property
    def primary_distance(self) -> float:
        """The first detector distance (used for spot position calculations)."""
        if not self.distances:
            raise ValueError("No detector distances configured (Lsd entries are missing)")
        return self.distances[0]

    @classmethod
    def from_paramfile(cls, path: Union[str, Path]) -> "DiffrSpotsParams":
        """Parse a MIDAS parameter file."""
        kwargs: dict = {
            "distances": [],
            "omega_ranges": [],
            "box_sizes": [],
            "rings_to_use": [],
        }
        # Six-tuple lattice (try LatticeParameter, fall back to LatticeConstant)
        latc: Optional[tuple] = None

        with open(path, "r") as f:
            for line in f:
                line = line.rstrip("\n")
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                key = parts[0]
                vals = parts[1:]
                if not vals:
                    continue

                if key == "Lsd":
                    kwargs["distances"].append(float(vals[0]))
                elif key == "RingsToUse":
                    kwargs["rings_to_use"].append(int(vals[0]))
                elif key == "DataDirectory":
                    kwargs["data_directory"] = vals[0]
                elif key == "OutputDirectory":
                    kwargs["output_directory"] = vals[0]
                elif key == "SeedOrientations":
                    kwargs["seed_orientations"] = vals[0]
                elif key == "NrOrientations":
                    kwargs["nr_orientations"] = int(vals[0])
                elif key == "SpaceGroup":
                    kwargs["space_group"] = int(vals[0])
                elif key == "ExcludePoleAngle":
                    kwargs["exclude_pole_angle"] = float(vals[0])
                elif key == "Wedge":
                    kwargs["wedge"] = float(vals[0])
                elif key in ("LatticeParameter", "LatticeConstant"):
                    if len(vals) >= 6:
                        latc = tuple(float(v) for v in vals[:6])
                elif key == "Wavelength":
                    kwargs["wavelength"] = float(vals[0])
                elif key == "px":
                    kwargs["px"] = float(vals[0])
                elif key == "MaxRingRad":
                    kwargs["max_ring_rad"] = float(vals[0])
                elif key == "OmegaRange":
                    if len(vals) >= 2:
                        kwargs["omega_ranges"].append(
                            (float(vals[0]), float(vals[1]))
                        )
                elif key == "BoxSize":
                    if len(vals) >= 4:
                        kwargs["box_sizes"].append(
                            tuple(float(v) for v in vals[:4])
                        )
        if latc is not None:
            kwargs["lattice_constant"] = latc
        return cls(**kwargs)

    def with_overrides(self, **kwargs) -> "DiffrSpotsParams":
        return replace(self, **kwargs)
