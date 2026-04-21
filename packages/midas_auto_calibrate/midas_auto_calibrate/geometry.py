"""DetectorGeometry dataclass — refined output from a calibration run.

This holds the fitted geometry + 15-parameter analytical distortion model in
a form that's easy to serialize (JSON) and round-trip through the MIDAS
Parameters.txt format that `MIDASCalibrant` consumes.

Field names mirror the ``CalibState`` dataclass in ``utils/AutoCalibrateZarr.py``
so extraction and re-seeding stay mechanical.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Iterable


# Every geometry key that round-trips through MIDAS Parameters.txt. Order
# matches what a human would expect to read in the file: primary geometry
# first, then distortion terms, then detector descriptors.
_PARAM_KEY_ORDER: tuple[str, ...] = (
    "Lsd",
    "ybc", "zbc",
    "tx", "ty", "tz",
    "p0", "p1", "p2", "p3", "p4", "p5",
    "p6", "p7", "p8", "p9", "p10", "p11", "p12", "p13", "p14",
    "RhoD",
    "Wavelength",
    "px",
    "NrPixelsY", "NrPixelsZ",
)

# MIDAS Parameters.txt uses `BC <ybc> <zbc>` as a single line rather than
# two separate keys.
_COMBINED_BC = ("ybc", "zbc")


@dataclass
class DetectorGeometry:
    """Refined detector geometry — the primary output of a calibration run.

    Distances are micrometers, angles are degrees, wavelength is angstroms.
    Matches MIDAS conventions. Distortion coefficients p0–p14 follow the
    15-parameter analytical model (tilt + spherical + dipole + trefoil +
    octupole) described in the calibration paper.
    """

    # Primary geometry
    lsd: float = 1_000_000.0
    ybc: float = 1024.0
    zbc: float = 1024.0
    tx: float = 0.0
    ty: float = 0.0
    tz: float = 0.0

    # 15-parameter distortion
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
    rhod: float = 0.0

    # Detector descriptors
    wavelength: float = 0.0
    px: float = 200.0
    nr_pixels_y: int = 0
    nr_pixels_z: int = 0

    # Fit quality (not geometry, but carried alongside for convenience)
    mean_strain: float = 0.0
    std_strain: float = 0.0

    @property
    def tilts(self) -> tuple[float, float, float]:
        """(tx, ty, tz) convenience accessor."""
        return (self.tx, self.ty, self.tz)

    @property
    def distortion(self) -> tuple[float, ...]:
        """(p0, p1, …, p14) — all 15 analytical distortion coefficients."""
        return tuple(getattr(self, f"p{i}") for i in range(15))

    # ------------------------------------------------------------------
    # JSON — native, lossless, symmetric
    # ------------------------------------------------------------------
    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "DetectorGeometry":
        allowed = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in allowed})

    def to_json(self, path: str | Path) -> Path:
        p = Path(path)
        p.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True))
        return p

    @classmethod
    def from_json(cls, path: str | Path) -> "DetectorGeometry":
        data = json.loads(Path(path).read_text())
        return cls.from_dict(data)

    # ------------------------------------------------------------------
    # MIDAS Parameters.txt — MIDASCalibrant consumes this format
    # ------------------------------------------------------------------
    def to_midas_params(
        self,
        path: str | Path,
        *,
        extra: dict[str, object] | None = None,
    ) -> Path:
        """Write the geometry as a MIDAS Parameters.txt.

        `extra` is an optional dict of additional ``key value`` pairs to append
        verbatim — useful for ring selection, I/O paths, or pre-processing
        flags that a downstream caller needs to add on top of pure geometry.
        """
        p = Path(path)
        with p.open("w") as f:
            f.write(f"Lsd {self.lsd}\n")
            f.write(f"BC {self.ybc} {self.zbc}\n")
            for key in ("tx", "ty", "tz"):
                f.write(f"{key} {getattr(self, key)}\n")
            for i in range(15):
                f.write(f"p{i} {getattr(self, f'p{i}')}\n")
            f.write(f"RhoD {self.rhod}\n")
            f.write(f"Wavelength {self.wavelength}\n")
            f.write(f"px {self.px}\n")
            if self.nr_pixels_y:
                f.write(f"NrPixelsY {self.nr_pixels_y}\n")
            if self.nr_pixels_z:
                f.write(f"NrPixelsZ {self.nr_pixels_z}\n")
            if extra:
                for key, value in extra.items():
                    f.write(_format_kv(key, value))
        return p

    @classmethod
    def from_midas_params(cls, path: str | Path) -> "DetectorGeometry":
        """Parse a MIDAS Parameters.txt back into a DetectorGeometry."""
        geom = cls()
        for key, values in _iter_params(Path(path)):
            cls._assign_from_param(geom, key, values)
        return geom

    @staticmethod
    def _assign_from_param(geom: "DetectorGeometry", key: str, values: list[str]) -> None:
        if key == "BC" and len(values) == 2:
            geom.ybc, geom.zbc = float(values[0]), float(values[1])
            return
        mapping = {
            "Lsd": ("lsd", float),
            "tx": ("tx", float), "ty": ("ty", float), "tz": ("tz", float),
            "RhoD": ("rhod", float),
            "Wavelength": ("wavelength", float),
            "px": ("px", float),
            "NrPixelsY": ("nr_pixels_y", int),
            "NrPixelsZ": ("nr_pixels_z", int),
        }
        if key in mapping and values:
            attr, cast = mapping[key]
            setattr(geom, attr, cast(values[0]))
            return
        if re.fullmatch(r"p\d{1,2}", key) and values:
            i = int(key[1:])
            if 0 <= i <= 14:
                setattr(geom, f"p{i}", float(values[0]))


def _format_kv(key: str, value: object) -> str:
    if isinstance(value, (list, tuple)):
        return f"{key} {' '.join(str(v) for v in value)}\n"
    return f"{key} {value}\n"


def _iter_params(path: Path) -> Iterable[tuple[str, list[str]]]:
    """Yield (key, tokens) from a MIDAS Parameters.txt, skipping blanks/comments."""
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        # Strip inline comments (MIDAS accepts `#` as comment start).
        if "#" in line:
            line = line.split("#", 1)[0].rstrip()
            if not line:
                continue
        tokens = line.split()
        if tokens:
            yield tokens[0], tokens[1:]
