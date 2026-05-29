"""Per-detector geometry config.

A pipeline run sees one or more ``DetectorConfig`` instances. They carry
the geometry needed to forward-model spots from each panel (Lsd, beam
center, tilts, distortion polynomial, zarr path).

Loadable from:
  * a JSON file (``DetectorConfig.load_many("detectors.json")``)
  * a paramstest-style file with ``DetParams 1 ...`` rows
    (``DetectorConfig.load_from_paramstest("paramstest.txt")``)
  * a paramstest with only single-detector globals
    (``DetectorConfig.single_from_paramstest("paramstest.txt")``)

Note: multi-detector ``DetParams`` rows are an FF-HEDM artifact. PF runs
always carry a single detector — but the same dataclass + loaders cover
both modes.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Sequence


@dataclass
class DetectorConfig:
    """Geometry for one detector panel."""

    det_id: int
    zarr_path: str                         # path to ``*.MIDAS.zip`` (per-panel raw data)
    lsd: float                             # sample-to-detector distance (μm)
    y_bc: float                            # beam center y in pixels
    z_bc: float                            # beam center z in pixels
    tx: float = 0.0                        # tilt about lab x-axis (deg)
    ty: float = 0.0                        # tilt about lab y-axis (deg)
    tz: float = 0.0                        # tilt about lab z-axis (deg)
    p_distortion: List[float] = field(default_factory=lambda: [0.0] * 11)

    @classmethod
    def load_many(cls, path: str | Path) -> List["DetectorConfig"]:
        """Read a list of detector configs from JSON.

        File format::

            [
              {"det_id": 1, "zarr_path": "...", "lsd": ..., ...},
              ...
            ]
        """
        path = Path(path)
        with path.open() as fp:
            entries = json.load(fp)
        if not isinstance(entries, list):
            raise ValueError(f"{path}: expected a JSON list, got {type(entries).__name__}")
        out: list[DetectorConfig] = []
        for i, e in enumerate(entries):
            if "det_id" not in e:
                e = {**e, "det_id": i + 1}
            out.append(cls(**e))
        out.sort(key=lambda d: d.det_id)
        return out

    @classmethod
    def load_from_paramstest(cls, path: str | Path,
                             zarr_path: str | None = None) -> List["DetectorConfig"]:
        """Read ``DetParams N a b c ...`` rows from a paramstest.txt.

        ff_MIDAS / FitMultipleGrains convention. The 18 floats per row are::

            DetParams <det_id> Lsd y_bc z_bc tx ty tz  p0..p10

        ``zarr_path`` is a single shared path — multi-detector workflows
        usually carry the same zip with multiple ``data/det{N}`` groups,
        OR a separate zip per detector. If you have separate zips, use
        ``load_many`` with a JSON file instead.
        """
        path = Path(path)
        out: list[DetectorConfig] = []
        with path.open() as fp:
            for raw in fp:
                line = raw.strip().rstrip(";").rstrip()
                if not line or line.startswith("#"):
                    continue
                tokens = line.split()
                if tokens[0] != "DetParams":
                    continue
                vals = [t.rstrip(";") for t in tokens[1:]]
                if len(vals) < 7:
                    raise ValueError(f"{path}: malformed DetParams row: {line!r}")
                det_id = int(float(vals[0]))
                lsd = float(vals[1])
                y_bc = float(vals[2])
                z_bc = float(vals[3])
                tx = float(vals[4])
                ty = float(vals[5])
                tz = float(vals[6])
                p_dist = [float(v) for v in vals[7:7 + 11]]
                while len(p_dist) < 11:
                    p_dist.append(0.0)
                out.append(cls(
                    det_id=det_id, zarr_path=zarr_path or "",
                    lsd=lsd, y_bc=y_bc, z_bc=z_bc,
                    tx=tx, ty=ty, tz=tz,
                    p_distortion=p_dist,
                ))
        out.sort(key=lambda d: d.det_id)
        return out

    @classmethod
    def single_from_paramstest(cls, path: str | Path,
                               zarr_path: str | None = None) -> "DetectorConfig":
        """Build one detector from the global ``Lsd`` / ``BC`` / ``tx`` keys.

        Used when a paramstest.txt carries only a single-detector setup
        (no ``DetParams`` rows), so we still produce a single-element
        detector list for the unified pipeline.
        """
        path = Path(path)
        keys = {
            "Lsd": 0.0, "Distance": 0.0,
            "BC": None,
            "tx": 0.0, "ty": 0.0, "tz": 0.0,
        }
        p_dist = [0.0] * 11
        ybc = None
        zbc = None
        with path.open() as fp:
            for raw in fp:
                line = raw.strip().rstrip(";").rstrip()
                if not line or line.startswith("#"):
                    continue
                tokens = [t.rstrip(";") for t in line.split()]
                if not tokens:
                    continue
                key = tokens[0]
                if key in ("Lsd", "Distance") and len(tokens) >= 2:
                    keys[key] = float(tokens[1])
                elif key == "BC" and len(tokens) >= 3:
                    ybc, zbc = float(tokens[1]), float(tokens[2])
                elif key in ("tx", "ty", "tz") and len(tokens) >= 2:
                    keys[key] = float(tokens[1])
                elif key in ("YBC", "y_bc"):
                    ybc = float(tokens[1])
                elif key in ("ZBC", "z_bc"):
                    zbc = float(tokens[1])
                elif key.startswith("p") and key[1:].isdigit():
                    idx = int(key[1:])
                    if 0 <= idx < 11 and len(tokens) >= 2:
                        p_dist[idx] = float(tokens[1])
        lsd = keys["Lsd"] if keys["Lsd"] > 0 else keys["Distance"]
        if lsd <= 0:
            raise ValueError(f"{path}: missing Lsd/Distance for single-detector config")
        if ybc is None or zbc is None:
            raise ValueError(f"{path}: missing BC (need 'BC ybc zbc' or YBC/ZBC keys)")
        return cls(
            det_id=1, zarr_path=zarr_path or "",
            lsd=lsd, y_bc=ybc, z_bc=zbc,
            tx=keys["tx"], ty=keys["ty"], tz=keys["tz"],
            p_distortion=p_dist,
        )

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def dump_many(detectors: Sequence["DetectorConfig"], path: str | Path) -> None:
        """Write a detectors list to JSON (the inverse of ``load_many``)."""
        with Path(path).open("w") as fp:
            json.dump([d.to_dict() for d in detectors], fp, indent=2)
