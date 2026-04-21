"""Detector-mapping wrapper around ``MIDASDetectorMapper``.

MIDASDetectorMapper pre-computes a pixel → (R, η) bin map (``Map.bin`` +
``nMap.bin``) from a geometry Parameters.txt. MIDASIntegrator then consumes
those artifacts to integrate many frames at the same geometry without
redoing the per-pixel binning math.

Public API:
    Mapper(config)              — create from an IntegrationConfig.
    Mapper.from_geometry(geom)  — create from a midas_auto_calibrate.DetectorGeometry.
    .build(work_dir) -> MapArtifacts

Artifacts are paths, not numpy arrays, because they're large and ultimately
consumed by the C integrator anyway. Use
:meth:`MapArtifacts.read_header` to get shape / bin metadata.
"""

from __future__ import annotations

import struct
import subprocess
from dataclasses import dataclass
from pathlib import Path

from ._binaries import midas_bin
from ._config import IntegrationConfig, write_params_file

__all__ = ["Mapper", "MapArtifacts", "MapHeader"]


@dataclass(frozen=True)
class MapHeader:
    """Parsed fields from the 256-byte header in Map.bin / nMap.bin.

    Header layout is defined in ``src/c/MapHeader.h`` — a compact "MIDASMAP"
    magic + version + detector dims + binning + pixel-pitch. Values are
    Little-Endian per the main MIDAS convention.
    """

    n_pixels_y: int
    n_pixels_z: int
    n_r_bins: int
    n_eta_bins: int
    r_min: float
    r_max: float
    r_bin_size: float
    eta_min: float
    eta_max: float
    eta_bin_size: float
    px_y: float
    px_z: float
    magic: bytes = b"MIDASMAP"
    version: int = 1


@dataclass(frozen=True)
class MapArtifacts:
    """Paths + metadata for a built Map.bin / nMap.bin pair."""

    work_dir: Path
    map_bin: Path
    n_map_bin: Path

    @property
    def header(self) -> MapHeader | None:
        """Parse the 256-byte header from ``Map.bin``; ``None`` if absent.

        Older MIDAS builds wrote Map.bin without a magic-header prefix, in
        which case there's no metadata to read. Callers that need shape
        info for those artifacts must fall back to the Parameters.txt that
        was used to build them.
        """
        return _maybe_read_header(self.map_bin)


class Mapper:
    """Build pixel → (R, η) binning maps via MIDASDetectorMapper.

    Parameters
    ----------
    config : IntegrationConfig
        Geometry + binning for the map. Reuse the same instance across
        Mapper + Integrator calls to guarantee they agree.
    """

    def __init__(self, config: IntegrationConfig):
        self.config = config

    @classmethod
    def from_geometry(cls, geometry, **config_overrides) -> "Mapper":
        """Shortcut: build straight from a refined ``DetectorGeometry``.

        Example
        -------
        >>> geom = mac.auto_calibrate(...).geometry
        >>> mapper = mi.Mapper.from_geometry(geom, r_max=2000, eta_bin_size=0.5)
        """
        return cls(IntegrationConfig.from_geometry(geometry, **config_overrides))

    def build(
        self,
        work_dir: str | Path,
        *,
        n_cpus: int = 4,
        bin_dir: str | Path | None = None,
        params_file_name: str = "Mapper.Parameters.txt",
    ) -> MapArtifacts:
        """Invoke MIDASDetectorMapper, return paths to the artefacts."""
        work = Path(work_dir).resolve()
        work.mkdir(parents=True, exist_ok=True)

        params_path = work / params_file_name
        write_params_file(params_path, self.config.to_params())

        exe = midas_bin("MIDASDetectorMapper", bin_dir=bin_dir)
        cmd = [str(exe), str(params_path), str(n_cpus)]
        proc = subprocess.run(
            cmd, capture_output=True, text=True, cwd=work, check=False,
        )
        (work / "mapper.stdout").write_text(proc.stdout)
        if proc.stderr:
            (work / "mapper.stderr").write_text(proc.stderr)

        if proc.returncode != 0:
            tail = "\n  ".join(proc.stderr.strip().splitlines()[-10:])
            raise RuntimeError(
                f"MIDASDetectorMapper exited {proc.returncode}.\n"
                f"Last stderr:\n  {tail}"
            )

        map_bin = work / "Map.bin"
        n_map_bin = work / "nMap.bin"
        for required in (map_bin, n_map_bin):
            if not required.exists():
                raise RuntimeError(
                    f"MIDASDetectorMapper succeeded but {required.name} was "
                    f"not produced in {work}. stdout tail:\n"
                    f"  " + "\n  ".join(proc.stdout.strip().splitlines()[-5:])
                )

        return MapArtifacts(work_dir=work, map_bin=map_bin, n_map_bin=n_map_bin)


# ---------------------------------------------------------------------------
# Map.bin header parsing — see src/c/MapHeader.h for the on-disk layout.
# ---------------------------------------------------------------------------

_MAGIC = b"MIDASMAP"
_HEADER_SIZE = 256


def _maybe_read_header(path: Path) -> MapHeader | None:
    """Parse the header prefix if present, else return None.

    Returns None (not an exception) because older MIDAS builds didn't write
    a header — callers can fall back to Parameters.txt for metadata.
    """
    if not path.exists() or path.stat().st_size < _HEADER_SIZE:
        return None
    with path.open("rb") as f:
        raw = f.read(_HEADER_SIZE)
    if raw[:8] != _MAGIC:
        return None

    # Layout (see src/c/MapHeader.h; little-endian):
    #   char[8]    magic
    #   uint32     version
    #   uint32     n_pixels_y, n_pixels_z
    #   uint32     n_r_bins, n_eta_bins
    #   double     r_min, r_max, r_bin_size
    #   double     eta_min, eta_max, eta_bin_size
    #   double     px_y, px_z
    fmt = "<8sI IIII dddddd dd"
    unpacked = struct.unpack(fmt, raw[: struct.calcsize(fmt)])
    (magic, version,
     ny, nz, nr, neta,
     rmin, rmax, rbin,
     emin, emax, ebin,
     pxy, pxz) = unpacked
    return MapHeader(
        magic=magic, version=version,
        n_pixels_y=ny, n_pixels_z=nz,
        n_r_bins=nr, n_eta_bins=neta,
        r_min=rmin, r_max=rmax, r_bin_size=rbin,
        eta_min=emin, eta_max=emax, eta_bin_size=ebin,
        px_y=pxy, px_z=pxz,
    )
