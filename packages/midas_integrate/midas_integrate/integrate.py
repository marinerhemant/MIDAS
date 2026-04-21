"""Integrator — wraps MIDASIntegrator (CPU) with a Pythonic API.

Usage sketch:

    >>> import midas_integrate as mi
    >>> mapper = mi.Mapper(cfg)
    >>> artefacts = mapper.build(workdir)
    >>> integ = mi.Integrator(cfg, map_artifacts=artefacts, backend="cpu")
    >>> result = integ.integrate(zarr_zip_path, n_cpus=4)
    >>> result.cake_path    # .caked.hdf HDF5 file next to the zarr.zip
    >>> cake = result.load_cake()   # dict of numpy arrays

The GPU backend is the ``midas-integrate-gpu`` wheel (v0.1.0 week 13);
installing it overlays a ``backend="gpu"`` path that calls the CUDA
streaming binary instead.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional, Union

from ._binaries import midas_bin
from ._config import IntegrationConfig
from .mapper import MapArtifacts

Backend = Literal["cpu", "gpu"]

__all__ = ["Integrator", "IntegrationResult"]


@dataclass
class IntegrationResult:
    """Output of a single ``Integrator.integrate`` call.

    Attributes
    ----------
    cake_path : Path
        HDF5 ``<stem>.caked.hdf`` with the 2D cake (``IntegrationResult/FrameNr_*``)
        + ``REtaMap`` (4, nR, nEta) metadata.
    peaks_path : Path | None
        ``<stem>.caked_peaks.h5`` when ``DoPeakFit=1`` was set (optional).
    work_dir : Path
        Directory containing the zarr.zip input; the binary writes output
        artefacts next to it.
    stdout : str
        Captured MIDASIntegrator stdout — retained for debugging.
    backend : str
        Which backend was used (``"cpu"`` or ``"gpu"``).
    """

    cake_path: Path
    peaks_path: Optional[Path] = None
    work_dir: Path = field(default_factory=Path)
    stdout: str = ""
    backend: Backend = "cpu"

    def load_cake(self) -> dict:
        """Load the .caked.hdf into a dict of numpy arrays.

        Returned keys:
            - ``R``         (nR,) radius bin centres (pixels)
            - ``Eta``       (nEta,) azimuthal bin centres (degrees)
            - ``I``         (n_frames, nEta, nR) intensity
            - ``area``      (nEta, nR) per-bin accumulated pixel area
        """
        import h5py
        with h5py.File(self.cake_path, "r") as f:
            reta = f["REtaMap"][...]                    # (4, nR, nEta)
            frames = sorted(
                k for k in f["IntegrationResult"].keys() if k.startswith("FrameNr_")
            )
            intensity = [f[f"IntegrationResult/{k}"][...] for k in frames]
        R = reta[0, :, 0]
        Eta = reta[1, 0, :]
        area = reta[3]
        return {
            "R": R, "Eta": Eta,
            "I": (intensity[0] if len(intensity) == 1
                  else __import__("numpy").stack(intensity)),
            "area": area,
        }


class Integrator:
    """Run MIDASIntegrator (CPU) or MIDASIntegratorGPU (GPU) on a zarr.zip.

    Parameters
    ----------
    config : IntegrationConfig
        Geometry + binning; must match what was passed to the Mapper that
        built ``map_artifacts``. Used only for bookkeeping by default —
        the binary reads parameters from inside the zarr.zip.
    map_artifacts : MapArtifacts
        Pre-computed Map.bin / nMap.bin from ``Mapper.build()``. The binary
        requires these files in the same folder as the zarr.zip.
    backend : "cpu" | "gpu"
        Which binary to invoke. ``"gpu"`` requires the
        ``midas-integrate-gpu`` wheel; raises a clear error otherwise.
    """

    def __init__(
        self,
        config: IntegrationConfig,
        map_artifacts: MapArtifacts,
        *,
        backend: Backend = "cpu",
    ):
        self.config = config
        self.map_artifacts = map_artifacts
        if backend not in ("cpu", "gpu"):
            raise ValueError(f"backend must be 'cpu' or 'gpu', got {backend!r}")
        self.backend = backend

    def integrate(
        self,
        zarr_zip: Union[str, Path],
        *,
        n_cpus: int = 4,
        peak_params_file: Union[str, Path, None] = None,
        bin_dir: Union[str, Path, None] = None,
    ) -> IntegrationResult:
        """Run the integrator on ``zarr_zip`` and return output paths.

        The Map.bin + nMap.bin produced by :meth:`Mapper.build` must be in
        the same directory as the zarr.zip (MIDASIntegrator reads them from
        ``"."`` when the input is a zarr.zip). We copy the artifacts in if
        they live elsewhere so the call site doesn't have to.
        """
        zarr_path = Path(zarr_zip).resolve()
        if not zarr_path.exists():
            raise FileNotFoundError(f"zarr.zip not found: {zarr_path}")
        work = zarr_path.parent

        self._ensure_map_in(work)

        bin_name = "MIDASIntegrator" if self.backend == "cpu" else "MIDASIntegratorGPU"
        try:
            exe = midas_bin(bin_name, bin_dir=bin_dir)
        except Exception as e:
            if self.backend == "gpu":
                raise RuntimeError(
                    "GPU backend requires the midas-integrate-gpu wheel "
                    "(pip install 'midas-integrate[gpu]'). Original error: "
                    f"{e}"
                ) from e
            raise

        cmd = [str(exe), zarr_path.name, str(n_cpus)]
        if peak_params_file is not None:
            cmd.append(str(Path(peak_params_file).resolve()))

        proc = subprocess.run(
            cmd, capture_output=True, text=True, cwd=work, check=False,
        )
        (work / f"{bin_name.lower()}.stdout").write_text(proc.stdout)
        if proc.stderr:
            (work / f"{bin_name.lower()}.stderr").write_text(proc.stderr)

        if proc.returncode != 0:
            tail = "\n  ".join(proc.stderr.strip().splitlines()[-10:])
            raise RuntimeError(
                f"{bin_name} exited {proc.returncode}.\nLast stderr:\n  {tail}"
            )

        cake_path = work / f"{zarr_path.stem}.caked.hdf"
        if not cake_path.exists():
            # Some MIDAS versions strip multiple dots; scan broadly.
            hits = list(work.glob("*.caked.hdf"))
            if not hits:
                raise RuntimeError(
                    f"{bin_name} exited 0 but no .caked.hdf was produced "
                    f"in {work}. Last stdout:\n"
                    + "\n".join(proc.stdout.strip().splitlines()[-5:])
                )
            cake_path = hits[0]

        peaks_path = cake_path.with_name(cake_path.stem + "_peaks.h5")
        peaks_path = peaks_path if peaks_path.exists() else None

        return IntegrationResult(
            cake_path=cake_path,
            peaks_path=peaks_path,
            work_dir=work,
            stdout=proc.stdout,
            backend=self.backend,
        )

    # ---- helpers ----

    def _ensure_map_in(self, work: Path) -> None:
        """Copy Map.bin / nMap.bin into ``work`` if not already there."""
        import shutil as _shutil

        for src in (self.map_artifacts.map_bin, self.map_artifacts.n_map_bin):
            dest = work / src.name
            if not dest.exists() or not dest.samefile(src):
                if dest.exists() and dest.samefile(src):
                    continue
                _shutil.copy(src, dest)
