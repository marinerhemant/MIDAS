"""Frame normalisation for sweep-mode integration.

Sweep frames have varying intensity due to:

- **Beam current decay** (or top-up): the beamline monitor reads the
  incident flux. Divide each frame's intensity by its monitor reading.
- **Exposure time**: not all frames have the same dwell.
- **Sample transmission**: an ion chamber upstream + downstream of the
  sample tells you absorption.

Convention: ``I_normalised = (I_raw - dark) / (monitor · exposure_s · transmission)``

A user-supplied callable can override the default formula for
beamline-specific normalisation chains.
"""
from __future__ import annotations

from typing import Callable, Dict, Optional

import numpy as np


class FrameNormalizer:
    """Per-frame normalisation factors and the rule that combines them.

    Parameters
    ----------
    monitor :
        Dict ``frame_id -> monitor_count`` (positive). Pass an empty
        dict / None to skip monitor normalisation.
    exposure_s :
        Dict ``frame_id -> exposure_seconds`` (positive). None to skip.
    transmission :
        Dict ``frame_id -> transmission_factor in (0, 1]``. None to skip.
    formula :
        Callable ``(image, monitor, exposure_s, transmission) -> image``.
        Defaults to ``image / (monitor · exposure_s · transmission)``
        with any missing factor treated as 1.0.
    dark :
        Optional 2-D dark frame; subtracted before normalisation.
    clip_negatives :
        If True (default), clip post-normalisation negatives to 0.
    """

    def __init__(
        self,
        *,
        monitor: Optional[Dict[str, float]] = None,
        exposure_s: Optional[Dict[str, float]] = None,
        transmission: Optional[Dict[str, float]] = None,
        formula: Optional[Callable[..., np.ndarray]] = None,
        dark: Optional[np.ndarray] = None,
        clip_negatives: bool = True,
    ):
        self.monitor = monitor or {}
        self.exposure_s = exposure_s or {}
        self.transmission = transmission or {}
        self._formula = formula or self._default_formula
        self.dark = dark
        self.clip_negatives = bool(clip_negatives)

    @staticmethod
    def _default_formula(image: np.ndarray,
                         monitor: float, exposure_s: float,
                         transmission: float) -> np.ndarray:
        denom = monitor * exposure_s * transmission
        return image / max(denom, 1e-30)

    @classmethod
    def from_nexus_h5(
        cls,
        path,
        *,
        i0_path: str = "entry/instrument/monitor/data",
        exposure_path: str = "entry/instrument/detector/count_time",
        transmission_path: Optional[str] = None,
        frame_id_prefix: str = "frame",
        formula: Optional[Callable[..., np.ndarray]] = None,
        dark: Optional[np.ndarray] = None,
    ) -> "FrameNormalizer":
        """Build a normaliser from a NeXus HDF5 file.

        Reads per-frame ``I0`` (monitor counts), ``count_time``
        (exposure), and optionally a transmission stream from canonical
        NeXus paths. Frame IDs are auto-generated as
        ``f"{prefix}_{i:05d}"`` to match the default
        :class:`HDF5FrameSource` naming.
        """
        try:
            import h5py
        except ImportError as e:
            raise ImportError(
                "from_nexus_h5 requires h5py; pip install h5py"
            ) from e
        monitor: dict = {}
        exposure: dict = {}
        transmission: dict = {}
        with h5py.File(path, "r") as f:
            if i0_path not in f:
                raise KeyError(f"NeXus file {path}: missing {i0_path!r}")
            i0 = np.asarray(f[i0_path][()], dtype=np.float64).reshape(-1)
            if exposure_path in f:
                expo = np.asarray(f[exposure_path][()], dtype=np.float64).reshape(-1)
            else:
                expo = np.ones_like(i0)
            if transmission_path is not None:
                if transmission_path in f:
                    trn = np.asarray(
                        f[transmission_path][()], dtype=np.float64
                    ).reshape(-1)
                else:
                    raise KeyError(
                        f"NeXus file {path}: missing {transmission_path!r}"
                    )
            else:
                trn = np.ones_like(i0)
        # Broadcast singleton exposures to per-frame
        if expo.shape[0] == 1 and i0.shape[0] > 1:
            expo = np.full_like(i0, float(expo[0]))
        if trn.shape[0] == 1 and i0.shape[0] > 1:
            trn = np.full_like(i0, float(trn[0]))
        n = i0.shape[0]
        for k in range(n):
            fid = f"{frame_id_prefix}_{k:05d}"
            monitor[fid] = float(i0[k]) if i0[k] > 0 else 1.0
            exposure[fid] = float(expo[k]) if expo[k] > 0 else 1.0
            transmission[fid] = float(trn[k]) if trn[k] > 0 else 1.0
        return cls(
            monitor=monitor, exposure_s=exposure, transmission=transmission,
            formula=formula, dark=dark,
        )

    def __call__(self, frame_id: str, image: np.ndarray) -> np.ndarray:
        if self.dark is not None:
            if self.dark.shape != image.shape:
                raise ValueError(
                    f"dark shape {self.dark.shape} != image shape "
                    f"{image.shape}"
                )
            image = image - self.dark
        m = self.monitor.get(frame_id, 1.0)
        e = self.exposure_s.get(frame_id, 1.0)
        t = self.transmission.get(frame_id, 1.0)
        out = self._formula(image, m, e, t)
        if self.clip_negatives:
            out = np.clip(out, 0.0, None)
        return out


__all__ = ["FrameNormalizer"]
