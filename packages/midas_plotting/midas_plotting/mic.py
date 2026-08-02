"""Reading MIDAS ``.mic`` reconstructions.

One place that knows the column layout, so analysis scripts stop re-deriving it
(and stop getting it wrong -- column 2 is ``RunTime``, which differs on every
run and has repeatedly been mistaken for a physical quantity when diffing two
reconstructions).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

__all__ = ["MicMap", "read_mic"]

# text .mic column layout (0-indexed)
_COL_X, _COL_Y = 3, 4
_COL_EULER = slice(7, 10)
_COL_CONF = 10
_COL_RUNTIME = 2


@dataclass
class MicMap:
    """A parsed ``.mic``.

    Attributes
    ----------
    x, y : (N,) float
        Voxel centre positions in microns.
    euler : (N, 3) float
        Bunge ZXZ Euler angles in **radians**.
    confidence : (N,) float
        FracOverlap in [0, 1].
    pitch : float
        Voxel pitch in microns, derived from the positions rather than read
        from a column -- the grid-size column does not track the actual
        spacing in multi-resolution output.
    path : Path
    """
    x: np.ndarray
    y: np.ndarray
    euler: np.ndarray
    confidence: np.ndarray
    pitch: float
    path: Path
    raw: np.ndarray

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def mask(self, cmin: float = 0.0, cmax: float = 1.01) -> np.ndarray:
        return (self.confidence >= cmin) & (self.confidence < cmax)

    def summary(self) -> str:
        c = self.confidence
        parts = [f"{len(self)} voxels", f"pitch {self.pitch:.2f} um",
                 f"maxC {c.max():.4f}", f"medC {np.median(c):.4f}"]
        parts += [f">={t}: {int((c >= t).sum())}" for t in (0.1, 0.3, 0.5)]
        return "  ".join(parts)


def read_mic(path: str | Path, *, skip_header: int = 4) -> MicMap:
    """Parse a text ``.mic``.

    Note the binary ``MicFileBinary`` is a different format: 11 **float64** per
    voxel, whose column 2 is ``RunTime``. Reading it as float32, or diffing that
    column between runs, produces spurious "changes" -- use this text reader
    for analysis.
    """
    path = Path(path)
    d = np.genfromtxt(path, skip_header=skip_header)
    if d.ndim == 1:
        d = d[None, :]
    if d.shape[1] <= _COL_CONF:
        raise ValueError(
            f"{path}: expected >{_COL_CONF + 1} columns, got {d.shape[1]}"
        )
    x, y = d[:, _COL_X], d[:, _COL_Y]
    uniq = np.unique(np.round(x, 4))
    pitch = float(2 * np.median(np.diff(uniq))) if uniq.size > 1 else 0.0
    return MicMap(
        x=x, y=y, euler=d[:, _COL_EULER], confidence=d[:, _COL_CONF],
        pitch=pitch, path=path, raw=d,
    )
