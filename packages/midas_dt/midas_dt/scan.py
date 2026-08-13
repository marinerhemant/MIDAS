"""The acquisition: which file holds which (translation, omega) frame.

A DT scan is one file per translation, each holding a full rotation. The files
are large -- 14.27 GB apiece for the 2022 U3O8 scan -- so everything here is
memory-mapped and nothing is read until it is indexed.

Raw layout (verified by arithmetic against the real files, not inferred):

    [ header: header_bytes ][ frame 0 ][ frame 1 ] ... [ frame n_frames-1 ]

with **one header per FILE**, not per frame. For U3O8:
``14,274,698,292 = 8192 + 1441 x (1475 x 1679 x 4)`` exactly, and the dark
``99,069,192 = 8192 + 10 x (...)``. Both divide with no remainder, which is
what makes the layout confirmed rather than assumed.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np

from .conventions import aps_1id_omega, unsnake

__all__ = ["RawFormat", "DTScan", "detect_snake", "frames_in_file"]

log = logging.getLogger(__name__)

#: The 2022 MPE U3O8 detector: Pilatus, int32, 8192-byte file header.
PILATUS_1475x1679 = None  # populated below, after RawFormat is defined


@dataclass(frozen=True)
class RawFormat:
    """Byte layout of one raw detector file."""

    n_pixels_y: int          # horizontal (fast axis within a row)
    n_pixels_z: int          # vertical
    dtype: np.dtype = field(default_factory=lambda: np.dtype("<i4"))
    header_bytes: int = 8192
    flip_vertical: bool = True   # legacy ImTransOpt 2 (Pilatus)

    @property
    def frame_bytes(self) -> int:
        return self.n_pixels_y * self.n_pixels_z * self.dtype.itemsize

    @property
    def frame_shape(self) -> tuple[int, int]:
        return (self.n_pixels_z, self.n_pixels_y)

    def n_frames(self, path: str | Path) -> int:
        """Frames in *path*, from its size. Raises if it does not divide."""
        size = Path(path).stat().st_size
        payload = size - self.header_bytes
        if payload <= 0:
            raise ValueError(
                f"{path} is {size} bytes, at or below the {self.header_bytes}-byte "
                f"header -- wrong format, or a truncated file"
            )
        n, rem = divmod(payload, self.frame_bytes)
        if rem:
            raise ValueError(
                f"{path}: {size} bytes = {self.header_bytes} header + "
                f"{payload} payload, which is not a whole number of "
                f"{self.frame_bytes}-byte frames ({n} frames + {rem} bytes over). "
                f"The detector dimensions or the header size are wrong."
            )
        return int(n)

    def memmap(self, path: str | Path) -> np.memmap:
        """Memory-map *path* as ``(n_frames, n_z, n_y)``. Nothing is read yet."""
        n = self.n_frames(path)
        return np.memmap(
            path, dtype=self.dtype, mode="r", offset=self.header_bytes,
            shape=(n, *self.frame_shape),
        )


PILATUS_1475x1679 = RawFormat(n_pixels_y=1475, n_pixels_z=1679)


def frames_in_file(path: str | Path, fmt: RawFormat = PILATUS_1475x1679) -> int:
    """Convenience wrapper around :meth:`RawFormat.n_frames`."""
    return fmt.n_frames(path)


@dataclass
class DTScan:
    """One DT acquisition: a set of per-translation files sharing a rotation.

    Attributes
    ----------
    files : list[Path]
        One per translation, in translation order.
    fmt : RawFormat
        Byte layout.
    omega_deg : ndarray
        Rotation angle of each frame, already in the sample frame -- i.e. the
        1-ID negation has been applied. See :func:`~midas_dt.conventions.aps_1id_omega`.
    snake : bool
        Alternate translations were scanned in the opposite rotation
        direction. Prefer :func:`detect_snake` over setting this by hand.
    dark_file : Path | None
        Dark reference; its frames are averaged.
    drop_first_frame : bool
        1-ID writes a throwaway first frame in every acquisition. Default True
        because that is the site rule, and a silently included junk frame
        biases the first projection.
    """

    files: list[Path]
    fmt: RawFormat = PILATUS_1475x1679
    omega_deg: np.ndarray = field(default_factory=lambda: np.empty(0))
    snake: bool = False
    dark_file: Path | None = None
    drop_first_frame: bool = True

    def __post_init__(self):
        self.files = [Path(f) for f in self.files]
        if not self.files:
            raise ValueError("a scan needs at least one translation file")
        missing = [f for f in self.files if not f.is_file()]
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} translation file(s) missing, first: {missing[0]}"
            )

    # ------------------------------------------------------------- geometry
    @property
    def n_translations(self) -> int:
        return len(self.files)

    @property
    def n_frames(self) -> int:
        """Usable frames per translation, after any throwaway is dropped."""
        raw = self.fmt.n_frames(self.files[0])
        return raw - 1 if self.drop_first_frame else raw

    @property
    def first_frame(self) -> int:
        return 1 if self.drop_first_frame else 0

    # ---------------------------------------------------------------- build
    @classmethod
    def from_stem(
        cls,
        directory: str | Path,
        stem: str,
        start_nr: int,
        end_nr: int,
        *,
        pad: int = 6,
        ext: str = ".raw",
        fmt: RawFormat = PILATUS_1475x1679,
        start_omega: float = 180.25,
        omega_step: float = -0.25,
        negate_omega: bool = True,
        snake: bool = False,
        dark_file: str | Path | None = None,
        drop_first_frame: bool = True,
    ) -> "DTScan":
        """Build from the ``<stem>_<nnnnnn><ext>`` naming the beamline writes.

        ``start_omega``/``omega_step`` are the NOMINAL motor values from the
        parameter file; the 1-ID negation is applied here, once, so callers
        never see un-negated angles.
        """
        directory = Path(directory)
        files = [directory / f"{stem}_{n:0{pad}d}{ext}"
                 for n in range(start_nr, end_nr + 1)]
        n_raw = fmt.n_frames(files[0]) if files and files[0].is_file() else 0
        n_use = n_raw - 1 if drop_first_frame else n_raw
        nominal = start_omega + omega_step * (
            np.arange(n_use, dtype=np.float64) + (1 if drop_first_frame else 0)
        )
        return cls(
            files=files, fmt=fmt,
            omega_deg=aps_1id_omega(nominal, negate=negate_omega),
            snake=snake,
            dark_file=Path(dark_file) if dark_file else None,
            drop_first_frame=drop_first_frame,
        )

    # ----------------------------------------------------------------- read
    def translation(self, index: int) -> np.memmap:
        """Memory-map one translation as ``(n_frames, n_z, n_y)``.

        The throwaway first frame is already excluded. Nothing is read from
        disk until the result is indexed.
        """
        if not 0 <= index < self.n_translations:
            raise IndexError(
                f"translation {index} out of range (0..{self.n_translations - 1})"
            )
        mm = self.fmt.memmap(self.files[index])
        return mm[self.first_frame:]

    def frame(self, translation: int, frame: int) -> np.ndarray:
        """One frame as a plain array, with the detector flip applied.

        Reading a frame is the only place ``flip_vertical`` is honoured, so
        every consumer sees the same orientation.
        """
        arr = np.asarray(self.translation(translation)[frame])
        return arr[::-1] if self.fmt.flip_vertical else arr

    def dark(self) -> np.ndarray | None:
        """Averaged dark frame, or ``None`` when no dark was given."""
        if self.dark_file is None:
            return None
        mm = self.fmt.memmap(self.dark_file)
        avg = np.asarray(mm, dtype=np.float64).mean(axis=0)
        return avg[::-1] if self.fmt.flip_vertical else avg

    def iter_frames(self, translation: int) -> Iterator[np.ndarray]:
        mm = self.translation(translation)
        for i in range(mm.shape[0]):
            arr = np.asarray(mm[i])
            yield arr[::-1] if self.fmt.flip_vertical else arr

    # -------------------------------------------------------------- summary
    def describe(self) -> str:
        gb = self.fmt.frame_bytes * self.fmt.n_frames(self.files[0]) / 1e9
        return (
            f"{self.n_translations} translations x {self.n_frames} frames "
            f"({self.fmt.frame_shape[0]}x{self.fmt.frame_shape[1]}, "
            f"{self.fmt.dtype}), {gb:.2f} GB/translation, "
            f"omega {self.omega_deg[0]:.2f}..{self.omega_deg[-1]:.2f} deg, "
            f"snake={self.snake}, drop_first_frame={self.drop_first_frame}"
        )


# ------------------------------------------------------------------- snake
def detect_snake(
    profiles: np.ndarray, *, min_gain: float = 1.05
) -> tuple[bool, float]:
    """Decide whether alternate translations run backwards in omega.

    Takes a cheap per-(translation, frame) summary -- a total intensity curve
    is enough -- shaped ``(n_translations, n_frames)``, and compares how well
    neighbouring translations agree as-is versus with alternate rows reversed.

    Returns ``(is_snake, gain)`` where *gain* is the ratio of the two
    agreements. Detection rather than configuration is deliberate: the legacy
    ``BadRotation`` flag was set by hand, and setting it wrongly in either
    direction gives a plausible reconstruction of the wrong object.

    ``min_gain`` is the margin required before claiming a snake; near 1.0 the
    two hypotheses are indistinguishable, which the caller should treat as
    "unknown", not "no".
    """
    p = np.asarray(profiles, dtype=np.float64)
    if p.ndim != 2:
        raise ValueError(f"profiles must be 2-D (translation, frame); got {p.shape}")
    if p.shape[0] < 3:
        raise ValueError(
            f"need at least 3 translations to compare neighbours, got {p.shape[0]}"
        )

    def _agreement(arr: np.ndarray) -> float:
        # Mean correlation between adjacent translations. A snake scan
        # decorrelates neighbours because their frames run opposite ways.
        cs = []
        for i in range(arr.shape[0] - 1):
            a, b = arr[i], arr[i + 1]
            if a.std() == 0 or b.std() == 0:
                continue
            cs.append(float(np.corrcoef(a, b)[0, 1]))
        return float(np.mean(cs)) if cs else 0.0

    as_is = _agreement(p)
    flipped = _agreement(unsnake(p, axis=0, frame_axis=1))

    # Guard the ratio: correlations can be near zero or negative.
    denom = max(abs(as_is), 1e-9)
    gain = abs(flipped) / denom
    is_snake = bool(abs(flipped) > abs(as_is) * min_gain)
    log.info(
        "snake detection: as-is agreement %.4f, un-snaked %.4f, gain %.2f -> %s",
        as_is, flipped, gain, "SNAKE" if is_snake else "not a snake",
    )
    return is_snake, gain
