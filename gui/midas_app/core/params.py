"""MIDAS PS.txt / ParametersFile parser.

Format: lines of ``KEY value [value ...]`` with ``#`` for comments and blank lines.
Values are stored as a list of raw tokens; helpers cast on demand.
"""

from __future__ import annotations
import os
from typing import Dict, List, Optional, Union


Number = Union[int, float]


class ParamFile:
    def __init__(self, path: Optional[str] = None):
        self.path: Optional[str] = path
        self.entries: Dict[str, List[str]] = {}
        if path is not None:
            self.load(path)

    # ── I/O ────────────────────────────────────────────────────────

    def load(self, path: str) -> None:
        self.path = path
        self.entries.clear()
        with open(path, 'r') as f:
            for raw in f:
                line = raw.split('#', 1)[0].strip()
                if not line:
                    continue
                tokens = line.split()
                if len(tokens) >= 1:
                    key = tokens[0]
                    self.entries[key] = tokens[1:]

    def write(self, path: Optional[str] = None) -> None:
        out = path or self.path
        if out is None:
            raise ValueError("no path given and ParamFile has no source path")
        with open(out, 'w') as f:
            for k, vs in self.entries.items():
                f.write(f"{k} {' '.join(vs)}\n")

    # ── Accessors ──────────────────────────────────────────────────

    def get_str(self, key: str, default: Optional[str] = None) -> Optional[str]:
        v = self.entries.get(key)
        return v[0] if v else default

    def get_int(self, key: str, default: Optional[int] = None) -> Optional[int]:
        v = self.entries.get(key)
        return int(v[0]) if v else default

    def get_float(self, key: str, default: Optional[float] = None) -> Optional[float]:
        v = self.entries.get(key)
        return float(v[0]) if v else default

    def get_floats(self, key: str) -> List[float]:
        return [float(t) for t in self.entries.get(key, [])]

    def get_ints(self, key: str) -> List[int]:
        return [int(t) for t in self.entries.get(key, [])]

    def has(self, key: str) -> bool:
        return key in self.entries

    def set(self, key: str, *values) -> None:
        self.entries[key] = [str(v) for v in values]

    # ── Convenience for common MIDAS keys ──────────────────────────

    @property
    def lsd(self) -> Optional[float]:
        return self.get_float('Lsd')

    @property
    def beam_center(self):  # (bcy, bcz)
        v = self.get_floats('BC')
        return tuple(v) if len(v) == 2 else None

    @property
    def wavelength(self) -> Optional[float]:
        return self.get_float('Wavelength')

    @property
    def px(self) -> Optional[float]:
        return self.get_float('px')

    @property
    def n_pixels(self):
        nrp = self.entries.get('NrPixels')
        if not nrp:
            return None
        if len(nrp) == 1:
            v = int(nrp[0])
            return (v, v)
        return (int(nrp[0]), int(nrp[1]))

    def __repr__(self):
        return f"ParamFile({self.path!r}, {len(self.entries)} entries)"


def find_param_file(directory: str) -> Optional[str]:
    """Find a typical MIDAS parameter file in ``directory``."""
    for cand in ('ps.txt', 'PS.txt', 'ParametersFile.txt', 'parameters.txt'):
        p = os.path.join(directory, cand)
        if os.path.isfile(p):
            return p
    return None
