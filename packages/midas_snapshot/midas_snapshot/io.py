"""Frame listing and reading (per-frame TIFF), orientation and validity."""
from __future__ import annotations

import glob
import os
import re
from typing import List, Optional

import numpy as np
import tifffile

_NUM = re.compile(r"(\d+)(?=\.[^.]+$)")


def list_frames(spec: str) -> List[str]:
    """Frames sorted by the trailing integer of the file name (not lexically)."""
    paths = glob.glob(os.path.join(spec, "*.tif*")) if os.path.isdir(spec) else glob.glob(spec)
    paths = [p for p in paths if _NUM.search(os.path.basename(p))]
    return sorted(paths, key=lambda p: int(_NUM.search(os.path.basename(p)).group(1)))


def orient(a: np.ndarray, flip: Optional[str]) -> np.ndarray:
    """Raw detector array -> geometry frame. The flip is a property of how the
    detector is read out and MUST be checked on a calibrant profile once."""
    if flip in (None, "", "none"):
        return a
    if flip == "ud":
        return np.flipud(a)
    if flip == "lr":
        return np.fliplr(a)
    if flip == "udlr":
        return np.flipud(np.fliplr(a))
    raise ValueError(f"unknown flip {flip!r}")


def load_mask(path: Optional[str], flip: Optional[str]) -> Optional[np.ndarray]:
    if not path:
        return None
    return orient(tifffile.imread(path).astype(bool), flip)


_INJ = {"key": None, "spec": None}


def _injection():
    """End-to-end injection control (testing only). If the environment variable
    SNAPSHOT_INJECT names a JSON spec, synthetic spots are added to frames AT READ, so every
    consumer (pipeline, window tests, raw-photon lifetime checks) sees the same data.
    Spec: {"frames": <folder or glob>, "seed": int, "spots": [{"row", "col", "flux" (counts per
    frame), "sigma" (px), "first", "last" (frame indices, inclusive, in list_frames order)}]}.
    Rows/cols are in the oriented (geometry) frame."""
    key = os.environ.get("SNAPSHOT_INJECT")
    if not key:
        return None
    if _INJ["key"] != key:
        import json
        from scipy.special import erf
        spec = json.load(open(key))
        idx = {os.path.basename(p): i for i, p in enumerate(list_frames(spec["frames"]))}
        stamps = []
        for sp in spec["spots"]:
            sg = float(sp["sigma"])
            h = int(np.ceil(4 * sg)) + 1
            r0, c0 = int(np.floor(sp["row"])), int(np.floor(sp["col"]))
            fr, fc = sp["row"] - r0, sp["col"] - c0
            o = np.arange(-h, h + 1, dtype=float)
            k = np.sqrt(2.0) * sg
            pr = 0.5 * (erf((o + 0.5 - fr) / k) - erf((o - 0.5 - fr) / k))
            pc = 0.5 * (erf((o + 0.5 - fc) / k) - erf((o - 0.5 - fc) / k))
            stamps.append((r0, c0, h, float(sp["flux"]) * np.outer(pr, pc), int(sp["first"]), int(sp["last"])))
        _INJ.update(key=key, spec=dict(seed=int(spec.get("seed", 0)), idx=idx, stamps=stamps))
    return _INJ["spec"]


def _apply_injection(a: np.ndarray, path: str, bad: np.ndarray) -> None:
    inj = _injection()
    if inj is None:
        return
    i = inj["idx"].get(os.path.basename(path))
    if i is None:
        return
    rng = np.random.default_rng([inj["seed"], i])
    H, Wd = a.shape
    for r0, c0, h, lam, first, last in inj["stamps"]:
        if not first <= i <= last:
            continue
        ra, rb, ca, cb = r0 - h, r0 + h + 1, c0 - h, c0 + h + 1
        if ra < 0 or ca < 0 or rb > H or cb > Wd:
            continue
        add = rng.poisson(lam).astype(np.float64)
        sub = a[ra:rb, ca:cb]
        sub += np.where(bad[ra:rb, ca:cb], 0.0, add)


def read_frame(path: str, flip: Optional[str], mask: Optional[np.ndarray] = None,
               invalid_below: float = 0.0) -> np.ndarray:
    """Oriented float frame; invalid pixels (sentinels, mask) are set to -1.
    (With SNAPSHOT_INJECT set, synthetic spots are added to valid pixels; see _injection.)"""
    a = orient(tifffile.imread(path).astype(np.float64), flip)
    bad = a < invalid_below
    if mask is not None:
        bad |= mask
    _apply_injection(a, path, bad)
    a[bad] = -1.0
    return a


def sum_frames(paths: List[str], flip, mask=None, invalid_below=0.0) -> np.ndarray:
    """Sum of frames; a pixel invalid in any frame is invalid (-1) in the sum.

    Unreadable or mis-shaped frames (truncated writes happen in fast acquisition) are
    skipped with a warning rather than failing the whole window."""
    import warnings
    acc, bad, shape = None, None, None
    for p in paths:
        try:
            a = read_frame(p, flip, mask, invalid_below)
        except Exception as e:                      # noqa: BLE001 -- any read failure
            warnings.warn(f"skipping unreadable frame {os.path.basename(p)}: {e}")
            continue
        if a.ndim != 2 or a.size == 0 or (shape is not None and a.shape != shape):
            warnings.warn(f"skipping frame {os.path.basename(p)} with shape {a.shape}")
            continue
        shape = a.shape
        b = a < 0
        acc = np.where(b, 0, a) if acc is None else acc + np.where(b, 0, a)
        bad = b if bad is None else (bad | b)
    if acc is None:
        raise ValueError("no readable frames in window")
    acc[bad] = -1.0
    return acc
