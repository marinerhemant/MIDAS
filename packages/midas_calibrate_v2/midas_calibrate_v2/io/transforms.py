"""MIDAS image transforms (``ImTransOpt``), in one place.

The transform that brings a raw detector frame into the geometry-model
orientation is three lines of array slicing and four separate correctness
traps, and it used to exist in three copies that each got a different subset
right:

  * ``pipelines/auto.py`` carried the complete version — image, dark and mask
    together, shape re-derived afterwards — and was the ONLY pipeline that
    accepted ``im_trans`` at all;
  * ``io/readers.py`` carried a second copy that moves image and mask but has
    no dark;
  * ``pipelines/ff_calibrate.py`` carried the ``ImTransOpt`` *parsing*.

Every other pipeline had none, so a detector needing a flip could not be
described to it, and callers re-implemented this by hand. Doing it partly
right fails silently — see :func:`apply_im_trans` for the specific ways.

The opcodes are MIDAS's own, applied in the order given:

===== ==================================================
code  meaning
===== ==================================================
1     flip Y (mirror the fast/column axis)
2     flip Z (mirror the slow/row axis)
3     transpose (swap Y and Z; **changes the shape**)
===== ==================================================

``0`` means "no transform" and is dropped during parsing, so ``ImTransOpt 0``
and a silent file both yield ``()``.
"""
from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np

__all__ = ["apply_im_trans", "parse_im_trans", "im_trans_from_v1"]


def parse_im_trans(raw) -> Tuple[int, ...]:
    """Normalise an ``ImTransOpt`` value to a tuple of opcodes.

    Accepts what the various sources actually hold: a tuple/list of ints, a
    whitespace-separated string (how a v1 paramstest line parses), a bare int,
    or ``None``. Zeros are dropped — they are MIDAS's "no transform" spelling,
    not an operation.
    """
    if raw is None:
        return ()
    if isinstance(raw, (list, tuple)):
        vals = list(raw)
    elif isinstance(raw, (int, np.integer)):
        vals = [int(raw)]
    else:
        vals = str(raw).split()
    out = []
    for x in vals:
        try:
            v = int(float(x))
        except (TypeError, ValueError):
            raise ValueError(f"ImTransOpt entry {x!r} is not an integer")
        if v == 0:
            continue
        if v not in (1, 2, 3):
            raise ValueError(
                f"ImTransOpt {v} is not a MIDAS transform code (1=flip Y, "
                "2=flip Z, 3=transpose, 0=none)")
        out.append(v)
    return tuple(out)


def im_trans_from_v1(v1) -> Tuple[int, ...]:
    """``ImTransOpt`` off a ``CalibrationParams``.

    ``CalibrationParams`` has no ``ImTransOpt`` field — the key lands in
    ``.extra`` if anywhere — so reading it with ``getattr`` returns nothing and
    the calibration silently runs with **no image transform**. On a file whose
    reconstruction uses ``ImTransOpt 2`` that mirrors Z, and the fit converges
    happily onto the mirrored beam centre with a good strain number. Measured:
    BC_z 1411.59 instead of 1467.46 (= 2879 − 1467.46) at 55.6 µε, reported
    PASS.
    """
    extra = getattr(v1, "extra", {}) or {}
    if "ImTransOpt" in extra:
        return parse_im_trans(extra["ImTransOpt"])
    return parse_im_trans(getattr(v1, "ImTransOpt", None))


def apply_im_trans(
    image: np.ndarray,
    dark: Optional[np.ndarray] = None,
    mask: Optional[np.ndarray] = None,
    im_trans: Sequence[int] = (),
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray], int, int]:
    """Apply ``im_trans`` to image, dark and mask together.

    Returns ``(image, dark, mask, NrPixelsY, NrPixelsZ)`` — all in the same
    frame, with the pixel counts read from the TRANSFORMED image so a caller
    cannot pick up the raw ones by mistake.

    Three things have to happen together, and each one fails silently on its
    own:

    * **the dark must ride along**, or the subtraction lands on the wrong
      pixels;
    * **the mask must ride along.** A mask left in the raw orientation while
      the image is flipped masks the wrong pixels — silently, and worse than
      no mask at all;
    * **``NrPixelsY``/``NrPixelsZ`` must come from the transformed shape.**
      Opcode 3 transposes, so on a non-square detector the raw counts are
      simply wrong afterwards.

    The image convention is ``(NrPixelsZ, NrPixelsY)`` — row = Z, column = Y —
    which is what every seed and forward module in this package assumes.
    """
    im_trans = parse_im_trans(im_trans)

    def _t(arr):
        if arr is None:
            return None
        for opt in im_trans:
            if opt == 1:
                arr = arr[:, ::-1]
            elif opt == 2:
                arr = arr[::-1, :]
            elif opt == 3:
                arr = arr.T
        return np.ascontiguousarray(arr)

    if im_trans:
        image = _t(image)
        dark = _t(dark)
        mask = _t(mask)

    if dark is not None and np.shape(dark) != np.shape(image):
        raise ValueError(
            f"dark shape {tuple(np.shape(dark))} != image shape "
            f"{tuple(np.shape(image))} (after im_trans={im_trans})")
    if mask is not None and np.shape(mask) != np.shape(image):
        raise ValueError(
            f"mask shape {tuple(np.shape(mask))} != image shape "
            f"{tuple(np.shape(image))} (after im_trans={im_trans})")

    NZ, NY = np.shape(image)
    return image, dark, mask, int(NY), int(NZ)
