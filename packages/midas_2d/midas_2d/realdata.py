"""Real-data hooks + an independent diffraction cross-check.

* :func:`load_profile` -- read a 1-D ``(q, I)`` profile from .npy / .npz / text.
* :func:`debye_reference_numpy` -- a deliberately naive, pure-NumPy double-loop
  Debye-equation implementation used to *independently* validate the tiled
  torch :func:`midas_2d.debye.debye_intensity` (a separate code path, so
  agreement is a real cross-check rather than a tautology).

When a real time-resolved dataset arrives, ``load_profile`` + the forwards in
this package + :mod:`midas_2d.inverse` are all that is needed to fit it.
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    import torch

__all__ = ["load_profile", "debye_reference_numpy"]

_FOUR_PI = 4.0 * math.pi


def load_profile(path):
    """Load a 1-D ``(q, I)`` profile.

    Supports ``.npy`` (2xN or Nx2 array), ``.npz`` (keys ``q``/``I``), or a
    whitespace/comma text file with two columns.  Returns ``(q, I)`` numpy
    arrays.
    """
    p = str(path)
    if p.endswith(".npz"):
        d = np.load(p)
        return np.asarray(d["q"], float), np.asarray(d["I"], float)
    if p.endswith(".npy"):
        arr = np.load(p)
        arr = np.asarray(arr, float)
        if arr.shape[0] == 2:
            return arr[0], arr[1]
        return arr[:, 0], arr[:, 1]
    # text
    arr = np.loadtxt(p, delimiter="," if p.endswith(".csv") else None)
    return arr[:, 0], arr[:, 1]


def debye_reference_numpy(coords, elements, q_mag):
    """Independent NumPy Debye-equation reference (slow, O(N^2 Q)).

    ``I(q) = sum_i sum_j f_i(q) f_j(q) sinc(q r_ij)`` with ``sinc(x)=sin(x)/x``.
    Form factors from ``midas_hkls`` (Cromer-Mann) at ``s = q/(4 pi)``.
    """
    from midas_hkls.form_factors import form_factor

    coords = np.asarray(_to_numpy(coords), float)
    q_mag = np.asarray(_to_numpy(q_mag), float).ravel()
    M = coords.shape[0]

    # pairwise distances
    diff = coords[:, None, :] - coords[None, :, :]
    rij = np.sqrt((diff ** 2).sum(-1))                       # (M, M)

    # unique-element form factors per q
    uniq = sorted(set(elements))
    s = q_mag / _FOUR_PI
    fmap = {el: np.array([float(form_factor(sv ** 2, el)) for sv in s]) for el in uniq}
    f = np.stack([fmap[el] for el in elements], axis=1)      # (Q, M)

    I = np.zeros_like(q_mag)
    for qi, q in enumerate(q_mag):
        qr = q * rij
        sinc = np.ones_like(qr)
        nz = qr > 1e-12
        sinc[nz] = np.sin(qr[nz]) / qr[nz]
        fi = f[qi]                                            # (M,)
        I[qi] = fi @ sinc @ fi
    return I


def _to_numpy(x):
    try:
        import torch
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
    except ImportError:
        pass
    return np.asarray(x)
