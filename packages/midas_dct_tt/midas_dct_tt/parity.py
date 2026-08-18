"""Separate lattice ROTATION from omega-independent contamination by parity.

The problem. In a topotomography rocking scan the per-pixel peak shift is not a
pure rotation measurement:

    d(theta_B) = w . a_lab  -  tan(theta) * eps_GG

so a lattice rotation about the sensitivity axis and an axial strain along ``G``
enter the same observable. A single reflection cannot separate them by fitting,
because both are static properties of the same voxel. Absorption, detector gain
drift and any signal-to-noise bias of the peak estimator enter the same way.

The separation, which needs no reconstruction and no model. Under
``psi -> psi + 180`` the sensitivity axis reverses exactly,

    a_s(psi + 180) = -a_s(psi),

while the ray through a given voxel is merely mirrored on the detector. So the
rotation part of the measurement is **odd** under an antipodal view swap, and
*any* psi-independent per-voxel scalar -- strain, absorption, estimator bias --
is **even**. Correlating a view against its mirrored antipode therefore returns
-1 for a pure rotation field and +1 for pure even contamination, with no fitted
parameter anywhere.

On the ESRF Ti-7Al dataset (grain 605, scan tt_1) this gives ``-0.820 +/- 0.010``
over 45 antipodal pairs, **all 45 negative**. (A scratchpad version of the same
test reported -0.840; the difference is the column used as the mirror axis --
here the orbit-fitted axis rather than the mean mask centroid. The verdict is
identical and 45/45 reproduces exactly.)

Read the bound carefully: :func:`even_power_fraction` returns a fraction of
POWER. ``rho = -0.820`` gives 0.090 of the power, which is
``sqrt(0.090) = 30%`` in AMPLITUDE. Quoting "10%" as an amplitude bound confuses
the two and understates the contamination by a factor of three.

The bound. Writing the measurement as ``m = odd + even`` with uncorrelated parts,
the antipodal correlation is ``-(P_odd - P_even)/(P_odd + P_even)`` in power, so

    P_even / P_total = (1 + rho) / 2

with ``rho`` the measured correlation. :func:`even_power_fraction` returns that.
"""
from __future__ import annotations

import math

import numpy as np

__all__ = ["antipodal_pairs", "parity_correlation", "even_power_fraction"]


def antipodal_pairs(psi_deg, *, tol_deg: float = 1.0):
    """Index pairs ``(i, j)`` whose scan angles differ by 180 degrees.

    Each unordered pair is returned once.
    """
    psi = np.asarray(psi_deg, dtype=float)
    out = []
    for i in range(len(psi)):
        for j in range(i + 1, len(psi)):
            d = abs((psi[j] - psi[i]) % 360.0 - 180.0)
            if d <= tol_deg:
                out.append((i, j))
    return out


def _ncc(a, b):
    a = a - a.mean()
    b = b - b.mean()
    d = math.sqrt(float((a * a).sum()) * float((b * b).sum()))
    return float((a * b).sum()) / d if d > 1e-30 else 0.0


def parity_correlation(images, psi_deg, *, valid=None, axis_u=None,
                       tol_deg: float = 1.0, min_pixels: int = 100):
    """Correlation of each view with its MIRRORED antipode.

    Parameters
    ----------
    images : (S, n_v, n_u) array
        Per-pixel rocking-peak shift (any units; the statistic is scale free).
    psi_deg : (S,) array
        Scan angle of each image.
    valid : (S, n_v, n_u) bool array, optional
        Per-view validity mask. Pixels must be valid in BOTH views of a pair.
    axis_u : float, optional
        Column of the rotation axis on the detector, about which the antipodal
        view is mirrored. Defaults to the centre of the frame.
    min_pixels : int
        Pairs with fewer jointly valid pixels than this are skipped.

    Returns
    -------
    mean : float
        Mean correlation over pairs. Near -1 means the signal is dominated by
        lattice rotation; near +1 means it is dominated by a psi-independent
        scalar such as strain or absorption.
    per_pair : (P,) ndarray
    pairs : list of (i, j)

    Notes
    -----
    This is a diagnostic on MEASURED data, so it is deliberately model-free: no
    geometry, no reconstruction and no fitted parameter enters. A positive or
    near-zero result invalidates interpreting the rocking shift as a rotation
    field, whatever a subsequent inversion reports.
    """
    im = np.asarray(images, dtype=float)
    if im.ndim != 3:
        raise ValueError("images must be (S, n_v, n_u)")
    n_u = im.shape[2]
    if axis_u is None:
        axis_u = (n_u - 1) / 2.0
    if valid is None:
        valid = np.ones(im.shape, dtype=bool)
    valid = np.asarray(valid, dtype=bool)
    if valid.shape != im.shape:
        raise ValueError("valid must match images in shape")

    # mirror about axis_u: column u maps to 2*axis_u - u
    src = np.rint(2.0 * axis_u - np.arange(n_u)).astype(int)
    keep = (src >= 0) & (src < n_u)

    pairs, vals = [], []
    for i, j in antipodal_pairs(psi_deg, tol_deg=tol_deg):
        bj = np.zeros_like(im[j])
        vj = np.zeros_like(valid[j])
        bj[:, keep] = im[j][:, src[keep]]
        vj[:, keep] = valid[j][:, src[keep]]
        m = valid[i] & vj
        if int(m.sum()) < min_pixels:
            continue
        pairs.append((i, j))
        vals.append(_ncc(im[i][m], bj[m]))
    if not pairs:
        return float("nan"), np.empty(0), []
    v = np.asarray(vals, dtype=float)
    return float(v.mean()), v, pairs


def even_power_fraction(rho: float) -> float:
    """Fraction of measured POWER carried by the even (non-rotation) part.

    ``rho`` is the antipodal correlation from :func:`parity_correlation`.
    Returns ``(1 + rho) / 2``, clipped to ``[0, 1]``.
    """
    return float(min(1.0, max(0.0, 0.5 * (1.0 + float(rho)))))
