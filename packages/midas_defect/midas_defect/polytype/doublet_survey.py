"""Full-sample 9R doublet survey -- automatic, label-free, null-baselined.

:mod:`satellite_doublet` resolves ONE hand-picked satellite rung into <=2 members.
This module is the **sample-wide** detector: given every satellite candidate in a
layer (or pooled over layers), it finds the two-variant 9R doublet automatically
and scores it against a null, without hand curation. It is the productized form of
the ``doublet_alllayers3.py`` (v3) analysis that confirmed the doublet across all
10 demk layers (14x over null).

The v3 discriminant (why v1/v2 failed)
--------------------------------------
The doublet is a **pure ω-split at the same detector pixel** (the two 9R maturity
variants are related by a small rotation about the ω / loading axis, so their
reflections land on the same pixel a few frames apart). Two traps:

* v1 under-segmented -- disordered-fault **relrods** bridge neighbouring satellites
  and drown the split. Fix: keep only **q-compact** candidates (reject anything
  extended along the activated <111>, i.e. a relrod) via ``q_extent_along_axis``.
* naive ω-pairing over-counts -- random mosaic and Friedel mates also produce ω
  differences. Fix: pair only **co-located** candidates (same pixel) and baseline
  the co-located Δω distribution against the **non-co-located** (null) pairs.

A real doublet shows a sharp excess of co-located pairs at Δω ≈ 6-8° over the
non-co-located null -- reported as an enrichment ratio.

Inputs are plain per-candidate arrays so this is independent of any particular
voxel/label pipeline; the caller supplies detector pixel, ω, and a compactness
metric (e.g. the q-extent along the activated axis from a connected-component
label). Pure NumPy.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

__all__ = ["DoubletSurvey", "survey_doublets"]


@dataclass
class DoubletSurvey:
    """Result of a sample-wide doublet survey.

    n_candidates : compact candidates surviving the relrod cut.
    pairs : co-located pairs, each a dict with ``i``, ``j``, ``d_omega_deg``,
        ``intensity_ratio`` (>=1), ``row``, ``col``.
    colocated_dw, null_dw : Δω (deg) of co-located vs non-co-located pairs.
    frac_colocated_in_window, frac_null_in_window : fraction of each set with
        Δω inside ``[dw_lo, dw_hi]``.
    enrichment : ``frac_colocated_in_window / frac_null_in_window`` (inf-safe).
    n_doublets : co-located pairs inside the Δω window (the detected doublets).
    verdict : "doublet-present" | "no-doublet" | "insufficient".
    """

    n_candidates: int
    pairs: list
    colocated_dw: NDArray[np.floating]
    null_dw: NDArray[np.floating]
    frac_colocated_in_window: float
    frac_null_in_window: float
    enrichment: float
    n_doublets: int
    verdict: str
    metadata: dict = field(default_factory=dict)


def _wrap_dw(a: float, b: float) -> float:
    """|a-b| wrapped to [0, 180] degrees (ω is 360-periodic; a Friedel mate is 180)."""
    d = abs(a - b) % 360.0
    return d if d <= 180.0 else 360.0 - d


def survey_doublets(
    row: NDArray[np.floating],
    col: NDArray[np.floating],
    omega_deg: NDArray[np.floating],
    *,
    q_extent_along_axis: NDArray[np.floating] | None = None,
    intensity: NDArray[np.floating] | None = None,
    rung: NDArray | None = None,
    compact_max: float | None = None,
    pixel_tol: float = 2.0,
    dw_lo: float = 4.5,
    dw_hi: float = 8.5,
    min_enrichment: float = 3.0,
    min_doublets: int = 3,
) -> DoubletSurvey:
    """Detect the 9R two-variant doublet sample-wide and score it against a null.

    Parameters
    ----------
    row, col, omega_deg : (M,)
        Detector pixel (row, col) and ω (deg) of each satellite candidate.
    q_extent_along_axis : (M,), optional
        Compactness metric -- the candidate's q-extent along the activated <111>.
        Relrods are extended (large); discrete satellites are compact (small). If
        given with ``compact_max``, candidates with ``q_extent_along_axis >
        compact_max`` (relrods) are dropped before pairing.
    intensity : (M,), optional
        Per-candidate intensity, used only to report each pair's intensity ratio.
    rung : (M,), optional
        Satellite-order label (e.g. "G/3", "2G/3"). If given, only same-rung
        candidates are paired (a doublet is same-rung by construction).
    compact_max : float, optional
        Relrod-rejection threshold on ``q_extent_along_axis`` (see above).
    pixel_tol : float
        Two candidates are "co-located" if |Δrow| <= pixel_tol and
        |Δcol| <= pixel_tol.
    dw_lo, dw_hi : float
        Δω window (deg) that defines a doublet (the demk value is ~6.5°).
    min_enrichment, min_doublets : float, int
        Verdict thresholds: "doublet-present" needs enrichment >= min_enrichment
        AND at least min_doublets co-located pairs inside the window.

    Returns
    -------
    DoubletSurvey
    """
    row = np.asarray(row, dtype=np.float64)
    col = np.asarray(col, dtype=np.float64)
    om = np.asarray(omega_deg, dtype=np.float64)
    M0 = row.shape[0]
    if not (col.shape[0] == om.shape[0] == M0):
        raise ValueError("row, col, omega_deg must have the same length")

    keep = np.ones(M0, dtype=bool)
    if q_extent_along_axis is not None and compact_max is not None:
        keep &= np.asarray(q_extent_along_axis, dtype=np.float64) <= compact_max
    idx = np.nonzero(keep)[0]
    n_cand = int(idx.size)

    if n_cand < 2:
        return DoubletSurvey(
            n_cand, [], np.empty(0), np.empty(0), float("nan"), float("nan"),
            float("nan"), 0, "insufficient",
            metadata={"n_input": M0, "reason": "fewer than 2 compact candidates"},
        )

    r = row[idx]
    c = col[idx]
    o = om[idx]
    inten = None if intensity is None else np.asarray(intensity, dtype=np.float64)[idx]
    rg = None if rung is None else np.asarray(rung)[idx]

    pairs = []
    colocated_dw = []
    null_dw = []
    for a in range(n_cand):
        for b in range(a + 1, n_cand):
            if rg is not None and rg[a] != rg[b]:
                continue
            dw = _wrap_dw(o[a], o[b])
            co = abs(r[a] - r[b]) <= pixel_tol and abs(c[a] - c[b]) <= pixel_tol
            if co:
                colocated_dw.append(dw)
                if dw_lo <= dw <= dw_hi:
                    if inten is not None and min(inten[a], inten[b]) > 0:
                        ratio = float(max(inten[a], inten[b]) / min(inten[a], inten[b]))
                    else:
                        ratio = float("nan")
                    pairs.append({
                        "i": int(idx[a]), "j": int(idx[b]),
                        "d_omega_deg": float(dw), "intensity_ratio": ratio,
                        "row": float(0.5 * (r[a] + r[b])),
                        "col": float(0.5 * (c[a] + c[b])),
                    })
            else:
                null_dw.append(dw)

    colocated_dw = np.asarray(colocated_dw, dtype=np.float64)
    null_dw = np.asarray(null_dw, dtype=np.float64)

    def _frac(d: NDArray[np.floating]) -> float:
        if d.size == 0:
            return float("nan")
        return float(np.mean((d >= dw_lo) & (d <= dw_hi)))

    fco = _frac(colocated_dw)
    fnull = _frac(null_dw)
    if fnull is not None and np.isfinite(fnull) and fnull > 0:
        enrichment = fco / fnull
    elif np.isfinite(fco) and fco > 0:
        enrichment = float("inf")
    else:
        enrichment = float("nan")

    n_doublets = len(pairs)
    if colocated_dw.size == 0:
        verdict = "insufficient"
    elif (np.isfinite(enrichment) and enrichment >= min_enrichment
          and n_doublets >= min_doublets) or (enrichment == float("inf")
                                              and n_doublets >= min_doublets):
        verdict = "doublet-present"
    else:
        verdict = "no-doublet"

    return DoubletSurvey(
        n_candidates=n_cand,
        pairs=pairs,
        colocated_dw=colocated_dw,
        null_dw=null_dw,
        frac_colocated_in_window=fco,
        frac_null_in_window=fnull,
        enrichment=float(enrichment),
        n_doublets=n_doublets,
        verdict=verdict,
        metadata={
            "n_input": M0,
            "n_colocated_pairs": int(colocated_dw.size),
            "n_null_pairs": int(null_dw.size),
            "dw_window_deg": [dw_lo, dw_hi],
            "pixel_tol": pixel_tol,
        },
    )
