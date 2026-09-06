"""Bragg / diffuse classification.

The entry point to the diffuse-metrology layer: given the q-space voxel cloud and
the orientations of the indexed grains, classify *every* above-threshold voxel by
its distance to the predicted reciprocal lattice ``⋃_g (U_g · G_hkl)``. Voxels
within ``tol`` of a predicted reflection are **Bragg**; the rest are **diffuse**.

Phase-agnostic: the reflection list comes from a `midas_hkls.Crystal`
(``fcc_cu_crystal()`` or ``cual2_crystal()``) via `lattice.bragg_shells`, and the
sample-frame reflection vectors from `seed_index.predict_q_from_U` (reused, not
re-ported).

Differentiability note (contract §1.1): the reflection-point *prediction* is the
differentiable torch path (`predicted_reflection_points`). The nearest-neighbour
*classification* is an inherently discrete spatial query, so it lives in the
`_discrete` cKDTree helper off the gradient path — the contract-sanctioned place
for numpy/scipy.
"""
from __future__ import annotations

from dataclasses import dataclass

import math
import numpy as np
import torch

from .seed_index import predict_q_from_U


def enumerate_hkls(crystal, *, q_max_inv_A: float = 8.5) -> np.ndarray:
    """All symmetry-allowed (h,k,l) up to ``q_max_inv_A``, full multiplicity.

    Unlike `lattice.bragg_shells` (which returns one ASU representative per |q|
    shell), this returns the **full signed reflection set** — every (h,k,l)
    that is not systematically absent for the crystal's space group — because
    classification needs each physical reflection, not a family rep. The
    allowance test is phase-agnostic: `SpaceGroup.is_systematically_absent`
    encodes FCC all-even/all-odd, I-centring, glides, etc.

    Assumes an orthogonal cell (cubic/tetragonal) for the |q| cutoff, matching
    `seed_index.predict_q_from_U`'s ``(2π h/a, 2π k/a, 2π l/c)`` map.
    """
    sg = crystal.space_group
    a = float(crystal.lattice.a)
    c = float(crystal.lattice.c)
    twopi = 2.0 * math.pi
    h_max = int(math.ceil(q_max_inv_A * max(a, c) / twopi)) + 1
    out = []
    for h in range(-h_max, h_max + 1):
        for k in range(-h_max, h_max + 1):
            for l in range(-h_max, h_max + 1):
                if h == 0 and k == 0 and l == 0:
                    continue
                qmag = twopi * math.sqrt((h / a) ** 2 + (k / a) ** 2 + (l / c) ** 2)
                if qmag > q_max_inv_A:
                    continue
                if sg.is_systematically_absent(h, k, l):
                    continue
                out.append((h, k, l))
    return np.asarray(out, dtype=np.int64)


def predicted_reflection_points(
    orientations,
    crystal,
    *,
    q_max_inv_A: float = 8.5,
    device=None,
    dtype=None,
) -> torch.Tensor:
    """Sample-frame predicted Bragg points ``⋃_g (U_g · G_hkl)``.

    Differentiable in the orientations and (implicitly) the cell constants.

    Parameters
    ----------
    orientations
        (G,3,3) or (G,9) array/tensor of grain orientation matrices U
        (crystal→sample), e.g. columns 1-9 of a MIDAS ``Grains.csv``.
    crystal
        `midas_hkls.Crystal`; supplies the allowed hkl list and the cell
        constants (a, c) for the orthogonal reciprocal map.
    q_max_inv_A
        Reflection-list cutoff in 1/Å.

    Returns
    -------
    (M, 3) torch.Tensor of predicted sample-frame q-vectors, M = G × n_hkl.
    """
    if isinstance(orientations, torch.Tensor):
        # preserve autograd; only move/cast if explicitly asked
        U = orientations
        if dtype is not None:
            U = U.to(dtype=dtype)
        if device is not None:
            U = U.to(device=device)
    else:
        U = torch.as_tensor(np.asarray(orientations),
                            dtype=dtype if dtype is not None else torch.float64,
                            device=device)
    if U.ndim == 2 and U.shape[-1] == 9:
        U = U.reshape(-1, 3, 3)
    elif U.ndim == 2 and U.shape == (3, 3):
        U = U.reshape(1, 3, 3)
    dt, dev = U.dtype, U.device
    hkls = torch.as_tensor(enumerate_hkls(crystal, q_max_inv_A=q_max_inv_A),
                           dtype=dt, device=dev)
    a = torch.as_tensor(float(crystal.lattice.a), dtype=dt, device=dev)
    c = torch.as_tensor(float(crystal.lattice.c), dtype=dt, device=dev)
    pts = [predict_q_from_U(U[g], hkls, a, c) for g in range(U.shape[0])]
    return torch.cat(pts, dim=0)


@dataclass
class BraggDiffuseSplit:
    """Per-voxel Bragg/diffuse classification + intensity summary."""
    dist_to_lattice: np.ndarray     # (N,) nearest-predicted-reflection distance, 1/Å
    on_lattice: np.ndarray          # (N,) bool — Bragg if dist < tol
    tol_inv_A: float
    n_voxels: int
    bragg_intensity_frac: float     # Σ I(Bragg) / Σ I(all)
    diffuse_intensity_frac: float

    @property
    def diffuse(self) -> np.ndarray:
        return ~self.on_lattice


def _discrete_nn_distance(q_sample: np.ndarray, predicted_pts: np.ndarray) -> np.ndarray:
    """Nearest-predicted-reflection distance for every voxel (scipy cKDTree).

    Off the gradient path by design (see module docstring).
    """
    from scipy.spatial import cKDTree

    tree = cKDTree(np.ascontiguousarray(predicted_pts, dtype=np.float64))
    dist, _ = tree.query(np.ascontiguousarray(q_sample, dtype=np.float64),
                         k=1, workers=-1)
    return dist


def classify_voxels(
    q_sample: np.ndarray,
    intensity: np.ndarray,
    predicted_pts,
    *,
    tol_inv_A: float = 0.05,
) -> BraggDiffuseSplit:
    """Split voxels into Bragg (on-lattice) vs diffuse (off-lattice).

    Parameters
    ----------
    q_sample
        (N,3) sample-frame q-vectors (the producer is responsible for using the
        full tilt+distortion detector model — e.g. `midas_transforms`'
        ``apply_tilt_distortion`` — so the q's are physically correct).
    intensity
        (N,) per-voxel intensity.
    predicted_pts
        (M,3) predicted reflection points from `predicted_reflection_points`
        (torch tensor or numpy array).
    tol_inv_A
        On-lattice distance threshold in 1/Å (demk production used 0.05).
    """
    q = np.asarray(q_sample, dtype=np.float64)
    I = np.asarray(intensity, dtype=np.float64)
    if isinstance(predicted_pts, torch.Tensor):
        predicted_pts = predicted_pts.detach().cpu().numpy()
    dist = _discrete_nn_distance(q, predicted_pts)
    on = dist < tol_inv_A
    tot = I.sum()
    bragg_frac = float(I[on].sum() / tot) if tot > 0 else 0.0
    return BraggDiffuseSplit(
        dist_to_lattice=dist,
        on_lattice=on,
        tol_inv_A=float(tol_inv_A),
        n_voxels=int(q.shape[0]),
        bragg_intensity_frac=bragg_frac,
        diffuse_intensity_frac=float(1.0 - bragg_frac) if tot > 0 else 0.0,
    )


def on_lattice_fraction(
    q_sample: np.ndarray,
    intensity: np.ndarray,
    predicted_pts,
    *,
    bright_percentile: float = 99.5,
    tol_inv_A: float = 0.1,
) -> float:
    """Geometry-validation QC: fraction of *bright* voxels within ``tol`` of the
    predicted lattice.

    With a correct detector model + orientations this is high (demk: 96.6 % at
    the 99.5th-percentile-bright cut, 0.1 1/Å tol; chance level ~5 %). A low
    value flags a broken geometry (wrong tilts/distortion, ω sign, etc.) — it
    was the discriminator that retired the old ``to_q_sam`` transform.
    """
    q = np.asarray(q_sample, dtype=np.float64)
    I = np.asarray(intensity, dtype=np.float64)
    if isinstance(predicted_pts, torch.Tensor):
        predicted_pts = predicted_pts.detach().cpu().numpy()
    bright = I >= np.percentile(I, bright_percentile)
    if bright.sum() == 0:
        return 0.0
    dist = _discrete_nn_distance(q[bright], predicted_pts)
    return float((dist < tol_inv_A).mean())


@dataclass
class ConventionVerdict:
    """Which orientation convention this voxel cloud is actually in."""
    convention: str | None      # "OM" | "OM.T" | None when undecidable
    frac_om: float              # on-lattice fraction using the matrix as given
    frac_om_t: float            # on-lattice fraction using its transpose
    margin: float               # ratio winner/loser (inf if loser is 0)
    decisive: bool
    note: str

    def __bool__(self) -> bool:      # so `if not verdict:` reads naturally
        return bool(self.decisive)


def check_orientation_convention(
    q_sample: np.ndarray,
    intensity: np.ndarray,
    orientations,
    crystal,
    *,
    q_max_inv_A: float | None = None,
    bright_percentile: float = 99.5,
    tol_inv_A: float = 0.1,
    min_margin: float = 3.0,
    min_frac: float = 0.30,
) -> ConventionVerdict:
    """Ask the **data** which orientation convention a voxel cloud is in.

    There is no universal answer, and assuming one is a live trap in this
    project: two demk voxel products from the *same* experiment require
    **opposite** conventions. ``all_labels_qvox.npz`` needs the raw MIDAS
    ``Grains.csv`` matrix as given (on-lattice 0.978 vs 0.212 for its
    transpose), while the ``demk_g1592_9r`` ladder fixture needs the transpose
    (bright voxels 0.21 Å⁻¹ from the ladder axis vs 4.47 Å⁻¹). Each analysis
    was right for its own product; a rule stated universally is wrong for one
    of them. Cf. ``LAB_NOTEBOOK.md`` R5, where a frame error of exactly this
    class produced a retracted result.

    Runs `on_lattice_fraction` both ways and returns the winner with its
    margin. **Call this before any per-grain attribution on a cloud whose
    provenance you have not personally checked.**

    Limitation you must respect
    ---------------------------
    This test is only valid for a cloud that *contains* Bragg intensity. On a
    curated **satellite** cloud the reflections are off-lattice by
    construction, both conventions score near zero, and the verdict comes back
    ``decisive=False`` — that is the honest answer, not a failure. Use the
    axis-based test there instead (see `polytype.ladder`): project the bright
    voxels onto ``U @ <hkl_axis>`` and compare perpendicular spread.

    Returns
    -------
    ConventionVerdict
        ``convention`` is ``"OM"`` (use the matrix as given), ``"OM.T"`` (pass
        the transpose), or ``None`` when the data cannot decide.
    """
    q = np.asarray(q_sample, dtype=np.float64)
    oms = np.asarray(
        orientations.detach().cpu().numpy() if isinstance(orientations, torch.Tensor)
        else orientations, dtype=np.float64).reshape(-1, 3, 3)
    if q_max_inv_A is None:
        q_max_inv_A = float(np.linalg.norm(q, axis=1).max()) + 0.5

    fracs = {}
    for name, mats in (("OM", oms), ("OM.T", np.transpose(oms, (0, 2, 1)))):
        pts = predicted_reflection_points(mats, crystal, q_max_inv_A=q_max_inv_A)
        fracs[name] = on_lattice_fraction(
            q, intensity, pts,
            bright_percentile=bright_percentile, tol_inv_A=tol_inv_A)

    win, lose = ("OM", "OM.T") if fracs["OM"] >= fracs["OM.T"] else ("OM.T", "OM")
    margin = float("inf") if fracs[lose] == 0 else fracs[win] / fracs[lose]
    decisive = bool(fracs[win] >= min_frac and margin >= min_margin)
    if decisive:
        note = (f"{win}: {fracs[win]:.3f} on-lattice vs {fracs[lose]:.3f} "
                f"({margin:.1f}x). Use the matrix "
                + ("as given." if win == "OM" else "TRANSPOSED."))
    else:
        note = (f"UNDECIDABLE: best {win} at {fracs[win]:.3f} on-lattice "
                f"({margin:.1f}x over {lose}), below the {min_frac:.2f}/"
                f"{min_margin:.1f}x bar. Expected for a satellite-only cloud "
                f"(off-lattice by construction) or a broken geometry — these "
                f"look identical here. Use the axis test, or check the geometry.")
    return ConventionVerdict(
        convention=win if decisive else None,
        frac_om=float(fracs["OM"]), frac_om_t=float(fracs["OM.T"]),
        margin=float(margin), decisive=decisive, note=note)
