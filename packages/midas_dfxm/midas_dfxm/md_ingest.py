"""Ingest a real MD atomic configuration into the DFXM forward chain.

Finalises the ``load_external_field`` stub in :mod:`midas_dfxm.io` (its
docstring said "finalised once the delivery format is known" -- the delivery
format used here, and the one a collaborator's own LAMMPS run will also
produce, is a plain LAMMPS dump).

MD output is atomic positions, not a deformation-gradient field, so getting to
the ``(N,3,3)`` + ``(N,3)`` contract :func:`midas_dfxm.generators.field_from_deformation_gradient`
wants needs two steps:

1. **Per-atom local best-fit deformation gradient** (Falk-Langer / Shimizu-Ogata-Li
   style): each atom's actual nearest-neighbour vectors are least-squares fit
   against a *fixed, ideal* FCC ``<110>`` neighbour-shell template that has been
   rotated to the reference grain's own orientation (found once, from one
   well-coordinated seed atom, by ICP/Procrustes against the same template --
   correspondence-free rotation recovery is not possible from the neighbour
   shell's second moments alone, because the ideal cuboctahedral shell is
   isotropic to that order).
2. **Coarse-grain onto a voxel grid.** Per-atom F is noisy and far denser than
   anything the DFXM forward needs; bin atoms into voxels and average.

**This is a from-scratch equivalent of OVITO's "Atomic strain" / Polyhedral
Template Matching modifiers, not a replacement for them.** OVITO's PTM is the
standard, production-grade tool (handles arbitrary local orientation per atom,
not a single fixed reference); what is here is the minimum needed to
demonstrate the ingestion *pattern* without adding a heavy optional dependency
to this package. It is deliberately conservative: a per-atom fit is accepted
only if it reproduces its own matched neighbour vectors to within
``residual_threshold`` of the bond length, and it uses **one fixed reference
orientation for the whole crop** -- atoms whose true local orientation is far
from that reference (a different grain, the interior of a high-angle grain
boundary) will fail the residual gate and drop out, which is the honest
outcome, not a bug to silence. Crop to one grain (+ its immediate defect
environment) before fitting; this is not intended to run on an entire
polycrystalline box.

Units: LAMMPS dump coordinates are read as Angstrom (the convention of the
example data this was built against); output positions are micrometers, as
:class:`midas_dfxm.field.DeformationField` requires.
"""
from __future__ import annotations

import gzip
from dataclasses import dataclass

import numpy as np
import torch
from scipy.spatial import cKDTree

from .field import DeformationField
from .generators import field_from_deformation_gradient

_ANGSTROM_PER_UM = 1.0e4

__all__ = [
    "LammpsDumpAtoms",
    "read_lammps_dump",
    "refine_fcc_lattice_constant",
    "fcc_neighbor_template",
    "find_reference_orientation",
    "atomic_deformation_gradients",
    "bin_atoms_to_voxels",
    "load_external_field",
]


@dataclass
class LammpsDumpAtoms:
    """One parsed LAMMPS dump snapshot (atomic-style, orthogonal box).

    Attributes
    ----------
    positions : (N, 3) ndarray
        Atom positions, Angstrom, shifted so the box lower bound is the origin.
    box_lengths : (3,) ndarray
        Orthogonal box edge lengths, Angstrom (periodic in all three axes).
    columns : dict[str, ndarray]
        Any additional per-atom columns from the dump (e.g. ``c_epot``), keyed
        by their header name, same atom order as ``positions``.
    """

    positions: np.ndarray
    box_lengths: np.ndarray
    columns: dict


def read_lammps_dump(path: str) -> LammpsDumpAtoms:
    """Read one snapshot of a LAMMPS ``ITEM: ATOMS`` text dump (plain or ``.gz``).

    Assumes an orthogonal box and a single timestep (the first one in the
    file); a multi-timestep trajectory only needs the first frame for this
    ingestion path, which treats the file as one static configuration.
    """
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as f:
        lines = f.read().splitlines()

    if lines[0].strip() != "ITEM: TIMESTEP":
        raise ValueError(f"{path}: does not start with 'ITEM: TIMESTEP' -- not a LAMMPS dump?")
    n_atoms = int(lines[3])
    if not lines[4].startswith("ITEM: BOX BOUNDS"):
        raise ValueError(f"{path}: expected 'ITEM: BOX BOUNDS' at line 5")
    bounds = np.array([[float(x) for x in lines[5 + k].split()[:2]] for k in range(3)])
    box_lengths = bounds[:, 1] - bounds[:, 0]

    header_line = lines[8]
    if not header_line.startswith("ITEM: ATOMS"):
        raise ValueError(f"{path}: expected 'ITEM: ATOMS <cols>' at line 9")
    cols = header_line.split()[2:]
    if not {"x", "y", "z"}.issubset(cols):
        raise ValueError(f"{path}: dump columns {cols} do not include x, y, z")

    data = np.loadtxt(lines[9:9 + n_atoms])
    col_idx = {name: i for i, name in enumerate(cols)}
    positions = data[:, [col_idx["x"], col_idx["y"], col_idx["z"]]] - bounds[:, 0]
    extra = {name: data[:, i] for name, i in col_idx.items() if name not in ("x", "y", "z")}
    return LammpsDumpAtoms(positions=positions, box_lengths=box_lengths, columns=extra)


def refine_fcc_lattice_constant(
    positions: np.ndarray, box_lengths: np.ndarray, *, a0_guess: float = 3.8
) -> float:
    """Self-consistent FCC lattice constant from the nearest-neighbour peak.

    Never trust a literature lattice constant for an MD-relaxed structure --
    the potential's own equilibrium value can differ from the experimental one
    by a few percent. Restricts the estimate to atoms with the bulk (12)
    coordination, where the neighbour-distance peak is sharpest, then converts
    the FCC nearest-neighbour distance ``a0/sqrt(2)`` back to ``a0``.
    """
    tree = cKDTree(positions, boxsize=box_lengths)
    cutoff = 1.2 * a0_guess / np.sqrt(2.0)
    dist, _ = tree.query(positions, k=13, distance_upper_bound=cutoff)
    coord = np.isfinite(dist[:, 1:]).sum(axis=1)
    bulk = coord == 12
    if bulk.sum() < 10:
        raise ValueError(
            "fewer than 10 atoms found with FCC bulk (12) coordination near "
            f"a0_guess={a0_guess}; pass a better a0_guess")
    nn = dist[bulk, 1:13]
    return float(np.median(nn) * np.sqrt(2.0))


def fcc_neighbor_template(a0: float) -> np.ndarray:
    """The 12 ideal FCC ``<110>`` nearest-neighbour vectors, cubic-axis-aligned.

    ``(12, 3)`` ndarray, each of length ``a0/sqrt(2)``.
    """
    perms = [(0, 1), (0, 2), (1, 2)]
    signs = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
    bond = a0 / np.sqrt(2.0)
    template = []
    for ax1, ax2 in perms:
        for s1, s2 in signs:
            v = np.zeros(3)
            v[ax1], v[ax2] = s1, s2
            template.append(v / np.linalg.norm(v) * bond)
    return np.array(template)


def _neighbor_vectors(positions, box_lengths, tree, index, cutoff, k=16):
    dist, jidx = tree.query(positions[index], k=k, distance_upper_bound=cutoff)
    valid = np.isfinite(dist[1:])
    nbr = jidx[1:][valid]
    v = positions[nbr] - positions[index]
    return v - box_lengths * np.round(v / box_lengths)


def find_reference_orientation(
    positions: np.ndarray,
    box_lengths: np.ndarray,
    template: np.ndarray,
    *,
    seed_center_angstrom,
    cutoff: float,
    n_seed_candidates: int = 8,
    n_iter: int = 15,
    seed: int = 0,
) -> np.ndarray:
    """Find ONE reference orientation ``R0`` (proper rotation) for a grain.

    Picks a well-coordinated (12-neighbour) atom near ``seed_center_angstrom``
    and runs ICP (nearest-direction correspondence + Kabsch/Procrustes,
    iterated to a fixed point) to align its neighbour shell to ``template``.
    Correspondence-free: the algorithm does not assume which observed vector
    is which ideal ``<110>`` direction, only that *some* proper rotation makes
    them line up -- true for any atom deep in a grain whose local lattice is
    (approximately) FCC, regardless of that grain's orientation.

    The recovered ``R0`` is one of the 24 crystallographically-equivalent
    answers (the cubic point group's ambiguity); this is fine here because it
    is used as a fixed, self-consistent reference for the *whole* crop, not
    compared against an external convention.

    Tries ``n_seed_candidates`` seeds (nearest bulk atoms to the requested
    centre) and keeps the one with the lowest final residual, since a single
    seed's shell can occasionally converge to a poor local optimum.
    """
    tree = cKDTree(positions, boxsize=box_lengths)
    center = np.asarray(seed_center_angstrom, dtype=float)
    dist_to_center, cand = tree.query(center, k=200)
    cand = np.atleast_1d(cand)

    dist, jidx = tree.query(positions[cand], k=13, distance_upper_bound=cutoff)
    coord = np.isfinite(dist[:, 1:]).sum(axis=1)
    bulk_cand = cand[coord == 12]
    if len(bulk_cand) == 0:
        raise ValueError("no 12-coordinated (bulk FCC) atom found near seed_center_angstrom")

    rng = np.random.default_rng(seed)
    tries = bulk_cand[:n_seed_candidates] if len(bulk_cand) > n_seed_candidates else bulk_cand

    template_dir = template / np.linalg.norm(template, axis=1, keepdims=True)

    def _random_rotation(rng):
        # Uniform-ish random proper rotation via QR of a Gaussian matrix.
        M = rng.normal(size=(3, 3))
        Q, r = np.linalg.qr(M)
        Q = Q * np.sign(np.diag(r))
        if np.linalg.det(Q) < 0:
            Q[:, 0] *= -1.0
        return Q

    def _icp(v, R_init, n_iter):
        R = R_init
        jmatch = None
        for _ in range(n_iter):
            rotated_dir = template_dir @ R.T
            v_dir = v / np.linalg.norm(v, axis=1, keepdims=True)
            jmatch = np.argmax(v_dir @ rotated_dir.T, axis=1)
            H = v.T @ template[jmatch]
            U, _, Vt = np.linalg.svd(H)
            d = np.sign(np.linalg.det(U @ Vt))
            R_new = U @ np.diag([1.0, 1.0, d]) @ Vt
            converged = np.linalg.norm(R_new - R) < 1e-12
            R = R_new
            if converged:
                break
        resid = float(np.linalg.norm(v - (R @ template[jmatch].T).T, axis=1).mean())
        return R, resid

    # Plain ICP is only LOCALLY convergent: starting from R=identity, a grain
    # whose true orientation is far from the cubic axes (a misalignment
    # comparable to the ~55-90 degree spacing between <110> template slots)
    # can converge to a self-consistent but WRONG fixed point -- verified on a
    # synthetic rotated perfect crystal, where an unlucky rotation angle made
    # the identity-seeded ICP converge with residual comparable to the
    # correct answer's, but to a rotation nowhere near any of the 24 cubic
    # symmetry equivalents of the truth. Multiple random-rotation restarts
    # per seed atom fixes this: the correct fixed point has residual ~ the
    # thermal/relaxation noise floor (near 0 for a perfect crystal), while a
    # wrong fixed point's residual is an O(1) fraction of the bond length, so
    # the restart with the lowest residual is reliably the right one.
    best_R, best_resid = None, np.inf
    for a_idx in tries:
        v = _neighbor_vectors(positions, box_lengths, tree, a_idx, cutoff)
        if len(v) < 8:
            continue
        for R_init in (np.eye(3), *(_random_rotation(rng) for _ in range(6))):
            R, resid = _icp(v, R_init, n_iter)
            if resid < best_resid:
                best_resid, best_R = resid, R
    if best_R is None:
        raise ValueError("no candidate seed atom had enough neighbours to fit an orientation")
    return best_R


def atomic_deformation_gradients(
    positions: np.ndarray,
    box_lengths: np.ndarray,
    R0: np.ndarray,
    template: np.ndarray,
    *,
    fit_index=None,
    cutoff: float,
    cos_threshold: float = 0.80,
    min_matches: int = 6,
    residual_threshold: float = 0.15,
):
    """Per-atom local best-fit deformation gradient against ``R0 @ template``.

    For each atom (default: all of them -- pass ``fit_index`` to restrict to a
    crop), matches its neighbour vectors to the nearest rotated-template
    direction, keeps matches within ``cos_threshold`` of that direction, and
    least-squares-fits ``F`` so ``F @ template_j ~= v_observed``. An atom is
    accepted only if it has at least ``min_matches`` matches AND the fit's own
    mean residual (relative to the bond length) is below
    ``residual_threshold`` -- **do not loosen this to raise the yield**: the
    residual distribution measured on real nanocrystalline Pd data is cleanly
    bimodal (a tight peak below ~0.15 for correct-grain atoms, a broad hump
    above ~0.25 for wrong-grain / grain-boundary-core atoms fit against the
    wrong template); loosening the gate mixes the two.

    Returns ``(fitted_index, F)``: ``fitted_index`` is the subset of
    ``fit_index`` (or ``range(N)``) that passed, ``F`` is ``(M, 3, 3)``.
    """
    tree = cKDTree(positions, boxsize=box_lengths)
    if fit_index is None:
        fit_index = np.arange(positions.shape[0])
    fit_index = np.asarray(fit_index)

    rot_template = template @ R0.T
    rot_template_dir = rot_template / np.linalg.norm(rot_template, axis=1, keepdims=True)
    bond = float(np.linalg.norm(template[0]))

    dist_all, jidx_all = tree.query(positions[fit_index], k=16, distance_upper_bound=cutoff)
    out_F, out_idx = [], []
    for a_local, gidx in enumerate(fit_index):
        dd, jj = dist_all[a_local], jidx_all[a_local]
        valid = np.isfinite(dd[1:])
        nbr = jj[1:][valid]
        if len(nbr) < min_matches:
            continue
        v = positions[nbr] - positions[gidx]
        v = v - box_lengths * np.round(v / box_lengths)
        v_dir = v / np.linalg.norm(v, axis=1, keepdims=True)
        sim = v_dir @ rot_template_dir.T
        jmatch = np.argmax(sim, axis=1)
        keep = sim[np.arange(len(v)), jmatch] > cos_threshold
        if keep.sum() < min_matches:
            continue
        T, V = rot_template[jmatch[keep]], v[keep]
        gram = T.T @ T
        if np.linalg.cond(gram) > 1e3:
            continue
        F = (V.T @ T) @ np.linalg.inv(gram)
        residual = float(np.linalg.norm(V - (F @ T.T).T, axis=1).mean() / bond)
        if residual < residual_threshold:
            out_F.append(F)
            out_idx.append(gidx)
    if not out_idx:
        raise ValueError(
            "no atom passed the correspondence + residual gate; check that "
            "seed_center_angstrom / crop_angstrom actually sit inside one grain")
    return np.asarray(out_idx), np.asarray(out_F)


def bin_atoms_to_voxels(
    positions: np.ndarray, F: np.ndarray, voxel_angstrom: float, *, min_atoms: int = 2
):
    """Coarse-grain per-atom ``F`` onto a regular voxel grid (mean per voxel).

    Returns ``(voxel_positions, voxel_F)`` with ``voxel_positions`` the
    atom-position centroid of each occupied voxel (Angstrom, same frame as the
    input). Empty voxels are simply absent -- the field this produces is
    unstructured (``shape=None``), matching a real, irregular defect region.
    Voxels with fewer than ``min_atoms`` contributing (gated) atoms are
    dropped: a 1-atom voxel's mean F is that single atom's fit noise, not a
    coarse-grained average, and shows up as salt-and-pepper outliers in the
    rendered image otherwise.
    """
    lo = positions.min(axis=0)
    bins = np.floor((positions - lo) / voxel_angstrom).astype(int)
    n_bins = bins.max(axis=0) + 1
    flat = (bins[:, 0] * n_bins[1] + bins[:, 1]) * n_bins[2] + bins[:, 2]
    uniq, inverse = np.unique(flat, return_inverse=True)
    n_vox = len(uniq)
    counts = np.bincount(inverse, minlength=n_vox)
    voxel_F = np.zeros((n_vox, 3, 3))
    voxel_pos = np.zeros((n_vox, 3))
    for k in range(9):
        i, j = divmod(k, 3)
        voxel_F[:, i, j] = np.bincount(inverse, weights=F[:, i, j], minlength=n_vox) / counts
    for k in range(3):
        voxel_pos[:, k] = np.bincount(inverse, weights=positions[:, k], minlength=n_vox) / counts
    keep = counts >= min_atoms
    return voxel_pos[keep], voxel_F[keep]


def load_external_field(
    path: str,
    *,
    crop_angstrom=None,
    seed_center_angstrom=None,
    a0_angstrom: float | None = None,
    voxel_angstrom: float = 3.0,
    min_atoms_per_voxel: int = 2,
    cos_threshold: float = 0.80,
    min_matches: int = 6,
    residual_threshold: float = 0.15,
    device=None,
    dtype: torch.dtype = torch.float64,
) -> DeformationField:
    """Load a real MD (LAMMPS dump) atomic configuration as a DFXM field.

    Finalises the former stub: ``path`` is a LAMMPS ``ITEM: ATOMS`` text dump
    (plain or ``.gz``), assumed cubic FCC.

    ``crop_angstrom`` is ``((x0,x1),(y0,y1),(z0,z1))`` -- **required in
    practice**: fitting a per-atom F against one fixed reference orientation
    only makes sense inside one grain, so crop to a single-grain region (plus
    whatever defect sits inside/at its edge) before calling this, never hand
    it a whole polycrystalline box. ``seed_center_angstrom`` (default: the
    crop's own centre) is where the reference-orientation seed atom is picked
    from -- put it deep in the bulk of the grain you are cropping.

    Returns a :class:`DeformationField` with ``positions`` converted to
    micrometers and ``reference_orientation = R0`` (the ICP-recovered
    reference grain orientation), ready for :func:`midas_dfxm.forward.dfxm_image`
    and the rest of the stack with zero further conversion.
    """
    atoms = read_lammps_dump(path)
    positions, box_lengths = atoms.positions, atoms.box_lengths

    a0 = a0_angstrom if a0_angstrom is not None else refine_fcc_lattice_constant(positions, box_lengths)
    template = fcc_neighbor_template(a0)
    cutoff = 1.2 * a0 / np.sqrt(2.0)

    if crop_angstrom is None:
        raise ValueError(
            "crop_angstrom is required: fitting against one fixed reference "
            "orientation is only valid inside a single grain (see the module "
            "docstring). Pick a crop that sits inside one grain of your sample.")
    (x0, x1), (y0, y1), (z0, z1) = crop_angstrom
    if seed_center_angstrom is None:
        seed_center_angstrom = (0.5 * (x0 + x1), 0.5 * (y0 + y1), 0.5 * (z0 + z1))

    R0 = find_reference_orientation(
        positions, box_lengths, template,
        seed_center_angstrom=seed_center_angstrom, cutoff=cutoff)

    # Pad the fit region so voxels near the crop boundary still see their real
    # neighbours (an atom just inside the crop can have a neighbour just outside it).
    pad = cutoff
    in_pad = (
        (positions[:, 0] >= x0 - pad) & (positions[:, 0] <= x1 + pad)
        & (positions[:, 1] >= y0 - pad) & (positions[:, 1] <= y1 + pad)
        & (positions[:, 2] >= z0 - pad) & (positions[:, 2] <= z1 + pad)
    )
    fit_index = np.where(in_pad)[0]

    fitted_idx, F = atomic_deformation_gradients(
        positions, box_lengths, R0, template, fit_index=fit_index, cutoff=cutoff,
        cos_threshold=cos_threshold, min_matches=min_matches,
        residual_threshold=residual_threshold)

    fitted_pos = positions[fitted_idx]
    in_crop = (
        (fitted_pos[:, 0] >= x0) & (fitted_pos[:, 0] <= x1)
        & (fitted_pos[:, 1] >= y0) & (fitted_pos[:, 1] <= y1)
        & (fitted_pos[:, 2] >= z0) & (fitted_pos[:, 2] <= z1)
    )
    voxel_pos_A, voxel_F = bin_atoms_to_voxels(
        fitted_pos[in_crop], F[in_crop], voxel_angstrom, min_atoms=min_atoms_per_voxel)

    positions_um = torch.as_tensor(voxel_pos_A / _ANGSTROM_PER_UM, device=device, dtype=dtype)
    F_t = torch.as_tensor(voxel_F, device=device, dtype=dtype)
    R0_t = torch.as_tensor(R0, device=device, dtype=dtype)
    return field_from_deformation_gradient(
        F_t, positions_um, orientation=R0_t,
        lattice_params=(a0, a0, a0, 90.0, 90.0, 90.0))
