"""Node-basis texture tomography: per-voxel ODFs as non-negative weights on a list of orientation nodes.

**Status: checked on simulation only; not a measured result; the engineering claim is provisional.** The gates are
``tests/test_texture_nodes.py`` and ``tests/test_odf_readout.py``; the README of the package lists the limits.

**The model.** Every voxel ``v`` carries non-negative weights ``c[v, k]`` on ``K`` orientation nodes. A node is a Gaussian on SO(3)
centred on ``U_k`` (crystal -> sample) with width ``sigma_k``, summed over the crystal symmetry. The measured quantity in the bin
``(omega, t, eta, ring)`` of a rotation series is the Radon projection (translation ``t``) of the voxel pole-figure value at the sample
direction that bin probes::

    data[o, t, e, r] = sum_{v, k} Radon_o[t, v] * c[v, k] * g_k[o, e, r]
    g_k[o, e, r]     = (1 / m_r) * mean_{sub-samples in the bin} sum_{h in {hkl}_r} f_sigma( angle(y(omega, eta, r), U_k h) )

``y = R(omega k)^T ghat_lab`` is the sample-frame direction diffracting into ``(omega, eta)`` on ring ``r`` and ``f_sigma`` is the
von Mises-Fisher density ``kappa exp(kappa (cos g - 1)) / (2 pi (1 - exp(-2 kappa)))``, ``kappa = 1 / sigma^2``. A Gaussian of per-axis
rotation-vector standard deviation ``sigma`` on SO(3) displaces a pole by ``sigma`` per tangent axis (the component of the rotation along
the pole does not move it), so this is its pole figure to first order in ``sigma``; ``tests/test_texture_nodes.py`` compares it with sampled
orientations (Matthies et al. 1987 treat pole figures of such radial kernels). The sum over the symmetry-equivalent, both-sign normals
``h`` of midas_hkls is the symmetry sum of the SO(3) kernel.

No ``eta``-dependent Jacobian or Lorentz factor appears, on purpose: the number of crystallites crossing the Ewald sphere inside a bin is
``P J d omega d eta`` with ``J`` the Jacobian of ``(omega, eta) -> sphere`` and each carries a Lorentz weight ``1 / J`` (times ``cos theta``),
so the integrated intensity per bin is ``cos(theta) P d omega d eta``: proportional to the pole density, with a per-ring constant that the
ring normalisation removes. Structure factors and multiplicities likewise drop out of ring-normalised data (:func:`ring_normalise`).

**Factorisation.** The voxel enters only through the Radon matrix and the node only through ``g``, so the operator is applied per node
batch: Radon-project the weight maps of ``K_b`` nodes (the sparse operator of :func:`midas_dt.direct.projection_matrix`), then scatter
through the block-diagonal (one block per omega frame) sparse ``g``. Intermediates scale with ``node_batch``, not with ``K``. The sparse
``g`` itself holds ``~K * (spots per node) * (footprint)`` entries; ``cache=False`` rebuilds it every call instead of keeping it.

**Bin-integrated sampling.** The pole figure is averaged over ``sub_omega x sub_eta`` sub-samples per detector bin. One sample per bin
(the default) is only adequate when the basis is wider than the bin; for ``sigma`` below the bin width it aliases (the response of a node
depends on where its spot falls inside the bin), so use ``sub_eta, sub_omega`` of about ``bin / sigma`` rounded up. Measured in
``tests/test_texture_nodes.py::test_subsampling_reduces_aliasing``. The Radon projection is evaluated at the frame
centre; the translation smear within one frame (``r * d omega``, at most ``0.2`` voxel at ``r = 50`` voxels for 0.5 deg frames) is not
modelled.

**Conventions.** Micrometre-free: the voxel edge is the unit of length and equals one translation bin. Angles in degrees at the API,
wavelength in angstrom. Orientation matrices map CRYSTAL -> SAMPLE (``g_sample = U h_crystal``); the crystal frame is the Cartesian frame
of midas_hkls (``a1`` along x). ``omega`` rotates sample vectors into the lab, ``v_lab = R(omega k) v_sample`` (right-handed about
``rot_axis``); apply :func:`midas_dt.conventions.aps_1id_omega` upstream where the stage needs it. The scan geometry is a
:class:`TextureScan` whose default lab directions are MIDAS-native (beam +x, rotation axis +z, ``eta`` = 0 along +z and 90 deg along -y);
``tests`` use the diffractom-style ``eta`` origin (-y) by passing the vectors explicitly.
Voxel images are ``(size, size)`` with ``x = col - (size - 1) / 2`` and ``y = row - (size - 1) / 2``, ``t = x sin(omega) + y cos(omega)``,
which is ``direct.projection_matrix``'s convention. The rotation axis must pass through the centre of the translation range.

Orientation math (symmetry operators, misorientation, orientation grids) is delegated to ``midas_stress``; hkl families and d-spacings to
``midas_hkls``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

__all__ = [
    "TextureScan",
    "Rings",
    "make_rings",
    "NodeBasis",
    "NodeMerge",
    "merge_nodes",
    "NodeOperator",
    "ring_normalise",
    "pole_figure_density",
]

_TWO_PI = 2.0 * math.pi


def _torch():
    import torch
    return torch


# ------------------------------------------------------------------ geometry
@dataclass(frozen=True)
class TextureScan:
    """Binning and lab geometry of a rotation series.

    Bins: omega frame ``i`` covers ``[omega_start + i * step, omega_start + (i + 1) * step)`` degrees; eta bin ``j`` covers
    ``[eta_start + j * 360 / n_eta, ...)`` and wraps. Uniform grids only. The omega range must not exceed 360 deg.

    ``beam`` is the incident direction, ``rot_axis`` the rotation axis (must be perpendicular to the beam), ``eta_zero`` / ``eta_ninety``
    the detector directions at ``eta = 0`` / ``90`` deg (both perpendicular to the beam). The diffracting direction is
    ``ghat_lab = cos(theta) e(eta) - sin(theta) beam`` with ``e(eta) = cos(eta) eta_zero + sin(eta) eta_ninety``.
    """

    n_omega: int
    omega_start_deg: float
    omega_step_deg: float
    n_eta: int
    n_trans: int
    wavelength_a: float
    eta_start_deg: float = 0.0
    beam: tuple = (1.0, 0.0, 0.0)
    rot_axis: tuple = (0.0, 0.0, 1.0)
    eta_zero: tuple = (0.0, 0.0, 1.0)
    eta_ninety: tuple = (0.0, -1.0, 0.0)

    def __post_init__(self):
        if min(self.n_omega, self.n_eta, self.n_trans) < 1:
            raise ValueError("n_omega, n_eta and n_trans must be positive")
        if not self.omega_step_deg > 0 or not self.wavelength_a > 0:
            raise ValueError("omega_step_deg and wavelength_a must be positive")
        if self.n_omega * self.omega_step_deg > 360.0 + 1e-9:
            raise ValueError("the omega range must not exceed 360 deg")
        b, k, d0, d9 = (self.vec(n) for n in ("beam", "rot_axis", "eta_zero", "eta_ninety"))
        if abs(b @ k) > 1e-9:
            raise ValueError("rot_axis must be perpendicular to the beam")
        if max(abs(d0 @ b), abs(d9 @ b), abs(d0 @ d9)) > 1e-9:
            raise ValueError("eta_zero and eta_ninety must be perpendicular to each other and to the beam")

    def vec(self, name: str) -> np.ndarray:
        v = np.asarray(getattr(self, name), dtype=float)
        return v / np.linalg.norm(v)

    @property
    def omega_deg(self) -> np.ndarray:
        """Frame centres (degrees)."""
        return self.omega_start_deg + (np.arange(self.n_omega) + 0.5) * self.omega_step_deg

    @property
    def eta_step_deg(self) -> float:
        return 360.0 / self.n_eta


@dataclass(frozen=True)
class Rings:
    """Ring set: unit plane normals (crystal Cartesian, both signs, symmetry-equivalent) and Bragg angles."""

    hkl: tuple
    normals: tuple          # n_rings arrays of shape (m_r, 3)
    theta_rad: np.ndarray
    d_spacing_a: np.ndarray

    @property
    def n_rings(self) -> int:
        return len(self.normals)


def make_rings(hkls, lattice, space_group: int, wavelength_a: float) -> Rings:
    """Rings from one representative ``hkl`` per family, for any space group (midas_hkls does the symmetry and the metric)."""
    from midas_hkls.lattice import Lattice
    from midas_hkls.point_group import as_lattice, plane_normals, proper_rotations_from_space_group

    lat = lattice if isinstance(lattice, Lattice) else as_lattice(lattice)
    rots = proper_rotations_from_space_group(int(space_group), lat)
    normals, theta, dsp = [], [], []
    for hkl in hkls:
        h = tuple(int(v) for v in hkl)
        d = lat.d_spacing(*h)
        s = wavelength_a / (2.0 * d)
        if not 0.0 < s < 1.0:
            raise ValueError(f"{h} is outside the Bragg cutoff at wavelength {wavelength_a} A (d = {d:.4f} A)")
        normals.append(plane_normals(h, rots, lat))
        theta.append(math.asin(s))
        dsp.append(d)
    return Rings(tuple(tuple(int(v) for v in h) for h in hkls), tuple(normals), np.array(theta), np.array(dsp))


# --------------------------------------------------------------- node basis
@dataclass
class NodeMerge:
    """Result of :func:`merge_nodes`: ``labels[i]`` is the cluster of input node ``i``; ``reps`` the input index kept for each cluster."""

    labels: np.ndarray
    reps: np.ndarray
    orientations: np.ndarray

    def merge_coefficients(self, coef) -> np.ndarray:
        """Sum the columns of ``coef`` (``(..., K_in)``) into the merged nodes (``(..., K_out)``): weight is conserved."""
        coef = np.asarray(coef)
        out = np.zeros(coef.shape[:-1] + (len(self.reps),), dtype=coef.dtype)
        for j in range(len(self.reps)):
            out[..., j] = coef[..., self.labels == j].sum(axis=-1)
        return out


def _quat_xyzw(mats: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation
    q = Rotation.from_matrix(mats).as_quat()
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


def _laue_matrices(space_group: int) -> np.ndarray:
    """Proper rotations of the Laue class as (n,3,3) matrices -- midas_stress's own operator list, so grid / misorientation / merge agree."""
    from midas_stress.orientation import make_symmetries, quat_to_orient_mat
    _, quats = make_symmetries(int(space_group))
    return np.array([np.array(quat_to_orient_mat(q), dtype=float).reshape(3, 3) for q in quats])


def merge_nodes(orientations, tol_deg: float, space_group: int, priority=None) -> NodeMerge:
    """Merge near-duplicate nodes: greedy, in descending ``priority``, every node within ``tol_deg`` (symmetry-reduced misorientation) of a
    kept node joins it.

    The kept nodes are exactly those :func:`midas_stress.group_orientations` would open for the same order and tolerance (a test pins
    this); the labels differ only in that a member is assigned to the FIRST kept node within tolerance rather than the nearest. The
    representative is an input node, not a mean. A dense node list inflates an ODF readout (many nodes within a tolerance of one truth look
    like a cloud); merge before counting atoms, and sum the coefficients with :meth:`NodeMerge.merge_coefficients`.

    Candidates for each kept node come from a KD-tree on the quaternions of its symmetry variants, and the decision is made by
    ``midas_stress.misorientation_om_batch`` -- the tree only prunes.
    """
    from scipy.spatial import cKDTree
    from midas_stress.orientation import misorientation_om_batch

    om = np.asarray(orientations, dtype=float).reshape(-1, 3, 3)
    n = len(om)
    order = np.arange(n) if priority is None else np.argsort(-np.asarray(priority, dtype=float), kind="stable")
    tol = math.radians(tol_deg)
    sym = _laue_matrices(space_group)
    q = _quat_xyzw(om)
    tree = cKDTree(np.vstack([q, -q]))
    chord = 2.0 * math.sin(min(tol, math.pi) / 4.0) * (1.0 + 1e-6) + 1e-12   # unit-quaternion chord for a rotation angle tol
    labels = np.full(n, -1, dtype=int)
    reps: list[int] = []
    rank = np.empty(n, dtype=int)
    rank[order] = np.arange(n)
    for r in order:
        if labels[r] >= 0:
            continue
        variants = _quat_xyzw(om[r] @ sym)
        cand = set()
        for v in variants:
            cand.update(i % n for i in tree.query_ball_point(v, chord))
        cand = np.array(sorted(i for i in cand if labels[i] < 0), dtype=int)
        d = misorientation_om_batch(np.broadcast_to(om[r], (len(cand), 3, 3)), om[cand], space_group)
        labels[cand[d < tol]] = len(reps)
        labels[r] = len(reps)
        reps.append(int(r))
    reps_a = np.array(reps, dtype=int)
    return NodeMerge(labels=labels, reps=reps_a, orientations=om[reps_a])


@dataclass
class NodeBasis:
    """Orientation nodes: ``orientations`` ``(K, 3, 3)`` crystal -> sample, kernel width ``sigma_deg`` (scalar or ``(K,)``; per-axis standard
    deviation of the rotation vector, which is the pole displacement per tangent axis), and the ``space_group`` whose Laue class symmetrises
    the kernel and defines misorientation."""

    orientations: np.ndarray
    sigma_deg: object
    space_group: int

    def __post_init__(self):
        om = np.asarray(self.orientations, dtype=float).reshape(-1, 3, 3)
        if len(om) == 0:
            raise ValueError("no nodes")
        if not np.allclose(om @ om.transpose(0, 2, 1), np.eye(3), atol=1e-6) or np.any(np.linalg.det(om) < 0):
            raise ValueError("orientations must be proper rotation matrices")
        self.orientations = om
        s = np.broadcast_to(np.asarray(self.sigma_deg, dtype=float), (len(om),)).copy()
        if np.any(s <= 0):
            raise ValueError("sigma_deg must be positive")
        self.sigma_deg = s

    def __len__(self) -> int:
        return len(self.orientations)

    @classmethod
    def from_grid(cls, step_deg: float, space_group: int, sigma_deg: float) -> "NodeBasis":
        """A quasi-uniform grid of the orientation space modulo the Laue class (``midas_stress.orientation_grid``, needs midas-stress >= 0.16)."""
        try:
            from midas_stress import orientation_grid
        except ImportError as exc:
            raise ImportError("NodeBasis.from_grid needs midas-stress >= 0.16.0 (orientation_grid)") from exc
        return cls(orientation_grid(step_deg, space_group), sigma_deg, space_group)

    def merge(self, tol_deg: float, priority=None):
        """Merge near-duplicate nodes (see :func:`merge_nodes`); returns ``(NodeBasis, NodeMerge)``. Kernel widths are those of the kept nodes."""
        m = merge_nodes(self.orientations, tol_deg, self.space_group, priority)
        return NodeBasis(m.orientations, self.sigma_deg[m.reps], self.space_group), m


# ------------------------------------------------------------ small helpers
def _rot(v: np.ndarray, k: np.ndarray, ang: np.ndarray) -> np.ndarray:
    """Active right-handed rotation of ``v`` (..., 3) about unit ``k`` by ``ang`` (broadcastable to ``v[..., 0]``)."""
    c = np.cos(ang)[..., None]
    s = np.sin(ang)[..., None]
    return v * c + np.cross(k, v) * s + k * (v @ k)[..., None] * (1.0 - c)


def _ghat(eta_rad: np.ndarray, theta: float, scan: TextureScan) -> np.ndarray:
    e = np.cos(eta_rad)[..., None] * scan.vec("eta_zero") + np.sin(eta_rad)[..., None] * scan.vec("eta_ninety")
    return math.cos(theta) * e - math.sin(theta) * scan.vec("beam")


def _vmf_norm(sigma_rad: np.ndarray) -> np.ndarray:
    kappa = 1.0 / np.asarray(sigma_rad, dtype=float) ** 2
    return kappa / (_TWO_PI * (-np.expm1(-2.0 * kappa)))


def pole_figure_density(centres, normals, theta_rad, omega_deg, eta_deg, sigma_deg, scan: TextureScan) -> np.ndarray:
    """Exact (unbinned, untruncated) pole-figure density of the nodes ``centres`` (``(K,3,3)`` crystal -> sample) at the points
    ``(omega_deg, eta_deg)`` of one ring: the symmetry-summed von Mises-Fisher density, family-averaged. Returns ``(K, n_points)``.
    This is the reference the operator is sub-sampled from; it exists for tests and diagnostics."""
    om = np.radians(np.atleast_1d(np.asarray(omega_deg, dtype=float)))
    et = np.radians(np.atleast_1d(np.asarray(eta_deg, dtype=float)))
    sig = np.broadcast_to(np.radians(np.asarray(sigma_deg, dtype=float)), (len(centres),))
    u = np.einsum("kij,mj->kmi", np.asarray(centres, dtype=float), normals)          # (K, m, 3)
    g = _ghat(et, theta_rad, scan)                                                    # (n, 3)
    uo = _rot(u[:, :, None, :], scan.vec("rot_axis"), om[None, None, :])              # (K, m, n, 3)
    cosg = np.einsum("kmni,ni->kmn", uo, g)
    kappa = 1.0 / sig ** 2
    f = _vmf_norm(sig)[:, None, None] * np.exp((cosg - 1.0) * kappa[:, None, None])
    return f.sum(axis=1) / len(normals)


def ring_normalise(data, *, q99: float | None = None):
    """Structure-factor-free ring normalisation of ``data[omega, t, eta, ring]``.

    Each ring is scaled to the same total (the mean of the ring totals), so multiplicity, ``|F|^2`` and the per-ring constant drop out and
    only the angular distribution is fitted. Optionally one global factor then sets the 99th percentile of the non-zero bins to ``q99``
    (use it so that a Huber ``delta`` has the same meaning across datasets). Returns ``(normalised, scale)`` with ``scale[ring]`` the total
    factor applied. A ring that sums to zero is left at zero.

    This is exact for an untextured sample and an approximation otherwise: in this model every (omega, eta) bin has a pole-density value
    and a uniform ODF gives every ring the same bin sum, but a textured one sums to ``integral P / J dA`` (``J`` the Jacobian of
    ``(omega, eta) -> sphere``, the Lorentz-type factor), which differs a little from ring to ring. The model rows are not rescaled to
    match; ``NodeOperator.ring_totals`` shows the ring-to-ring scatter of single nodes.
    """
    d = np.asarray(data, dtype=np.float64)
    tot = d.sum(axis=(0, 1, 2))
    live = tot > 0
    scale = np.zeros_like(tot)
    scale[live] = tot[live].mean() / tot[live]
    out = d * scale
    if q99 is not None:
        nz = out[out > 0]
        if nz.size:
            g = q99 / np.quantile(nz, 0.99)
            out *= g
            scale = scale * g
    return out.astype(np.float32), scale


# ----------------------------------------------------------------- operator
class NodeOperator:
    """The linear map from node weights to ring-binned data, and its exact adjoint.

    Parameters
    ----------
    scan, rings, basis
        :class:`TextureScan`, :class:`Rings` (see :func:`make_rings`), :class:`NodeBasis`.
    size
        Reconstruction grid edge; ``mask`` ``(size, size)`` bool selects the voxels that exist (default all). Unknowns are
        ``C[n_vox, K]`` with voxels in ``np.flatnonzero(mask.ravel())`` order (row-major image order).
    sub_omega, sub_eta
        Sub-samples per detector bin in omega and eta (bin-integrated sampling; see the module docstring). Default 1 = one point per bin.
    node_batch
        Nodes handled per pass; peak intermediates are ``O(node_batch)`` (the Radon sinograms of that batch, ``n_omega * n_trans *
        node_batch`` numbers) and do not grow with ``K``.
    cutoff_sigma
        Each kernel is truncated at this many sigma on the sphere (4.5 loses ~4e-5 of its mass; bins of slow, grazing crossings
        amplify the tail by ~1/J, so there the truncation error is larger: measured 2e-3 relative at 4.5 and 3e-6 at 8 for the {420} ring
        at sigma 0.4 deg. Raise it if those bins matter).
    cache
        Keep the sparse pole-figure blocks of every batch (memory ``~ nnz``); ``False`` rebuilds them on every call.
    ring_weights
        Optional per-ring factors on the model (default 1).
    dtype, device
        torch dtype (``float32`` default; use ``float64`` for exact adjoint tests) and device.

    Data layout ``(n_omega, n_trans, n_eta, n_rings)``.
    """

    _CHUNK_ELEMENTS = 3_000_000            # sub-sample values evaluated per pass of the footprint build; a class attribute so a test can force chunking

    def __init__(self, scan: TextureScan, rings: Rings, basis: NodeBasis, *, size: int, mask=None,
                 sub_omega: int = 1, sub_eta: int = 1, node_batch: int = 256, cutoff_sigma: float = 4.5,
                 cache: bool = True, ring_weights=None, dtype=None, device=None):
        torch = _torch()
        from .direct import projection_matrix
        import scipy.sparse as sp

        self.scan, self.rings, self.basis = scan, rings, basis
        self.size = int(size)
        self.sub_omega, self.sub_eta = int(sub_omega), int(sub_eta)
        if min(self.sub_omega, self.sub_eta, int(node_batch)) < 1:
            raise ValueError("sub_omega, sub_eta and node_batch must be >= 1")
        self.node_batch = int(node_batch)
        self.cutoff_sigma = float(cutoff_sigma)
        self.cache = bool(cache)
        self.dtype = dtype or torch.float32
        self.device = torch.device(device or "cpu")
        self.K = len(basis)
        self.n_rings = rings.n_rings
        self.ring_weights = np.ones(self.n_rings) if ring_weights is None else np.asarray(ring_weights, dtype=float)
        mask = np.ones((size, size), bool) if mask is None else np.asarray(mask, dtype=bool)
        if mask.shape != (size, size):
            raise ValueError(f"mask must be {(size, size)}, got {mask.shape}")
        self.mask = mask
        self.voxels = np.flatnonzero(mask.ravel())
        self.n_vox = len(self.voxels)
        self._NB = scan.n_eta * self.n_rings
        self.data_shape = (scan.n_omega, scan.n_trans, scan.n_eta, self.n_rings)

        # Radon operator: the package's own sparse projector, restricted to the existing voxels
        A = projection_matrix(size, scan.omega_deg, scan.n_trans, dtype=torch.float64).coalesce()
        idx, val = A.indices().numpy(), A.values().numpy()
        A = sp.coo_matrix((val, (idx[0], idx[1])), shape=A.shape).tocsr()[:, self.voxels]
        self._A = self._to_torch(A)
        self._At = self._to_torch(A.T.tocsr())
        self._blocks = {}
        self.batches = [(k0, min(k0 + self.node_batch, self.K)) for k0 in range(0, self.K, self.node_batch)]

    # ------------------------------------------------------------- plumbing
    def _to_torch(self, m):
        import warnings
        torch = _torch()
        m = m.tocsr()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Sparse CSR tensor support is in beta")
            t = self._csr(torch, m)
        return t.to(dtype=self.dtype, device=self.device)

    @staticmethod
    def _csr(torch, m):
        return torch.sparse_csr_tensor(torch.from_numpy(m.indptr.astype(np.int64)), torch.from_numpy(m.indices.astype(np.int64)),
                                    torch.from_numpy(m.data), size=m.shape, dtype=torch.float64)

    def _block(self, b: int):
        if b in self._blocks:
            return self._blocks[b]
        import scipy.sparse as sp
        k0, k1 = self.batches[b]
        rows, cols, vals = self._triplets(k0, k1)
        G = sp.coo_matrix((vals, (rows, cols)), shape=(self.scan.n_omega * self._NB, self.scan.n_omega * (k1 - k0))).tocsr()
        blk = (self._to_torch(G), self._to_torch(G.T.tocsr()))
        if self.cache:
            self._blocks[b] = blk
        return blk

    @property
    def nnz(self) -> int:
        """Stored entries of the pole-figure blocks (builds them if needed; with ``cache=False`` they are discarded again)."""
        return int(sum(self._block(b)[0]._nnz() for b in range(len(self.batches))))

    def ring_totals(self) -> np.ndarray:
        """Sum over all data bins of a unit-weight node, per ring, averaged over nodes -- the quantity :func:`ring_normalise` assumes equal."""
        torch = _torch()
        tot = np.zeros(self.n_rings)
        for b, (k0, k1) in enumerate(self.batches):
            G = self._block(b)[0]
            ones = torch.ones((G.shape[1], 1), dtype=self.dtype, device=self.device)
            rowsum = (G @ ones).view(self.scan.n_omega, self.scan.n_eta, self.n_rings).sum(dim=(0, 1))
            tot += rowsum.cpu().numpy() * (1.0 / max(self.K, 1))
        return tot

    # -------------------------------------------------- pole-figure blocks
    def _triplets(self, k0: int, k1: int):
        """Sparse entries ``(row, col, value)`` of the omega-block-diagonal pole-figure matrix for nodes ``k0:k1``.

        Row ``o * n_eta * n_rings + e * n_rings + r``, column ``o * K_b + k``. For each (node, normal, Ewald crossing) the kernel is
        evaluated at the sub-samples of all bins inside the (linearised) footprint of its cutoff cap; entries of the same bin and node from
        different normals add when the matrix is assembled.
        """
        sc, rg = self.scan, self.rings
        kb = k1 - k0
        U = self.basis.orientations[k0:k1]
        sig = np.radians(self.basis.sigma_deg[k0:k1])
        kax, beam, d0, d9 = sc.vec("rot_axis"), sc.vec("beam"), sc.vec("eta_zero"), sc.vec("eta_ninety")
        pxk = np.cross(beam, kax)
        d_om, d_et = math.radians(sc.omega_step_deg), math.radians(sc.eta_step_deg)
        So, Se = self.sub_omega, self.sub_eta
        so_off = (np.arange(So) + 0.5) / So
        se_off = (np.arange(Se) + 0.5) / Se
        cap = self.cutoff_sigma * sig                                    # radians, (kb,)
        cos_cap = np.cos(np.minimum(cap, math.pi))
        kappa = 1.0 / sig ** 2
        norm = _vmf_norm(sig)
        out_r, out_c, out_v, out_m = [], [], [], []
        mmax = max(len(n) for n in rg.normals)
        for r in range(self.n_rings):
            nrm = rg.normals[r]
            m = len(nrm)
            th = float(rg.theta_rad[r])
            u = np.einsum("kij,mj->kmi", U, nrm)                         # (kb, m, 3)
            A = u @ beam
            B = u @ pxk
            C = -math.sin(th)
            R = np.hypot(A, B)
            ok = np.abs(C) <= R
            phi = np.arctan2(B, A)
            dlt = np.arccos(np.clip(C / np.where(R > 0, R, 1.0), -1.0, 1.0))
            kk = np.broadcast_to(np.arange(kb)[:, None], (kb, m))
            mm = np.broadcast_to(np.arange(m)[None, :], (kb, m))
            h_om_cap = max((int(round(360.0 / sc.omega_step_deg)) - 1) // 2, 1)      # < 360 deg: copies shifted by 360 never overlap
            h_et_cap = (sc.n_eta - 1) // 2
            cands = []                                                            # (exists, omega, eta, half-width in omega frames, in eta bins)
            for sgn in (1.0, -1.0):
                w0 = (phi + sgn * dlt) % _TWO_PI
                g = _rot(u, kax, w0)                                     # ghat_lab at the crossing
                eta0 = np.arctan2(g @ d9, g @ d0)
                # linearised footprint of the cap: y_omega = k x g, y_eta = cos(th) e'(eta)
                a = np.cross(kax, g)
                ep = -np.sin(eta0)[..., None] * d0 + np.cos(eta0)[..., None] * d9
                b = math.cos(th) * ep
                J = np.maximum(np.linalg.norm(np.cross(a, b), axis=-1), 1e-9)
                cp = cap[:, None]
                dw = 1.5 * cp * np.linalg.norm(b, axis=-1) / J
                de = 1.5 * cp * np.linalg.norm(a, axis=-1) / J
                cands.append((ok, w0, eta0, np.minimum(np.ceil(dw / d_om) + 1, h_om_cap).astype(int),
                              np.minimum(np.ceil(de / d_et) + 1, h_et_cap).astype(int)))
            # Normals inside the polar cap (angle alpha from the rotation axis < theta) have NO Ewald crossing, but the tail of their kernel
            # can still reach the measured (omega, eta) surface, which stays theta away from the axis. Their footprint is centred where the
            # pole is closest to it (omega = phi + pi); necessary bounds: the polar angle of a surface point within the cap lies in
            # [theta, alpha + rho] (eta half-width: cos(alpha + rho) / cos(theta) = cos(d_eta)), and its azimuth differs from the pole's by at
            # most max over that polar-angle range of arccos((cos rho - cos a cos b) / (sin a sin b)) (omega half-width).
            alpha = np.arcsin(np.clip(R, 0.0, 1.0))
            cp = np.minimum(cap[:, None], math.pi / 2)
            sp = (~ok) & (alpha + cp >= th)
            if sp.any():
                w0 = (phi + math.pi) % _TWO_PI
                g = _rot(u, kax, w0)
                eta0 = np.arctan2(g @ d9, g @ d0)
                d_eta = np.arccos(np.clip(np.cos(np.minimum(alpha + cp, math.pi / 2)) / math.cos(th), -1.0, 1.0))
                d_phi = np.zeros_like(alpha)
                for t in np.linspace(0.0, 1.0, 17):
                    beta = th + t * np.maximum(alpha + cp - th, 0.0)
                    den = np.sin(alpha) * np.sin(beta)
                    c = (np.cos(cp) - np.cos(alpha) * np.cos(beta)) / np.where(den > 1e-12, den, 1.0)
                    d_phi = np.maximum(d_phi, np.where(den > 1e-12, np.arccos(np.clip(c, -1.0, 1.0)), math.pi))
                cands.append((sp, w0, eta0, np.minimum(np.ceil(1.2 * d_phi / d_om) + 1, h_om_cap).astype(int),
                              np.minimum(np.ceil(1.2 * d_eta / d_et) + 1, h_et_cap).astype(int)))
            for ok_c, w0, eta0, h_om, h_et in cands:
                eta_f = (np.degrees(eta0) - sc.eta_start_deg) / sc.eta_step_deg
                j0 = np.floor(eta_f).astype(int)
                for shift in (-360.0, 0.0, 360.0):
                    f = (np.degrees(w0) + shift - sc.omega_start_deg) / sc.omega_step_deg
                    i0 = np.floor(f).astype(int)
                    sel = ok_c & (i0 + h_om >= 0) & (i0 - h_om <= sc.n_omega - 1)
                    if not sel.any():
                        continue
                    s_k, s_u, s_m = kk[sel], u[sel], mm[sel]
                    s_i0, s_j0 = i0[sel], j0[sel]
                    s_hom, s_het = h_om[sel], h_et[sel]
                    cls = s_hom * (sc.n_eta + 1) + s_het
                    for c in np.unique(cls):
                        pick = np.flatnonzero(cls == c)
                        hom, het = int(s_hom[pick[0]]), int(s_het[pick[0]])
                        nf, ne = 2 * hom + 1, 2 * het + 1
                        per = nf * So * ne * Se
                        step = max(1, int(self._CHUNK_ELEMENTS // per))
                        for st in range(0, len(pick), step):
                            sl = pick[st:st + step]
                            kq, mq = s_k[sl], s_m[sl]
                            fr = s_i0[sl][:, None] + np.arange(-hom, hom + 1)[None, :]          # (n, nf)
                            eb = s_j0[sl][:, None] + np.arange(-het, het + 1)[None, :]          # (n, ne)
                            om_s = np.radians(sc.omega_start_deg + (fr[:, :, None] + so_off) * sc.omega_step_deg)   # (n, nf, So)
                            et_s = np.radians(sc.eta_start_deg + (eb[:, :, None] + se_off) * sc.eta_step_deg)       # (n, ne, Se)
                            uo = _rot(s_u[sl][:, None, None, :], kax, om_s)                      # (n, nf, So, 3)
                            gl = _ghat(et_s, th, sc)                                             # (n, ne, Se, 3)
                            cosg = np.einsum("nfsa,neta->nfset", uo, gl)
                            val = norm[kq][:, None, None, None, None] * np.exp((cosg - 1.0) * kappa[kq][:, None, None, None, None])
                            val = np.where(cosg >= cos_cap[kq][:, None, None, None, None], val, 0.0)
                            val = val.mean(axis=(2, 4)) * (self.ring_weights[r] / m)             # (n, nf, ne)
                            valid = (fr >= 0) & (fr < sc.n_omega)
                            rows = fr[:, :, None] * self._NB + (eb[:, None, :] % sc.n_eta) * self.n_rings + r
                            cols = np.broadcast_to(fr[:, :, None] * kb + kq[:, None, None], val.shape)
                            keep = valid[:, :, None] & (val > 0)
                            out_r.append(rows[keep])
                            out_c.append(cols[keep])
                            out_v.append(val[keep])
                            out_m.append(np.broadcast_to(mq[:, None, None], val.shape)[keep])
        if not out_r:
            return np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0)
        rows, cols, vals, mem = (np.concatenate(a) for a in (out_r, out_c, out_v, out_m))
        # The two crossings of one normal are separate footprints, and they overlap where the crossing is nearly tangent: the same
        # (normal, sample point) would then be added twice. Its value is identical both times, so keep one (key = bin, node-frame, normal).
        key = (rows.astype(np.int64) * (sc.n_omega * kb) + cols) * mmax + mem
        _, first = np.unique(key, return_index=True)
        return rows[first], cols[first], vals[first]

    # ------------------------------------------------------- forward/adjoint
    def forward(self, C):
        """``C[n_vox, K] -> data[n_omega, n_trans, n_eta, n_rings]``."""
        torch = _torch()
        C = torch.as_tensor(C, dtype=self.dtype, device=self.device)
        if C.shape != (self.n_vox, self.K):
            raise ValueError(f"C must be {(self.n_vox, self.K)}, got {tuple(C.shape)}")
        no, My = self.scan.n_omega, self.scan.n_trans
        out = torch.zeros((no * self._NB, My), dtype=self.dtype, device=self.device)
        for b, (k0, k1) in enumerate(self.batches):
            kb = k1 - k0
            G = self._block(b)[0]
            S = (self._A @ C[:, k0:k1].contiguous()).view(no, My, kb).permute(0, 2, 1).reshape(no * kb, My)
            out += G @ S
            del S
        return out.view(no, self.scan.n_eta, self.n_rings, My).permute(0, 3, 1, 2).contiguous()

    def adjoint(self, Y):
        """``data[n_omega, n_trans, n_eta, n_rings] -> [n_vox, K]``; the exact transpose of :meth:`forward` (same blocks)."""
        torch = _torch()
        Y = torch.as_tensor(Y, dtype=self.dtype, device=self.device)
        if tuple(Y.shape) != self.data_shape:
            raise ValueError(f"Y must be {self.data_shape}, got {tuple(Y.shape)}")
        no, My = self.scan.n_omega, self.scan.n_trans
        Yf = Y.permute(0, 2, 3, 1).reshape(no * self._NB, My)
        out = torch.zeros((self.n_vox, self.K), dtype=self.dtype, device=self.device)
        for b, (k0, k1) in enumerate(self.batches):
            kb = k1 - k0
            Gt = self._block(b)[1]
            Z = (Gt @ Yf).view(no, kb, My).permute(0, 2, 1).reshape(no * My, kb)
            out[:, k0:k1] = self._At @ Z
            del Z
        return out
