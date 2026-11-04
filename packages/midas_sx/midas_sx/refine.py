"""Weighted least-squares refinement of a structure against merged F².

Model:  F²_calc(hkl) = k · |F(hkl; x, U_iso)|²   with ``midas_hkls.structure_factors`` (differentiable).

Parameters
----------
* the scale ``k`` (> 0);
* for every site, the coordinate shifts that the SITE SYMMETRY leaves free. They are derived, not typed in: a
  shift δ is allowed iff ``R δ = δ`` for every operation that fixes the site, i.e. the null space of the stacked
  ``(R − I)`` over the stabilizer. A site on a 4-fold axis gets one shift along it; a site on a mirror gets two; a
  site on an inversion centre gets none. Shifts are expressed along unit CARTESIAN directions, so a shift
  parameter is a displacement in Å and its σ needs no conversion;
* one isotropic U (Å²) per site, bounded to ``[u_min, u_max]``.
* optionally (``refine_twin=True``) the fraction m of an exactly overlapping merohedral twin component (SHELX TWIN/BASF):
  ``F²_calc(h) = k [(1 - m) |F(h)|² + m |F(T h)|²]`` with ``T h = twin_matrix @ h`` (default the 90° rotation about c, (k, -h, l)); m is bounded to
  [0, 0.5] (domain 1 is the majority, which removes the label-swap degeneracy) and its lower bound is the physical zero, not a rail.
  Valid when T normalises the point group used to merge (PREREGISTER A24). ``refine(..., partner=, flags=)`` gives each observation its own
  partner index and component weights (flags (1, 1) = blended pair, (1, 0) / (0, 1) = one resolved component; PREREGISTER A26).
* optionally (``n_domains=2``) one scale per crystal domain (``scale``, ``scale.d2``, ...): the observations carry a domain label and
  ``F²_calc = k_domain · |F|²`` (the twin fraction is k_2 / k_1 when the domains are resolved; PREREGISTER A23).
* optionally (``aniso_sites=("La1", ...)``) a site-symmetry-constrained ANISOTROPIC U for the named atoms instead of one U_iso: the allowed
  Cartesian tensor is the null space of ``U - R_c U R_c^T`` over the stabilizer of the site (derived, like the coordinate shifts), the
  parameters are its independent entries ``Uxx Uyy Uzz Uxy Uxz Uyz`` in Å² (greedy pivots in that order, so a 4mm site gets ``Uxx`` and ``Uzz``),
  converted to the CIF U_ij that ``midas_hkls`` takes. Positive-definiteness is NOT imposed (reported: ``absolute`` has the smallest
  eigenvalue). PREREGISTER A22.
* optionally (``refine_extinction=True``) one primary-extinction parameter, the effective block thickness t_b (µm): every calculated
  F² is multiplied by the exact perfect-crystal symmetric-Laue factor ``midas_hkls.laue_extinction_factor(t_b, Λ_hkl)`` with Λ_hkl
  from the CURRENT |F_hkl| (σ polarization, C = 1). It is carried as ``e = t_b²`` (µm²) because the factor is analytic in t_b², so the
  Jacobian column stays finite and non-zero at "no extinction"; its lower bound 1e-8 µm² is the physical zero and is NOT counted as a
  rail (an upper-bound hit is). PREREGISTER A21.
Not yet implemented (README lists the plan): secondary extinction, a robust pseudo-merohedral (split-spot) twin refinement (the A26 test failed its
convergence criterion), anisotropic U on sites that are not named in ``aniso_sites`` and positive-definiteness constraints. Occupancies are fixed.

What "converged" means here
---------------------------
A refinement is CONVERGED only if (a) the optimiser reports success, (b) no parameter sits on a bound (a parameter on
its bound is a rail, not an answer), and (c) the column-scaled Jacobian has full numerical rank. Anything else is
returned with ``converged = False`` and the reason; callers count those, they never drop them.

Uncertainties are the formal ones, ``(JᵀJ)⁻¹`` with the supplied σ as weights (no goodness-of-fit rescaling); the
rescaled variant is returned separately and labelled. Parameters in a numerical null space get ``σ = inf``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from midas_hkls import Crystal
from midas_hkls.tables import STBF

__all__ = ["site_free_directions", "RefineModel", "RefinementResult", "refine", "perturbed_crystal", "EIGHT_PI2"]

EIGHT_PI2 = 8.0 * np.pi ** 2
EXT_E_MIN = 1e-8                                           # µm²: the physical zero of the extinction parameter e = t_b²
BOUND_LR_CHI2 = 2.71                                       # one-sided 5 % likelihood-ratio threshold for ONE parameter on a bound (a boundary parameter: chi2 = 1/2 chi2_0 + 1/2 chi2_1)


def site_free_directions(space_group, fract: Sequence[float], *, tol: float = 1e-4) -> np.ndarray:
    """Orthonormal basis (k, 3) of the FRACTIONAL shifts a site's symmetry leaves free.

    Empty (shape (0, 3)) for a site fixed by its symmetry. The stabilizer is every operation (centring included)
    that maps the site onto itself modulo the lattice.
    """
    x = np.asarray(fract, float)
    rows = []
    for op in space_group.operations:
        R = np.array(op.R, float).reshape(3, 3)
        t = np.array(op.t, float) / float(STBF)
        d = R @ x + t - x
        if np.all(np.abs(d - np.rint(d)) < tol):
            rows.append(R - np.eye(3))
    if not rows:
        return np.eye(3)
    M = np.vstack(rows)
    _, s, vt = np.linalg.svd(M)
    rank = int(np.sum(s > 1e-9))
    return vt[rank:]                                   # (3 - rank, 3)


def origin_gauge_directions(space_group, *, tol: float = 1e-9) -> np.ndarray:
    """Orthonormal basis (g, 3) of the FRACTIONAL translations delta with R delta = delta for EVERY operation: shifting all atoms by delta leaves every
    |F_hkl| unchanged, so the refinement cannot determine it. g = 0 for a centrosymmetric or any group with a fixed point, 1 for a polar group with one polar axis
    (P2_1, P2_1ma, Pmc2_1 ...), 3 for P1."""
    M = [np.array(op.R, float).reshape(3, 3) - np.eye(3) for op in space_group.operations]
    _, s, vt = np.linalg.svd(np.vstack(M))
    return vt[int(np.sum(s > tol)):]


_U_ORDER = ("xx", "yy", "zz", "xy", "xz", "yz")
_U_IDX = ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))


def _sym_basis():
    out = []
    for i, j in _U_IDX:
        E = np.zeros((3, 3)); E[i, j] = 1.0; E[j, i] = 1.0
        out.append(E)
    return out                                              # E_xx has a single 1; off-diagonals are symmetric 1/1


def _vec6(M: np.ndarray) -> np.ndarray:
    return np.array([M[i, j] for i, j in _U_IDX])


def _stabilizer_fractional(space_group, fract: Sequence[float], tol: float = 1e-4) -> List[np.ndarray]:
    x = np.asarray(fract, float)
    out = []
    for op in space_group.operations:
        R = np.array(op.R, float).reshape(3, 3)
        t = np.array(op.t, float) / float(STBF)
        d = R @ x + t - x
        if np.all(np.abs(d - np.rint(d)) < tol):
            out.append(R)
    return out or [np.eye(3)]


def aniso_constraints(lattice, space_group, fract: Sequence[float], *, tol: float = 1e-4):
    """Site-symmetry-constrained anisotropic U of one atom.

    Returns ``dict(pivots, names, B_cart (6, k), to_cif (6, 6))``: the k free parameters are the Cartesian U entries ``pivots``
    (indices into ``xx yy zz xy xz yz``, Å²); ``U_cart6 = B_cart @ p``; ``U_cif6 = to_cif @ U_cart6`` in the order U11 U22 U33 U12 U13 U23
    of ``midas_hkls``. Cartesian frame = that of ``lattice.cartesian_vectors()`` (rows = a, b, c).
    """
    A = lattice.cartesian_vectors()                          # rows = direct vectors
    Ainv_T = np.linalg.inv(A).T
    E = _sym_basis()
    rows = []
    for Rf in _stabilizer_fractional(space_group, fract, tol):
        Rc = A.T @ Rf @ Ainv_T                               # x_c = A^T x_f
        M = np.stack([_vec6(Rc @ Ek @ Rc.T) for Ek in E], axis=1)   # (6, 6): column k = image of basis k
        rows.append(M - np.eye(6))
    _, s_, vt = np.linalg.svd(np.vstack(rows))
    rank = int(np.sum(s_ > 1e-9))
    N = vt[rank:].T                                          # (6, k) null space of the constraints
    k = N.shape[1]
    piv: List[int] = []
    for i in range(6):                                       # greedy pivots in xx yy zz xy xz yz order
        if len(piv) < k and np.linalg.matrix_rank(N[piv + [i], :], tol=1e-9) > len(piv):
            piv.append(i)
    B = N @ np.linalg.inv(N[piv, :])                          # U_cart6 = B @ (entries at pivots)
    astar = np.linalg.norm(lattice.reciprocal_cartesian_vectors(), axis=1)
    to_cif = np.zeros((6, 6))
    for kk, Ek in enumerate(E):                               # U_cif_ij = (A^-T U A^-1)_ij / (a*_i a*_j)
        Uf = Ainv_T @ Ek @ Ainv_T.T
        to_cif[:, kk] = _vec6(Uf / np.outer(astar, astar))
    return dict(pivots=piv, names=[f"U{_U_ORDER[i]}" for i in piv], B_cart=B, to_cif=to_cif)


def u_cart_of_cif(lattice, u6_cif: Sequence[float]) -> np.ndarray:
    """Cartesian 3x3 U (Å²) of a CIF U_ij six-vector (U11 U22 U33 U12 U13 U23)."""
    A = lattice.cartesian_vectors()
    astar = np.linalg.norm(lattice.reciprocal_cartesian_vectors(), axis=1)
    Uc = np.array([[u6_cif[0], u6_cif[3], u6_cif[4]], [u6_cif[3], u6_cif[1], u6_cif[5]], [u6_cif[4], u6_cif[5], u6_cif[2]]])
    return A.T @ (Uc * np.outer(astar, astar)) @ A


def perturbed_crystal(crystal: Crystal, rng: np.random.Generator, *, shift_frac: Optional[float] = 0.02,
                      shift_A: Optional[float] = None, dB: float = 0.3) -> Crystal:
    """Copy of ``crystal`` displaced along each site's ALLOWED shifts (random sign) and with B raised by ``dB`` Å².

    Two ways to say how far (PREREGISTER A4 and A8): ``shift_frac`` moves by a FRACTIONAL amount, which is a very
    different physical distance on a short and a long axis (0.02 is 0.07 Å on a 3.6 Å axis and 0.41 Å on a 20.5 Å axis);
    ``shift_A`` moves by a physical distance in Å and takes precedence when given. Coordinates stay on their special
    positions, B_iso goes up by ``dB``.
    """
    from dataclasses import replace
    A = crystal.lattice.cartesian_vectors()
    atoms = []
    for atom in crystal.atoms:
        f = np.array(atom.fract, float)
        for d in site_free_directions(crystal.space_group, atom.fract):
            sign = rng.choice([-1.0, 1.0])
            if shift_A is not None:
                f = f + sign * shift_A * d / np.linalg.norm(d @ A)
            else:
                f = f + sign * shift_frac * d / np.max(np.abs(d))
        atoms.append(replace(atom, fract=tuple(float(v) for v in f), B_iso=atom.B_iso + dB))
    return Crystal(crystal.lattice, crystal.space_group, atoms, crystal.name)


@dataclass
class RefineModel:
    """A crystal plus the parameterisation that is refined.

    ``crystal`` holds the STARTING coordinates and ADPs. Parameter vector layout:
    ``[scale | shifts (Å, site by site) | U_iso (Å², one per site) | e = t_b² (µm², only with refine_extinction)]``.
    """
    crystal: Crystal
    refine_coords: bool = True
    refine_adp: bool = True
    aniso_sites: Tuple[str, ...] = ()
    refine_twin: bool = False
    twin_start: float = 0.15
    twin_matrix: Tuple[Tuple[int, int, int], ...] = ((0, 1, 0), (-1, 0, 0), (0, 0, 1))
    n_domains: int = 1                                      # one scale per crystal domain (twin fractions); observations carry a domain label
    refine_extinction: bool = False
    wavelength_A: Optional[float] = None                    # required with refine_extinction
    ext_t_start_um: float = 5.0
    ext_t_max_um: float = 200.0
    u_bounds: Tuple[float, float] = (1e-4, 0.2)
    shift_bound_A: float = 1.0
    fixed_atoms: Tuple[str, ...] = ()                       # labels whose coordinates and U stay at the crystal's values (e.g. hydrogens); they still scatter
    fix_origin: bool = True                                 # drop one coordinate per polar axis (the REFINED atoms' parameter that most nearly translates them; see origin_gauge_directions): the origin then stays at the start value of that coordinate
    names: List[str] = field(default_factory=list)
    kinds: List[str] = field(default_factory=list)          # scale | coord | U
    atom_of: List[int] = field(default_factory=list)
    basis_frac: List[np.ndarray] = field(default_factory=list)   # per shift param: fractional direction per Å
    to_A: List[float] = field(default_factory=list)              # per param: Å per unit of the parameter (1 for shifts)

    def __post_init__(self):
        lat = self.crystal.lattice
        A = lat.cartesian_vectors()                              # rows = direct vectors, Å
        self.names, self.kinds, self.atom_of, self.basis_frac = ["scale"], ["scale"], [-1], [None]
        all_labels = [a.label or f"{a.element}{i}" for i, a in enumerate(self.crystal.atoms)]
        bad_fixed = [n for n in self.fixed_atoms if n not in all_labels]
        if bad_fixed:
            raise ValueError(f"fixed_atoms not in the crystal: {bad_fixed}")
        fixed = set(self.fixed_atoms)
        for ia, atom in enumerate(self.crystal.atoms):
            label = atom.label or f"{atom.element}{ia}"
            if self.refine_coords and label not in fixed:
                for j, d in enumerate(site_free_directions(self.crystal.space_group, atom.fract)):
                    cart = d @ A                                  # Å per unit fractional shift
                    d_per_A = d / np.linalg.norm(cart)            # fractional shift per Å along the same direction
                    if d_per_A[np.argmax(np.abs(d_per_A))] < 0:
                        d_per_A = -d_per_A
                    nm = "xyz"[int(np.argmax(np.abs(d))) ] if np.sum(np.abs(d) > 1e-6) == 1 else f"d{j}"
                    self.names.append(f"{label}.{nm}")
                    self.kinds.append("coord"); self.atom_of.append(ia); self.basis_frac.append(d_per_A)
        self.origin_fixed: List[str] = []                          # coordinate parameters removed because they only translate the structure along a polar axis
        gauge = origin_gauge_directions(self.crystal.space_group) if (self.refine_coords and self.fix_origin) else np.zeros((0, 3))
        if len(gauge):                                              # also with fixed atoms: they scatter, so the direction is not exactly null, but a few weak hydrogens anchor it so loosely that the refined atoms drift as a rigid body (thiourea tier C: all six heavy atoms 0.28 A along the polar axis)
            from scipy.linalg import qr
            idx = [i for i, k in enumerate(self.kinds) if k == "coord"]
            N = np.array([[float(self.basis_frac[i] @ dlt) / float(self.basis_frac[i] @ self.basis_frac[i]) for i in idx] for dlt in gauge])     # (g, n_coord): the parameter values that realise a pure translation
            piv = qr(N, pivoting=True)[2][:len(gauge)]
            drop = sorted(idx[j] for j in piv)
            self.origin_fixed = [self.names[i] for i in drop]
            for lst in (self.names, self.kinds, self.atom_of, self.basis_frac):
                for i in reversed(drop):
                    del lst[i]
        self.aniso: Dict[int, dict] = {}
        self.ent_of: List[int] = [-1] * len(self.names)           # per parameter: Cartesian-U entry index for aniso parameters (scale + coords: -1)
        if self.refine_adp:
            labels = [a.label or a.element + str(i) for i, a in enumerate(self.crystal.atoms)]
            unknown = [n for n in self.aniso_sites if n not in labels]
            if unknown:
                raise ValueError(f"aniso_sites not in the crystal: {unknown}")
            for ia, atom in enumerate(self.crystal.atoms):
                label = labels[ia]
                if label in fixed:
                    continue
                if label in self.aniso_sites:
                    c = aniso_constraints(lat, self.crystal.space_group, atom.fract)
                    self.aniso[ia] = c
                    for nm, i_ent in zip(c["names"], c["pivots"]):
                        self.names.append(f"{label}.{nm}")
                        self.kinds.append("Uaniso_diag" if i_ent < 3 else "Uaniso_off")
                        self.atom_of.append(ia); self.basis_frac.append(None); self.ent_of.append(i_ent)
                else:
                    self.names.append(f"{label}.U")
                    self.kinds.append("U"); self.atom_of.append(ia); self.basis_frac.append(None); self.ent_of.append(-1)
        if self.refine_extinction:
            if self.wavelength_A is None or not self.wavelength_A > 0:
                raise ValueError("refine_extinction needs wavelength_A")
            self.names.append("ext.t2"); self.kinds.append("ext"); self.atom_of.append(-1); self.basis_frac.append(None); self.ent_of.append(-1)
        if self.refine_twin:
            self.names.append("twin.m"); self.kinds.append("twin"); self.atom_of.append(-1); self.basis_frac.append(None); self.ent_of.append(-1)
        self.scale_idx = [0]
        for d in range(1, int(self.n_domains)):
            self.names.append(f"scale.d{d + 1}"); self.kinds.append("scale"); self.atom_of.append(-1); self.basis_frac.append(None)
            self.ent_of.append(-1); self.scale_idx.append(len(self.names) - 1)
        self._kinds = np.array(self.kinds)

    # ---------------------------------------------------------------- vectors
    @property
    def n_par(self) -> int:
        return len(self.names)

    def x0(self) -> np.ndarray:
        x = np.zeros(self.n_par)
        x[0] = 1.0
        for i, k in enumerate(self.kinds):
            if k == "U":
                x[i] = self.crystal.atoms[self.atom_of[i]].B_iso / EIGHT_PI2
            elif k in ("Uaniso_diag", "Uaniso_off"):
                u_iso = self.crystal.atoms[self.atom_of[i]].B_iso / EIGHT_PI2
                x[i] = u_iso if self.ent_of[i] < 3 else 0.0        # start isotropic: Uxx = Uyy = Uzz = U_iso, off-diagonals 0
            elif k == "ext":
                x[i] = max(self.ext_t_start_um ** 2, 10.0 * EXT_E_MIN)                 # t_b = 0 is the singular point of sqrt(e): never start there
            elif k == "twin":
                x[i] = self.twin_start
        return x

    def bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        lo = np.full(self.n_par, -self.shift_bound_A)
        hi = np.full(self.n_par, self.shift_bound_A)
        for i, k in enumerate(self.kinds):
            if k == "scale":
                lo[i], hi[i] = 1e-12, np.inf
            elif k == "U":
                lo[i], hi[i] = self.u_bounds
            elif k == "Uaniso_diag":
                lo[i], hi[i] = self.u_bounds
            elif k == "Uaniso_off":
                lo[i], hi[i] = -self.u_bounds[1], self.u_bounds[1]
            elif k == "ext":
                lo[i], hi[i] = EXT_E_MIN, self.ext_t_max_um ** 2
            elif k == "twin":
                lo[i], hi[i] = 0.0, 0.5
        return lo, hi

    def shift_directions_A(self) -> Dict[str, np.ndarray]:
        return {n: b for n, b in zip(self.names, self.basis_frac) if b is not None}

    def absolute(self, x: np.ndarray) -> pd.DataFrame:
        """Final fractional coordinates and U per site for parameter vector x."""
        rows = []
        for ia, atom in enumerate(self.crystal.atoms):
            f = np.array(atom.fract, float)
            for i, k in enumerate(self.kinds):
                if k == "coord" and self.atom_of[i] == ia:
                    f = f + x[i] * self.basis_frac[i]
            U = next((x[i] for i, k in enumerate(self.kinds) if k == "U" and self.atom_of[i] == ia),
                     atom.B_iso / EIGHT_PI2)
            row = dict(atom=atom.label or atom.element, x=f[0], y=f[1], z=f[2])
            if ia in self.aniso:
                c = self.aniso[ia]
                p_ = np.array([x[i] for i in range(self.n_par) if self.atom_of[i] == ia and self.kinds[i].startswith("Uaniso")])
                u6 = c["B_cart"] @ p_
                Um = np.array([[u6[0], u6[3], u6[4]], [u6[3], u6[1], u6[5]], [u6[4], u6[5], u6[2]]])
                U = float(np.trace(Um) / 3.0)                    # equivalent isotropic U (Cartesian trace / 3)
                row.update({f"U{n}": u6[i] for i, n in enumerate(_U_ORDER)}, U_min_eig=float(np.linalg.eigvalsh(Um)[0]))
            row["U_iso"] = U
            rows.append(row)
        return pd.DataFrame(rows)

    # ------------------------------------------------------------ torch model
    def _torch_state(self):
        import torch
        from dataclasses import replace
        crystal = self.crystal
        if self.aniso:                                            # give the anisotropic atoms a U_aniso so the tensor carries the (n_asu, 6) handle
            crystal = Crystal(crystal.lattice, crystal.space_group,
                              [replace(a, U_aniso=(1e-4,) * 3 + (0.0,) * 3) if i in self.aniso else a for i, a in enumerate(crystal.atoms)],
                              crystal.name)
        ct = crystal.to_torch(dtype=torch.float64)
        f0 = torch.tensor(np.array([a.fract for a in self.crystal.atoms]), dtype=torch.float64)
        return ct, f0

    def make_f2(self, hkl: np.ndarray, domain: Optional[np.ndarray] = None, partner: Optional[np.ndarray] = None,
                flags: Optional[np.ndarray] = None):
        """Return ``f2(x_tensor) -> k_domain * |F|^2`` (differentiable in x); ``domain`` is a 0-based label per observation."""
        import torch
        from midas_hkls import structure_factors
        ct, f0 = self._torch_state()
        hkl_t = torch.as_tensor(np.array(hkl, dtype=int))                  # a COPY: pandas 3 hands out read-only arrays and torch warns (once per process) about them
        kinds, atom_of = self.kinds, self.atom_of
        basis = [None if b is None else torch.tensor(b, dtype=torch.float64) for b in self.basis_frac]
        n_atoms = len(self.crystal.atoms)
        u_default = torch.tensor([a.B_iso / EIGHT_PI2 for a in self.crystal.atoms], dtype=torch.float64)
        dom_t = None
        if domain is not None:
            domain = np.asarray(domain, int)
            if domain.shape != (len(hkl),) or domain.min() < 0 or domain.max() >= self.n_domains:
                raise ValueError("domain must be one label in [0, n_domains) per observation")
            dom_t = torch.as_tensor(domain)
            scale_pos = torch.as_tensor(self.scale_idx)
        twin_i = kinds.index("twin") if "twin" in kinds else -1
        if twin_i >= 0:
            hkl_partner_np = np.array(hkl, dtype=int) @ np.asarray(self.twin_matrix, int).T if partner is None else np.array(partner, dtype=int)
            hkl_partner = torch.as_tensor(hkl_partner_np)
            if flags is None:
                fl1 = fl2 = None
            else:
                flags = np.asarray(flags, float)
                if flags.shape != (len(hkl), 2):
                    raise ValueError("flags must be (n, 2): the weight of the first and of the partner component of each observation")
                fl1, fl2 = torch.as_tensor(flags[:, 0]), torch.as_tensor(flags[:, 1])
        ext_i = kinds.index("ext") if "ext" in kinds else -1
        if ext_i >= 0:
            from midas_hkls import extinction_length_um, laue_extinction_factor
            lat = self.crystal.lattice
            V = float(lat.volume())
            d = np.array([lat.d_spacing(int(h), int(k), int(l)) for h, k, l in np.asarray(hkl, int)])
            theta_deg = torch.tensor(np.degrees(np.arcsin(np.clip(self.wavelength_A / (2.0 * d), 0.0, 1.0))), dtype=torch.float64)
            theta_deg_p = theta_deg
            if twin_i >= 0:                                             # the partner T h has its own d (equal to d(h) only when a = b): b/a 1.01 -> up to 5e-5, 1.03 -> 1.5e-4 relative
                dp = np.array([lat.d_spacing(int(h), int(k), int(l)) for h, k, l in hkl_partner_np])
                theta_deg_p = torch.tensor(np.degrees(np.arcsin(np.clip(self.wavelength_A / (2.0 * dp), 0.0, 1.0))), dtype=torch.float64)
            lam_A = float(self.wavelength_A)

        aniso_par = {ia: [i for i in range(self.n_par) if atom_of[i] == ia and kinds[i].startswith("Uaniso")] for ia in self.aniso}
        aniso_map = {ia: torch.tensor(c["to_cif"] @ c["B_cart"], dtype=torch.float64) for ia, c in self.aniso.items()}

        def f2(x):
            shift = torch.zeros_like(f0)
            U = u_default.clone()
            for i, k in enumerate(kinds):
                if k == "coord":
                    shift = shift.index_add(0, torch.tensor([atom_of[i]]), (x[i] * basis[i]).unsqueeze(0))
                elif k == "U":
                    U = U.index_put((torch.tensor([atom_of[i]]),), x[i].unsqueeze(0))
            ct.fract_asu = f0 + shift
            ct.B_iso_asu = EIGHT_PI2 * U
            if aniso_par:
                ct.U_aniso_asu = torch.stack([aniso_map[ia] @ x[aniso_par[ia]] if ia in aniso_par else torch.zeros(6, dtype=torch.float64)
                                              for ia in range(n_atoms)])
            F = structure_factors(ct, hkl_t)
            F2 = F.real ** 2 + F.imag ** 2

            def _extinct(F2c, theta=None):
                # primary extinction of ONE block: Lambda from that block's own |F|^2 (a twin domain is its own block; its intensity is attenuated
                # before the domains add, 2026-10-21: it used to multiply the twin SUM by one factor computed from the sum)
                Lam = extinction_length_um(torch.sqrt(F2c + 1e-12), unit_cell_volume_A3=V, wavelength_A=lam_A, theta_deg=theta_deg if theta is None else theta)
                return F2c * laue_extinction_factor(torch.sqrt(torch.clamp(x[ext_i], min=EXT_E_MIN)), Lam)      # the clamp keeps the Jacobian finite at e = 0

            if twin_i >= 0:
                Fp = structure_factors(ct, hkl_partner)
                F2p = Fp.real ** 2 + Fp.imag ** 2
                if ext_i >= 0:
                    F2, F2p = _extinct(F2), _extinct(F2p, theta_deg_p)
                if fl1 is None:
                    F2 = (1.0 - x[twin_i]) * F2 + x[twin_i] * F2p
                else:                                           # per-observation component weights (resolved / blended pairs, A26)
                    F2 = (1.0 - x[twin_i]) * fl1 * F2 + x[twin_i] * fl2 * F2p
            elif ext_i >= 0:
                F2 = _extinct(F2)
            if dom_t is None:
                return x[0] * F2
            return x[scale_pos][dom_t] * F2
        return f2


@dataclass
class RefinementResult:
    names: List[str]
    kinds: List[str]
    x: np.ndarray
    sigma: np.ndarray                    # formal, weights = supplied sigma; inf in a numerical null space
    sigma_rescaled: np.ndarray           # sigma * GooF  (labelled extra, NOT used for the preregistered pull)
    cov: np.ndarray
    converged: bool
    reason: str
    status: int
    n_obs: int
    n_par: int
    chi2: float
    goof: float
    wR2: float
    R1: float
    R_F2: float
    at_bound: List[str]
    full_rank: bool
    sv_ratio: float
    null_params: List[str]
    f2_calc: np.ndarray
    absolute: pd.DataFrame
    nfev: int
    rejected: np.ndarray = field(default_factory=lambda: np.zeros(0, int))     # rows removed by ``outlier_sigma``
    at_bound_ignored: List[str] = field(default_factory=list)                   # bound hits of ``gate_free`` parameters
    at_bound_harmless: List[str] = field(default_factory=list)                 # bound hits the data do not mind (likelihood-ratio test): UNDETERMINED parameters
    bound_delta_chi2: float = 0.0                                               # the largest single-bound chi2 drop (see ``bound_delta_chi2_by_param``), when something was on a bound
    bound_delta_chi2_by_param: Dict[str, float] = field(default_factory=dict)   # chi2(bounds kept) - chi2(THIS bound relaxed, the others held), per parameter on a bound

    @property
    def twin_fraction(self) -> float:
        return float(self.x[self.names.index("twin.m")]) if "twin.m" in self.names else float("nan")

    @property
    def sigma_twin_fraction(self) -> float:
        return float(self.sigma[self.names.index("twin.m")]) if "twin.m" in self.names else float("nan")

    @property
    def t_b_um(self) -> float:
        """Refined effective block thickness (µm); nan when the extinction parameter was not refined."""
        return float(np.sqrt(self.x[self.names.index("ext.t2")])) if "ext.t2" in self.names else float("nan")

    @property
    def sigma_t_b_um(self) -> float:
        """Delta-method sigma of t_b from the sigma of e = t_b² (large and unreliable as t_b -> 0)."""
        if "ext.t2" not in self.names:
            return float("nan")
        i = self.names.index("ext.t2")
        return float(self.sigma[i] / (2.0 * max(np.sqrt(self.x[i]), 1e-9)))

    def correlation(self) -> np.ndarray:
        """Parameter correlation matrix from the formal covariance (nan where a variance is zero or infinite)."""
        sd = np.sqrt(np.diag(self.cov))
        sd = np.where(np.isfinite(self.sigma) & (sd > 0), sd, np.nan)
        return self.cov / np.outer(sd, sd)

    @property
    def poor_fit(self) -> bool:
        """Converged by the optimiser gates but with GooF > 3: a false minimum that touches no bound looks like this."""
        return bool(self.goof > 3.0)

    def table(self) -> pd.DataFrame:
        return pd.DataFrame(dict(name=self.names, kind=self.kinds, value=self.x, sigma=self.sigma,
                                 sigma_rescaled=self.sigma_rescaled))


def refine(model: RefineModel, hkl: np.ndarray, f2_obs: np.ndarray, sigma: np.ndarray, *,
           x_start: Optional[np.ndarray] = None, scale_start: Optional[float] = None, domain: Optional[np.ndarray] = None,
           partner: Optional[np.ndarray] = None, flags: Optional[np.ndarray] = None, max_nfev: int = 200, rank_tol: float = 1e-8, rail_tol: float = 1e-6,
           outlier_sigma: Optional[float] = None, gate_free: Sequence[str] = (), bound_test: bool = True) -> RefinementResult:
    """Refine ``model`` against ``f2_obs`` with weights ``1/sigma**2``. See the module docstring for the gates.

    ``outlier_sigma`` (default None = every row counts): WORST-FIRST rejection on the INTERNALLY STUDENTIZED residual ``|r_i| / sqrt(1 - h_ii)`` (``h_ii`` the hat
    value from the weighted Jacobian). After an ordinary weighted fit the row with the largest value is removed if it exceeds ``outlier_sigma``, the fit is repeated,
    and so on (at most 10 rows, never below ``n_par + 5`` rows). One row at a time and studentized, because a row whose quoted sigma is far too small for what is
    wrong with it (a box that missed its spot has a tiny Poisson sigma) has a leverage near 1: the fit bends towards it, its RAW residual stays small and the other
    rows look bad (masking); the plain worst-residual rule and a robust loss both remove the wrong rows. The rejected rows are reported (``rejected``); GooF, the formal
    sigma and the dof use the kept rows only.
    ``bound_test`` (default True): a parameter on a bound is a rail only if relaxing THAT bound (the others held) improves chi2 by more than 2.71 (one-sided 5 %
    likelihood-ratio test, per parameter); otherwise it is UNDETERMINED, listed in ``at_bound_harmless`` and does not fail the convergence verdict
    (``bound_delta_chi2_by_param`` holds the improvements). False restores the old rule (every bound hit fatal).
    ``gate_free``: parameter names whose bound hit is reported (``at_bound_ignored``) but does not fail the convergence verdict (parameters the data cannot determine,
    e.g. an oxygen U); the default keeps every bound hit fatal."""
    import torch
    from scipy.optimize import least_squares

    hkl = np.asarray(hkl, int)
    f2_obs = np.asarray(f2_obs, float)
    sigma = np.asarray(sigma, float)
    if not (len(hkl) == len(f2_obs) == len(sigma)):
        raise ValueError("hkl, f2_obs, sigma must have the same length")
    if np.any(sigma <= 0) or not np.all(np.isfinite(sigma)):
        raise ValueError("sigma must be positive and finite")
    f2 = model.make_f2(hkl, domain, partner, flags)
    if model.n_domains > 1 and domain is None:
        raise ValueError("a model with n_domains > 1 needs a domain label per observation")
    w = 1.0 / sigma
    lo, hi = model.bounds()
    x0 = model.x0() if x_start is None else np.asarray(x_start, float).copy()
    if scale_start is None or model.n_domains > 1:
        xs = x0.copy()
        for i_s in model.scale_idx:
            xs[i_s] = 1.0
        with torch.no_grad():
            calc = f2(torch.tensor(xs)).numpy()
        ww = w ** 2
        if model.n_domains > 1:
            dom = np.asarray(domain, int)
            for d, i_s in enumerate(model.scale_idx):
                m_ = dom == d
                x0[i_s] = max(float((ww[m_] * calc[m_] * f2_obs[m_]).sum() / (ww[m_] * calc[m_] ** 2).sum()), 1e-9)
        else:
            x0[0] = max(float((ww * calc * f2_obs).sum() / (ww * calc * calc).sum()), 1e-9)
    if scale_start is not None and model.n_domains == 1:
        x0[0] = scale_start
    x0 = np.clip(x0, lo + 1e-12, np.where(np.isfinite(hi), hi, x0) )

    keep = np.ones(len(f2_obs), bool)

    def resid(x):
        with torch.no_grad():
            return (f2_obs - f2(torch.tensor(x)).numpy()) * w * keep

    def jac(x):
        xt = torch.tensor(x, requires_grad=False)
        J = torch.autograd.functional.jacobian(lambda z: f2(z), xt)          # d f2_calc / d x   (M, P)
        return -(J.numpy() * w[:, None]) * keep[:, None]

    lsq = dict(jac=jac, bounds=(lo, hi), method="trf", x_scale="jac", ftol=1e-14, xtol=1e-14, gtol=1e-14, max_nfev=max_nfev)
    sol = least_squares(resid, x0, **lsq)
    if outlier_sigma is not None:
        for _ in range(10):                                                     # worst-first, one row per round
            with torch.no_grad():
                r_raw = (f2_obs - f2(torch.tensor(sol.x)).numpy()) * w
            Jk = jac(sol.x)
            cs = np.linalg.norm(Jk, axis=0); cs[cs == 0] = 1.0
            Uq, sq, _ = np.linalg.svd(Jk / cs, full_matrices=False)
            hat = np.sum(Uq[:, sq > 1e-9 * sq[0]] ** 2, axis=1)                  # diag of the hat matrix on the kept rows
            r_all = np.where(keep, np.abs(r_raw) / np.sqrt(np.maximum(1.0 - hat, 1e-4)), 0.0)
            j_worst = int(np.argmax(r_all))
            if r_all[j_worst] <= float(outlier_sigma) or keep.sum() - 1 < model.n_par + 5:
                break
            keep[j_worst] = False
            sol = least_squares(resid, sol.x, **lsq)
    x = sol.x
    J = jac(x)
    r = resid(x)
    n, p = int(keep.sum()), model.n_par
    # rank / null space on the column-scaled Jacobian
    cn = np.linalg.norm(J, axis=0)
    cn[cn == 0] = 1.0
    Js = J / cn
    # FULL svd: with fewer reflections than parameters the null space lives in the rows of vt beyond len(s)
    u, s, vt = np.linalg.svd(Js, full_matrices=True)
    r_rank = int(np.sum(s > rank_tol * s[0])) if len(s) else 0
    sv_ratio = float(s[-1] / s[0]) if (len(s) == p and s[0] > 0) else 0.0
    full_rank = bool(n >= p and r_rank == p)
    inv = (vt[:r_rank].T / (s[:r_rank] ** 2)) @ vt[:r_rank]          # pinv(Js^T Js) on the identified subspace
    cov = inv / np.outer(cn, cn)
    var = np.diag(cov).copy()
    null_dir = vt[r_rank:]                                            # (p - rank, p) directions with no information
    null_mask = np.zeros(p, bool)
    if null_dir.size:
        null_mask = np.any(np.abs(null_dir) > 0.3, axis=0)
    var[null_mask] = np.inf
    sig_formal = np.sqrt(np.maximum(var, 0.0))
    chi2 = float((r ** 2).sum())
    dof = max(n - p, 1)
    goof = float(np.sqrt(chi2 / dof))

    at_bound = []
    for i in range(p):
        span = (hi[i] - lo[i]) if np.isfinite(hi[i]) else max(abs(x[i]), 1.0)
        lower = (x[i] - lo[i] <= rail_tol * span) and model.kinds[i] not in ("ext", "twin")      # e = 0 / m = 0 are "no extinction" / "no twin", not rails
        if lower or (np.isfinite(hi[i]) and hi[i] - x[i] <= rail_tol * span):
            at_bound.append(model.names[i])
    # A parameter on a bound is a RAIL (the model wants a value outside the physical range) or just UNDETERMINED (the data do not care, the fit wandered to the
    # edge of a flat direction; e.g. an oxygen U, or Uxx - Uyy of a cation in a twin whose law swaps x and y). Decide by a likelihood-ratio test: refit with
    # those bounds relaxed; if chi2 improves by less than BOUND_LR_CHI2 per parameter the bound distorts nothing and the hit is reported, not fatal.
    harmless, d_chi2, d_by = [], 0.0, {}
    if at_bound and bound_test:
        chi2_kept = float((resid(x) ** 2).sum())
        for i in [i for i in range(p) if model.names[i] in set(at_bound)]:
            # ONE bound relaxed at a time, the others held: a joint test with threshold 2.71 x k would let a real rail hide behind an innocent parameter that is
            # also on a bound (demonstrated by the physics lens of /verify claim 97f96459b1e5)
            span = (hi[i] - lo[i]) if np.isfinite(hi[i]) else max(abs(x[i]), 1.0)
            lo2, hi2 = lo.copy(), hi.copy()
            if x[i] - lo[i] <= rail_tol * span:
                lo2[i] = lo[i] - span
            else:
                hi2[i] = hi[i] + span
            sol_free = least_squares(resid, x, **dict(lsq, bounds=(lo2, hi2)))
            d_by[model.names[i]] = max(chi2_kept - float((resid(sol_free.x) ** 2).sum()), 0.0)
        d_chi2 = max(d_by.values())
        harmless = [nm for nm in at_bound if d_by[nm] <= BOUND_LR_CHI2]
        at_bound = [nm for nm in at_bound if d_by[nm] > BOUND_LR_CHI2]
    ignored = [nm for nm in at_bound if nm in set(gate_free)]
    at_bound = [nm for nm in at_bound if nm not in set(gate_free)]
    reasons = []
    if sol.status <= 0:
        reasons.append(f"optimiser status {sol.status}: {sol.message}")
    if at_bound:
        reasons.append("on bound: " + ",".join(at_bound))
    if not full_rank:
        reasons.append(f"rank deficient (sv ratio {sv_ratio:.2e}, n_obs {n} vs n_par {p})")
    converged = not reasons

    f2c = f2(torch.tensor(x)).detach().numpy()
    Fo, Fc = np.sqrt(np.maximum(f2_obs, 0.0)), np.sqrt(np.maximum(f2c, 0.0))
    R1 = float(np.abs(Fo - Fc).sum() / max(Fo.sum(), 1e-300))
    wR2 = float(np.sqrt((w ** 2 * (f2_obs - f2c) ** 2).sum() / (w ** 2 * f2_obs ** 2).sum()))
    R_F2 = float(np.abs(f2_obs - f2c).sum() / max(np.abs(f2_obs).sum(), 1e-300))
    return RefinementResult(
        names=list(model.names), kinds=list(model.kinds), x=x, sigma=sig_formal,
        sigma_rescaled=sig_formal * goof, cov=cov, converged=converged,
        reason="; ".join(reasons) if reasons else "ok", status=int(sol.status), n_obs=n, n_par=p,
        chi2=chi2, goof=goof, wR2=wR2, R1=R1, R_F2=R_F2, at_bound=at_bound, full_rank=bool(full_rank),
        sv_ratio=sv_ratio, null_params=[model.names[i] for i in np.flatnonzero(null_mask)],
        f2_calc=f2c, absolute=model.absolute(x), nfev=int(sol.nfev), rejected=np.flatnonzero(~keep), at_bound_ignored=ignored,
        at_bound_harmless=harmless, bound_delta_chi2=d_chi2, bound_delta_chi2_by_param=d_by)
