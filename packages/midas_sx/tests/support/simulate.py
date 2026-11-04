"""Independent forward simulator: mono-beam single-crystal rotation series in a DAC.

DEV ONLY. Never imported by a package. Its job is to produce frames whose truth we
know, through code that shares NOTHING with the corrections under test
(PREREGISTER.md, "Independence of the simulator"):

* |F|^2 truth: gemmi, not midas_hkls.
* Lorentz: no closed form. Monte-Carlo mosaic blocks cross a finite-thickness
  Ewald shell; the time spent in the shell, binned into frames, IS the Lorentz
  factor (1/|d f / d omega|), and it is checked against the closed form below.
* Polarization: Thomson amplitude |eps . s|^2, incoherent mixture of the two
  transverse states.
* Absorption: explicit ray / slab intersection lengths through two diamond
  anvils in the rotating cell frame, a hard opening cone, and a random-depth
  Monte-Carlo through the crystal plate.
* Rotations: scipy.spatial.transform, not midas_hkls.corrections.rotate_about_axis.

Lab frame (MIDAS): x along the beam, z up, y completes it. Detector pixel
convention = the MIDAS one used by ``midas_defect.geometry`` (checked against it in the
reduction tests): ``Y = -(col - bcy)*px``, ``Z = (row - bcz)*px``, i.e.
``col = cx - y/pixel``, ``row = cz + z/pixel``; pixel centres at integer indices. Sample rotation by +omega is right-handed
about +z; the DAC rotates with the sample.

Units: Angstrom, degrees, mm, counts. q is in 1/Angstrom WITHOUT 2 pi, so the
Ewald radius is k = 1/lambda.

Not simulated (so a pass here says nothing about them): extinction, multiple
scattering, twinning, other grains, diamond Bragg spikes, gasket powder rings,
detector nonlinearity/parallax, crystal off the rotation axis.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import gemmi
import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.spatial.transform import Rotation

AXIS = np.array([0.0, 0.0, 1.0])
S0 = np.array([1.0, 0.0, 0.0])


# --------------------------------------------------------------------- structure

@dataclass
class Site:
    label: str
    element: str
    fract: Tuple[float, float, float]
    u_iso: float                 # Angstrom^2
    occ: float = 1.0
    u_aniso: Optional[Tuple[float, float, float, float, float, float]] = None   # U11 U22 U33 U12 U13 U23, CIF convention, A^2 (PREREGISTER A22)


@dataclass
class Structure:
    cell: Tuple[float, float, float, float, float, float]
    spacegroup: str
    sites: List[Site]
    name: str = ""

    def to_gemmi(self) -> gemmi.SmallStructure:
        """Both gemmi traps are handled here (PREREGISTER A2): the space group must be attached
        with setup_cell_images() (else the cell holds ONE atom and forbidden reflections appear),
        and occupancies must be converted to the crystallographic convention (else |F|^2 is
        multiplicity^2 too large on special positions)."""
        ss = gemmi.SmallStructure()
        ss.cell = gemmi.UnitCell(*self.cell)
        ss.spacegroup = gemmi.SpaceGroup(self.spacegroup)
        for s in self.sites:
            g = gemmi.SmallStructure.Site()
            g.label, g.element = s.label, gemmi.Element(s.element)
            g.fract = gemmi.Fractional(*s.fract)
            g.occ, g.u_iso = s.occ, s.u_iso
            if s.u_aniso is not None:
                g.aniso = gemmi.SMat33d(*s.u_aniso)
            ss.add_site(g)
        ss.setup_cell_images()
        ss.change_occupancies_to_crystallographic()
        return ss


def structure_cu(u_iso: float = 0.005) -> Structure:
    """S1: fcc Cu-like, one free ADP."""
    return Structure((3.6, 3.6, 3.6, 90, 90, 90), "F m -3 m",
                     [Site("Cu", "Cu", (0, 0, 0), u_iso)], "S1_Cu")


def structure_s2_bilayer() -> Structure:
    """S2: bilayer Ruddlesden-Popper, I4/mmm, a = 3.83, c = 20.5 (synthetic truth).

    A bilayer-oxide-LIKE test structure (La, Ni, O sites): coordinates are plausible (checked for interatomic distances) but are NOT a
    published refinement. 6 La + 4 Ni + 14 O per cell; O is a few percent of |F|^2.
    """
    return Structure((3.83, 3.83, 20.5, 90, 90, 90), "I 4/m m m", [
        Site("La1", "La", (0, 0, 0.5), 0.006),
        Site("La2", "La", (0, 0, 0.3190), 0.006),
        Site("Ni", "Ni", (0, 0, 0.0950), 0.005),
        Site("O1", "O", (0, 0.5, 0.0950), 0.010),     # equatorial, 8g
        Site("O2", "O", (0, 0, 0.2000), 0.010),       # outer apical, 4e
        Site("O3", "O", (0, 0, 0.0), 0.010),          # inner apical, 2a
    ], "S2_bilayer_like")


def structure_s2_bilayer_aniso() -> Structure:
    """S2A (PREREGISTER A22): S2 with anisotropic cations, U33/U11 = 3.0 (La1), 2.2 (La2), 2.3 (Ni); O unchanged."""
    st = structure_s2_bilayer()
    aniso = {"La1": (0.0040, 0.0040, 0.0120, 0.0, 0.0, 0.0), "La2": (0.0045, 0.0045, 0.0100, 0.0, 0.0, 0.0),
             "Ni": (0.0035, 0.0035, 0.0080, 0.0, 0.0, 0.0)}
    sites = [Site(x.label, x.element, x.fract, x.u_iso, x.occ, aniso.get(x.label)) for x in st.sites]
    return Structure(st.cell, st.spacegroup, sites, "S2A_bilayer_like_aniso")


def structure_s2o() -> Structure:
    """S2O (PREREGISTER A24): the S2 cell in I m m m with the equatorial O split in z (O1a 0.105, O1b 0.085), so that F^2(h) != F^2(T h) for the
    90 deg twin law T h = (k, -h, l) on the metrically tetragonal lattice."""
    return Structure((3.83, 3.83, 20.5, 90, 90, 90), "I m m m", [
        Site("La1", "La", (0, 0, 0.5), 0.006),
        Site("La2", "La", (0, 0, 0.3190), 0.006),
        Site("Ni", "Ni", (0, 0, 0.0950), 0.005),
        Site("O1a", "O", (0.5, 0, 0.1050), 0.010),
        Site("O1b", "O", (0, 0.5, 0.0850), 0.010),
        Site("O2", "O", (0, 0, 0.2000), 0.010),
        Site("O3", "O", (0, 0, 0.0), 0.010),
    ], "S2O_immm")


def twin_f2(st: Structure, m: float):
    """``f2_override`` for a 90 deg merohedral twin of fraction m: F2_tw(h) = (1 - m) F2(h) + m F2(k, -h, l)."""
    def f(hkl, F2):
        h = np.asarray(hkl)
        return (1.0 - m) * F2 + m * f2_truth(st, np.stack([h[:, 1], -h[:, 0], h[:, 2]], axis=1))
    return f


def structure_s2oa() -> Structure:
    """S2OA (PREREGISTER A33): S2O (Immm, split equatorial O) with anisotropic cations as in S2A (U11 = U22 = 0.0040 / 0.0045 / 0.0035, U33 = 0.0120 / 0.0100 / 0.0080
    for La1 / La2 / Ni; the site symmetry mmm / mm2 would allow U11 != U22, the truth uses the tetragonal special case); O unchanged."""
    st = structure_s2o()
    aniso = {"La1": (0.0040, 0.0040, 0.0120, 0.0, 0.0, 0.0), "La2": (0.0045, 0.0045, 0.0100, 0.0, 0.0, 0.0), "Ni": (0.0035, 0.0035, 0.0080, 0.0, 0.0, 0.0)}
    return Structure(st.cell, st.spacegroup, [Site(x.label, x.element, x.fract, x.u_iso, x.occ, aniso.get(x.label)) for x in st.sites], "S2OA_immm_aniso")


def joint_f2(st: Structure, m: float, factor):
    """``f2_override`` for the JOINT case (PREREGISTER A33): a 90 deg merohedral twin of fraction m whose two domains are each a primary-extinction block:
    F2_obs(h) = (1 - m) F2(h) y(h) + m F2(T h) y(T h), y from that domain's OWN |F|^2 (``factor`` = extinction_truth.make_factor(...)). Anisotropic U enters through
    ``f2_truth`` (gemmi Site.aniso)."""
    def f(hkl, F2_unused):
        h = np.asarray(hkl)
        part = np.stack([h[:, 1], -h[:, 0], h[:, 2]], axis=1)
        a, b = f2_truth(st, h), f2_truth(st, part)
        return (1.0 - m) * a * np.asarray(factor(h, a), float) + m * b * np.asarray(factor(part, b), float)
    return f


def structure_s2p(b_over_a: float = 1.01) -> Structure:
    """S2P (PREREGISTER A26): S2O with b = a * b_over_a (orthorhombic, a != b), so that the two components of a 90 deg twin about c are split on the
    detector by delta-q = (1/b - 1/a) (h, -k, 0) (zero only for 00l)."""
    st = structure_s2o()
    a = st.cell[0]
    return Structure((a, a * b_over_a, st.cell[2], 90, 90, 90), st.spacegroup, st.sites, f"S2P_immm_b{b_over_a:g}")


def min_interatomic_distance(st: Structure) -> float:
    """Smallest distance between unit-cell atoms [A], periodic images included."""
    ss = st.to_gemmi()
    cell = ss.cell
    pos = [cell.orthogonalize(gemmi.Fractional(*[x for x in (s.fract.x, s.fract.y, s.fract.z)]))
           for s in ss.get_all_unit_cell_sites()]
    fr = np.array([[s.fract.x, s.fract.y, s.fract.z] for s in ss.get_all_unit_cell_sites()])
    orth = np.array(cell.orth.mat.tolist())
    best = 1e9
    shifts = np.array([[i, j, k] for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)])
    for a in range(len(fr)):
        for b in range(len(fr)):
            d = (fr[b] - fr[a])[None, :] + shifts
            r = np.linalg.norm(d @ orth.T, axis=1)
            if a == b:
                r = r[r > 1e-6]
            best = min(best, r.min())
    return float(best)


# -------------------------------------------------------------------- experiment

@dataclass
class Experiment:
    wavelength_A: float = 0.4133            # 30 keV
    det_n: int = 512
    pixel_mm: float = 0.3
    dist_mm: float = 100.0
    centre_rc: Tuple[float, float] = (255.5, 255.5)      # (row, col) of the beam on the detector
    omega_start: float = -40.0
    omega_step: float = 0.25
    n_frames: int = 320
    mosaic_fwhm_deg: float = 0.10
    bandwidth_rel: float = 1e-3             # Ewald-shell width (effective: bandwidth + divergence). The integrated result is independent of
                                            # it, but the rocking width is not: at 3e-3 a low-angle spot spans ~0.9 deg of omega and its tail
                                            # is clipped by the DAC opening cone up to ~1 deg before the nominal edge (see check_outliers.py)
    pol_fraction: float = 0.99              # linear, horizontal
    dac_tilt_deg: Tuple[float, float] = (0.0, 0.0)      # (about y, about z) tilt of the DAC axis off the beam at omega = 0
    t_plus_mm: float = 2.5                  # anvil thickness on the +c side of the culet
    t_minus_mm: float = 2.5
    mu_diamond_per_mm: float = 0.0900       # 30 keV, NIST via midas_hkls (0.9005 cm^-1); see make_experiment
    half_opening_deg: float = 35.0
    plate_h_um: float = 8.0
    mu_sample_per_cm: float = 0.0           # crystal plate self-absorption (0 = off)
    sigma_psf_px: float = 0.7
    bg0: float = 30.0
    bg1: float = 60.0
    d_min_A: float = 0.0                    # 0 = whatever the detector corner allows

    @property
    def k(self) -> float:
        return 1.0 / self.wavelength_A

    @property
    def omega_end(self) -> float:
        return self.omega_start + self.omega_step * self.n_frames

    def dac_axis0(self) -> np.ndarray:
        ty, tz = self.dac_tilt_deg
        return Rotation.from_euler("yz", [ty, tz], degrees=True).apply(S0)

    def tth_azi_maps(self) -> Tuple[np.ndarray, np.ndarray]:
        r, c = np.indices((self.det_n, self.det_n), dtype=float)
        y = (self.centre_rc[1] - c) * self.pixel_mm
        z = (r - self.centre_rc[0]) * self.pixel_mm
        tth = np.degrees(np.arctan(np.hypot(y, z) / self.dist_mm))
        azi = np.degrees(np.arctan2(-y, z))               # MIDAS eta
        return tth, azi

    def background_map(self) -> np.ndarray:
        tth, _ = self.tth_azi_maps()
        return self.bg0 + self.bg1 * np.exp(-tth / 8.0)

    def tth_max_deg(self) -> float:
        edge = (self.det_n / 2.0 - 3.0) * self.pixel_mm
        return math.degrees(math.atan(edge / self.dist_mm))


def make_experiment(**kw) -> Experiment:
    """Experiment with the diamond mu taken from the NIST table at the requested wavelength."""
    exp = Experiment(**kw)
    if "mu_diamond_per_mm" not in kw:
        from midas_hkls.absorption import linear_absorption_coefficient
        exp.mu_diamond_per_mm = float(linear_absorption_coefficient(
            "C", exp.wavelength_A, density_g_cm3=3.515)) / 10.0
    return exp


# ------------------------------------------------------------------ ray tracing

def _slab_length(d: np.ndarray, c: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Length of the ray r = tau * d (tau >= 0, |d| = 1) inside the slab lo <= c . r <= hi.

    d, c: (n, 3). Explicit plane intersections; no cos(angle) shortcut.
    """
    a = np.sum(c * d, axis=1)
    out = np.zeros(len(a))
    ok = np.abs(a) > 1e-12
    t1 = np.full(len(a), np.nan)
    t2 = np.full(len(a), np.nan)
    t1[ok] = lo / a[ok]
    t2[ok] = hi / a[ok]
    tl = np.fmin(t1, t2)
    th = np.fmax(t1, t2)
    start = np.fmax(tl, 0.0)
    out[ok] = np.fmax(th[ok] - start[ok], 0.0)
    return out


def _thomson_pol(s: np.ndarray, f: float) -> np.ndarray:
    eps_h = np.array([0.0, 1.0, 0.0])
    eps_v = np.array([0.0, 0.0, 1.0])
    return 0.5 * (1 + f) * (1 - (s @ eps_h) ** 2) + 0.5 * (1 - f) * (1 - (s @ eps_v) ** 2)


def _rot(rotvec: np.ndarray) -> Rotation:
    return Rotation.from_rotvec(rotvec)


# ------------------------------------------------------------------ reflections

def _reciprocal_basis(st: Structure) -> np.ndarray:
    """B such that q_cartesian = B @ hkl, 1/A, no 2 pi."""
    A = np.array(gemmi.UnitCell(*st.cell).orth.mat.tolist())
    return np.linalg.inv(A).T


def enumerate_hkl(st: Structure, d_min: float) -> np.ndarray:
    a = [gemmi.UnitCell(*st.cell).a, gemmi.UnitCell(*st.cell).b, gemmi.UnitCell(*st.cell).c]
    n = [int(math.ceil(x / d_min)) + 1 for x in a]
    h, k, l = np.meshgrid(*[np.arange(-m, m + 1) for m in n], indexing="ij")
    hkl = np.stack([h.ravel(), k.ravel(), l.ravel()], axis=1)
    hkl = hkl[np.any(hkl != 0, axis=1)]
    B = _reciprocal_basis(st)
    inv_d = np.linalg.norm(hkl @ B.T, axis=1)
    return hkl[inv_d <= 1.0 / d_min]


def f2_truth(st: Structure, hkl: np.ndarray) -> np.ndarray:
    ss = st.to_gemmi()
    calc = gemmi.StructureFactorCalculatorX(ss.cell)
    return np.array([abs(calc.calculate_sf_from_small_structure(ss, [int(x) for x in h])) ** 2
                     for h in hkl])


def random_orientation(seed: int) -> np.ndarray:
    return Rotation.random(random_state=np.random.RandomState(seed)).as_matrix()


# ----------------------------------------------------------------------- engine

@dataclass
class SimResult:
    exp: Experiment
    structure: Structure
    U: np.ndarray
    signal: np.ndarray                      # (n_frames, n, n) expected counts, background excluded
    truth: pd.DataFrame
    scale: float

    def frames(self, seed: int = 0) -> np.ndarray:
        """Poisson frames = signal + background. Fresh noise per seed; the signal is fixed."""
        rng = np.random.default_rng(seed)
        bg = self.exp.background_map()
        out = np.empty(self.signal.shape, dtype=np.float32)
        for i in range(self.signal.shape[0]):
            out[i] = rng.poisson(self.signal[i] + bg).astype(np.float32)
        return out


def _crossings(q0: np.ndarray, exp: Experiment, step: float = 0.02):
    """Nominal omega roots of f(omega) = |k s0 + R(omega) q0| - k inside the scan, with |slope|."""
    om = np.arange(exp.omega_start, exp.omega_end + step, step)
    qs = Rotation.from_rotvec(np.radians(om)[:, None] * AXIS).apply(q0)
    f = np.linalg.norm(exp.k * S0 + qs, axis=1) - exp.k
    idx = np.flatnonzero(f[:-1] * f[1:] < 0)
    roots = []
    for i in idx:
        t = f[i] / (f[i] - f[i + 1])
        roots.append((om[i] + t * step, abs(f[i + 1] - f[i]) / step))     # (omega*, |df/domega| per deg)
    return roots


def simulate(st: Structure, exp: Experiment, *, orientation_seed: int = 1, mc_seed: int = 2,
             n_int_max: float = 1e5, target_hits: int = 200_000, batch: int = 200_000,
             max_pairs: int = 6_000_000, keep_every_hkl: int = 1, verbose: bool = False,
             f2_factor=None, scale_override: Optional[float] = None, f2_override=None, U_override: Optional[np.ndarray] = None) -> SimResult:
    rng = np.random.default_rng(mc_seed)
    U = random_orientation(orientation_seed) if U_override is None else np.asarray(U_override, float)
    B = _reciprocal_basis(st)
    d_min = max(exp.d_min_A,
                exp.wavelength_A / (2.0 * math.sin(math.radians(exp.tth_max_deg() / 2.0))))
    hkl_all = enumerate_hkl(st, d_min)
    F2 = f2_truth(st, hkl_all)
    if f2_factor is not None:                                 # e.g. primary extinction (extinction_truth.py): F2 -> F2 * y(hkl)
        F2 = F2 * np.asarray(f2_factor(hkl_all, F2), float)
    if f2_override is not None:                               # e.g. a merohedral twin (PREREGISTER A24): F2 -> (1 - m) F2(h) + m F2(T h)
        F2 = np.asarray(f2_override(hkl_all, F2), float)
    keep = F2 > 1e-6 * F2.max()
    hkl_all, F2 = hkl_all[keep][::keep_every_hkl], F2[keep][::keep_every_hkl]

    k = exp.k
    w_q = exp.bandwidth_rel * k                         # shell thickness [1/A]
    sigma_m = math.radians(exp.mosaic_fwhm_deg / 2.3548)
    c0 = exp.dac_axis0()
    cos_open = math.cos(math.radians(exp.half_opening_deg))
    mu_mm = exp.mu_diamond_per_mm
    h_cm = exp.plate_h_um * 1e-4
    n, D, px = exp.det_n, exp.dist_mm, exp.pixel_mm
    cr, cc = exp.centre_rc

    records = []           # truth rows
    patches = []           # (f0, r0, c0, local array of unscaled weights)
    for ih, (h, F2h) in enumerate(zip(hkl_all, F2)):
        q0 = U @ (B @ h)
        for om_star, slope in _crossings(q0, exp):
            if slope < 1e-4:                                         # near-tangent: window ill-defined
                continue
            theta_m = 5.0 * sigma_m
            W = 1.5 * (np.linalg.norm(q0) * theta_m + w_q) / slope
            pairs = 0
            sums = dict(L=0.0, tot=0.0)
            fr, rr, cc_, ww = [], [], [], []
            n_hit = 0
            lost = 0.0
            while n_hit < target_hits and pairs < max_pairs:
                om = rng.uniform(om_star - W, om_star + W, batch)
                rv = rng.normal(0.0, sigma_m, (batch, 3))
                q = _rot(np.radians(om)[:, None] * AXIS).apply(_rot(rv).apply(q0))
                v = k * S0 + q
                vn = np.linalg.norm(v, axis=1)
                hit = np.abs(vn - k) < w_q / 2.0
                pairs += batch
                if not hit.any():
                    continue
                omh, vh, vnh = om[hit], v[hit], vn[hit]
                s = vh / vnh[:, None]
                n_hit += int(hit.sum())
                base = np.ones(len(omh))                              # x (2W/M)/w_q applied after the loop
                sums["L"] += base.sum()
                # polarization
                P = _thomson_pol(s, exp.pol_fraction)
                # DAC axis at this omega (rotates with the sample)
                c = _rot(np.radians(omh)[:, None] * AXIS).apply(c0)
                s0b = np.broadcast_to(S0, s.shape)
                ci, co = np.abs(np.sum(c * s0b, axis=1)), np.abs(np.sum(c * s, axis=1))
                inside = (ci >= cos_open) & (co >= cos_open)
                din = -s0b
                path = (_slab_length(din, c, -exp.t_minus_mm, 0.0) + _slab_length(din, c, 0.0, exp.t_plus_mm)
                        + _slab_length(s, c, 0.0, exp.t_plus_mm) + _slab_length(s, c, -exp.t_minus_mm, 0.0))
                Tdac = np.where(inside, np.exp(-mu_mm * path), 0.0)
                # crystal plate self-absorption, random depth
                if exp.mu_sample_per_cm > 0.0:
                    z = rng.uniform(0.0, 1.0, len(omh)) * h_cm
                    laue = np.sign(np.sum(c * s, axis=1)) == np.sign(np.sum(c * s0b, axis=1))
                    p_in = z / np.maximum(ci, 1e-12)
                    p_out = np.where(laue, (h_cm - z), z) / np.maximum(co, 1e-12)
                    Ts = np.exp(-exp.mu_sample_per_cm * (p_in + p_out))
                else:
                    Ts = 1.0
                wt = base * P * Tdac * Ts
                # detector position and frame
                sx = s[:, 0]
                ok = sx > 1e-6
                yy = np.where(ok, D * s[:, 1] / np.where(ok, sx, 1.0), 0.0)
                zz = np.where(ok, D * s[:, 2] / np.where(ok, sx, 1.0), 0.0)
                col = cc - yy / px
                row = cr + zz / px
                fr_ = np.floor((omh - exp.omega_start) / exp.omega_step).astype(int)
                on = ok & (fr_ >= 0) & (fr_ < exp.n_frames) & (np.abs(row - cr) < n / 2 - 3) & (np.abs(col - cc) < n / 2 - 3)
                lost += float(wt[~on].sum())
                fr.append(fr_[on]); rr.append(row[on]); cc_.append(col[on]); ww.append(wt[on])
                sums["tot"] += wt.sum()
            if n_hit < 1000:
                continue
            norm = (2.0 * W / pairs) / w_q                           # deg per (1/A)
            fr_a, rr_a, cc_a, ww_a = (np.concatenate(x) if x else np.array([]) for x in (fr, rr, cc_, ww))
            ww_a = ww_a * norm
            tot_w = sums["tot"] * norm
            if tot_w <= 0.0:
                continue                                              # blocked by the opening cone
            complete = (lost * norm) < 1e-3 * tot_w
            s_nom = (k * S0 + _rot(np.radians(om_star) * AXIS).apply(q0)) / k
            # emergent Lorentz: integral of (1/w_q) 1[in shell] d omega = 1/|df/domega|; compare later
            lor_emergent = sums["L"] * norm * k * math.pi / 180.0     # = L if the closed form is right
            # histogram into local patch
            if len(ww_a) == 0:
                continue
            f0, f1 = fr_a.min(), fr_a.max()
            r0, r1 = int(np.floor(rr_a.min())) - 3, int(np.ceil(rr_a.max())) + 3
            c0_, c1 = int(np.floor(cc_a.min())) - 3, int(np.ceil(cc_a.max())) + 3
            patch, _ = np.histogramdd(
                np.stack([fr_a, np.rint(rr_a), np.rint(cc_a)], axis=1), weights=ww_a,
                bins=[np.arange(f0, f1 + 2) - 0.5, np.arange(r0, r1 + 2) - 0.5, np.arange(c0_, c1 + 2) - 0.5])
            patch = ndimage.gaussian_filter(patch, sigma=(0, exp.sigma_psf_px, exp.sigma_psf_px), mode="constant")
            patches.append((f0, r0, c0_, patch.astype(np.float32)))
            records.append(dict(h=int(h[0]), k=int(h[1]), l=int(h[2]), F2=float(F2h), omega=float(om_star),
                                row=float(np.average(rr_a, weights=ww_a)), col=float(np.average(cc_a, weights=ww_a)),
                                w_total=float(patch.sum()), w_tot_all=tot_w, complete=bool(complete),
                                n_hits=int(n_hit), lorentz_emergent=float(lor_emergent),
                                sx=float(s_nom[0]), sy=float(s_nom[1]), sz=float(s_nom[2]),
                                tth=float(math.degrees(math.acos(np.clip(s_nom[0], -1.0, 1.0))))))
        if verbose and ih % 200 == 0:
            print(f"  hkl {ih}/{len(hkl_all)}  spots so far {len(records)}", flush=True)

    truth = pd.DataFrame(records)
    if truth.empty:
        raise RuntimeError("no reflection crossed the Ewald sphere inside the scan and opening cone")
    truth["expected_unscaled"] = truth["F2"] * truth["w_total"]
    scale = n_int_max / truth.loc[truth["complete"], "expected_unscaled"].max()
    if scale_override is not None:                            # a second domain: same absolute scale as domain 1 times its fraction (PREREGISTER A23)
        scale = float(scale_override)
    signal = np.zeros((exp.n_frames, n, n), dtype=np.float32)
    for t, (f0, r0, c0_, patch) in zip(truth.itertuples(), patches):
        fs, rs, cs = patch.shape
        rlo, clo = max(r0, 0), max(c0_, 0)
        rhi, chi = min(r0 + rs, n), min(c0_ + cs, n)
        if rhi <= rlo or chi <= clo:
            continue
        sub = patch[:, rlo - r0:rhi - r0, clo - c0_:chi - c0_] * np.float32(scale * t.F2)
        for j in range(fs):
            signal[f0 + j, rlo:rhi, clo:chi] += sub[j]
    truth["expected_counts"] = truth["expected_unscaled"] * scale
    return SimResult(exp=exp, structure=st, U=U, signal=signal, truth=truth, scale=float(scale))
