"""Atomistic SAXS source: small-angle intensity straight from MD atomic positions.

Registered design: kept in the author's notes, not shipped; the README section
"Scattering from MD atomic positions" states what was checked (read it before changing
anything here). Everything below is **kinematic, single-scattering, transmission**.

What is computed
----------------
For one periodic MD box with N atoms the amplitude at the box's own reciprocal-lattice points
``G = h1 b1 + h2 b2 + h3 b3`` (``b_i`` the reciprocal basis, ``h`` integers) is::

    A(h) = sum_i f_{s(i)}(|G|) exp(2 pi i  h . s_i),     s_i = fractional coordinates

Because ``G . r = 2 pi h . s``, wrapping an atom by a lattice vector changes nothing, and a
rigid translation changes only a phase, so ``|A|^2`` is exactly invariant to both (gate G4).

**The box defines intensity only at those points** (Paddison, arXiv:1809.07088; the same
statement the dislocation code makes, KL-111). Pixels between them get the un-normalised
(Parseval) Gaussian window average of KL-112::

    I(q) = sum_{G != 0} W(q - G) |A(G)|^2 / ( int W d3k / Omega* ),   Omega* = (2 pi)^3 / V

so that I(q)/n is unchanged when the same structure is stored in an n-fold supercell. ``I`` is
electrons^2 *per box*; the caller divides by the box volume to get a per-A^3 intensity, which
``simulate_frame`` multiplies by ``sample_volume_A3`` (the same contract as the particle terms).
That scaling assumes defects in different boxes are uncorrelated.

Two ways to get ``A(h)``
------------------------
* ``direct``  exact sum, cost ``N_atoms x N_G``. Reference.
* ``fft``     cardinal B-spline deposition of each species onto a fractional-coordinate grid
  and a 3-D FFT, which returns ``A(h)`` on the integer lattice directly (no interpolation),
  deconvolved by ``sinc(h/n)^order``. Cost ``N_atoms x order^3 + n^3 log n``. Valid only for
  ``|h_i| <= n_i / 8`` (|q| <= q_Nyquist / 4, the range registered for gate G1); aliased Bragg
  intensity is suppressed by ``sinc^order`` of the alias index. That leakage still grows as N^2
  (Bragg intensity does): at most ``N^2 (h/n)^(2 order)`` in units of one atom's scattering f^2,
  reported as ``info['fft_leakage_bound_f2']`` with a warning when it exceeds 1. Measured on
  perfect Cu (gate G2): 1.5e-8 f^2 at N = 2048 in a cubic box, 6.1e-5 f^2 for the same crystal
  in a sheared cell. The order (6) and grid spacing (1 A) are FIXED here, not tuned on the gate.

Only the lattice points that some pixel's window can reach are ever computed.
"""
from __future__ import annotations

import math
import time
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from .md_io import AtomicFrame

__all__ = [
    "reciprocal_basis",
    "lattice_amplitudes",
    "atomistic_intensity",
    "default_resolution_fwhm_inv_A",
]

_FWHM_PER_SIGMA = 2.0 * math.sqrt(2.0 * math.log(2.0))
_FFT_MAX_FRACTION = 0.125           # |h_i| <= n_i/8, i.e. |q_i| <= q_Nyquist/4 (registered, G1)


# ----------------------------------------------------------------------------- geometry

def reciprocal_basis(cell) -> np.ndarray:
    """Rows ``b_i`` with ``a_i . b_j = 2 pi delta_ij`` for lattice rows ``a_i``. 1/A."""
    cell = np.asarray(cell, dtype=np.float64)
    return 2.0 * math.pi * np.linalg.inv(cell).T


def default_resolution_fwhm_inv_A(cell) -> float:
    """Two reciprocal-lattice spacings, the KL-112 floor for the Parseval average."""
    return 2.0 * float(np.linalg.norm(reciprocal_basis(cell), axis=1).max())


def _heights(cell: np.ndarray) -> np.ndarray:
    """Perpendicular spacing of the cell faces, A."""
    a, b, c = cell
    vol = abs(np.linalg.det(cell))
    return vol / np.array([np.linalg.norm(np.cross(b, c)),
                           np.linalg.norm(np.cross(c, a)),
                           np.linalg.norm(np.cross(a, b))])


def _fast_len(n: int) -> int:
    """Smallest integer >= n whose only prime factors are 2, 3, 5 (FFT-friendly)."""
    while True:
        m = n
        for p in (2, 3, 5):
            while m % p == 0:
                m //= p
        if m == 1:
            return n
        n += 1


# ----------------------------------------------------------------------------- form factors

def _form_factors(syms: Sequence[str], g_norm: torch.Tensor) -> Dict[str, torch.Tensor]:
    """``f_s(|G|)`` per species, Cromer-Mann, evaluated at s^2 = (|G| / 4 pi)^2."""
    from midas_hkls.form_factors import form_factor

    s2 = (g_norm / (4.0 * math.pi)) ** 2
    return {s: form_factor(s2, s) for s in syms}


# ----------------------------------------------------------------------------- direct sum

def _direct_species_sum(s_frac: torch.Tensor, h: torch.Tensor, tile_elements: int) -> torch.Tensor:
    """``sum_i exp(2 pi i h . s_i)`` for every row of ``h``. Tiled over (G, atoms)."""
    n_g = h.shape[0]
    out = torch.zeros(n_g, dtype=torch.complex128, device=h.device)
    gb = min(n_g, 2048)
    ab = max(1, tile_elements // gb)
    two_pi = 2.0 * math.pi
    for g0 in range(0, n_g, gb):
        hg = h[g0:g0 + gb] * two_pi
        re = torch.zeros(hg.shape[0], dtype=torch.float64, device=h.device)
        im = torch.zeros_like(re)
        for a0 in range(0, s_frac.shape[0], ab):
            ph = hg @ s_frac[a0:a0 + ab].T
            re += torch.cos(ph).sum(dim=1)
            im += torch.sin(ph).sum(dim=1)
        out[g0:g0 + gb] = torch.complex(re, im)
    return out


# ----------------------------------------------------------------------------- FFT path

def _bspline_weights(t: torch.Tensor, order: int) -> torch.Tensor:
    """The ``order`` nonzero values ``N_order(t + k)``, k = 0..order-1, of the cardinal
    B-spline, for fractional offsets ``t`` in [0, 1). Rows sum to 1. Cox-de Boor."""
    w = torch.ones(t.shape[0], 1, dtype=t.dtype, device=t.device)
    for q in range(2, order + 1):
        k = torch.arange(q, dtype=t.dtype, device=t.device)
        x = t[:, None] + k[None, :]                                  # (N, q)
        new = torch.zeros(t.shape[0], q, dtype=t.dtype, device=t.device)
        new[:, :q - 1] += x[:, :q - 1] / (q - 1) * w
        new[:, 1:] += (q - x[:, 1:]) / (q - 1) * w
        w = new
    return w


def _deposit(s_frac: torch.Tensor, n: Tuple[int, int, int], order: int,
             block: int = 1 << 14) -> torch.Tensor:
    """Cardinal B-spline deposition of unit point charges at fractional ``s_frac``."""
    n1, n2, n3 = n
    grid = torch.zeros(n1 * n2 * n3, dtype=torch.float64, device=s_frac.device)
    nn = torch.tensor(n, dtype=torch.float64, device=s_frac.device)
    half = order // 2
    for a0 in range(0, s_frac.shape[0], block):
        u = s_frac[a0:a0 + block] * nn                                # (b, 3) grid units
        i0 = torch.floor(u)
        t = u - i0
        i0 = i0.long()
        k = torch.arange(order, device=u.device)
        ws, js = [], []
        for ax in range(3):
            ws.append(_bspline_weights(t[:, ax], order))              # (b, order)
            js.append(torch.remainder(i0[:, ax, None] - k[None, :] + half, n[ax]))
        w = ws[0][:, :, None, None] * ws[1][:, None, :, None] * ws[2][:, None, None, :]
        idx = (js[0][:, :, None, None] * n2 + js[1][:, None, :, None]) * n3 \
            + js[2][:, None, None, :]
        grid.index_add_(0, idx.reshape(-1), w.reshape(-1))
    return grid.reshape(n1, n2, n3)


def _fft_species_sum(s_frac: torch.Tensor, h: torch.Tensor, n: Tuple[int, int, int],
                     order: int) -> torch.Tensor:
    grid = _deposit(s_frac, n, order)
    # ifftn carries 1/N: multiplying by N gives sum_j g_j exp(+2 pi i h j / n)
    F = torch.fft.ifftn(grid) * float(n[0] * n[1] * n[2])
    del grid
    hi = h.long()
    S = F[torch.remainder(hi[:, 0], n[0]), torch.remainder(hi[:, 1], n[1]),
          torch.remainder(hi[:, 2], n[2])]
    nn = torch.tensor(n, dtype=torch.float64, device=h.device)
    decon = torch.prod(torch.sinc(h / nn) ** order, dim=1)
    return S / decon


# ----------------------------------------------------------------------------- amplitudes

def _fft_grid(cell: np.ndarray, spacing: float) -> Tuple[int, int, int]:
    return tuple(_fast_len(int(math.ceil(hgt / spacing))) for hgt in _heights(cell))  # type: ignore


def lattice_amplitudes(
    frame: AtomicFrame,
    hkl,
    *,
    method: str = "auto",
    device=None,
    fft_spacing_A: float = 1.0,
    fft_order: int = 6,
    tile_elements: int = 1 << 22,
    max_grid_bytes: float = 4e9,
) -> Tuple[torch.Tensor, str]:
    """``A(h)`` at integer lattice indices ``hkl`` (n_G, 3). Returns ``(A complex128, method)``.

    ``method`` is ``"direct"``, ``"fft"`` or ``"auto"`` (FFT when every index is inside
    ``n/8`` and the direct cost ``N x N_G`` clearly exceeds the FFT cost; else direct).
    """
    if method not in ("auto", "direct", "fft"):
        raise ValueError("method must be 'auto', 'direct' or 'fft'")
    if fft_order % 2:
        raise ValueError("fft_order must be even")
    h = torch.as_tensor(np.asarray(hkl), dtype=torch.float64, device=device)
    if h.ndim != 2 or h.shape[1] != 3:
        raise ValueError("hkl must be (n_G, 3)")
    cell = frame.cell
    n = _fft_grid(cell, fft_spacing_A)
    n_t = torch.tensor(n, dtype=torch.float64, device=h.device)
    inside = bool((h.abs() <= n_t * _FFT_MAX_FRACTION).all()) if h.shape[0] else True
    grid_bytes = 24.0 * n[0] * n[1] * n[2]               # real grid + complex FFT
    if method == "fft":
        if not inside:
            raise ValueError(
                f"the FFT path is valid only for |h_i| <= n_i/8 = {[x // 8 for x in n]}; "
                f"the needed lattice points reach {h.abs().max(dim=0).values.tolist()}. "
                f"Use method='direct' or a finer fft_spacing_A.")
        use = "fft"
    elif method == "direct":
        use = "direct"
    else:
        cost_direct = frame.n_atoms * float(h.shape[0])
        cost_fft = frame.n_atoms * fft_order ** 3 + n[0] * n[1] * n[2] * 20.0
        use = "fft" if (inside and grid_bytes <= max_grid_bytes
                        and cost_direct > 20.0 * cost_fft) else "direct"
    if use == "fft" and grid_bytes > max_grid_bytes:
        raise ValueError(f"the FFT grid {n} needs {grid_bytes / 1e9:.1f} GB (> max_grid_bytes); "
                         f"raise fft_spacing_A or use method='direct'")

    s = torch.as_tensor(frame.fractional(), dtype=torch.float64, device=h.device)
    s = s - torch.floor(s)                                # exact: G.r = 2 pi h.s
    g_norm = torch.linalg.vector_norm(h @ torch.as_tensor(reciprocal_basis(cell),
                                                          dtype=torch.float64, device=h.device),
                                      dim=1)
    syms = frame.species_list()
    ff = _form_factors(syms, g_norm)
    A = torch.zeros(h.shape[0], dtype=torch.complex128, device=h.device)
    for sym in syms:
        sel = torch.as_tensor(frame.species == sym, device=h.device)
        s_sp = s[sel]
        if use == "direct":
            S = _direct_species_sum(s_sp, h, tile_elements)
        else:
            S = _fft_species_sum(s_sp, h, n, fft_order)
        A = A + ff[sym].to(torch.complex128) * S
    return A, use


# ----------------------------------------------------------------------------- the frame term

def _offsets(cell: np.ndarray, B: np.ndarray, c: float) -> np.ndarray:
    """Integer offsets d from a pixel's rounded lattice index that can hold a point within c."""
    a_len = np.linalg.norm(cell, axis=1)
    reach = np.ceil(c * a_len / (2.0 * math.pi) + 0.5).astype(int)
    grids = np.meshgrid(*[np.arange(-r, r + 1) for r in reach], indexing="ij")
    d = np.stack([g.reshape(-1) for g in grids], axis=1)
    slack = 0.5 * np.linalg.norm(B, axis=1).sum()
    keep = np.linalg.norm(d @ B, axis=1) <= c + slack
    return d[keep]


def atomistic_intensity(
    q_lab: torch.Tensor,
    frames: Union[AtomicFrame, Sequence[AtomicFrame]],
    *,
    resolution_fwhm_inv_A: Optional[float] = None,
    lab_from_sample=None,
    method: str = "auto",
    device=None,
    fft_spacing_A: float = 1.0,
    fft_order: int = 6,
    cutoff_sigma: float = 5.0,
    tile_elements: int = 1 << 22,
    max_grid_bytes: float = 4e9,
    pixel_chunk: int = 1 << 15,
) -> Tuple[torch.Tensor, dict]:
    """Per-A^3 SAXS intensity (electrons^2 / A^3) at lab-frame ``q_lab`` (n, 3), 1/A.

    Parameters
    ----------
    frames
        One snapshot or several of the SAME box (same cell and species order); intensities
        are averaged over them, which is where thermal disorder comes from.
    resolution_fwhm_inv_A
        FWHM of the Gaussian intensity window, 1/A. Default two reciprocal-lattice spacings
        (KL-112); below that the lattice average ripples and a warning is returned.
    lab_from_sample
        (3, 3) rotation with ``r_lab = R r_sample``. Default identity, i.e. the cell's x axis is
        the beam. Transmission SAXS has no omega, so the sample orientation is whatever you set.
    method, fft_spacing_A, fft_order
        See :func:`lattice_amplitudes`.

    Returns
    -------
    ``(I, info)``. ``info`` has the method, the number of lattice points evaluated, the window
    in use, the reported q floor (about three reciprocal-lattice spacings, below which pixels
    are not representative), the fraction of pixels under it, and a ``warnings`` list.
    """
    fr = [frames] if isinstance(frames, AtomicFrame) else list(frames)
    if not fr:
        raise ValueError("no frames")
    f0 = fr[0]
    for f in fr[1:]:
        if (not np.allclose(f.cell, f0.cell, rtol=0, atol=1e-9) or f.n_atoms != f0.n_atoms
                or not np.array_equal(f.species, f0.species)):
            raise ValueError("frames must share the cell, atom count and species order")
    if not all(f0.pbc):
        raise ValueError("the lattice-point estimator needs a fully periodic box "
                         f"(pbc = {f0.pbc}); a free cluster needs a different estimator")

    t_start = time.perf_counter()
    dev = q_lab.device if device is None else torch.device(device)
    q_lab = q_lab.to(device=dev, dtype=torch.float64)
    cell = f0.cell
    B = reciprocal_basis(cell)
    vol = f0.volume
    a_star = float(np.linalg.norm(B, axis=1).max())
    fwhm = default_resolution_fwhm_inv_A(cell) if resolution_fwhm_inv_A is None \
        else float(resolution_fwhm_inv_A)
    if fwhm <= 0:
        raise ValueError("resolution_fwhm_inv_A must be positive")
    sig = fwhm / _FWHM_PER_SIGMA
    c = cutoff_sigma * sig
    warnings: List[str] = []
    if fwhm < 2.0 * a_star * (1 - 1e-9):
        warnings.append(
            f"the window FWHM {fwhm:.3g} 1/A is below two reciprocal-lattice spacings "
            f"({2 * a_star:.3g}): the lattice average ripples with the lattice (KL-112). "
            f"Raise it or use a bigger box.")

    R = np.eye(3) if lab_from_sample is None else np.asarray(lab_from_sample, dtype=np.float64)
    if R.shape != (3, 3) or not np.allclose(R @ R.T, np.eye(3), atol=1e-8):
        raise ValueError("lab_from_sample must be a 3x3 rotation matrix")
    q = q_lab @ torch.as_tensor(R, dtype=torch.float64, device=dev)    # q_sample = R^T q_lab

    cell_t = torch.as_tensor(cell, dtype=torch.float64, device=dev)
    B_t = torch.as_tensor(B, dtype=torch.float64, device=dev)
    two_pi = 2.0 * math.pi

    # ---- which lattice points can any pixel window reach
    offs = _offsets(cell, B, c)
    offs_t = torch.as_tensor(offs, dtype=torch.long, device=dev)
    ub_list = []
    for p0 in range(0, q.shape[0], pixel_chunk):
        eta = q[p0:p0 + pixel_chunk] @ cell_t.T / two_pi
        ub_list.append(torch.unique(torch.round(eta).long(), dim=0))
    ub = torch.unique(torch.cat(ub_list), dim=0)
    half = (ub.abs().max(dim=0).values + offs_t.abs().max(dim=0).values).cpu().numpy()
    dims = tuple(int(2 * h_ + 1) for h_ in half)
    if np.prod(dims, dtype=np.float64) > 4e8:
        raise ValueError(f"the reachable reciprocal-lattice cube {dims} is too large for this "
                         f"box; lower the q range or the window")
    H = torch.as_tensor(half, dtype=torch.long, device=dev)
    needed = torch.zeros(dims, dtype=torch.bool, device=dev)
    for d in offs_t:
        idx = ub + d + H
        needed[idx[:, 0], idx[:, 1], idx[:, 2]] = True
    needed[tuple(int(x) for x in half)] = False                         # G = 0 is excluded
    hkl = torch.nonzero(needed).to(torch.long) - H
    if hkl.shape[0] == 0:
        raise ValueError("no reciprocal-lattice points inside the window of any pixel")

    t_select = time.perf_counter()
    # ---- |A(G)|^2 averaged over frames, on the dense integer lattice
    mean_I = torch.zeros(hkl.shape[0], dtype=torch.float64, device=dev)
    used = "direct"
    for f in fr:
        A, used = lattice_amplitudes(f, hkl, method=method, device=dev,
                                     fft_spacing_A=fft_spacing_A, fft_order=fft_order,
                                     tile_elements=tile_elements, max_grid_bytes=max_grid_bytes)
        mean_I += (A.real ** 2 + A.imag ** 2) / len(fr)
    dense = torch.zeros(dims, dtype=torch.float64, device=dev)
    sh = hkl + H
    dense[sh[:, 0], sh[:, 1], sh[:, 2]] = mean_I

    t_amp = time.perf_counter()
    # ---- window average onto pixels (Parseval form)
    denom = (two_pi * sig ** 2) ** 1.5 * vol / two_pi ** 3
    out = torch.zeros(q.shape[0], dtype=torch.float64, device=dev)
    c2 = c * c
    inv2s2 = 1.0 / (2.0 * sig * sig)
    # Per pixel the offset from its rounded lattice cell is computed once, then every neighbouring
    # lattice point is evaluated in blocks: r^2 = |delta|^2 - 2 delta.(dB) + |dB|^2, with one BLAS
    # matmul per block and a single flat gather from the dense |A|^2 lattice.
    strides = torch.tensor([dims[1] * dims[2], dims[2], 1], dtype=torch.long, device=dev)
    flat = dense.reshape(-1)
    d_lin = offs_t @ strides                                   # (n_off,)
    dB = offs_t.to(torch.float64) @ B_t                        # (n_off, 3)
    dB2 = (dB * dB).sum(dim=1)
    off_block = max(1, min(offs_t.shape[0], (1 << 21) // max(1, pixel_chunk)))
    for p0 in range(0, q.shape[0], pixel_chunk):
        qc = q[p0:p0 + pixel_chunk]
        base = torch.round(qc @ cell_t.T / two_pi).long()
        delta = qc - base.to(torch.float64) @ B_t              # (n, 3), q minus its base lattice point
        d2 = (delta * delta).sum(dim=1)
        base_lin = (base + H) @ strides
        acc = torch.zeros(qc.shape[0], dtype=torch.float64, device=dev)
        for o0 in range(0, offs_t.shape[0], off_block):
            o1 = o0 + off_block
            r2 = d2[:, None] - 2.0 * (delta @ dB[o0:o1].T) + dB2[None, o0:o1]
            w = torch.exp(-r2 * inv2s2) * (r2 <= c2)
            acc += (w * flat[base_lin[:, None] + d_lin[None, o0:o1]]).sum(dim=1)
        out[p0:p0 + pixel_chunk] = acc / denom

    t_win = time.perf_counter()
    leak_bound = None
    if used == "fft":
        n_grid = np.array(_fft_grid(cell, fft_spacing_A), dtype=np.float64)
        eps_alias = float((hkl.abs().to(torch.float64).max(dim=0).values.cpu().numpy() / n_grid).max())
        leak_bound = float(f0.n_atoms) ** 2 * eps_alias ** (2 * fft_order)
        if leak_bound > 1.0:
            warnings.append(
                f"FFT path: aliased Bragg leakage could reach {leak_bound:.2g} x one atom's "
                f"scattering (f^2) at the outer lattice points if the crystal's Bragg peaks alias "
                f"there (it grows as N^2). That is not small next to a few-atom defect; use "
                f"method='direct', a finer fft_spacing_A, or a higher fft_order.")
    floor = 3.0 * a_star
    below = float((torch.linalg.vector_norm(q, dim=1) < floor).double().mean())
    if below > 0:
        warnings.append(
            f"{100 * below:.1f} % of pixels have |q| < {floor:.3g} 1/A, about three "
            f"reciprocal-lattice spacings. There the excluded G = 0 term and the box's own "
            f"periodicity dominate, so those pixels are not representative; a bigger MD box "
            f"lowers the floor.")
    info = dict(
        method=used, n_lattice_points=int(hkl.shape[0]), n_frames=len(fr), n_atoms=f0.n_atoms,
        volume_A3=vol, resolution_fwhm_inv_A=fwhm, window_sigma_inv_A=sig,
        a_star_inv_A=a_star, q_floor_inv_A=floor, fraction_below_floor=below,
        mean_points_under_window=float(denom), warnings=warnings,
        fft_grid=_fft_grid(cell, fft_spacing_A) if used == "fft" else None,
        fft_order=fft_order if used == "fft" else None,
        fft_leakage_bound_f2=leak_bound,
        timings_s=dict(select_lattice=t_select - t_start, amplitudes=t_amp - t_select,
                       window=t_win - t_amp),
    )
    return out / vol, info
