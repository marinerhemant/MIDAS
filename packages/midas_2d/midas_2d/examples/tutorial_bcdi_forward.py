"""Bragg CDI forward model: a strained nanocrystal to photon counts on a detector.

Answers, concretely, "how does the Fourier transform of a grain become the
intensity signal on the detector?" -- and checks every link against something
analytic rather than against itself.

The chain
---------
1. **The object is not the atomic density.** Near one Bragg peak the far field is
   the transform of a slowly-varying complex *envelope*::

       psi(r) = s(r) * exp(-i G . u(r))

   with ``s`` the grain shape and ``u`` the lattice displacement field. Voxels
   are nanometres, not angstroms. The phase carries only ``G . u`` -- one
   projection of ``u`` -- so a single reflection gives one strain component and
   three non-coplanar reflections are needed for the full tensor.

2. **Far field.** ``A(q) = sum psi(r) exp(-i q.r)``, ``I = |A|^2``, with ``q``
   measured *from* the Bragg peak. Fraunhofer holds easily (``D >> a^2/lambda``:
   ~8 mm for a 1 um grain at 10 keV, against a metre of detector distance).

3. **The part that bites: q is sampled on a non-orthogonal grid.** The measured
   array is indexed by (detector column, detector row, rocking step), and those
   three directions are not mutually perpendicular in q. For the geometry below
   the shear is 17 degrees -- the Bragg angle -- so a plain FFT does *not*
   reconstruct onto a Cartesian grid. ``conjugate_real_basis`` gives the grid it
   does use; ``sheared_to_lab`` removes the shear from the reconstructed object
   at the very end. Never interpolate the measured intensity before phasing.

4. **Detector.** ``|F_hkl|^2``, polarisation and solid angle, partial coherence,
   flux, then Poisson counts.

Sign convention, stated once
----------------------------
``A(q) = sum psi(r) exp(-i q.r)`` -- plain ``torch.fft.fftn`` -- so the phase is
``-G.u``. Flip either half of that pair and you swap tension for compression.
The error is not loud: conjugating ``psi`` maps ``I(q) -> I(-q)``, i.e. the
conjugate-twin ambiguity of phase retrieval *is* the strain-sign ambiguity.

Run
---
    python -m midas_2d.examples.tutorial_bcdi_forward
    python -m midas_2d.examples.tutorial_bcdi_forward --n 128 --dislocation

Writes a six-panel figure plus the simulated arrays. The optional
``--dislocation`` field needs ``midas-dfxm`` (``pip install midas-2d[dfxm]``);
without it an analytic inhomogeneous strain field is used, which exercises the
same chain.
"""
from __future__ import annotations

import argparse
import math
import os

import torch

import midas_2d as m2d

DT = torch.float64
TWOPI = 2.0 * math.pi
ANGSTROM_PER_UM = 1.0e4

# Au (111) at 9 keV -- the canonical BCDI system (gold nanocrystals).
A_AU = 4.0782
ENERGY_EV = 9000.0
HKL = (1, 1, 1)


# =========================================================== 1. the geometry
def build_geometry(grain_size_A, shape, *, distance_mm=None, target_sigma=4.0):
    """q-space basis ``B``, its real-space conjugate ``C``, and the Bragg vector.

    ``distance_mm=None`` solves for the detector distance that gives
    ``target_sigma`` oversampling -- which is how the choice is actually made at
    a beamline: it is set by the grain size you expect, not by the reflection.
    """
    from midas_hkls import energy_eV_to_wavelength

    lam = float(energy_eV_to_wavelength(ENERGY_EV))
    d_hkl = A_AU / math.sqrt(sum(h * h for h in HKL))

    if distance_mm is None:
        distance_mm = m2d.detector_distance_for_oversampling(
            lam, grain_size_A, 0.055, target_sigma)
    # Match the rocking step to the detector sampling so all three axes agree.
    step = m2d.rocking_step_for_oversampling(lam, d_hkl, grain_size_A, target_sigma)

    B = m2d.q_basis(lam, d_hkl, distance_mm=distance_mm, pixel_mm=0.055,
                    rocking_step_deg=step, dtype=DT)
    C = m2d.conjugate_real_basis(B, shape)
    geom = m2d.bragg_geometry(lam, d_hkl, dtype=DT)
    return {"B": B, "C": C, "G": geom["G"], "wavelength_A": lam,
            "d_hkl_A": d_hkl, "theta_deg": math.degrees(geom["theta_rad"]),
            "distance_mm": distance_mm, "rocking_step_deg": step, "shape": shape}


def describe(g, grain_size_A):
    sig = m2d.oversampling(g["B"], grain_size_A)
    shear = m2d.shear_angles_deg(g["B"])
    vox = torch.linalg.norm(g["C"], dim=0)
    extent = vox * torch.tensor(g["shape"], dtype=DT)
    return "\n".join([
        f"  beam            {ENERGY_EV/1e3:.3f} keV, lambda = {g['wavelength_A']:.5f} A",
        f"  reflection      {HKL}, d = {g['d_hkl_A']:.4f} A, "
        f"|G| = {float(torch.linalg.norm(g['G'])):.4f} 1/A",
        f"  Bragg angle     theta = {g['theta_deg']:.3f} deg "
        f"(2theta = {2*g['theta_deg']:.3f})",
        f"  detector        D = {g['distance_mm']:.1f} mm, 55 um pixels, "
        f"array {tuple(g['shape'])}",
        f"  rocking         {g['rocking_step_deg']:.5f} deg/step about y",
        "",
        "  oversampling    " + ", ".join(f"{float(x):.2f}" for x in sig)
        + f"   (grain {grain_size_A/10:.0f} nm; needs > 2 in every axis)",
        "  q-basis angles  " + ", ".join(f"{float(x):.1f}" for x in shear)
        + " deg   <- 90 would mean orthogonal. It is not: the",
        "                  detector/rocking pair is off by the Bragg angle.",
        "  voxel size      " + ", ".join(f"{float(x):.1f}" for x in vox) + " A",
        "  array covers    " + ", ".join(f"{float(x)/10:.0f}" for x in extent) + " nm",
    ])


# ============================================================= 2. the object
def real_space_grid(g):
    """(N1, N2, N3, 3) positions in Angstrom on the *conjugate* basis C.

    Index-centred: element ``N//2`` sits at r = 0. This is the grid the FFT
    actually pairs with -- not a Cartesian one.
    """
    axes = [torch.arange(n, dtype=DT) - n // 2 for n in g["shape"]]
    m = torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1)
    return m @ g["C"].transpose(0, 1)


def grain_support(r, size_A, shape="cuboid"):
    """0/1 occupancy for a grain of edge/diameter ``size_A``, centred at r = 0."""
    h = size_A / 2.0
    if shape == "cuboid":
        return (r.abs() <= h).all(dim=-1).to(r.dtype)
    if shape == "ellipsoid":
        return ((r / h).pow(2).sum(dim=-1) <= 1.0).to(r.dtype)
    if shape == "octahedron":                       # {111}-faceted, BCDI-typical
        return (r.abs().sum(dim=-1) <= h * 1.5).to(r.dtype)
    raise ValueError(f"unknown shape {shape!r}")


def analytic_displacement(r, size_A, G, peak_phase_rad=2.0):
    """A smooth inhomogeneous displacement field, in Angstrom.

    Parameterised by the peak phase it produces rather than by a displacement
    amplitude, because the phase is what the measurement sees and it is the
    quantity with a natural scale: real BCDI grains show order 1 rad across the
    object, and the ``exp(-i G.u)`` envelope model assumes ``|q.u| << 1``. A raw
    coefficient does not scale with grain size and quietly becomes absurd -- at
    400 nm an innocuous-looking one gave 153 rad, wrapping the phase 24 times.

    ``u_x = amp (x/L)^3``, deliberately *odd* in x: an even field on a
    centrosymmetric support makes ``psi`` centrosymmetric, and then the two
    strain signs give a bit-identical pattern. The twin ambiguity becomes exact
    and the sign is unrecoverable from one reflection at any noise level.
    """
    L = size_A / 2.0
    # phase = -G.u = -G_x u_x, so the peak is |G_x| * amp.
    amp = float(peak_phase_rad) / max(abs(float(G[0])), 1e-30)
    ux = amp * (r[..., 0] / L) ** 3
    return torch.stack([ux, torch.zeros_like(ux), torch.zeros_like(ux)], dim=-1)


def dislocation_displacement(r, *, core_radius_A=50.0):
    """``u(r)`` from one anisotropic (Stroh) edge dislocation, via midas-dfxm.

    Full anisotropic elasticity rather than an isotropic Volterra approximation.
    Returns ``None`` if midas-dfxm is not installed.

    The core sits at a voxel face centre, half a voxel off the sampling lattice:
    the Stroh displacement goes as ``ln(x1 + p x2)``, which is singular exactly
    on the line. This is physically honest anyway -- BCDI real-space resolution
    is 10-30 nm, so a real measurement does not resolve the core either.
    """
    try:
        import midas_dfxm as mdx
    except ImportError:
        return None

    C6 = mdx.cubic_stiffness(192.9, 163.8, 41.5)          # Au, GPa
    disl = mdx.stroh_dislocation(
        C6, burgers=[1, 1, 0], slip_normal=[1, 1, 1], character="edge",
        burgers_length_A=A_AU / math.sqrt(2.0),
        core_radius_um=core_radius_A / ANGSTROM_PER_UM,
        # "compact" so u and grad(u) agree exactly outside the core, which is
        # required whenever the exit-wave phase G.u is consumed.
        core_model="compact")
    pos_um = (r.reshape(-1, 3) / ANGSTROM_PER_UM).to(C6.dtype)
    u = (disl.displacement(pos_um) * ANGSTROM_PER_UM).reshape(r.shape).to(r.dtype)
    if not torch.isfinite(u).all():
        raise RuntimeError("non-finite displacement: a voxel landed on the "
                           "dislocation line; shift the core or change --n")
    return u


def build_object(g, *, grain_size_A, shape="cuboid", use_dislocation=False,
                 peak_phase_rad=2.0):
    """``psi = s * exp(-i G.u)`` on the conjugate basis."""
    r = real_space_grid(g)
    s = grain_support(r, grain_size_A, shape)

    u = None
    if use_dislocation:
        core = 0.5 * (g["C"][:, 0] + g["C"][:, 1])        # half-voxel face centre
        u = dislocation_displacement(r - core)
        if u is None:
            print("  NOTE: midas-dfxm not installed -- falling back to the "
                  "analytic field.\n        pip install midas-2d[dfxm] for the "
                  "Stroh dislocation.")
    if u is None:
        u = analytic_displacement(r, grain_size_A, g["G"], peak_phase_rad)

    # The convention. A = sum psi exp(-i q.r) pairs with phase = -G.u.
    phase = -(u * g["G"]).sum(dim=-1) * s                 # phase is meaningless outside
    return {"psi": torch.polar(s, phase), "support": s, "phase": phase, "u": u, "r": r}


# ============================================================ 3. the forward
def forward(g, obj, *, photons_per_peak=1e6, coherence_length_A=1500.0, seed=0):
    """psi -> |A|^2 -> expected detector rate -> Poisson counts.

    Everything up to and including the rate is differentiable. ``sample_counts``
    draws random numbers and is not; to fit measured data you hold the counts
    fixed and differentiate the Poisson likelihood of the rate.
    """
    from midas_hkls import (Atom, Crystal, Lattice, SpaceGroup,
                            structure_factor_intensity, structure_factors)

    crystal = Crystal(
        lattice=Lattice(A_AU, A_AU, A_AU, 90, 90, 90),
        space_group=SpaceGroup.from_number(225),          # Fm-3m
        atoms=[Atom(element="Au", fract=(0.0, 0.0, 0.0), B_iso=0.6)], name="Au")
    F2 = float(structure_factor_intensity(structure_factors(
        crystal.to_torch(), [list(HKL)], wavelength_A=g["wavelength_A"]))[0])

    A = m2d.object_to_amplitude(obj["psi"])
    I_coherent = A.real * A.real + A.imag * A.imag

    rate = m2d.detector_signal(
        I_coherent,
        Q=m2d.q_grid(g["B"], g["shape"], offset=g["G"]),
        wavelength_A=g["wavelength_A"],
        structure_factor_sq=F2,
        # Partial coherence is the step people skip. Omit it and every simulated
        # pattern looks sharper than any real one, and the reconstruction gets a
        # spuriously crisp surface you will misread as a real interface.
        coherence_length_A=coherence_length_A or None,
        real_basis=g["C"],
        photons_per_peak=photons_per_peak)
    counts = m2d.sample_counts(rate, generator=torch.Generator().manual_seed(seed))
    return {"I_coherent": I_coherent, "rate": rate, "counts": counts, "F2": F2}


# =========================================================== 4. self-checks
def self_checks(g, obj, fwd, grain_size_A):
    """Each link against something analytic. Returns (all_ok, list of lines)."""
    out, ok_all = [], True

    def record(name, ok, detail):
        nonlocal ok_all
        ok_all &= ok
        out.append(f"  [{'PASS' if ok else 'FAIL'}] {name}\n         {detail}")

    # -- the identity that makes a plain FFT the right transform between grids
    N = torch.tensor(g["shape"], dtype=DT)
    err = float((g["B"].transpose(0, 1) @ g["C"] - TWOPI * torch.diag(1 / N)).abs().max())
    record("conjugate basis  B^T C = 2pi/N", err < 1e-15,
           f"max deviation {err:.2e} (machine precision)")

    # -- closed form: |FFT(box)|^2 is a product of Dirichlet kernels
    n_supp = (7, 5, 9)
    box = torch.zeros(g["shape"], dtype=torch.complex128)
    box[:n_supp[0], :n_supp[1], :n_supp[2]] = 1.0
    I_box = m2d.bcdi_forward(box)
    axes = [(torch.arange(n, dtype=DT) - n // 2) / n for n in g["shape"]]
    x = torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1)
    I_ana = m2d.interference_factor(x, torch.tensor(n_supp, dtype=DT))
    rel = float((I_box - I_ana).abs().max() / I_ana.max())
    record("analytic Laue    FFT of a box vs closed form", rel < 1e-10,
           f"max relative difference {rel:.2e}")

    # -- the theorem behind the factor-of-2 oversampling rule
    ac = torch.fft.fftshift(torch.fft.ifftn(
        torch.fft.ifftshift(fwd["I_coherent"]))).abs()
    thr = 1e-6 * float(ac.max())

    def extent(mask):
        spans = []
        for d in range(3):
            o = tuple(i for i in range(3) if i != d)
            nz = torch.nonzero(mask.amax(dim=o[1]).amax(dim=o[0])).flatten()
            spans.append(int(nz[-1] - nz[0]) + 1)
        return spans

    e_obj, e_ac = extent(obj["support"] > 0), extent(ac > thr)
    want = [2 * n - 1 for n in e_obj]
    dev = max(abs(a - p) / p for a, p in zip(e_ac, want))
    record("autocorrelation  support = 2x object - 1", dev < 0.06,
           f"object {e_obj} -> autocorrelation {e_ac}, expected {want}")

    # -- the flat-Ewald approximation, in pixels
    n0, n1 = g["shape"][0], g["shape"][1]
    corners = torch.tensor([[i, j, 0] for i in (-n0 // 2, n0 // 2 - 1)
                            for j in (-n1 // 2, n1 // 2 - 1)], dtype=DT)
    k = TWOPI / g["wavelength_A"]
    ki = torch.tensor([0.0, 0.0, k], dtype=DT)
    kf = ki + g["G"] + corners @ g["B"].transpose(0, 1)
    px = float((torch.linalg.norm(kf, dim=-1) - k).abs().max()
               / torch.linalg.norm(g["B"][:, 0]))
    record("Ewald curvature  cost of the linearised basis", px < 0.5,
           f"{px:.3f} pixel at the array corner (this is what we neglect)")

    # -- sampling
    sig = m2d.oversampling(g["B"], grain_size_A)
    record("oversampling     sigma >= 2 in every axis", bool((sig >= 2).all()),
           "sigma = " + ", ".join(f"{float(s):.2f}" for s in sig))

    # -- the shear correction, with a control so "extents agree" means something
    def extents_A(mask, vox):
        spans = []
        for d in range(3):
            o = tuple(i for i in range(3) if i != d)
            nz = torch.nonzero(mask.amax(dim=o[1]).amax(dim=o[0])).flatten()
            spans.append((int(nz[-1] - nz[0]) + 1) * vox)
        return spans

    probe = grain_support(obj["r"], grain_size_A, "cuboid")     # cube, any --shape
    raw = extents_A(probe > 0, float(torch.linalg.norm(g["C"], dim=0).mean()))
    lab = m2d.sheared_to_lab(probe, g["C"])
    cor = extents_A(lab["obj"] > 0.5, lab["voxel_A"])
    s_raw, s_cor = max(raw) / min(raw) - 1, max(cor) / min(cor) - 1
    tol = max(0.10, 2.5 * lab["voxel_A"] / grain_size_A)        # discretisation floor
    record("shear correction sheared_to_lab un-shears a cube",
           s_raw > 0.2 and s_cor < tol,
           f"uncorrected {[round(v) for v in raw]} A (spread {s_raw*100:.0f}%) -> "
           f"corrected {[round(v) for v in cor]} A (spread {s_cor*100:.0f}%, "
           f"tol {tol*100:.0f}%); true edge {grain_size_A:.0f} A")

    return ok_all, out


# ============================================================== 5. the figure
def make_figure(g, obj, fwd, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    N = g["shape"]
    mid, cz = N[2] // 2, N[2] // 2
    fig, ax = plt.subplots(2, 3, figsize=(15, 9))

    ax[0, 0].imshow(obj["support"][:, :, mid].T.numpy(), origin="lower", cmap="gray")
    ax[0, 0].set_title("object |psi| (central slice)")

    ph = obj["phase"][:, :, mid].clone()
    ph[obj["support"][:, :, mid] == 0] = float("nan")
    im = ax[0, 1].imshow(ph.T.numpy(), origin="lower", cmap="twilight_shifted")
    ax[0, 1].set_title("phase = -G.u  (rad)\nonly the component of u along G "
                       "is measurable", fontsize=9)
    plt.colorbar(im, ax=ax[0, 1], fraction=0.046)

    um = obj["u"][:, :, mid].norm(dim=-1).clone()
    um[obj["support"][:, :, mid] == 0] = float("nan")
    im = ax[0, 2].imshow(um.T.numpy(), origin="lower", cmap="magma")
    ax[0, 2].set_title("|u| (Angstrom)")
    plt.colorbar(im, ax=ax[0, 2], fraction=0.046)

    Ic = fwd["I_coherent"][:, :, cz]
    im = ax[1, 0].imshow(Ic.T.numpy(), origin="lower", cmap="viridis",
                         norm=LogNorm(vmin=max(float(Ic.max()) * 1e-7, 1e-12),
                                      vmax=float(Ic.max())))
    ax[1, 0].set_title("|FFT(psi)|^2 (detector slice, log)")
    plt.colorbar(im, ax=ax[1, 0], fraction=0.046)

    cts = fwd["counts"][:, :, cz]
    im = ax[1, 1].imshow(cts.T.numpy(), origin="lower", cmap="inferno",
                         norm=LogNorm(vmin=0.5, vmax=max(float(cts.max()), 1.0)))
    ax[1, 1].set_title("Poisson counts (what the detector records)")
    plt.colorbar(im, ax=ax[1, 1], fraction=0.046)

    B = g["B"]
    u = (B / torch.linalg.norm(B, dim=0, keepdim=True)).numpy()
    labels = ["dq1 (detector column)", "dq2 (detector row)", "dq3 (rocking)"]
    for i, c in enumerate(["tab:red", "tab:green", "tab:blue"]):
        if math.hypot(u[0, i], u[2, i]) < 1e-6:          # dq2 is along y
            ax[1, 2].plot(0, 0, "o", color=c, ms=11, mfc="none", mew=2.5,
                          label=labels[i] + ", out of page")
            continue
        ax[1, 2].arrow(0, 0, u[0, i], u[2, i], color=c, width=0.012,
                       length_includes_head=True, label=labels[i])
    sh = m2d.shear_angles_deg(B)
    ax[1, 2].set_xlim(-1.2, 1.2); ax[1, 2].set_ylim(-1.2, 1.2)
    ax[1, 2].set_aspect("equal"); ax[1, 2].legend(fontsize=8, loc="lower left")
    ax[1, 2].grid(alpha=0.3)
    ax[1, 2].set_xlabel("qx"); ax[1, 2].set_ylabel("qz")
    ax[1, 2].set_title(f"q-space basis, projected on the scattering plane\n"
                       f"angles {sh[0]:.0f}, {sh[1]:.0f}, {sh[2]:.0f} deg "
                       f"-> shear of {abs(float(sh[1])-90):.0f} deg", fontsize=9)

    for a in ax.flat[:5]:
        a.set_xticks([]); a.set_yticks([])
    fig.suptitle(f"BCDI forward model -- Au {HKL} at {ENERGY_EV/1e3:.0f} keV",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ==================================================================== driver
def main(out_dir=None, *, seed=0, n=64, grain_nm=400.0, shape="cuboid",
         use_dislocation=False, peak_phase_rad=2.0, photons=1e6,
         coherence_nm=150.0, figure=True):
    """Run the whole chain and return the output directory.

    Parameters mirror the command line; see ``--help``.
    """
    torch.manual_seed(seed)
    grain_A = grain_nm * 10.0
    arr = (n, n, n)

    print("=" * 74)
    print("BCDI FORWARD MODEL")
    print("=" * 74)
    g = build_geometry(grain_A, arr)
    print(describe(g, grain_A))

    print("\n  building the object ...")
    obj = build_object(g, grain_size_A=grain_A, shape=shape,
                       use_dislocation=use_dislocation,
                       peak_phase_rad=peak_phase_rad)
    n_vox = int(obj["support"].sum())
    print(f"    support {n_vox} voxels ({100*n_vox/n**3:.2f}% of the array); "
          f"peak |G.u| = {float(obj['phase'].abs().max()):.2f} rad")

    print("  forward ...")
    fwd = forward(g, obj, photons_per_peak=photons,
                  coherence_length_A=coherence_nm * 10.0, seed=seed)
    nz = fwd["I_coherent"][fwd["I_coherent"] > 0]
    print(f"    |F_{''.join(map(str, HKL))}|^2 = {fwd['F2']:.1f} e^2 (midas_hkls)")
    print(f"    dynamic range {float(fwd['I_coherent'].max()/nz.min()):.2e}, "
          f"total counts {float(fwd['counts'].sum()):.4g}")

    ok, lines = self_checks(g, obj, fwd, grain_A)
    print("\n" + "=" * 74 + "\nSELF-CHECKS\n" + "=" * 74)
    print("\n".join(lines))
    print("=" * 74)
    print("  " + ("ALL CHECKS PASSED" if ok else "*** SOME CHECKS FAILED ***"))
    print("=" * 74)

    if out_dir is None:
        out_dir = os.path.join(os.getcwd(), "bcdi_output")
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    data_path = os.path.join(out_dir, "bcdi_forward.pt")
    torch.save({"counts": fwd["counts"].to(torch.float32),
                "I_coherent": fwd["I_coherent"].to(torch.float32),
                "rate": fwd["rate"].to(torch.float32),
                "support": obj["support"].to(torch.uint8),
                "phase": obj["phase"].to(torch.float32),
                "q_basis_invA": g["B"], "real_basis_A": g["C"], "G_invA": g["G"],
                "wavelength_A": g["wavelength_A"]}, data_path)
    print(f"\n  data   -> {data_path}")

    if figure:
        fig_path = os.path.join(out_dir, "bcdi_forward.png")
        make_figure(g, obj, fwd, fig_path)
        print(f"  figure -> {fig_path}")
    return out_dir


def _cli():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir")
    p.add_argument("--n", type=int, default=64, help="array size per axis")
    p.add_argument("--grain-nm", type=float, default=400.0)
    p.add_argument("--shape", default="cuboid",
                   choices=["cuboid", "ellipsoid", "octahedron"])
    p.add_argument("--dislocation", action="store_true",
                   help="anisotropic Stroh edge dislocation (needs midas-dfxm)")
    p.add_argument("--peak-phase-rad", type=float, default=2.0,
                   help="peak |G.u| for the analytic field; order 1 is realistic")
    p.add_argument("--photons", type=float, default=1e6)
    p.add_argument("--coherence-nm", type=float, default=150.0,
                   help="transverse coherence length; 0 disables")
    p.add_argument("--no-figure", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    main(a.out_dir, seed=a.seed, n=a.n, grain_nm=a.grain_nm, shape=a.shape,
         use_dislocation=a.dislocation, peak_phase_rad=a.peak_phase_rad,
         photons=a.photons,
         coherence_nm=a.coherence_nm, figure=not a.no_figure)


if __name__ == "__main__":
    _cli()
