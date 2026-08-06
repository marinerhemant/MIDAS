"""BCDI from data you already have: an external array, or atomic coordinates.

Two entry points that are not "simulate everything from scratch".

**A. An array you already have** (``--from-file``) -- your own FFT, a
reconstructed object, or measured intensity -- pushed through the detector chain
in consistent conventions.

**B. Atomic coordinates** (``--from-md``) -- from MD, a relaxed structure, or a
builder -- turned into a detector signal.

Both are differentiable end to end: a loss defined on what the detector records
backpropagates to the object (A) or to every atom (B). The only step that is not
is drawing Poisson counts, which is a random sample. To fit real data you hold
the counts fixed and differentiate the *likelihood* of the predicted rate --
``--grad-demo`` shows exactly that.

Reading your file
-----------------
Three things cannot be inferred, and are asked for explicitly rather than
guessed, because a wrong guess yields a plausible wrong answer instead of an
error:

``--kind``
    ``object`` | ``amplitude`` | ``intensity``. A *real* array is taken as
    intensity. A *complex* one is genuinely ambiguous: a real-space object needs
    a Fourier transform, a far-field amplitude does not. Getting it wrong
    applies or skips a transform and the result still looks like speckle.
``--transpose``
    This package indexes ``(detector column, detector row, rocking step)``.
    Files stored rocking-first -- the acquisition order, and common -- want
    ``--transpose 1,2,0``.
``--uncentered``
    Pass it if q = 0 sits at index 0 rather than at the array centre.

``--list-datasets`` shows what is inside an HDF5/CXI file.

Two routes from atoms, and the choice is real
---------------------------------------------
``speckle_from_atoms`` evaluates ``sum_i f_i exp(i Q.r_i)`` exactly, with no
envelope or small-strain approximation. It costs ``O(N_atoms * N_q)``, which
caps it near 10 nm: a 60 nm crystal is 6.7 M atoms and ~1.75e12 terms.

``atoms_to_object`` bins the same coordinates in ``O(N_atoms)`` into the envelope
object ``psi = occupancy * exp(-i G.u)``, which scales to real grain sizes --
6.7 M atoms in about a second. That is how MD couples to BCDI in practice: MD
supplies the displacement field, the envelope model supplies the diffraction.

Use the exact sum to *validate* the envelope model on a small crystal, then use
the envelope model at the real size. ``--cross-check`` does the validation, with
controls, and ``atom_sum_cost`` tells you which route a given problem needs.

Run
---
    python -m midas_2d.examples.tutorial_bcdi_from_data                  # both, self-contained
    python -m midas_2d.examples.tutorial_bcdi_from_data --from-file my.npy --kind object
    python -m midas_2d.examples.tutorial_bcdi_from_data --from-file scan.cxi --list-datasets
    python -m midas_2d.examples.tutorial_bcdi_from_data --from-md frames.xyz --reference-frame 0
    python -m midas_2d.examples.tutorial_bcdi_from_data --from-md synthetic --grad-demo
"""
from __future__ import annotations

import argparse
import math
import os

import torch

import midas_2d as m2d

DT = torch.float64
A_AU = 4.0782
ENERGY_EV = 9000.0
HKL = (1, 1, 1)
PIXEL_MM = 0.055


# ------------------------------------------------------------------- helpers
def geometry(object_size_A, shape, *, target_sigma=3.0):
    """q-basis, real-space conjugate basis, Bragg vector -- sized to the object."""
    from midas_hkls import energy_eV_to_wavelength

    lam = float(energy_eV_to_wavelength(ENERGY_EV))
    d_hkl = A_AU / math.sqrt(sum(h * h for h in HKL))
    D = m2d.detector_distance_for_oversampling(lam, object_size_A, PIXEL_MM,
                                               target_sigma)
    step = m2d.rocking_step_for_oversampling(lam, d_hkl, object_size_A, target_sigma)
    B = m2d.q_basis(lam, d_hkl, distance_mm=D, pixel_mm=PIXEL_MM,
                    rocking_step_deg=step, dtype=DT)
    C = m2d.conjugate_real_basis(B, shape)
    G = m2d.bragg_geometry(lam, d_hkl, dtype=DT)["G"]
    print(f"  geometry      D = {D:.2f} mm, rocking {step:.5f} deg/step, "
          f"array {tuple(shape)}")
    print("  sampling      sigma = "
          + ", ".join(f"{float(x):.2f}" for x in m2d.oversampling(B, object_size_A))
          + "   q-basis angles "
          + ", ".join(f"{float(x):.0f}" for x in m2d.shear_angles_deg(B)) + " deg")
    return {"B": B, "C": C, "G": G, "wavelength_A": lam, "shape": tuple(shape)}


def detector_chain(I, g, *, photons=1e6, coherence_A=None, seed=0):
    """|A|^2 -> expected rate (differentiable) -> Poisson counts (not)."""
    from midas_hkls import (Atom, Crystal, Lattice, SpaceGroup,
                            structure_factor_intensity, structure_factors)

    crystal = Crystal(lattice=Lattice(A_AU, A_AU, A_AU, 90, 90, 90),
                      space_group=SpaceGroup.from_number(225),
                      atoms=[Atom(element="Au", fract=(0, 0, 0), B_iso=0.6)],
                      name="Au")
    F2 = float(structure_factor_intensity(structure_factors(
        crystal.to_torch(), [list(HKL)], wavelength_A=g["wavelength_A"]))[0])

    rate = m2d.detector_signal(
        I, Q=m2d.q_grid(g["B"], g["shape"], offset=g["G"]),
        wavelength_A=g["wavelength_A"], structure_factor_sq=F2,
        coherence_length_A=coherence_A, real_basis=g["C"],
        photons_per_peak=photons)
    counts = m2d.sample_counts(rate, generator=torch.Generator().manual_seed(seed))
    nz = rate[rate > 0]
    print(f"  |F_{''.join(map(str, HKL))}|^2     {F2:.1f} e^2 (midas_hkls)")
    print(f"  detector      peak rate {float(rate.max()):.4g}, total counts "
          f"{float(counts.sum()):.4g}, dynamic range "
          f"{float(rate.max()/nz.min()):.3e}")
    return rate, counts


def ncc(x, y):
    """Normalised cross-correlation, for comparing two patterns."""
    x, y = x.flatten(), y.flatten()
    x, y = x - x.mean(), y - y.mean()
    return float((x * y).sum() / (x.norm() * y.norm()))


def invert_pattern(I):
    """``q -> -q`` on an fftshift-ed array of even length.

    ``flip`` alone is off by one voxel and reads as a real discrepancy -- 0.62
    instead of 1.00 when comparing a pattern against its own conjugate twin.
    """
    return torch.roll(torch.flip(I, dims=(0, 1, 2)), shifts=(1, 1, 1),
                      dims=(0, 1, 2))


# ============================================================== A: from file
def from_file(args, out_dir):
    if args.list_datasets:
        print(f"3-D datasets in {args.from_file}:")
        for row in m2d.list_datasets(args.from_file):
            print("   ", row)
        return None

    transpose = tuple(int(x) for x in args.transpose.split(",")) if args.transpose else None
    raw_shape = tuple(int(x) for x in args.raw_shape.split(",")) if args.raw_shape else None
    data = m2d.load_bcdi(args.from_file, kind=args.kind, dataset=args.dataset,
                         centered=not args.uncentered, transpose=transpose,
                         dtype=args.raw_dtype, shape=raw_shape)
    print("\nloaded")
    print(data.summary())

    # to_intensity applies exactly the right amount of processing for the kind:
    # object -> FFT then |.|^2 ; amplitude -> |.|^2 ; intensity -> unchanged.
    I = data.to_intensity().to(DT)
    shape = tuple(I.shape)

    print(f"\n  NOTE: the file carries no geometry, so --object-nm "
          f"({args.object_nm:g} nm) is used to size it.\n"
          f"        Pass the real value for meaningful sigma and voxel numbers; "
          f"it does not\n        change the intensity, only the reported "
          f"sampling.")
    g = geometry(args.object_nm * 10.0, shape)
    rate, counts = detector_chain(I, g, photons=args.photons,
                                  coherence_A=(args.coherence_nm * 10.0) or None,
                                  seed=args.seed)

    dest = os.path.join(out_dir, "bcdi_from_file.pt")
    torch.save({"intensity": I.to(torch.float32), "rate": rate.to(torch.float32),
                "counts": counts.to(torch.float32), "q_basis_invA": g["B"],
                "real_basis_A": g["C"], "source": data.source}, dest)
    print(f"\n  saved -> {dest}")
    return dest


# ================================================================ B: from MD
def synthetic_frame(radius_A, R):
    """An fcc Au sphere with an inhomogeneous displacement field.

    A stand-in for an MD frame, oriented so (111) satisfies Bragg in the lab
    frame. Returns ``(reference, deformed)`` so the binned route can form
    ``u = deformed - reference``.

    The field is *odd* in x on purpose: with an even field on a centrosymmetric
    support the two strain signs give a bit-identical pattern, so the
    cross-check below would have no power to detect a sign error.
    """
    n = int(2 * radius_A / A_AU) + 3
    ijk = torch.stack(torch.meshgrid(*[torch.arange(n, dtype=DT)] * 3,
                                     indexing="ij"), -1).reshape(-1, 3)
    basis = torch.tensor([[0, 0, 0], [.5, .5, 0], [.5, 0, .5], [0, .5, .5]], dtype=DT)
    c = ((ijk[:, None, :] + basis[None, :, :]).reshape(-1, 3) * A_AU)
    c = (c - c.mean(0)) @ R.T
    ref = c[c.pow(2).sum(-1).sqrt() <= radius_A]
    ux = 5e-2 * (ref[:, 0] ** 3) / (radius_A * radius_A)
    return ref, ref + torch.stack([ux, torch.zeros_like(ux),
                                   torch.zeros_like(ux)], -1)


def from_md(args, out_dir):
    from midas_hkls import energy_eV_to_wavelength

    lam = float(energy_eV_to_wavelength(ENERGY_EV))
    d_hkl = A_AU / math.sqrt(sum(h * h for h in HKL))
    # Atom coordinates live in the crystal frame; the q-grid lives in the lab
    # frame. Without this rotation the Bragg peak lands off the array entirely.
    R = m2d.rotation_to_bragg((2 * math.pi / A_AU) * torch.tensor(
        [float(h) for h in HKL], dtype=DT), lam, d_hkl)

    reference = None
    if args.from_md == "synthetic":
        radius = args.object_nm * 10.0 / 2
        reference, coords = synthetic_frame(radius, R)
        elements = ["Au"] * coords.shape[0]
        print(f"\nsynthetic MD frame: {coords.shape[0]:,} Au atoms in a "
              f"{2*radius:.0f} A sphere, inhomogeneous strain")
    else:
        frames, elements = m2d.load_xyz_frames(args.from_md, dtype=DT)
        coords = frames[args.frame] - frames[args.frame].mean(0)
        print(f"\nloaded {args.from_md}: {frames.shape[0]} frame(s), "
              f"{coords.shape[0]:,} atoms; using frame {args.frame}")
        if not args.no_orient:
            coords = coords @ R.T
            print(f"  oriented so {HKL} satisfies Bragg "
                  "(--no-orient if already in the lab frame)")
        if args.reference_frame is not None:
            ref = frames[args.reference_frame]
            reference = ref - ref.mean(0)
            if not args.no_orient:
                reference = reference @ R.T
            print(f"  frame {args.reference_frame} is the undeformed reference "
                  "-> u = deformed - reference")

    size_A = float((coords.max(0).values - coords.min(0).values).max())
    shape = (args.n,) * 3
    print(f"  extent        {size_A:.1f} A")
    g = geometry(size_A, shape)

    # State the cost before spending it -- the route choice is a real one.
    cost = m2d.atom_sum_cost(args.n ** 3, coords.shape[0])
    print(f"\n  direct sum    {args.n**3:,} q-points x {coords.shape[0]:,} atoms "
          f"= {cost['terms']:.3g} terms, ~{cost['seconds']:.0f} s "
          f"({cost['advice']})")

    route = args.route
    if route == "auto":
        route = "direct" if cost["terms"] < 2e9 else "binned"
        print(f"  route         auto -> {route}")

    if route == "binned":
        if reference is None:
            print("  WARNING: no reference positions, so the binned object "
                  "carries SHAPE ONLY\n           and no G.u phase. Pass "
                  "--reference-frame, or use --route direct\n           on a "
                  "small crystal.")
        ob = m2d.atoms_to_object(coords, g["C"], shape, reference=reference,
                                 G=g["G"] if reference is not None else None)
        if ob["n_outside"]:
            print(f"  WARNING: {ob['n_outside']:,} atoms fell outside the array "
                  "-- raise --n, or the object is off-centre")
        print(f"  binned        {int((ob['occupancy'] > 0).sum()):,} filled "
              "voxels, O(N_atoms) with no q-loop")
        A = m2d.object_to_amplitude(ob["psi"])
        I = A.real ** 2 + A.imag ** 2
    else:
        I = m2d.speckle_from_atoms(coords, elements,
                                   m2d.q_grid(g["B"], shape, offset=g["G"]))

    rate, counts = detector_chain(I, g, photons=args.photons,
                                  coherence_A=(args.coherence_nm * 10.0) or None,
                                  seed=args.seed)

    if args.cross_check and reference is not None:
        cross_check(coords, elements, reference, g, shape, I, route)

    dest = os.path.join(out_dir, "bcdi_from_md.pt")
    torch.save({"intensity": I.to(torch.float32), "rate": rate.to(torch.float32),
                "counts": counts.to(torch.float32), "q_basis_invA": g["B"],
                "real_basis_A": g["C"], "n_atoms": int(coords.shape[0])}, dest)
    print(f"\n  saved -> {dest}")

    if args.grad_demo:
        gradient_demo(coords, elements, g, shape, counts)
    return dest


def cross_check(coords, elements, reference, g, shape, I_current, route):
    """Exact atomic sum against the binned envelope object, with controls.

    The two use *opposite* Fourier sign conventions -- ``exp(+iQ.r)`` for the
    atomic sum, ``exp(-iq.r)`` for ``fftn`` -- which are correctly paired only if
    the envelope is built as ``psi = s exp(-i G.u)``. So this tests the
    linearisation and the sign convention at once.

    Both controls matter. Without them, a high correlation could just mean the
    comparison is insensitive: the shape alone already gets you most of the way.
    """
    direct = (I_current if route == "direct" else
              m2d.speckle_from_atoms(coords, elements,
                                     m2d.q_grid(g["B"], shape, offset=g["G"])))
    if route == "binned":
        binned = I_current
    else:
        ob = m2d.atoms_to_object(coords, g["C"], shape, reference=reference,
                                 G=g["G"])
        A = m2d.object_to_amplitude(ob["psi"])
        binned = A.real ** 2 + A.imag ** 2

    shape_only = m2d.atoms_to_object(coords, g["C"], shape)["psi"]
    A0 = m2d.object_to_amplitude(shape_only)

    print("\n  cross-check: exact atomic sum vs the binned envelope object")
    print(f"    corr(exact, envelope)      {ncc(direct, binned):+.5f}"
          "   <- the linearisation AND the sign pairing")
    print(f"    corr(exact, inverted)      {ncc(direct, invert_pattern(binned)):+.5f}"
          "   <- control: a sign error would win here")
    print(f"    corr(exact, shape only)    "
          f"{ncc(direct, A0.real ** 2 + A0.imag ** 2):+.5f}"
          "   <- control: without the G.u phase")


def gradient_demo(coords, elements, g, shape, counts):
    """d(Poisson NLL)/d(atom positions): the point of the whole chain.

    The loss lives on what the detector records; the gradient lands on the
    atomic coordinates. A structure refinement against measured speckle is an
    optimiser on top of this.
    """
    print("\n  gradient demo: d(Poisson NLL) / d(atom positions)")
    c = coords.clone().requires_grad_(True)
    pred = m2d.detector_signal(
        m2d.speckle_from_atoms(c, elements, m2d.q_grid(g["B"], shape, offset=g["G"])),
        photons_per_peak=float(counts.max()))
    nll = m2d.poisson_nll(pred, counts)
    nll.backward()
    grad = c.grad
    print(f"    NLL           {float(nll.detach()):.6g}")
    print(f"    gradient      {tuple(grad.shape)} -- one 3-vector per atom")
    print(f"    |grad|        max {float(grad.norm(dim=-1).max()):.4g}, "
          f"mean {float(grad.norm(dim=-1).mean()):.4g}, "
          f"all finite {bool(torch.isfinite(grad).all())}")


# =============================================================== self-contained
def write_demo_object(path):
    """A small complex object, so --from-file has something real to read."""
    import numpy as np

    shape = (32, 32, 32)
    g = geometry(300.0, shape)
    axes = [torch.arange(n, dtype=DT) - n // 2 for n in shape]
    r = torch.stack(torch.meshgrid(*axes, indexing="ij"), -1) @ g["C"].transpose(0, 1)
    s = (r.pow(2).sum(-1).sqrt() <= 150.0).to(DT)
    # peak phase ~1 rad, odd in x (see synthetic_frame for why odd)
    amp = 1.0 / abs(float(g["G"][0]))
    phi = -(amp * (r[..., 0] / 150.0) ** 3) * g["G"][0] * s
    np.save(path, torch.polar(s, phi).numpy())
    return path


def main(out_dir=None, *, seed=0, n=32, object_nm=6.0):
    """Run both entry points self-contained, and return the output directory.

    Writes a demo object, reads it back through the file path, then runs the MD
    path on a synthetic frame with the cross-check and the gradient demo.
    """
    if out_dir is None:
        out_dir = os.path.join(os.getcwd(), "bcdi_output")
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 74)
    print(f"BCDI FROM DATA -- Au {HKL} at {ENERGY_EV/1e3:.0f} keV")
    print("=" * 74)

    print("\n--- A. an array you already have " + "-" * 40)
    demo = write_demo_object(os.path.join(out_dir, "demo_object.npy"))
    print(f"wrote a demo object -> {demo}\nreading it back:")
    args = argparse.Namespace(
        from_file=demo, kind="object", dataset=None, transpose=None,
        uncentered=False, list_datasets=False, raw_dtype=None, raw_shape=None,
        object_nm=30.0, photons=1e6, coherence_nm=0.0, seed=seed)
    from_file(args, out_dir)

    print("\n--- B. atomic coordinates " + "-" * 47)
    args = argparse.Namespace(
        from_md="synthetic", frame=0, reference_frame=None, no_orient=False,
        route="auto", cross_check=True, n=n, object_nm=object_nm,
        photons=1e6, coherence_nm=0.0, grad_demo=True, seed=seed)
    from_md(args, out_dir)

    print(f"\nall outputs in {out_dir}")
    return out_dir


def _cli():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group()
    src.add_argument("--from-file",
                     help="3-D array: .npy .npz .h5 .hdf5 .cxi .nxs .mat .tif .bin")
    src.add_argument("--from-md",
                     help="atomic coordinates (.xyz), or 'synthetic'")

    f = p.add_argument_group("reading a file")
    f.add_argument("--kind", choices=["object", "amplitude", "intensity"],
                   help="what the numbers mean; required for a complex array")
    f.add_argument("--dataset", help="name/path inside .npz/.h5/.cxi/.mat")
    f.add_argument("--transpose", help="axis order, e.g. 1,2,0 for rocking-first")
    f.add_argument("--uncentered", action="store_true",
                   help="q = 0 is at index 0, not the array centre")
    f.add_argument("--list-datasets", action="store_true",
                   help="list 3-D datasets in an HDF5/CXI file and exit")
    f.add_argument("--raw-dtype", help="headerless .bin/.raw only")
    f.add_argument("--raw-shape", help="headerless .bin/.raw only, e.g. 128,128,128")

    m = p.add_argument_group("atomic coordinates")
    m.add_argument("--frame", type=int, default=0, help="MD frame index")
    m.add_argument("--reference-frame", type=int, default=None,
                   help="index of the UNDEFORMED frame; without it the binned "
                        "route has no displacement field and carries shape only")
    m.add_argument("--no-orient", action="store_true",
                   help="coordinates are already in the lab frame")
    m.add_argument("--route", choices=["auto", "direct", "binned"], default="auto",
                   help="direct = exact O(N_atoms*N_q), caps near 10 nm; "
                        "binned = O(N_atoms) envelope, scales to real grains")
    m.add_argument("--cross-check", action="store_true",
                   help="run both routes and correlate, with controls")
    m.add_argument("--grad-demo", action="store_true",
                   help="gradient of a detector-level loss w.r.t. every atom")
    m.add_argument("--n", type=int, default=32, help="array size per axis")

    p.add_argument("--out-dir")
    p.add_argument("--object-nm", type=float, default=6.0,
                   help="object size. The DIRECT atom sum scales as size^3: "
                        "6 nm is ~7k atoms (seconds), 30 nm ~830k (minutes), "
                        "a real 300 nm grain ~1e9 (hopeless -- use --route binned)")
    p.add_argument("--photons", type=float, default=1e6)
    p.add_argument("--coherence-nm", type=float, default=0.0,
                   help="transverse coherence length; 0 disables")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    if not a.from_file and not a.from_md:
        main(a.out_dir, seed=a.seed, n=a.n, object_nm=a.object_nm)
        return

    out_dir = os.path.abspath(a.out_dir or os.path.join(os.getcwd(), "bcdi_output"))
    os.makedirs(out_dir, exist_ok=True)
    print("=" * 74)
    print(f"BCDI FROM DATA -- Au {HKL} at {ENERGY_EV/1e3:.0f} keV")
    print("=" * 74)
    from_file(a, out_dir) if a.from_file else from_md(a, out_dir)


if __name__ == "__main__":
    _cli()
