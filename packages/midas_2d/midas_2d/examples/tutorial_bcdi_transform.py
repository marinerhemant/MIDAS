"""The transformation on its own: complex object -> intensity on the detector.

One file, no CLI, no options. Everything that is actually the transformation and
nothing that is not, so it can be read top to bottom in a couple of minutes and
copied into your own code.

The whole thing is five steps:

    1. geometry    B  = where each array element sits in q
    2. object      psi = s * exp(-i G.u)          on the grid conjugate to B
    3. transform   A  = FFT(psi),  I = |A|^2      <- the actual answer
    4. detector    |F|^2, polarisation, coherence, flux, Poisson counts
    5. inverse     undo the shear, on the object, at the end

Step 3 is one line. Steps 1 and 5 are the ones that catch people out, because
the array is indexed by (detector column, detector row, rocking step) and those
three directions are *not* perpendicular in q -- for the geometry below they are
sheared by 17 degrees, which is the Bragg angle. So a plain FFT does not
reconstruct onto a Cartesian grid, and the grid it does use is

    B^T C = 2*pi*diag(1/N)      ->      C = 2*pi * B^-T diag(1/N)

Sign convention, which has to match or the strain comes out backwards:
``A(q) = sum psi(r) exp(-i q.r)`` is plain ``torch.fft.fftn``, and it pairs with
phase ``= -G.u``.

Needs midas-2d >= 0.3.1:  pip install -U midas-2d

    python -m midas_2d.examples.tutorial_bcdi_transform
"""
import math

import torch

import midas_2d as m2d

DT = torch.float64

# Au (111) at 9 keV, a 400 nm grain on a 64^3 array.
A_AU, LAMBDA_A = 4.0782, 1.3776
D_HKL = A_AU / math.sqrt(3.0)
GRAIN_A = 4000.0
SHAPE = (64, 64, 64)
PIXEL_MM = 0.055


def main():
    # === 1. GEOMETRY ========================================================
    # Detector distance and rocking step chosen for 4x oversampling. This is
    # backwards from the grain size, which is how you pick them at a beamline.
    distance_mm = m2d.detector_distance_for_oversampling(
        LAMBDA_A, GRAIN_A, PIXEL_MM, target=4.0)
    rocking_deg = m2d.rocking_step_for_oversampling(
        LAMBDA_A, D_HKL, GRAIN_A, target=4.0)

    # B: columns are the q-step per unit step of (det column, det row, rocking).
    B = m2d.q_basis(LAMBDA_A, D_HKL, distance_mm=distance_mm,
                    pixel_mm=PIXEL_MM, rocking_step_deg=rocking_deg, dtype=DT)
    # C: the real-space grid the FFT actually pairs with. NOT Cartesian.
    C = m2d.conjugate_real_basis(B, SHAPE)
    G = m2d.bragg_geometry(LAMBDA_A, D_HKL, dtype=DT)["G"]

    print(f"detector distance   {distance_mm:.1f} mm")
    print(f"rocking step        {rocking_deg:.5f} deg")
    print("q-basis angles      "
          + ", ".join(f"{float(a):.1f}" for a in m2d.shear_angles_deg(B))
          + " deg   <- not 90, the grid is sheared")
    print("oversampling        "
          + ", ".join(f"{float(s):.2f}" for s in m2d.oversampling(B, GRAIN_A)))
    print("voxel size          "
          + ", ".join(f"{float(v):.1f}" for v in torch.linalg.norm(C, dim=0))
          + " A")

    # === 2. OBJECT ==========================================================
    # Real-space positions of every voxel: r = C @ m, with m the centred index.
    axes = [torch.arange(n, dtype=DT) - n // 2 for n in SHAPE]
    m = torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1)
    r = m @ C.transpose(0, 1)

    s = (r.abs() <= GRAIN_A / 2).all(dim=-1).to(DT)          # cuboid grain

    # A displacement field. Scaled to a peak phase of 2 rad, which is the sort
    # of number a real grain shows; the envelope model assumes |G.u| is not huge.
    L = GRAIN_A / 2
    u_x = (2.0 / abs(float(G[0]))) * (r[..., 0] / L) ** 3
    u = torch.stack([u_x, torch.zeros_like(u_x), torch.zeros_like(u_x)], dim=-1)

    psi = torch.polar(s, -(u * G).sum(dim=-1) * s)           # phase = -G.u
    print(f"\nobject              {int(s.sum())} voxels, "
          f"peak |G.u| = {float(psi.angle().abs().max()):.2f} rad")

    # === 3. THE TRANSFORMATION ==============================================
    A = m2d.object_to_amplitude(psi)        # fftshift(fftn(ifftshift(psi)))
    I = A.real ** 2 + A.imag ** 2           # <- intensity. That is the answer.

    # And this is where each element of I sits in reciprocal space:
    #   q = B @ (i - N1/2, j - N2/2, k - N3/2)      (deviation from the peak)
    #   Q = q + G                                    (absolute)
    Q = m2d.q_grid(B, SHAPE, offset=G)
    print(f"intensity           {tuple(I.shape)}, "
          f"dynamic range {float(I.max() / I[I > 0].min()):.2e}")

    # The centre of the array must be exactly the Bragg peak.
    c = (SHAPE[0] // 2, SHAPE[1] // 2, SHAPE[2] // 2)
    assert torch.allclose(Q[c], G, atol=1e-12)
    print(f"array centre        Q = {[round(float(x), 4) for x in Q[c]]} 1/A "
          f"= G exactly")

    # Cross-check against the independent ray tracer in midas_2d.instrument.
    # Careful reading the pixel number: project_to_detector intersects a plane
    # perpendicular to the BEAM at z = distance, so at 2theta = 34 deg the Bragg
    # peak lands D*tan(2theta)/p ~ 7800 pixels off-axis. Nothing is wrong -- that
    # is the beam-normal convention, and it is exactly why a real BCDI detector
    # is mounted perpendicular to k_f instead, which is the plane B is built in.
    from midas_2d.instrument import project_to_detector
    pix, valid = project_to_detector(Q, wavelength_A=LAMBDA_A,
                                     distance_mm=distance_mm, pixel_mm=PIXEL_MM)
    assert bool(valid.all()), "some rays scatter backwards"
    expected = distance_mm * math.tan(2 * m2d.bragg_geometry(
        LAMBDA_A, D_HKL, dtype=DT)["theta_rad"]) / PIXEL_MM
    print(f"ray-trace check     beam-normal plane puts it at "
          f"{float(pix[c][0]):.0f} px = D*tan(2theta)/p = {expected:.0f} px")
    assert abs(float(pix[c][0]) - expected) < 1.0

    # === 4. DETECTOR ========================================================
    # |F_hkl|^2, polarisation and solid angle, partial coherence, then flux.
    # detector_signal returns the EXPECTED rate and is differentiable;
    # sample_counts draws Poisson counts and is not.
    from midas_hkls import (Atom, Crystal, Lattice, SpaceGroup,
                            structure_factor_intensity, structure_factors)
    crystal = Crystal(lattice=Lattice(A_AU, A_AU, A_AU, 90, 90, 90),
                      space_group=SpaceGroup.from_number(225),
                      atoms=[Atom(element="Au", fract=(0, 0, 0), B_iso=0.6)])
    F2 = float(structure_factor_intensity(structure_factors(
        crystal.to_torch(), [[1, 1, 1]], wavelength_A=LAMBDA_A))[0])

    rate = m2d.detector_signal(I, Q=Q, wavelength_A=LAMBDA_A,
                               structure_factor_sq=F2,
                               coherence_length_A=1500.0, real_basis=C,
                               photons_per_peak=1e6)
    counts = m2d.sample_counts(rate, generator=torch.Generator().manual_seed(0))
    print(f"detector            |F111|^2 = {F2:.0f} e^2, "
          f"peak rate {float(rate.max()):.3g}, "
          f"total counts {float(counts.sum()):.4g}")

    # === 5. THE INVERSE =====================================================
    # Undo the shear. On the OBJECT, at the end -- never on the measured
    # intensity before phasing. Here we un-shear psi itself; in a real
    # reconstruction you would un-shear whatever phase retrieval returned.
    lab = m2d.sheared_to_lab(psi.abs(), C)
    ext = []
    for d in range(3):
        o = tuple(i for i in range(3) if i != d)
        nz = torch.nonzero((lab["obj"] > 0.5).amax(dim=o[1]).amax(dim=o[0])).flatten()
        ext.append((int(nz[-1] - nz[0]) + 1) * lab["voxel_A"])
    # The three extents come out equal: the shear is gone. They sit ~1 voxel
    # over the true edge because thresholding a mask costs half a voxel a face,
    # not because anything is still sheared.
    print(f"\nun-sheared to lab   voxel {lab['voxel_A']:.1f} A, "
          f"grain measures {[round(e) for e in ext]} A "
          f"(true edge {GRAIN_A:.0f} A, +1 voxel from thresholding)")
    assert max(ext) / min(ext) - 1 < 0.02, "still sheared"

    return {"I": I, "rate": rate, "counts": counts, "Q": Q, "B": B, "C": C}


if __name__ == "__main__":
    main()
