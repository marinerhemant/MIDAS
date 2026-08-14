"""Differentiable FP-DFXM reconstruction recovers phase + super-resolution.

Locks the core proven in the FP MVP (fp_dfxm_mvp.py: phase RMS 7.4 deg, vortex
winding 1.00, super-resolution 28x): from a stack of aperture-shifted images of a
complex phantom (amplitude grating finer than the single-aperture NA + a phase
domain-wall + a dislocation phase vortex), reconstruct_fp recovers (1) the phase to
low RMS, (2) the vortex winding (g.b), and (3) spectral content past the single-
aperture cutoff that the central image alone cannot see. Small grid / few iters so
it runs in CI; the full-resolution demo lives in the analysis dir.
"""
import math

import torch

from midas_dfxm import aperture_grid, disk_pupil, fp_forward, reconstruct_fp
from midas_dfxm.fourier_ptychography import align_global_phase

DT = torch.float64
CDT = torch.complex128
N = 64
GRATING_PX = 5.0          # 0.20 cyc/px: outside single-NA cutoff 0.10, inside synth ~0.26
K_PUPIL = 0.10


def _phantom():
    ax = torch.arange(N, dtype=DT) - N / 2
    X, Y = torch.meshgrid(ax, ax, indexing="xy")
    R = torch.hypot(X, Y)
    support = (R < 0.42 * N).to(DT)
    grating = 0.5 + 0.5 * torch.cos(2 * math.pi * X / GRATING_PX)
    amp = support * (0.6 + 0.4 * grating)
    wall = 0.6 * math.pi * torch.tanh(X / 2.0)
    vortex = torch.atan2(Y - 4.0, X + 4.0)            # winding +1 about (-4, 4)
    O = (amp * torch.exp(1j * (wall + vortex))).to(CDT)
    return O, support.bool()


def _winding(ph, xc, yc, r=10):
    ang = torch.linspace(0, 2 * math.pi, 241)[:-1]
    xs = (N / 2 + xc + r * torch.cos(ang)).round().long().clamp(0, N - 1)
    ys = (N / 2 + yc + r * torch.sin(ang)).round().long().clamp(0, N - 1)
    p = ph[ys, xs]
    d = torch.angle(torch.exp(1j * (p.roll(-1) - p)))
    return float(d.sum() / (2 * math.pi))


def _spectral_amp(field):
    rows = field.abs()[N // 2 - 4:N // 2 + 4]
    F = torch.fft.fft(rows, dim=1).abs().mean(0)
    return float(F[round(N / GRATING_PX)])


def test_fp_recovers_phase_winding_and_superresolution():
    torch.manual_seed(0)
    O_true, mask = _phantom()
    pupil = disk_pupil(N, K_PUPIL, dtype=DT)
    shifts = aperture_grid(half=3, step_px=int(0.06 * N))    # 7x7 overlapping apertures

    I_meas = fp_forward(O_true, pupil, shifts).detach()
    I_meas = (I_meas + 0.002 * I_meas.mean() * torch.randn_like(I_meas)).clamp_min(0)

    rec = reconstruct_fp(I_meas, pupil, shifts, iters=250, lr=0.05)
    rec = align_global_phase(rec, O_true, mask)

    # (1) phase recovered to low RMS on the support
    dphi = torch.angle(torch.exp(1j * (torch.angle(rec) - torch.angle(O_true))))
    ph_rms_deg = float(dphi[mask].pow(2).mean().sqrt()) * 180 / math.pi
    assert ph_rms_deg < 20.0, f"phase RMS {ph_rms_deg:.1f} deg too high"

    # (2) dislocation phase vortex winding recovered (g.b)
    w_true = _winding(torch.angle(O_true), -4, 4)
    w_rec = _winding(torch.angle(rec), -4, 4)
    assert abs(w_true - 1.0) < 0.2 and abs(w_rec - w_true) < 0.3, \
        f"winding true {w_true:.2f} rec {w_rec:.2f}"

    # (3) super-resolution: the grating sits beyond the single-aperture cutoff, so the
    # central image is blind to it; FP recovers it.
    central = torch.fft.ifft2(pupil * torch.fft.fft2(O_true))
    s_single = _spectral_amp(central.abs())
    s_fp = _spectral_amp(rec)
    s_true = _spectral_amp(O_true)
    assert s_fp > 5.0 * s_single, f"no super-resolution: FP {s_fp:.2f} vs single {s_single:.2f}"
    assert s_fp > 0.5 * s_true, f"grating under-recovered: FP {s_fp:.2f} vs true {s_true:.2f}"


def test_fp_recovers_real_dislocation_exit_wave_winding():
    """FP recovers the g.b phase vortex from a PHYSICALLY-GENERATED exit wave --
    the anisotropic Stroh displacement of a real edge dislocation, not a phantom.

    dislocation_exit_wave produces amplitude sqrt(R) (Bragg Lorentzian) x phase g.u.
    Reflection (2,0,0) has g.b = 1 (visible, charge-1 vortex); (2,2,0) has g.b = 0
    (invisible) as a negative control. The reconstruction must recover both windings
    from the aperture-shifted image stack alone. Proven in analysis fp_from_tt_exitwave.py
    (true/rec winding +1.00/+1.00, phase RMS 3.4 deg).

    Uses an ample-overlap config (49 apertures, step 0.07*M, pupil 0.14) that reaches the
    global minimum. Vortex recovery is NOT robust to the default zero-phase init under weak
    overlap / an under-resolved core (winding-0 stagnation) -- see reconstruct_fp's LIMITATION
    docstring. This test also asserts the fit reached the noise floor, so a future regression
    into the stall basin (loss ~40x higher, winding 0) fails loudly rather than silently.
    """
    import math as _m

    from midas_dfxm import cubic_stiffness, stroh_dislocation
    from midas_dfxm.coherence import dislocation_exit_wave

    torch.manual_seed(0)
    M = 64
    fov, a = 3.0, 3.6156
    px = fov / M
    latc = torch.tensor([a] * 3 + [90.0] * 3, dtype=DT)
    C = cubic_stiffness(168.4, 121.4, 75.4, dtype=DT)
    pupil = disk_pupil(M, 0.14, dtype=DT)
    shifts = aperture_grid(half=3, step_px=int(0.07 * M))

    def _exit(hkl):
        d = stroh_dislocation(C, burgers=(1, -1, 0), slip_normal=(1, 1, 1),
                              character="edge", burgers_length_A=a / _m.sqrt(2),
                              core_radius_um=0.05, core_model="compact")
        g = (torch.arange(M, dtype=DT) - M / 2) * px
        GX, GY = torch.meshgrid(g, g, indexing="xy")
        flat = torch.stack([GX.reshape(-1), GY.reshape(-1),
                            torch.zeros(M * M, dtype=DT)], -1)
        O = dislocation_exit_wave(flat @ d.M, d, hkl, lattice_params=latc,
                                  res_width=0.02, shape=(M, M))
        return O.to(CDT)

    def _wind(ph, r=12):
        ang = torch.linspace(0, 2 * _m.pi, 241)[:-1]
        xs = (M / 2 + r * torch.cos(ang)).round().long().clamp(0, M - 1)
        ys = (M / 2 + r * torch.sin(ang)).round().long().clamp(0, M - 1)
        p = ph[ys, xs]
        return float(torch.angle(torch.exp(1j * (p.roll(-1) - p))).sum() / (2 * _m.pi))

    def _n_singularities(ph):
        """Number of phase singularities (nonzero plaquette windings) in the field.
        The recovered winding is only trustworthy when the reconstruction is CLEAN;
        a /verify pass showed ill-conditioned configs sprout ~100 spurious vortices."""
        d1 = torch.angle(torch.exp(1j * (ph[1:, :-1] - ph[:-1, :-1])))
        d2 = torch.angle(torch.exp(1j * (ph[1:, 1:] - ph[1:, :-1])))
        d3 = torch.angle(torch.exp(1j * (ph[:-1, 1:] - ph[1:, 1:])))
        d4 = torch.angle(torch.exp(1j * (ph[:-1, :-1] - ph[:-1, 1:])))
        return int(((d1 + d2 + d3 + d4) / (2 * _m.pi)).round().abs().gt(0.5).sum())

    for hkl, expect in [((2, 0, 0), 1.0), ((2, 2, 0), 0.0)]:
        O_true = _exit(hkl)
        I = fp_forward(O_true, pupil, shifts).detach()
        I = (I + 0.002 * I.mean() * torch.randn_like(I)).clamp_min(0)
        rec_raw = reconstruct_fp(I, pupil, shifts, iters=300, lr=0.05)
        # the fit must reach the noise floor; a stall sits ~40x higher (loss ~1e-3)
        loss = float(((fp_forward(rec_raw, pupil, shifts).clamp_min(0).sqrt()
                       - I.clamp_min(0).sqrt()) ** 2).mean())
        assert loss < 5e-4, f"{hkl}: solve stalled (loss {loss:.1e}); global min not reached"
        rec = align_global_phase(rec_raw, O_true,
                                 O_true.abs() > 0.05 * float(O_true.abs().max()))
        # ample overlap => CLEAN reconstruction: singularity count matches the true object.
        # This is what makes the winding here trustworthy (unlike the weak-overlap regime).
        n_true, n_rec = _n_singularities(torch.angle(O_true)), _n_singularities(torch.angle(rec))
        assert n_rec == n_true, f"{hkl}: {n_rec} singularities (true {n_true}); contaminated"
        w_true, w_rec = _wind(torch.angle(O_true)), _wind(torch.angle(rec))
        assert abs(w_true - expect) < 0.2, f"{hkl}: true winding {w_true:.2f} != {expect}"
        assert abs(w_rec - expect) < 0.3, f"{hkl}: recovered winding {w_rec:.2f} != {expect}"


def test_multistart_escapes_amplitude_stall_but_not_charge():
    """`restarts` reliably escapes the AMPLITUDE stall (reaches the loss floor) but does
    NOT reliably recover the vortex CHARGE in the weak-overlap regime -- the honest result
    a /verify pass established (2026-08). Single-start stalls at loss ~4.5e-3; restarts=6
    reaches the floor ~4e-5. But reaching the floor is winding-degenerate here: the phase is
    contaminated by many spurious singularities and the core charge is NOT reliably +1, so
    this test deliberately does NOT assert a recovered winding (asserting it would be
    seed-cherry-picking). It also locks backward compatibility of restarts=1.
    """
    import math as _m

    from midas_dfxm import cubic_stiffness, stroh_dislocation
    from midas_dfxm.coherence import dislocation_exit_wave

    M, fov, a = 64, 8.0, 3.6156                 # weak/moderate-overlap stall regime (pupil 0.12)
    px = fov / M
    latc = torch.tensor([a] * 3 + [90.0] * 3, dtype=DT)
    C = cubic_stiffness(168.4, 121.4, 75.4, dtype=DT)
    d = stroh_dislocation(C, burgers=(1, -1, 0), slip_normal=(1, 1, 1), character="edge",
                          burgers_length_A=a / _m.sqrt(2), core_radius_um=0.05,
                          core_model="compact")
    g = (torch.arange(M, dtype=DT) - M / 2) * px
    GX, GY = torch.meshgrid(g, g, indexing="xy")
    flat = torch.stack([GX.reshape(-1), GY.reshape(-1), torch.zeros(M * M, dtype=DT)], -1)
    O_true = dislocation_exit_wave(flat @ d.M, d, (2, 0, 0), lattice_params=latc,
                                   res_width=0.02, shape=(M, M)).to(CDT)
    pupil = disk_pupil(M, 0.12, dtype=DT)
    shifts = aperture_grid(half=3, step_px=int(0.05 * M))

    torch.manual_seed(0)
    I = fp_forward(O_true, pupil, shifts).detach()
    I = (I + 0.002 * I.mean() * torch.randn_like(I)).clamp_min(0)

    # single start stalls: loss well above the floor
    _, loss1 = reconstruct_fp(I, pupil, shifts, iters=400, lr=0.05, return_loss=True)
    assert loss1 > 1e-3, f"expected a stall for this config, got loss {loss1:.1e}"

    # multi-start reaches the amplitude floor (the reliable capability)
    _, lossM = reconstruct_fp(I, pupil, shifts, iters=400, lr=0.05,
                              restarts=6, seed=0, return_loss=True)
    assert lossM < 2e-4, f"multi-start did not reach the amplitude floor (loss {lossM:.1e})"
    # NB: no winding assertion here -- charge is not reliably recovered in this regime.

    # restarts=1 == the single-start path exactly (backward compatibility)
    r_a = reconstruct_fp(I, pupil, shifts, iters=50, lr=0.05)
    r_b = reconstruct_fp(I, pupil, shifts, iters=50, lr=0.05, restarts=1)
    assert torch.allclose(r_a, r_b)


def test_fp_forward_shapes_and_differentiability():
    O_true, _ = _phantom()
    pupil = disk_pupil(N, K_PUPIL, dtype=DT)
    shifts = aperture_grid(half=1, step_px=4)               # 3x3 = 9
    obj = O_true.clone().requires_grad_(True)
    imgs = fp_forward(obj, pupil, shifts)
    assert imgs.shape == (9, N, N)
    assert (imgs >= 0).all()
    imgs.sum().backward()
    assert obj.grad is not None and torch.isfinite(obj.grad).all()
