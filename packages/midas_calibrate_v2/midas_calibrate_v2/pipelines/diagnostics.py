"""Calibration diagnostics — auto-detection of the three known
failure modes from B1 (basis incompleteness), B6 (basin escape),
and v1's silent strain blow-up.

Three lightweight gates:

  1. ``cross_validation_gate(fits, unpacked, n_train_rings)`` — splits
     the fits by ring and asks whether the held-out residual is
     statistically distinguishable from the training residual.  A
     clean pass means the analytical model generalises.

  2. ``strain_cap_check(history, threshold_uE)`` — flags runs whose
     converged strain exceeds a calibrant threshold.  Catches every
     basin escape in the B6 sweep (strain ≥ 800 μϵ in failure cases).

  3. ``basin_check(v1_init, unpacked)`` — measures the Lsd / BC drift
     between seed and converged values.  Drift outside the B6 basin
     (±0.3 % Lsd, ±1.5 px BC) is suspicious — either the seed was
     way off, or LM walked into a side basin.

Each gate returns a :class:`DiagnosticResult` with a ``severity`` of
``"ok"``, ``"warn"``, or ``"fail"`` plus a one-line explanation.  Use
:func:`run_all_gates` to evaluate all three in one call.

One more gate runs **before** any of those, and needs no fit at all:

  4. ``seed_provenance_gate(...)`` — was the geometry seeded by the validated
     seeder, or by the last-resort fallback?  ``basin_check`` structurally
     cannot answer this: it flags LARGE seed-to-MAP drift, so a refiner that
     never leaves a wrong seed scores a clean pass.

  5. ``detector_scope_gate(...)`` — pure geometry.  Does the calibrant put
     enough rings on this panel, at this distance and wavelength, for a
     geometry to be determinable?  Everything above assumes the answer is yes.
     When it is not, the fitter still converges — onto parasitic scatter — and
     every downstream gate then judges a meaningless answer.  Measured on the
     1-ID archive: of 42 exposures this gate halts, 26 had produced a
     plausible-looking calibration, including a SAXS detector (25 × 63 mm at
     1.8–3.3 m) whose CeO2 (111) ring sits six times past the panel edge.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, List

import torch

from ..forward.panels import PanelLayout
from ..loss.pseudo_strain import pseudo_strain_residual


@dataclass
class DiagnosticResult:
    name: str
    severity: str            # "ok" | "warn" | "fail"
    message: str
    metrics: Dict[str, float]


def seed_provenance_gate(
    *,
    seed_method: str,
    seed_note: str = "",
    seed_BC_y: Optional[float] = None,
    seed_BC_z: Optional[float] = None,
    NrPixelsY: Optional[int] = None,
    NrPixelsZ: Optional[int] = None,
) -> DiagnosticResult:
    """Was the geometry seeded by the validated seeder, or by the last resort?

    This exists because :func:`basin_check` cannot see the failure it is most
    likely to be asked about.  It measures seed-to-MAP drift and flags LARGE
    drift, so a refiner that never moves scores a clean pass — which is exactly
    what happens when the seed is wrong: LM has no gradient telling it to leave.
    Measured on the 1-ID archive: a frame seeded by the fallback converged with
    ``ΔLsd=+0 µm, ΔBC=0.0 px`` and collected a green ``basin_check`` tick
    alongside a beam centre 103 px off the edge of the detector.

    So the provenance of the seed is itself a diagnostic, independent of
    anything the fit reports about itself.
    """
    off_panel = False
    if None not in (seed_BC_y, seed_BC_z, NrPixelsY, NrPixelsZ):
        off_panel = not (0 <= seed_BC_y <= NrPixelsY - 1
                         and 0 <= seed_BC_z <= NrPixelsZ - 1)
    metrics = {"off_panel": float(off_panel)}

    if seed_method == "fallback":
        extra = (" The seed beam centre is OFF THE PANEL, which the refiner will "
                 "not recover from." if off_panel else "")
        return DiagnosticResult(
            name="seed_provenance", severity="fail",
            message=("seeded by the LAST-RESORT chord-only arc seed, not the "
                     "validated seeder (%s). That fallback is a materially "
                     "weaker estimate and the refiner tends to stay where it is "
                     "put, so basin_check's zero drift is not reassurance here.%s"
                     % (seed_note or "reason not recorded", extra)),
            metrics=metrics)

    if off_panel:
        return DiagnosticResult(
            name="seed_provenance", severity="warn",
            message=("seeded by %s, but the seed beam centre lies off the panel "
                     "— legitimate for a wedge geometry, wrong for a centred one; "
                     "confirm which this is" % seed_method),
            metrics=metrics)

    if seed_method in ("make_seed", "user"):
        return DiagnosticResult(
            name="seed_provenance", severity="ok",
            message="seeded by %s (%s)" % (seed_method, seed_note or "no detail"),
            metrics=metrics)

    return DiagnosticResult(
        name="seed_provenance", severity="warn",
        message="seed provenance not recorded (seed_method=%r)" % seed_method,
        metrics=metrics)


def detector_scope_gate(
    *,
    wavelength_A: float,
    Lsd_um: float,
    pxY_um: float,
    NrPixelsY: int,
    NrPixelsZ: int,
    BC_y: Optional[float] = None,
    BC_z: Optional[float] = None,
    pxZ_um: Optional[float] = None,
    lattice_a_A: float = 5.41153,          # CeO2
    space_group: int = 225,
    min_rings: int = 3,
) -> DiagnosticResult:
    """Preflight: can this detector see enough of the calibrant to be calibrated?

    Pure geometry — no image, no fit, no residual — so it cannot be fooled by a
    fitter converging onto parasitic scatter, and it costs nothing to run first.

    A ring at Bragg angle 2θ lands at ``R = Lsd·tan(2θ)``.  The panel reaches at
    most its beam-centre-to-farthest-corner distance, so any ring beyond that is
    simply not recorded.  With fewer than ``min_rings`` on the panel the
    geometry is not determinable, however well the optimiser reports it did.

    ``BC_y`` / ``BC_z`` default to the panel centre, which is the most generous
    assumption: it maximises the reach and so makes the gate conservative.

    Ring positions come from ``midas_hkls`` — the same generator the calibration
    itself uses — not from a hardcoded d-spacing list.
    """
    import math

    from midas_hkls import SpaceGroup, Lattice, generate_hkls

    pz = float(pxZ_um) if pxZ_um else float(pxY_um)
    bcy = float(NrPixelsY) / 2.0 if BC_y is None else float(BC_y)
    bcz = float(NrPixelsZ) / 2.0 if BC_z is None else float(BC_z)

    dy = max(bcy, NrPixelsY - bcy) * float(pxY_um)
    dz = max(bcz, NrPixelsZ - bcz) * pz
    R_reach = math.hypot(dy, dz)
    two_theta_max = math.degrees(math.atan(R_reach / float(Lsd_um)))

    refs = generate_hkls(SpaceGroup.from_number(space_group),
                         Lattice(lattice_a_A, lattice_a_A, lattice_a_A, 90, 90, 90),
                         wavelength_A=float(wavelength_A),
                         two_theta_max_deg=max(two_theta_max, 1e-6))
    radii = sorted(float(Lsd_um) * math.tan(math.radians(float(r.two_theta_deg)))
                   for r in refs)
    on_panel = [R for R in radii if R <= R_reach]
    n = len(on_panel)

    # innermost ring the calibrant can produce at all, for the message
    sin_t = float(wavelength_A) / (2.0 * lattice_a_A / math.sqrt(3.0))
    R_first = (float(Lsd_um) * math.tan(2.0 * math.asin(sin_t))
               if sin_t < 1.0 else float("inf"))

    metrics = {"n_rings_on_panel": float(n),
               "R_reach_mm": R_reach / 1000.0,
               "R_innermost_mm": R_first / 1000.0,
               "two_theta_max_deg": two_theta_max}

    if n < min_rings:
        return DiagnosticResult(
            name="detector_scope", severity="fail",
            message=(
                "only %d calibrant ring(s) reach this panel — innermost ring at "
                "R=%.0f mm, panel reaches %.0f mm at Lsd=%.0f mm. The geometry is "
                "NOT determinable from this exposure; a fit will converge on "
                "parasitic scatter. Check the detector is the one the calibrant "
                "was shot for (a small-angle detector at a long distance sees no "
                "powder rings) and that Lsd and the wavelength are right."
                % (n, R_first / 1000.0, R_reach / 1000.0, float(Lsd_um) / 1000.0)),
            metrics=metrics)

    sev = "warn" if n < min_rings + 2 else "ok"
    return DiagnosticResult(
        name="detector_scope", severity=sev,
        message=("%d calibrant rings reach the panel (innermost %.0f mm, reach "
                 "%.0f mm, 2θ_max %.1f°)"
                 % (n, R_first / 1000.0, R_reach / 1000.0, two_theta_max)),
        metrics=metrics)


def cross_validation_gate(
    fits,                                 # FittedDataset with ring_idx
    unpacked: Dict[str, torch.Tensor],
    *,
    n_train_rings: Optional[int] = None,  # default: split at floor(n_rings * 2/3)
    panel_layout: Optional[PanelLayout] = None,
    fail_med_ratio: float = 1.5,         # test/train median > this → fail
    warn_med_ratio: float = 1.2,         # test/train median > this → warn
    ks_p_threshold: float = 0.01,        # KS p-value below this is suspicious
) -> DiagnosticResult:
    """Held-out-ring cross-validation gate.

    Calibrate on rings 0..N_train-1 implicit in ``unpacked`` (the caller
    should have done this); evaluate the residual on *all* fits at the
    converged geometry, then split by ring index and compare.

    Note: this gate does NOT re-fit.  It assumes ``unpacked`` is the
    converged MAP and the fits include held-out rings that were
    *not* used in the LM.  To use it as a true CV gate, call the
    calibration with ``spec.max_ring_number = n_train_rings`` (so the
    cake produces only train rings), then call this function with
    a separate fits dataset that includes the test rings.

    For an in-line check on a single calibration that used all rings,
    this gate degenerates to comparing the upper-third of rings vs the
    lower two-thirds — still informative, less rigorous.
    """
    rho_d = fits.rho_d
    with torch.no_grad():
        r_all = pseudo_strain_residual(
            fits.Y_pix, fits.Z_pix, fits.ring_two_theta_deg, unpacked,
            rho_d=rho_d, panel_layout=panel_layout,
            panel_idx=getattr(fits, "panel_idx", None),
        )
    r_uE = (r_all.abs() * 1e6).cpu().numpy()
    ring_arr = fits.ring_idx.cpu().numpy()
    rings_present = sorted(set(ring_arr))
    if len(rings_present) < 4:
        return DiagnosticResult(
            name="cross_validation",
            severity="warn",
            message=f"too few rings ({len(rings_present)}) to run CV gate",
            metrics={"n_rings": float(len(rings_present))},
        )

    if n_train_rings is None:
        n_train_rings = max(1, int(len(rings_present) * 2 / 3))
    train_mask = ring_arr < n_train_rings
    test_mask = ~train_mask
    if test_mask.sum() < 5:
        return DiagnosticResult(
            name="cross_validation",
            severity="warn",
            message=f"only {int(test_mask.sum())} test fits — split too narrow",
            metrics={"n_test": float(test_mask.sum())},
        )

    train_uE = r_uE[train_mask]
    test_uE = r_uE[test_mask]
    med_train = float(_np_median(train_uE))
    med_test = float(_np_median(test_uE))
    ratio = med_test / max(med_train, 1e-3)

    ks_p = _ks_2samp_p(train_uE, test_uE)

    if ratio > fail_med_ratio and ks_p < ks_p_threshold:
        sev = "fail"
        msg = (f"held-out median {med_test:.2f} μϵ vs train {med_train:.2f} μϵ "
               f"({ratio:+.2f}× , KS p={ks_p:.2e}) — analytical basis is "
               f"incomplete on rings ≥{n_train_rings}")
    elif ratio > warn_med_ratio and ks_p < ks_p_threshold:
        sev = "warn"
        msg = (f"held-out residual {ratio:.2f}× train, KS p={ks_p:.2e} — "
               f"borderline systematic on rings ≥{n_train_rings}")
    else:
        sev = "ok"
        msg = (f"held-out residual {ratio:.2f}× train, KS p={ks_p:.2e} — "
               f"model generalises across rings")

    return DiagnosticResult(
        name="cross_validation",
        severity=sev,
        message=msg,
        metrics={
            "med_train_uE": med_train,
            "med_test_uE": med_test,
            "ratio": ratio,
            "ks_pvalue": ks_p,
            "n_train": float(train_mask.sum()),
            "n_test": float(test_mask.sum()),
            "n_train_rings": float(n_train_rings),
        },
    )


def strain_cap_check(
    history,                              # list of IterRecord with mean_strain_uE
    *,
    threshold_uE: float = 100.0,
    warn_uE: float = 50.0,
    n_rings: Optional[int] = None,
    min_rings_for_ok: int = 3,
) -> DiagnosticResult:
    """Strain-cap rejection.  B6 showed that all 24 basin-escape failures
    have converged strain ≥ 800 μϵ.  A 100 μϵ cap rejects every escape
    while never tripping on a real calibrant (all four reference
    datasets converge at < 35 μϵ).

    ``n_rings`` is the number of distinct rings the fit actually rested on.
    A low strain computed from one or two rings means nothing — there are as
    many free parameters as constraints, so the residual can be driven to zero
    at an arbitrary geometry.

    MEASURED: a synthetic CeO2 image whose seeder locked onto a SINGLE ring
    converged at **11.92 μϵ with Lsd wrong by a factor of 7.9**
    (300 000 µm true, 2 362 886 µm fitted). The old gate returned "ok" on
    that, and 11.92 μϵ reads as an excellent calibration to anyone applying
    the <100 μϵ rule. The ring count was already known to the other gates --
    ``cross_validation`` said "too few rings (1)" in the same run -- but the
    headline number did not inherit their caution. It does now: below
    ``min_rings_for_ok`` distinct rings this can return at best "warn", no
    matter how small the strain.
    """
    if not history:
        return DiagnosticResult(
            name="strain_cap",
            severity="warn",
            message="no iterations recorded",
            metrics={"strain_uE": float("nan")},
        )
    final = history[-1]
    strain = float(getattr(final, "mean_strain_uE", float("nan")))
    if strain != strain:                   # NaN
        return DiagnosticResult(
            name="strain_cap",
            severity="fail",
            message="converged strain is NaN — LM diverged",
            metrics={"strain_uE": float("nan")},
        )
    if strain > threshold_uE:
        return DiagnosticResult(
            name="strain_cap",
            severity="fail",
            message=(f"converged strain {strain:.1f} μϵ exceeds calibrant cap "
                     f"{threshold_uE:.0f} μϵ — likely basin escape (B6 failure mode)"),
            metrics={"strain_uE": strain, "threshold_uE": threshold_uE},
        )
    if strain > warn_uE:
        return DiagnosticResult(
            name="strain_cap",
            severity="warn",
            message=(f"converged strain {strain:.1f} μϵ above warn level "
                     f"{warn_uE:.0f} μϵ — review residual distribution"),
            metrics={"strain_uE": strain, "warn_uE": warn_uE},
        )
    if n_rings is not None and n_rings < min_rings_for_ok:
        # A small strain on this little support is not evidence of a good
        # calibration; it is evidence that the residual had nothing to
        # disagree with. Do not let the number read as "ok".
        return DiagnosticResult(
            name="strain_cap",
            severity="warn",
            message=(f"converged strain {strain:.2f} μϵ, but the fit rests on "
                     f"only {n_rings} ring{'s' if n_rings != 1 else ''} "
                     f"(< {min_rings_for_ok}) — a low strain on this little "
                     f"support does not mean the geometry is right; check the "
                     f"ring overlay before using it"),
            metrics={"strain_uE": strain, "n_rings": float(n_rings),
                     "min_rings_for_ok": float(min_rings_for_ok)},
        )
    return DiagnosticResult(
        name="strain_cap",
        severity="ok",
        message=(f"converged strain {strain:.2f} μϵ — within calibrant range"
                 + (f" ({n_rings} rings)" if n_rings is not None else "")),
        metrics={"strain_uE": strain,
                 **({"n_rings": float(n_rings)} if n_rings is not None else {})},
    )


def basin_check(
    v1_init,                              # V1Params at the seed (before LM)
    unpacked: Dict[str, torch.Tensor],
    *,
    fail_lsd_pct: float = 5.0,           # |ΔLsd|/Lsd > this → fail
    warn_lsd_pct: float = 1.0,           # > this → warn (B6 basin edge)
    fail_bc_px: float = 50.0,            # |ΔBC| > this → fail
    warn_bc_px: float = 5.0,             # > this → warn (B6 basin edge)
) -> DiagnosticResult:
    """Compare seed (v1_init) vs converged geometry.  B6 found that
    LM always converges (100 % rate across the 48-trial sweep) but
    walks into a side basin once the seed perturbation exceeds
    ±1 % Lsd or ±5 px BC.  This gate flags runs where the converged
    geometry sits far from the seed — usually a sign that the v1
    seed was stale or LM jumped basins.
    """
    Lsd_seed = float(v1_init.Lsd)
    Lsd_final = float(unpacked["Lsd"])
    dLsd = Lsd_final - Lsd_seed
    pct = 100.0 * abs(dLsd) / max(abs(Lsd_seed), 1.0)

    BCy_seed = float(v1_init.BC_y)
    BCz_seed = float(v1_init.BC_z)
    BCy_final = float(unpacked["BC_y"])
    BCz_final = float(unpacked["BC_z"])
    dBC = ((BCy_final - BCy_seed) ** 2 + (BCz_final - BCz_seed) ** 2) ** 0.5

    metrics = {
        "Lsd_seed": Lsd_seed, "Lsd_final": Lsd_final,
        "delta_Lsd_um": dLsd, "delta_Lsd_pct": pct,
        "BC_seed_y": BCy_seed, "BC_seed_z": BCz_seed,
        "BC_final_y": BCy_final, "BC_final_z": BCz_final,
        "delta_BC_px": dBC,
    }
    if pct > fail_lsd_pct or dBC > fail_bc_px:
        return DiagnosticResult(
            name="basin_check",
            severity="fail",
            message=(f"converged geometry walked far from seed: "
                     f"ΔLsd={dLsd:+.0f} μm ({pct:+.2f} %), ΔBC={dBC:.1f} px — "
                     f"likely basin escape (B6 ≥ 1× failure regime)"),
            metrics=metrics,
        )
    if pct > warn_lsd_pct or dBC > warn_bc_px:
        return DiagnosticResult(
            name="basin_check",
            severity="warn",
            message=(f"seed-to-MAP drift: ΔLsd={dLsd:+.0f} μm ({pct:+.2f} %), "
                     f"ΔBC={dBC:.1f} px — outside the safe basin "
                     f"(±{warn_lsd_pct:.1f} % / ±{warn_bc_px:.0f} px), "
                     f"verify before downstream use"),
            metrics=metrics,
        )
    return DiagnosticResult(
        name="basin_check",
        severity="ok",
        message=(f"seed-to-MAP drift: ΔLsd={dLsd:+.0f} μm ({pct:+.2f} %), "
                 f"ΔBC={dBC:.1f} px — within safe basin"),
        metrics=metrics,
    )


def _observed_eta_deg(fits, unpacked: Dict[str, torch.Tensor]) -> "np.ndarray":
    """Azimuth of every fitted point, in the forward model's own convention."""
    import numpy as np
    from ..forward.geometry import pixel_to_REta
    from ..forward.distortion import build_p_coeffs

    dt = fits.Y_pix.dtype
    zero = torch.zeros((), dtype=dt, device=fits.Y_pix.device)

    def g(name, default=None):
        v = unpacked.get(name)
        if v is None:
            return zero if default is None else torch.as_tensor(
                default, dtype=dt, device=fits.Y_pix.device)
        return v

    with torch.no_grad():
        out = pixel_to_REta(
            fits.Y_pix, fits.Z_pix,
            Lsd=g("Lsd"), BC_y=g("BC_y"), BC_z=g("BC_z"),
            tx=g("tx"), ty=g("ty"), tz=g("tz"),
            p_coeffs=build_p_coeffs(unpacked, dtype=dt,
                                    device=fits.Y_pix.device),
            parallax=g("Parallax"),
            pxY=g("pxY", 200.0), pxZ=g("pxZ", 200.0), rho_d=fits.rho_d,
        )
    return out.eta_deg.detach().cpu().numpy().astype(float)


def azimuth_coverage_gate(
    fits,
    unpacked: Dict[str, torch.Tensor],
    *,
    spec=None,
    warn_fraction: float = 0.50,
    fail_fraction: float = 0.25,
    bin_deg: float = 5.0,
) -> DiagnosticResult:
    """Is there enough azimuth to determine what is being refined?

    An off-axis or offset detector sees only a wedge of every Debye-Scherrer
    ring.  Over a narrow wedge the azimuthal distortion harmonics
    (``a1..a6`` / ``phi1..phi6``) stop being separable from the geometry that
    already produces azimuthal modulation — the beam centre generates a
    1-fold term and the tilts a 2-fold one — and from each other.  The fit
    then puts the harmonics on their bounds and the alternating E<->M loop
    stops converging.

    This is not a hypothetical.  On a 1-ID ge1 frame whose beam centre lies
    off the panel corner, the rings are covered over ~70 deg of azimuth, and
    the shipped calibration had **7 of its 15 distortion coefficients pinned
    at +-0.002**.  Refitting with the harmonics frozen converged immediately;
    refitting with them free oscillated between 84 and 4692 ue.

    Note a second calibrant does NOT help here — both powders illuminate the
    same wedge, so multi-phase adds rows to the Jacobian but no new azimuth.
    """
    import numpy as np
    eta = _observed_eta_deg(fits, unpacked)
    if eta.size == 0:
        return DiagnosticResult(
            name="azimuth_coverage", severity="warn",
            message="no fitted points — cannot assess azimuthal coverage",
            metrics={})

    # Coverage from the GAPS between neighbouring fitted azimuths, not from
    # occupancy of fixed bins.  Bin occupancy conflates "the detector does not
    # see this azimuth" with "the cake was sampled coarsely": at EtaBinSize=10
    # deg and 5 deg bins, a fully-covered ring reports every other bin empty
    # and the contiguous-arc metric collapses to one bin.
    # Distinct azimuths only: every ring contributes a fit at the SAME set of
    # eta bins, so the raw list is mostly duplicates and their zero gaps would
    # drag the median to 0 and make every real gap look like a hole.
    a = np.unique(np.round(np.mod(eta, 360.0), 6))
    if a.size < 2:
        return DiagnosticResult(
            name="azimuth_coverage", severity="warn",
            message=(f"only {a.size} distinct azimuth(s) among "
                     f"{eta.size} fits — cannot assess coverage"),
            metrics={"covered_fraction": 0.0, "n_distinct_eta": float(a.size)})
    gaps = np.diff(np.concatenate([a, a[:1] + 360.0]))
    med_gap = float(np.median(gaps))
    # A gap is a genuine HOLE only if it is much larger than the typical
    # spacing between neighbouring fits.
    hole_deg = max(5.0 * med_gap, bin_deg)
    holes = gaps > hole_deg
    covered_deg = float(360.0 - gaps[holes].sum()) if holes.any() else 360.0
    frac = covered_deg / 360.0

    # Longest contiguous arc = longest run of non-hole gaps, wrap-aware.
    if not holes.any():
        span_deg = 360.0
    else:
        run, best = 0.0, 0.0
        for g, is_hole in zip(np.concatenate([gaps, gaps]),
                              np.concatenate([holes, holes])):
            run = 0.0 if is_hole else run + float(g)
            best = max(best, min(run, 360.0))
        span_deg = best

    harmonics = [n for n in ("a1", "a2", "a3", "a4", "a5", "a6")
                 if n in unpacked]
    refined_harmonics: List[str] = []
    if spec is not None and getattr(spec, "parameters", None):
        refined_harmonics = [n for n in harmonics
                             if n in spec.parameters and spec.parameters[n].refined]
    else:
        # No spec: treat a non-zero harmonic as evidence it was refined.
        refined_harmonics = [n for n in harmonics
                             if float(torch.as_tensor(unpacked[n]).reshape(-1)[0]) != 0.0]

    metrics = {"covered_fraction": frac, "covered_deg": covered_deg,
               "longest_arc_deg": span_deg,
               "n_distinct_eta": float(a.size),
               "n_refined_harmonics": float(len(refined_harmonics))}

    if frac >= warn_fraction and not refined_harmonics:
        sev, msg = "ok", (f"azimuthal coverage {covered_deg:.0f}° "
                          f"({frac * 100:.0f} % of the ring, longest arc "
                          f"{span_deg:.0f}°)")
    elif frac >= warn_fraction:
        sev, msg = "ok", (f"azimuthal coverage {covered_deg:.0f}° "
                          f"({frac * 100:.0f} %) — enough for the "
                          f"{len(refined_harmonics)} refined harmonics")
    elif frac >= fail_fraction:
        sev = "warn" if refined_harmonics else "ok"
        msg = (f"azimuthal coverage only {covered_deg:.0f}° "
               f"({frac * 100:.0f} %, longest arc {span_deg:.0f}°)"
               + (f" while refining {', '.join(refined_harmonics)} — these are "
                  f"weakly separable from BC (1-fold) and tilts (2-fold); "
                  f"consider refine_distortion='radial'"
                  if refined_harmonics else
                  " — geometry is determined over a limited wedge"))
    else:
        sev = "fail" if refined_harmonics else "warn"
        msg = (f"azimuthal coverage {covered_deg:.0f}° "
               f"({frac * 100:.0f} %, longest arc {span_deg:.0f}°)"
               + (f" is too narrow to determine {', '.join(refined_harmonics)}; "
                  f"use refine_distortion='radial' (or --refine-distortion "
                  f"radial) to keep the isotropic terms and freeze the "
                  f"harmonics, or add a second detector position. A second "
                  f"calibrant will NOT help — both phases share this wedge."
                  if refined_harmonics else
                  " — a narrow wedge; treat tilts and beam centre as weakly "
                  "determined"))
    return DiagnosticResult(name="azimuth_coverage", severity=sev,
                            message=msg, metrics=metrics)


def rho_d_scaling_gate(
    fits,
    unpacked: Dict[str, torch.Tensor],
    *,
    spec=None,
    warn_ratio: float = 1.5,
    fail_ratio: float = 3.0,
) -> DiagnosticResult:
    """Is RhoD scaled so the radial distortion terms have any lever?

    The distortion polynomial is evaluated in ``rho = R_um / RhoD``, so RhoD is
    a normalisation, not a measurement — but it decides the dynamic range of
    every radial term.  When RhoD is set far beyond the outermost ring, rho
    stays small and the high powers collapse: at rho_max = 0.28, ``rho^6`` is
    4e-4, and ``iso_R6`` becomes unidentifiable (observed 1-sigma of 5 to 15 on
    a coefficient of order 1e-3, railed at its bound).  Setting RhoD to the
    actual outer ring radius fixed it.
    """
    import numpy as np
    rho_d_um = float(torch.as_tensor(fits.rho_d).reshape(-1)[0])
    pxY = float(torch.as_tensor(unpacked.get(
        "pxY", torch.as_tensor(200.0))).reshape(-1)[0])
    pxZ = float(torch.as_tensor(unpacked.get(
        "pxZ", torch.as_tensor(pxY))).reshape(-1)[0])
    px = 0.5 * (pxY + pxZ)
    bc_y = float(torch.as_tensor(unpacked["BC_y"]).reshape(-1)[0])
    bc_z = float(torch.as_tensor(unpacked["BC_z"]).reshape(-1)[0])
    Y = fits.Y_pix.detach().cpu().numpy()
    Z = fits.Z_pix.detach().cpu().numpy()
    r_max_um = float(np.max(np.hypot(bc_y - Y, Z - bc_z))) * px if Y.size else 0.0
    if r_max_um <= 0 or rho_d_um <= 0:
        return DiagnosticResult(
            name="rho_d_scaling", severity="warn",
            message="cannot evaluate RhoD scaling (no fits or RhoD <= 0)",
            metrics={})
    ratio = rho_d_um / r_max_um
    rho_max = r_max_um / rho_d_um
    metrics = {"RhoD_um": rho_d_um, "r_max_um": r_max_um,
               "ratio": ratio, "rho_max": rho_max,
               "rho_max_pow6": rho_max ** 6}

    radial = [n for n in ("iso_R2", "iso_R4", "iso_R6") if n in unpacked]
    refined_radial = ([n for n in radial
                       if spec is not None and n in spec.parameters
                       and spec.parameters[n].refined]
                      if spec is not None else radial)
    base = (f"RhoD={rho_d_um / 1000:.1f} mm vs outermost fitted ring "
            f"{r_max_um / 1000:.1f} mm (ratio {ratio:.2f}, ρ_max={rho_max:.2f})")
    if ratio >= fail_ratio and refined_radial:
        return DiagnosticResult(
            name="rho_d_scaling", severity="fail",
            message=(f"{base} — ρ⁶ is only {rho_max ** 6:.1e}, so "
                     f"{', '.join(refined_radial)} have essentially no lever and "
                     f"will rail at their bounds. Set RhoD to the outer ring "
                     f"radius in µm (~{r_max_um:.0f})."),
            metrics=metrics)
    if ratio >= warn_ratio:
        return DiagnosticResult(
            name="rho_d_scaling", severity="warn",
            message=(f"{base} — the high-order radial terms are weakly "
                     f"determined; consider RhoD ≈ {r_max_um:.0f} µm."),
            metrics=metrics)
    return DiagnosticResult(name="rho_d_scaling", severity="ok",
                            message=f"{base} — radial terms well scaled",
                            metrics=metrics)


def n_rings_from_fits(fits) -> Optional[int]:
    """Number of DISTINCT rings a fitted dataset rests on, or None.

    Feeds :func:`strain_cap_check` so the headline strain number carries the
    same caution the coverage gates already have. Deliberately a named
    function rather than three lines inside ``run_all_gates``: the first
    version was inline, used a bare ``np.unique`` — numpy is imported
    per-function in this module, not at module scope — and the resulting
    ``NameError`` was swallowed by a blanket ``except``, so the count silently
    stayed None and the gate went on reporting "ok". Pulled out so it can be
    tested on its own.
    """
    import numpy as _np

    ring_idx = getattr(fits, "ring_idx", None) if fits is not None else None
    if ring_idx is None:
        return None
    arr = (ring_idx.detach().cpu().numpy()
           if hasattr(ring_idx, "detach") else _np.asarray(ring_idx))
    return int(_np.unique(arr).size)


def run_all_gates(
    *,
    v1_init,
    unpacked: Dict[str, torch.Tensor],
    history,
    fits=None,
    spec=None,
    panel_layout: Optional[PanelLayout] = None,
    n_train_rings: Optional[int] = None,
    strain_threshold_uE: float = 100.0,
    strain_warn_uE: float = 50.0,
) -> List[DiagnosticResult]:
    """Run every gate, returning a list of DiagnosticResult.  The gates that
    need fitted points are skipped if ``fits`` is None.
    """
    out: List[DiagnosticResult] = []
    # Ring count feeds the strain gate so the headline number carries the same
    # caution the coverage gates already have.
    n_rings_used = n_rings_from_fits(fits)
    out.append(strain_cap_check(history,
                                  threshold_uE=strain_threshold_uE,
                                  warn_uE=strain_warn_uE,
                                  n_rings=n_rings_used))
    out.append(basin_check(v1_init, unpacked))
    if fits is not None:
        out.append(cross_validation_gate(fits, unpacked,
                                           n_train_rings=n_train_rings,
                                           panel_layout=panel_layout))
        out.append(azimuth_coverage_gate(fits, unpacked, spec=spec))
        out.append(rho_d_scaling_gate(fits, unpacked, spec=spec))
    return out


def summarise(diagnostics: List[DiagnosticResult]) -> str:
    """Compact human-readable summary of a gate-result list."""
    icon = {"ok": "✓", "warn": "⚠", "fail": "✗"}
    lines = ["Calibration diagnostics:"]
    for d in diagnostics:
        lines.append(f"  {icon.get(d.severity, '?')} [{d.name}] {d.message}")
    return "\n".join(lines)


def worst_severity(diagnostics: List[DiagnosticResult]) -> str:
    """Return ``"fail"`` if any gate failed, else ``"warn"`` if any warned,
    else ``"ok"`` — and ``"unknown"`` when NO gate ran.

    The empty case used to return ``"ok"`` (``max(..., default=0)``), so a run
    whose diagnostics failed to compute reported the same clean bill of health
    as one that passed every gate. ``auto.py`` builds its gate list inside a
    ``try/except`` that only prints under ``verbose``, so an exception there
    left an empty list and a green "ok" with nothing on stdout — a gate that
    could not fail, the same shape as the ``basin_check`` defect.

    ``"unknown"`` sorts as neither pass nor fail; callers that test
    ``!= "fail"`` still behave as before for a genuinely empty list, but the
    string now says what happened.
    """
    if not diagnostics:
        return "unknown"
    sev_rank = {"ok": 0, "warn": 1, "fail": 2}
    rank_sev = {0: "ok", 1: "warn", 2: "fail"}
    worst = max((sev_rank.get(d.severity, 0) for d in diagnostics), default=0)
    return rank_sev[worst]


# ------------------------------------------------------------------- helpers


def _np_median(arr) -> float:
    import numpy as np
    return float(np.median(arr))


def _ks_2samp_p(a, b) -> float:
    """Two-sample KS test p-value.  Uses scipy if available; falls
    back to Kolmogorov asymptotic p-value computed by hand (stdlib
    only) so this module has no hard scipy dep."""
    try:
        from scipy.stats import ks_2samp
        return float(ks_2samp(a, b).pvalue)
    except ImportError:
        pass
    import math
    import numpy as np
    a = np.sort(np.asarray(a, dtype=float))
    b = np.sort(np.asarray(b, dtype=float))
    na, nb = len(a), len(b)
    if na == 0 or nb == 0:
        return 1.0
    pooled = np.sort(np.concatenate([a, b]))
    cdf_a = np.searchsorted(a, pooled, side="right") / na
    cdf_b = np.searchsorted(b, pooled, side="right") / nb
    D = float(np.max(np.abs(cdf_a - cdf_b)))
    en = math.sqrt(na * nb / (na + nb))
    # Kolmogorov asymptotic CDF.
    lam = (en + 0.12 + 0.11 / en) * D
    p = 2.0 * sum((-1) ** (k - 1) * math.exp(-2 * (lam ** 2) * (k ** 2))
                  for k in range(1, 50))
    return max(min(p, 1.0), 0.0)


__all__ = [
    "DiagnosticResult",
    "seed_provenance_gate",
    "detector_scope_gate",
    "cross_validation_gate",
    "strain_cap_check",
    "basin_check",
    "azimuth_coverage_gate",
    "rho_d_scaling_gate",
    "run_all_gates",
    "summarise",
    "worst_severity",
]
