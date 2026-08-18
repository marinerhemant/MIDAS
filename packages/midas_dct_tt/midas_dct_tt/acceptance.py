"""Reciprocal-space acceptance for TT and DCT: the objective-free limit.

Phase 1 of ``implementation_plan.md``.

.. warning::
   **Use :class:`ObjectiveFreeAcceptance`. The ``na = 0`` forms below
   (:func:`tt_resolution`, :func:`tt_resolution_aniso`) are the WRONG LIMIT and
   are retained only for reproducing earlier results.**

   The reasoning that led to them -- given below, and left in place because the
   error is instructive -- is that ``na`` enters Poulsen's widths additively, so
   ``na = 0`` "removes the objective". It does the opposite. Those widths describe
   an objective that accepts a vanishing range of ``k_out``, i.e. an *infinitely
   selective* one. A bare detector accepts **every** ``k_out``, which is the
   opposite limit and is not reachable by setting a parameter to zero in that
   formula.

   Consequences measured: the true roll:rock anisotropy is **200**, not the 21 the
   ``na = 0`` form gives; and the true acceptance is a **slab perpendicular to
   k_out** (correlated across any par/rock/roll frame), not a diagonal ellipsoid,
   so no choice of three widths can represent it.

.. warning::
   **The ``na = 0`` premise below is WRONG, and this module is built on it.**
   Recorded here rather than quietly fixed because everything downstream inherits
   it. In :func:`midas_dfxm.resolution.poulsen_resolution_widths` the objective
   enters as an **additive** variance (``div_v^2 + na^2``): a larger aperture
   *widens* the acceptance. So ``na = 0`` is the **narrowest** case -- an
   infinitely selective objective accepting one ``k_out`` -- whereas a bare TT/DCT
   detector accepts **all** ``k_out``, which is the opposite limit. Verified
   numerically: ``na = 0 -> 1e-3 -> 5e-3`` takes ``sigma_rock`` from 1.06e-03 to
   2.26e-03 to 1.01e-02, monotonically wider.

   Measured against an exact objective-free acceptance (Q accepted iff *some* ray
   in the beam satisfies ``2 k_in.Q + |Q|^2 = 0``, no constraint on ``k_out``),
   derived and cross-checked two independent ways:

   ==========  ===================  ==============  =======
   direction   true objective-free  ``na=0`` model  ratio
   ==========  ===================  ==============  =======
   ``e_rock``  1.83213e-03          9.15967e-04     2.00
   ``e_par``   3.80388e-02          1.92357e-02     1.98
   ``e_roll``  3.64608e-01          1.92514e-02     **18.94**
   ==========  ===================  ==============  =======

   So the true ``sigma_roll/sigma_rock`` is **199, not 21**, and the true
   (par, rock) block is strongly correlated (**-0.9996**) -- a tilted *strip*, not
   an axis-aligned ellipse. At 5 ``sigma_rock`` off-axis in the (par, rock) plane
   the model gives weight 3.7e-6 where the truth gives 4.4e-2, **four orders of
   magnitude**. :class:`AnisotropicTTResolution` is therefore still a substantially
   wrong acceptance -- less wrong than :func:`tt_resolution`, but not right.
   Implementing the objective-free form is open work.

The widths below are the ``na = 0`` (perfect-objective) limit of Poulsen's
expressions, kept because everything measured so far used them and because they
remain the correct baseline for a *DFXM-like* geometry.

The widths (Poulsen 2017 Eqs. 58-63, at ``na = 0``)::

    sigma_rock = (|Q|/2) div_v
    sigma_roll = (|Q|/2) div_h / sin(theta)
    sigma_par  = (|Q|/2) sqrt(4 eps^2 + cot^2(theta) div_v^2)

Note ``sigma_roll`` diverges as ``theta -> 0``: at the low Bragg angles that give
TT its *best* tomographic coverage (the missing cone is ``theta``), the roll
acceptance is at its *worst*. That tension is real, it is not an artefact of this
implementation, and reflection selection has to trade the two off.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from midas_dfxm.resolution import ResolutionFunction, aligned_resolution, poulsen_resolution_widths

__all__ = [
    "AnisotropicTTResolution",
    "ObjectiveFreeAcceptance",
    "orientation_resolution_deg",
    "tt_resolution",
    "tt_resolution_aniso",
    "tt_resolution_widths",
]


def tt_resolution_widths(
    q_mag: float,
    *,
    two_theta_deg: float,
    div_v: float = 0.53e-3,
    div_h: float = 0.53e-3,
    energy_spread: float = 1.4e-4,
) -> dict:
    """Principal acceptance widths with the objective removed (``na = 0``).

    Thin wrapper over :func:`midas_dfxm.resolution.poulsen_resolution_widths`;
    exists to make ``na = 0`` explicit and documented at every call site rather
    than a magic argument. Returns ``{'sigma_rock', 'sigma_roll', 'sigma_par'}``
    in 1/Angstrom.

    .. warning::
       These are **not** the acceptance widths of a bare detector -- see the module
       warning. They remain correct as *Poulsen widths at na = 0*, which is what
       :func:`~midas_dct_tt.orientation_resolution_deg` uses them for (the rock and
       roll angular bounds, where the ``na`` term cancels out of the ratio anyway).
       For an acceptance, use :class:`ObjectiveFreeAcceptance`.

       They also **assume a vertical scattering plane**: ``div_v`` is hard-wired to
       the rocking direction. That is exact when ``div_h == div_v`` (the default,
       where the azimuth cannot matter by symmetry) and wrong otherwise -- at
       azimuth 0 with ``div_h = 0.31`` mrad the correct rock width is
       ``|Q| * div_h = 1.07176e-3`` while this returns ``9.15967e-4``.
       :func:`orientation_resolution_deg` detects the anisotropic case and routes
       around it.
    """
    return poulsen_resolution_widths(
        q_mag,
        two_theta_deg=two_theta_deg,
        div_v=div_v,
        div_h=div_h,
        na=0.0,
        eps=energy_spread,
    )


def tt_resolution(
    alignment,
    *,
    div_v: float = 0.53e-3,
    div_h: float = 0.53e-3,
    energy_spread: float = 1.4e-4,
) -> ResolutionFunction:
    """Gaussian acceptance centred on a TTAlignment. **Superseded** -- see below.

    .. warning::
       Wrong limit for TT/DCT; use :class:`ObjectiveFreeAcceptance`. Retained for
       reproducing earlier results and for the case where an objective really is
       present. See the module warning.

    Returns a :class:`midas_dfxm.resolution.ResolutionFunction` centred on the
    aligned ``G_lab``, with ``sigma_par`` from the exact width and ``sigma_perp``
    the **geometric mean of rock and roll**.

    That last step is an approximation and worth stating plainly:
    ``ResolutionFunction`` is isotropic transverse, while the true acceptance is
    a thin plate (``sigma_rock << sigma_roll``, often by 10x or more at small
    ``theta``).

    .. warning::
       An earlier version of this docstring claimed the geometric mean "preserves
       the transverse *area*, so integrated intensities are right". **That is only
       true at zero strain.** Preserving area is not preserving
       ``<exp(-chi^2/2)>``: measured at ``|H| = 1e-3``, the psi-averaged plate
       ``m0`` is **0.756 / 0.712 / 0.829x** the isotropic ``m0`` for
       (111)/(200)/(220) -- 17-29% low. The zero-strain control gives 1.000000.
       Integrated intensities from the isotropic form are wrong for a strained
       grain, not just the rocking-curve shape.
    """
    q_mag = float(alignment.G_lab.detach().norm())
    two_theta = 2.0 * float(alignment.theta_deg.detach())
    w = tt_resolution_widths(
        q_mag, two_theta_deg=two_theta, div_v=div_v, div_h=div_h, energy_spread=energy_spread
    )
    sigma_perp = math.sqrt(w["sigma_rock"] * w["sigma_roll"])
    res = aligned_resolution(alignment.G_lab, sigma_par=w["sigma_par"], sigma_perp=sigma_perp)
    # Attached for the caller's judgement, not used internally.
    res.anisotropy = w["sigma_roll"] / w["sigma_rock"]
    return res


_EPS = 1e-12


@dataclass
class AnisotropicTTResolution:
    """Anisotropic (rock/roll) acceptance. **Superseded** -- see below.

    .. warning::
       Right that the acceptance is anisotropic, wrong about how much and about its
       shape: it is built on the ``na = 0`` widths (understating roll:rock as 21
       against a true **200**) and is diagonal in the (par, rock, roll) frame, which
       cannot represent a slab perpendicular to ``k_out``. Use
       :class:`ObjectiveFreeAcceptance`.

    The original description follows, since the psi-dependence argument it makes is
    correct and is what motivated the objective-free treatment.

    The acceptance as the **thin plate it actually is**, not an isotropic disc.

    :func:`tt_resolution` collapses the transverse acceptance to a single width
    (the geometric mean of rock and roll). That preserves the transverse *area*,
    so integrated intensities come out right, but it destroys the *shape* -- and
    the shape is where the physics is. Measured at 71.7 keV on low-index fcc
    reflections, ``sigma_roll / sigma_rock`` is **15-24x**, so the true acceptance
    is a blade, not a disc.

    What the anisotropy actually buys, counted honestly
    ---------------------------------------------------
    Under this (diagonal) plate, ``log m0(psi)`` is **exactly**
    ``a + b cos(2 psi) + c sin(2 psi)`` -- fitted residual 9.4e-13 against a signal
    sd of 0.24, with the 1st, 3rd and 4th harmonics all below 2e-13. So a
    ``psi`` scan of *any* length measures **3 numbers per reflection**, of which
    ``a`` is the ``psi``-independent scale already available under isotropy. **The
    anisotropy adds 2 numbers per reflection -- 6 over three reflections.** An
    earlier version of this docstring said "144 numbers instead of 3", which
    counted *samples* rather than *degrees of freedom* and overstated the gain by
    more than 20x.

    Two further honesty notes:

    * The **exact 180-degree period is a property of this diagonal approximation**,
      not of the physics. Under the exact objective-free acceptance (module
      warning) the first harmonic is **41%** of the second, because ``k_out.u``
      mixes the ``psi``-invariant ``u_par`` with the rotating ``u_rock`` in a
      single quadratic form.
    * The **phase** of the modulation depends on the strain orientation, so the
      correlation with ``cos(2 psi)`` specifically is *not* an invariant: rotating
      one strain about lab ``z`` sweeps it +0.598 / +0.723 / +0.300 / -0.003 /
      +0.278 while the 2-``psi`` *power* stays 0.94-1.00. Test the power or the
      3-term fit, never the ``cos(2 psi)`` correlation.

    Why the difference is not cosmetic
    ----------------------------------
    In a TT scan the acceptance is **fixed in the lab** (``q_nom`` and ``k_in``
    are both fixed) while the sample rotates about ``G_lab``. For an unstrained
    grain ``Q_lab`` sits exactly at ``q_nom`` for every ``psi``. Under strain,
    ``Q_lab = R(psi) R_align F^-T G0`` has a transverse offset that **rotates**
    in the plane perpendicular to ``G_lab`` as ``psi`` advances.

    * With an isotropic transverse acceptance only ``|d_transverse|`` matters, so
      that rotation is invisible: the accepted intensity is **exactly
      psi-independent** (measured to ~1e-13), the whole intensity channel
      collapses to one scalar per reflection, and a rotation-only fit to it is
      degenerate.
    * With the true plate, the rotating offset sweeps alternately along the tight
      (rock) and loose (roll) axis, so the intensity **oscillates at the 2-psi
      harmonic**, at an amplitude set by the transverse strain components.

    So the anisotropy is what turns the integrated-intensity channel from empty
    into informative. Any conclusion drawn about that channel under
    :func:`tt_resolution` is conditional on an approximation this class removes.

    Frame
    -----
    ``e_par`` along ``q_nom``; ``e_roll`` perpendicular to the scattering plane
    (``k_in x q_nom``); ``e_rock`` in the scattering plane and perpendicular to
    ``q_nom``. That assignment is forced by the physics rather than chosen: a
    crystal rotation about the scattering-plane normal moves ``G`` within the
    plane and sweeps straight through the Bragg condition (tight, ``sigma_rock``),
    while a rotation about the beam moves ``G`` out of the plane and leaves
    ``k . G`` unchanged to first order (loose, ``sigma_roll``).
    """

    q_nom: torch.Tensor
    k_in: torch.Tensor
    sigma_par: float = 5e-3
    sigma_rock: float = 5e-4
    sigma_roll: float = 5e-3

    def frame(self) -> tuple:
        """``(e_par, e_rock, e_roll)`` -- orthonormal, lab frame.

        Raises if ``k_in`` is parallel to ``q_nom``: the scattering plane is then
        undefined and rock/roll cannot be told apart. That is exact backscatter,
        which TT never uses, so it is an error rather than a fallback.
        """
        q = self.q_nom
        e_par = q / torch.linalg.vector_norm(q)
        out = torch.linalg.cross(self.k_in.to(dtype=q.dtype, device=q.device), q)
        n = torch.linalg.vector_norm(out)
        if bool(n.detach() < _EPS):
            raise ValueError(
                "k_in is parallel to q_nom: the scattering plane is undefined, so "
                "rock and roll are indistinguishable. Exact backscatter is not a "
                "TT geometry."
            )
        e_roll = out / n
        e_rock = torch.linalg.cross(e_roll, e_par)
        return e_par, e_rock, e_roll

    def weight(self, Q_lab: torch.Tensor) -> torch.Tensor:
        """Acceptance weight in ``[0, 1]``. Drop-in for
        :meth:`midas_dfxm.resolution.ResolutionFunction.weight`.

        Differentiable in ``Q_lab``, ``q_nom``, ``k_in`` and all three widths.
        """
        e_par, e_rock, e_roll = self.frame()
        d = Q_lab - self.q_nom
        dt = Q_lab.dtype
        dev = Q_lab.device

        def _w(x):
            return torch.as_tensor(x, dtype=dt, device=dev)

        chi2 = ((d @ e_par) / _w(self.sigma_par)) ** 2 \
            + ((d @ e_rock) / _w(self.sigma_rock)) ** 2 \
            + ((d @ e_roll) / _w(self.sigma_roll)) ** 2
        return torch.exp(-0.5 * chi2)

    @property
    def anisotropy(self) -> float:
        """``sigma_roll / sigma_rock``. 1.0 would be the isotropic disc."""
        return float(self.sigma_roll) / float(self.sigma_rock)

    @property
    def sigma_perp_equivalent(self) -> float:
        """The isotropic width with the same transverse area, ``sqrt(rock*roll)``.

        This is exactly what :func:`tt_resolution` uses, so the two agree on
        transverse area (hence on integrated intensity averaged over ``psi``) and
        differ only in shape.
        """
        return math.sqrt(float(self.sigma_rock) * float(self.sigma_roll))


def tt_resolution_aniso(
    alignment,
    *,
    div_v: float = 0.53e-3,
    div_h: float = 0.53e-3,
    energy_spread: float = 1.4e-4,
) -> AnisotropicTTResolution:
    """Anisotropic (thin-plate) acceptance for a :class:`TTAlignment`.

    Same widths as :func:`tt_resolution`, but kept as separate rock and roll axes
    instead of collapsed to their geometric mean. Prefer this whenever the
    ``psi``-dependence of intensity matters -- which is whenever you intend to use
    intensity at all.
    """
    q_mag = float(alignment.G_lab.detach().norm())
    w = tt_resolution_widths(
        q_mag, two_theta_deg=2.0 * float(alignment.theta_deg.detach()),
        div_v=div_v, div_h=div_h, energy_spread=energy_spread,
    )
    return AnisotropicTTResolution(
        q_nom=alignment.G_lab, k_in=alignment.k_in,
        sigma_par=w["sigma_par"], sigma_rock=w["sigma_rock"], sigma_roll=w["sigma_roll"],
    )


def orientation_resolution_deg(
    alignment=None,
    *,
    theta_deg: float = None,
    div_v: float = 0.53e-3,
    div_h: float = 0.53e-3,
) -> dict:
    """The **instrumental** orientation resolution of a fixed TT setting, in degrees.

    Returns ``{'rock', 'roll', 'ratio', 'curvature_scale'}``.

    .. important::
       **Not the same quantity as Liu et al. (2025) "orientation sampling
       resolution"** -- the names collide and the two are easy to confuse.

       * Liu et al., arXiv:2510.08712 Sec. 2.7.2, derive an **upper bound on the
         orientation sampling INTERVAL** used inside a reconstruction algorithm: a
         discretisation/Nyquist criterion set by detector pixel size, detector
         distance and base-tilt step, giving 0.013-0.021 deg at ID11. It answers
         *"how finely must my algorithm sample orientation space?"*
       * This function returns a **physical measurement limit** set by beam
         divergence: the smallest lattice rotation the instrument can distinguish
         at a fixed goniometer setting. It answers *"what can the measurement
         actually tell apart?"*

       They are complementary and mutually consistent: Liu's sampling intervals
       (0.013-0.021 deg) sit just below the 0.0304 deg resolution computed here,
       which is what Nyquist requires of a sampling interval relative to a
       resolution. Cite Liu for the former; do not present this as the same result.

    Two results, and the second is the striking one
    -----------------------------------------------
    1. **rock = div_v, exactly, and independent of the reflection.** Measured
       ``rock / div_v = 1.0000`` for (111), (200), (220), (311), (222) and (400):
       **0.03037 deg** at 0.53 mrad. ``|Q|`` cancels, so a higher-index reflection
       buys longitudinal (strain) resolution and **nothing** in orientation about
       the rocking axis.
    2. **roll is not bounded at all.** A lattice rotation about ``k_in`` preserves
       both ``|Q|`` and ``k_in . Q``, so it leaves ``rho`` invariant **exactly, at
       every angle** -- not merely to first order. At a fixed goniometer setting a
       TT measurement constrains orientation about the rocking axis and is
       completely blind to rotation about the beam. ``roll`` is returned as
       ``inf``.

    ``curvature_scale`` is the finite number sometimes mistaken for a roll width:
    the angular distance over which the Ewald surface departs from its tangent
    plane by one slab thickness, obtained by walking a straight line off the
    ``|Q| = const`` sphere. It is a validity radius for the planar-slab picture,
    **not** an orientation bound, and it does depend on the reflection
    (6.50 deg at (111) down to 4.27 deg at (400)).

    Corrected 2026-08-04
    --------------------
    An earlier version returned ``rock = div_v/2`` and
    ``roll = div_h/(2 sin theta)``. Both came from the Poulsen ``na = 0`` widths,
    which are the wrong limit for a bare detector (see the module warning): they
    understate ``rock`` by exactly **2x** and describe ``roll`` as a finite width
    when the true acceptance does not constrain it at all.
    """
    if alignment is None:
        if theta_deg is None:
            raise ValueError("pass an alignment (preferred) or an explicit theta_deg")
        if abs(div_h - div_v) > 1e-15 * max(div_h, div_v, 1e-30):
            raise ValueError(
                "div_h != div_v makes the bound azimuth-dependent; pass an "
                "alignment so the scattering-plane orientation is known."
            )
        rock = math.degrees(div_v)
        return {"rock": rock, "roll": float("inf"), "ratio": float("inf"),
                "curvature_scale": float("nan")}

    q = alignment.G_lab
    qmag = float(torch.linalg.vector_norm(q))
    acc = ObjectiveFreeAcceptance(k_in=alignment.k_in, q_nom=q, div_v=div_v,
                                  div_h=div_h, energy_spread=0.0)
    e_par = q / qmag
    out = torch.linalg.cross(alignment.k_in.to(q.dtype), q)
    e_roll = out / torch.linalg.vector_norm(out)
    e_rock = torch.linalg.cross(e_roll, e_par)
    rock = math.degrees(acc.effective_sigma(q, e_rock) / qmag)
    curv = math.degrees(acc.effective_sigma(q, e_roll) / qmag)
    return {"rock": rock, "roll": float("inf"), "ratio": float("inf"),
            "curvature_scale": curv}


@dataclass
class ObjectiveFreeAcceptance:
    """The acceptance a **bare detector** actually has -- no objective at all.

    Replaces the ``na = 0`` construction, which is the *opposite* limit (module
    warning): ``na`` enters Poulsen's widths additively, so ``na = 0`` describes an
    infinitely selective objective, whereas TT and DCT put a detector straight in
    the diffracted beam and accept **every** ``k_out``.

    The condition is that *some* ray in the incident beam scatters elastically::

        rho(Q) = |k_in + Q|^2 - |k_in|^2 = 2 k_in . Q + |Q|^2 = 0

    i.e. the Ewald condition with ``k_out`` free. Since
    ``grad_Q rho = 2 (k_in + Q) = 2 k_out``, the acceptance is a **slab
    perpendicular to k_out**. (Note that the strong par/rock correlation this
    implies is *already* present in the ``na = 0`` marginals -- corr -0.9996 -- so
    it is a defect of any diagonal-widths API, not something the objective-free
    limit creates.)

    Width from the beam's own spread::

        sigma_rho = 2 |k_in| sqrt( eps^2 Qx^2 + div_h^2 Qy^2 + div_v^2 Qz^2 )

    with lab ``x`` along the beam, ``y`` horizontal, ``z`` vertical. Using the
    **exact** ``rho`` rather than a linearisation means the out-of-plane behaviour
    falls out on its own.

    Validated against exact 2-D quadrature of ``<delta(rho)>`` and a 4e7-ray Monte
    Carlo, agreeing to **6 digits**: rock **1.832131e-03**, par **3.803882e-02**,
    roll **3.646085e-01**.

    On the roll direction, precisely
    --------------------------------
    A lattice rotation about ``k_in`` preserves both ``|Q|`` and ``k_in . Q``, so it
    leaves ``rho`` invariant **exactly, at any angle** -- the physical roll degree
    of freedom is unconstrained outright, not merely to first order. The finite
    "roll width" is what you get walking a *straight line* off the ``|Q| = const``
    sphere, so it measures the **Ewald-sphere curvature scale**
    (``~ sqrt(|Q| * sigma_par)``, 0.55% here), i.e. the radius over which the
    planar-slab picture is valid. Any "roll:rock anisotropy" is therefore
    contour-dependent, ``ratio(c) = ratio(1)/sqrt(c)`` (200 at 1 sigma, 90 at 5,
    72 at 8), because roll is quadratic in the offset where rock is linear.
    **Quote the mechanism, not the ratio.**

    Monochromator coupling
    ----------------------
    By default energy spread and vertical divergence are treated as **independent**,
    which is what an undulator beam before optics looks like. After a
    vertically-diffracting monochromator they are not: differentiating Bragg's law
    gives ``d(eps) = -cot(theta_M) d(alpha_v)``, so each vertical ray carries its
    own energy and the two contributions add **coherently**. Set
    ``mono_bragg_deg`` (and ``mono_sense`` = +/-1 for the dispersion sense) to model
    it.

    This is load-bearing on the narrow axis. For Si(111) at 71.7 keV
    (``theta_M = 1.58 deg``, ``cot = 36.2``) the ``Qx`` term dominates ``Qz``, and
    ``sigma_rock`` scales by **2.73x or 0.73x** depending on sense -- a factor of
    3.7 between the two, on the very axis that carries the orientation information.
    If you quote a rock resolution for a real beamline, state which sense you assumed.

    Still a Gaussian beam profile, and no absorption or extinction.
    """

    k_in: torch.Tensor
    q_nom: torch.Tensor = None
    div_v: float = 0.53e-3
    div_h: float = 0.53e-3
    energy_spread: float = 1.4e-4
    domain_size_um: float = None
    mono_bragg_deg: float = None
    mono_sense: float = +1.0

    def sigma_domain(self) -> float:
        """Reciprocal-space broadening from a finite coherent domain, 1/Angstrom.

        ``2*pi / L``. Zero when ``domain_size_um`` is None.

        This is a property of the **scatterer**, not the beam, and it is not
        optional at small sizes -- measured against the beam-only rock width of
        1.832e-03 1/A at 71.7 keV:

        ==========  =================  ==============
        domain      as % of rock       net widening
        ==========  =================  ==============
        3 um        11.4%              1.01x
        1 um        34.3%              1.06x
        0.5 um      **68.6%**          **1.21x**
        ==========  =================  ==============

        So it is ignorable for whole-grain work at a few microns and emphatically
        not for sub-micron intragranular structure -- which is exactly what a
        per-voxel inverse claims to resolve. Note it does **not** make the roll
        direction finite: a rotation about ``k_in`` still maps the (broadened)
        reciprocal-lattice point onto itself, so ``rho`` remains exactly invariant.
        """
        if self.domain_size_um is None:
            return 0.0
        if self.domain_size_um <= 0:
            raise ValueError("domain_size_um must be > 0")
        return 2.0 * math.pi / (float(self.domain_size_um) * 1e4)

    def rho(self, Q_lab: torch.Tensor) -> torch.Tensor:
        """``2 k_in . Q + |Q|^2`` -- zero on the Ewald sphere."""
        k = self.k_in.to(dtype=Q_lab.dtype, device=Q_lab.device)
        return 2.0 * (Q_lab @ k) + (Q_lab * Q_lab).sum(-1)

    def sigma_rho(self, Q_lab: torch.Tensor) -> torch.Tensor:
        """Spread of ``rho`` induced by the beam's divergence and bandwidth."""
        k = self.k_in.to(dtype=Q_lab.dtype, device=Q_lab.device)
        kmag = torch.linalg.vector_norm(k)
        if self.mono_bragg_deg is None:
            # eps and div_v treated as INDEPENDENT -- see the note on mono coupling.
            v2 = ((self.energy_spread * Q_lab[..., 0]) ** 2
                  + (self.div_h * Q_lab[..., 1]) ** 2
                  + (self.div_v * Q_lab[..., 2]) ** 2)
        else:
            # After a vertically-diffracting monochromator the two are locked:
            # differentiating Bragg gives d(eps) = -cot(theta_M) d(alpha_v), so the
            # vertical fan carries a correlated energy fan and they add coherently
            # rather than in quadrature.
            cot_m = 1.0 / math.tan(math.radians(float(self.mono_bragg_deg)))
            coupled = Q_lab[..., 2] - float(self.mono_sense) * cot_m * Q_lab[..., 0]
            v2 = ((self.div_v * coupled) ** 2 + (self.div_h * Q_lab[..., 1]) ** 2)
        beam = 2.0 * kmag * torch.sqrt(v2.clamp_min(0))
        sd = self.sigma_domain()
        if sd:
            # A finite domain spreads Q itself; rho varies as 2*k_out . dQ, so the
            # two add in quadrature along the slab normal.
            k_out = k + Q_lab
            dom = 2.0 * torch.linalg.vector_norm(k_out, dim=-1) * sd
            beam = torch.sqrt(beam ** 2 + dom ** 2)
        return beam + _EPS

    def weight(self, Q_lab: torch.Tensor) -> torch.Tensor:
        """Acceptance weight. Drop-in for the other resolution classes.

        This is the **density of** ``rho`` **at zero**, not merely a Gaussian in
        ``rho``: the beam average ``<delta(rho)>`` carries a ``1 / sigma_rho(Q)``
        prefactor, and since ``sigma_rho`` varies with ``Q`` that is not an overall
        constant -- it tilts the profile. Omitting it made the ``par`` width
        **1.11% too wide**; with it, all three widths match exact quadrature and
        Monte Carlo to 6 digits. Normalised to 1 at ``q_nom``.
        """
        if self.q_nom is None:
            raise ValueError(
                "q_nom is required: the density form carries a 1/sigma_rho(Q) "
                "prefactor, so the weight has no natural scale without a nominal "
                "reflection to normalise at. Pass q_nom=alignment.G_lab."
            )
        s_q = self.sigma_rho(Q_lab)
        ref = self.sigma_rho(self.q_nom.unsqueeze(0)).reshape(())
        return torch.exp(-0.5 * (self.rho(Q_lab) / s_q) ** 2) * (ref / s_q)

    def effective_sigma(self, Q_nom: torch.Tensor, direction: torch.Tensor,
                        *, rtol: float = 1e-10, max_iter: int = 200) -> float:
        """Numerically measured 1-sigma half-width along ``direction``.

        Measured rather than derived because the response is not the same order in
        every direction: linear in ``rho`` along rock and par, quadratic along roll.
        Bracket-and-bisect on ``weight = exp(-1/2)``; a fixed linear grid cannot do
        it, since the widths span two decades in one geometry.
        """
        u = direction / torch.linalg.vector_norm(direction)
        target = math.exp(-0.5)

        def w_at(t):
            return float(self.weight((Q_nom + t * u).unsqueeze(0))[0])

        if w_at(0.0) < target:
            return 0.0
        hi = 1e-8
        for _ in range(max_iter):
            if w_at(hi) < target:
                break
            hi *= 2.0
        else:
            return float("inf")
        lo = 0.0
        for _ in range(max_iter):
            mid = 0.5 * (lo + hi)
            if w_at(mid) >= target:
                lo = mid
            else:
                hi = mid
            if hi - lo <= rtol * max(hi, 1e-30):
                break
        return 0.5 * (lo + hi)
