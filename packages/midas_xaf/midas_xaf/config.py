"""Configuration for Cross-Axis Faceted HEDM (XAF-HEDM) simulations.

A single :class:`XAFConfig` dataclass carries every knob of the measurement so
the sweep tool can vary them one at a time.  The defaults describe the baseline
cubic diamond anvil cell design:

* a **full-cone ~15 deg opening** on all six faces -> max 2theta ~= 7.5 deg
  (the diffracted beam must clear the same opening, so the *exit* cone caps
  the accessible scattering angle at the opening half-angle);
* four narrow omega wedges at 0, +/-90, 180 deg, accessed through the four
  equatorial faces in each mounting;
* two orthogonal-axis mountings (the second obtained by a 90 deg remount about
  the beam axis) merged into one reciprocal-space reconstruction.

Units follow the MIDAS convention: micrometres, degrees, and angstroms (for the
wavelength / lattice parameters).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple

# X-ray photon-energy <-> wavelength conversion (keV * A).
_HC_KEV_A = 12.398419843320026


@dataclass
class XAFConfig:
    """All parameters of an XAF-HEDM measurement + sample.

    Anything left as ``None`` that has an ``auto`` rule (currently ``Lsd_um``)
    is resolved by :mod:`midas_xaf.geometry` from the other knobs.
    """

    # --- Beam / energy -----------------------------------------------------
    energy_keV: float = 80.0

    # --- Cell access geometry ---------------------------------------------
    opening_full_deg: float = 15.0            # full cone of each face opening
    #: omega half-width actually collected per wedge.  ``None`` -> opening
    #: half-angle (the incident beam is gated by the same opening).
    wedge_halfwidth_deg: Optional[float] = None
    wedge_centers_deg: Tuple[float, ...] = (0.0, 90.0, 180.0, 270.0)
    #: exit-aperture model.  "cone" (default): the diffracted beam must clear a
    #: face opening whose transmitting cone rotates with omega -- captures the
    #: omega/eta-dependent shadowing (different detector sectors go dark at
    #: different omega).  "tth_cap": first-order 2theta<=opening_half disk
    #: (valid only at wedge centre; faster, for comparison).
    exit_model: str = "cone"

    # --- Detector ----------------------------------------------------------
    px_um: float = 75.0                       # Eiger2 CdTe pixel
    n_pixels_y: int = 2048
    n_pixels_z: int = 2048
    #: sample-detector distance (um).  ``None`` -> auto: place ``tth_max`` at
    #: ``detector_fill_frac`` of the detector half-width for best 2theta
    #: (strain) resolution while still capturing the full exit cone.
    Lsd_um: Optional[float] = None
    detector_fill_frac: float = 0.9
    min_eta_deg: float = 0.0
    #: dead-region model. "none" = perfect detector; "pilatus2m" = the 3x8
    #: module tiling with inter-module gaps (spots in gaps are lost).
    detector_type: str = "none"
    #: central beamstop radius (pixels); spots inside are lost.  0 = none.
    beamstop_radius_px: float = 0.0
    #: spot-centroid precision on the detector, in pixels (for the strain CRLB).
    sigma_det_px: float = 1.0

    # --- omega sampling (frame mapping; geometry uses the wedge mask) ------
    omega_step_deg: float = 0.25
    #: omega-centroid precision, in omega steps (for the strain CRLB).
    sigma_omega_steps: float = 1.0

    # --- Crystal / material ------------------------------------------------
    material: str = "zirconia_monoclinic"     # key into midas_xaf.crystal.MATERIALS

    # --- Two-mounting remount ---------------------------------------------
    #: axis (lab frame: beam=+x, rotation-axis=+z, transverse=+y) and angle of
    #: the rigid remount that brings the top/bottom faces to the equator.
    remount_axis: Tuple[float, float, float] = (1.0, 0.0, 0.0)
    remount_angle_deg: float = 90.0
    n_mountings: int = 2
    #: explicit per-mounting remounts (from mounting 0), one (axis, angle_deg)
    #: per mounting >=1.  ``None`` composes the single ``remount_axis`` instead.
    #: For 3 orthogonal rotation axes use
    #: ``(((1,0,0),90.0), ((0,1,0),90.0))`` with ``n_mountings=3``.
    remount_specs: Optional[Tuple[Tuple[Tuple[float, float, float], float], ...]] = None

    # --- Beam mode ---------------------------------------------------------
    beam_mode: str = "box"                    # "box" | "line" | "point"
    beam_size_um: float = 2.0                 # line/point beam height (um)
    #: beam translation positions (um) for line/point scanning.  ``None`` ->
    #: auto grid spanning +/- sample_radius_um at ``beam_size_um`` step.
    scan_positions_um: Optional[Sequence[float]] = None

    # --- Sample population -------------------------------------------------
    n_grains: int = 100
    sample_radius_um: float = 50.0            # grains within this radius
    strain_rms: float = 1.0e-3               # RMS of random per-grain strain
    seed: int = 0

    # --- Fiducials ---------------------------------------------------------
    n_fiducials: int = 3
    fiducial_material: str = "tungsten"       # single-crystal high-Z markers
    fiducial_type: str = "single_crystal"     # "single_crystal" | "absorbing"

    # --- Compute -----------------------------------------------------------
    device: str = "cpu"                       # "cpu" | "mps" | "cuda"
    dtype_double: bool = True                 # use float64 in the input pipeline

    # ------------------------------------------------------------------ #
    #  Derived quantities                                                #
    # ------------------------------------------------------------------ #
    @property
    def wavelength_A(self) -> float:
        return _HC_KEV_A / self.energy_keV

    @property
    def opening_half_deg(self) -> float:
        return 0.5 * self.opening_full_deg

    @property
    def tth_max_deg(self) -> float:
        """Max accessible scattering angle: the diffracted beam must clear the
        same face opening, so 2theta is capped at the opening half-angle."""
        return self.opening_half_deg

    @property
    def wedge_half_deg(self) -> float:
        return (self.opening_half_deg if self.wedge_halfwidth_deg is None
                else float(self.wedge_halfwidth_deg))

    def resolved_Lsd_um(self) -> float:
        """Auto-couple distance to opening + detector when ``Lsd_um`` is None.

        The exit cone caps 2theta at ``tth_max``; we place that ring at
        ``detector_fill_frac`` of the detector half-width so the few
        accessible rings spread across as many pixels as possible (best
        strain resolution) while remaining on the detector.
        """
        import math
        if self.Lsd_um is not None:
            return float(self.Lsd_um)
        half_width_um = 0.5 * min(self.n_pixels_y, self.n_pixels_z) * self.px_um
        target_um = self.detector_fill_frac * half_width_um
        return target_um / math.tan(math.radians(self.tth_max_deg))
