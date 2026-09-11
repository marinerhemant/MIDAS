"""midas-saxs — differentiable small-angle X-ray scattering for MIDAS.

Two source terms, one detector.

**Density contrast** (:mod:`midas_saxs.particles`, :mod:`midas_saxs.form_factors`,
:mod:`midas_saxs.model`, :mod:`midas_saxs.core_shell`, :mod:`midas_saxs.wide_band`)
— voids, bubbles, precipitates; sphere / ellipsoid / cylinder form factors,
lognormal polydispersity, Percus-Yevick S(Q), Guinier and Porod analysis. These
came from ``midas_pdf.saxs``, which now re-exports them from here, so there is
one definition of "SAXS form factor" in MIDAS.

**Strain contrast** (:mod:`midas_saxs.strain_source`) — dislocation loops, driven
by a :mod:`midas_ddd` network read from ExaDiS. Optional extra; plain particle
SAXS does not pull a dislocation-dynamics stack.

    pip install 'midas-saxs[dislocations]'

Quick start
-----------
    from midas_saxs import SAXSGeometry, SpherePopulation, simulate_frame

    geom = SAXSGeometry(lsd_um=2.0e6, bcy_px=512, bcz_px=512, px_um=75.0,
                        wavelength_A=0.7293, n_pix_y=1024, n_pix_z=1024,
                        beamstop_radius_px=30)
    voids = SpherePopulation(radius_A=50.0, number_density_per_A3=1e-8,
                             delta_rho_e_per_A3=-2.45, label="voids")
    frame = simulate_frame(geom, particles=[voids], sample_volume_A3=1e15)

The one number to keep in mind
------------------------------
A dislocation loop of radius R has relaxation volume ``pi R^2 b``, while a void
of the same radius displaces ``4 pi R^3 / 3``. At R = 5 nm in Cu that is a factor
26 in volume, hence 680 in intensity from volume alone, and the loop loses a
further ``(1 - kappa)^2`` even in its own plane (below). If an irradiated sample
has voids or bubbles, they dominate the small-angle image, and a
dislocation-only simulation will not resemble the measurement.

Aligned loops also differ from voids in **shape**. A loop's small-angle
amplitude is the distortion term plus the Laue term (Ehrhart, Trinkaus & Larson,
Phys. Rev. B 25 (1982) 834), which as q -> 0 is ``dV (1 - kappa) sin^2(theta)``,
theta measured from the loop normal, ``kappa = lambda/(lambda + 2 mu)``: zero
along the normal, largest in the loop plane. Voids are isotropic. The null lives
on the 2-D frame and is destroyed by radial averaging — see
:func:`midas_saxs.detector.azimuthal_profile`. It is not a general
loop-versus-void test: loops with isotropically distributed normals scatter
isotropically too.
"""

__version__ = "0.1.2"

from .core_shell import (
    core_shell_sphere_form_factor_squared,
    multi_shell_sphere_form_factor_squared,
)
from .detector import (
    Frame,
    azimuthal_profile,
    radial_average,
    simulate_frame,
)
from .form_factors import (
    cylinder_form_factor_squared,
    ellipsoid_form_factor_squared,
    percus_yevick_S,
    sphere_form_factor_squared,
)
from .geometry import (
    SAXSGeometry,
    inv_A_to_inv_um,
    inv_um_to_inv_A,
    pixel_to_q,
    q_magnitude_grid,
    two_theta_to_q,
)
from .model import SAXSModel, lognormal_quadrature_nodes
from .particles import (
    SpherePopulation,
    loop_equivalent_sphere_radius_A,
    void_intensity,
)
from .wide_band import (
    GuinierFit,
    PorodFit,
    guinier_fit,
    kratky_plot,
    porod_fit,
    porod_invariant,
    worm_like_chain_form_factor_squared,
)

# `midas_saxs.strain_source` is NOT imported here: it needs midas_ddd, which is
# an optional extra. Import it explicitly, or go through `simulate_frame`, which
# only reaches for it when a network is actually passed.

__all__ = [
    # geometry
    "SAXSGeometry", "pixel_to_q", "q_magnitude_grid", "two_theta_to_q",
    "inv_A_to_inv_um", "inv_um_to_inv_A",
    # detector
    "Frame", "simulate_frame", "radial_average", "azimuthal_profile",
    # particles
    "SpherePopulation", "void_intensity", "loop_equivalent_sphere_radius_A",
    # form factors (migrated from midas_pdf.saxs)
    "sphere_form_factor_squared", "ellipsoid_form_factor_squared",
    "cylinder_form_factor_squared", "percus_yevick_S",
    "core_shell_sphere_form_factor_squared", "multi_shell_sphere_form_factor_squared",
    "SAXSModel", "lognormal_quadrature_nodes",
    "GuinierFit", "guinier_fit", "PorodFit", "porod_fit", "porod_invariant",
    "kratky_plot", "worm_like_chain_form_factor_squared",
    "__version__",
]
