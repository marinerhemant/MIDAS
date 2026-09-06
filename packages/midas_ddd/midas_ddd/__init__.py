"""midas-ddd — dislocation networks for MIDAS.

Ingests a discrete-dislocation-dynamics network (ExaDiS / ParaDiS) and turns it
into the displacement fields that three MIDAS forward models consume:

    real space   beta(r)   ->  midas_dfxm      (DFXM contrast)
    Fourier      u~(q)     ->  midas_saxs      (small-angle, evaluated at q)
    Fourier      u~(q)     ->  midas_defect    (near-Bragg diffuse, at G + q)

One kernel, three evaluation points. Everything is torch-differentiable.

    from midas_ddd import cubic_stiffness, fcc_slip_systems

Why this package exists
-----------------------
The anisotropic-elasticity (Stroh) primitives here were previously private to
``midas_defect.contrast_factor``, which already carried a "single source of
truth, do NOT re-port" contract because ``midas_dfxm`` imported them across a
package boundary. Two more consumers made that arrangement untenable: a SAXS
package should not have to depend on an FF-HEDM metrology package to build a
stiffness matrix. ``midas_defect.contrast_factor`` now re-exports every name,
so no historical import path broke.

ExaDiS itself requires a Kokkos/CMake build and is never a pip dependency. The
ParaDiS ``.data`` file bridge works without it; the in-process ``pyexadis`` path
is optional and imported lazily.
"""

__version__ = "0.1.1"

from .elasticity import (
    bcc_slip_systems,
    cubic_stiffness,
    fcc_slip_systems,
    hexagonal_stiffness,
)
from .generate import (
    combine,
    polygon_area_exact_um2,
    prismatic_loop,
    straight_line,
)
from .fourier import (
    FourierResult,
    acoustic_tensor,
    isotropic_stiffness,
    loop_small_q_limit,
    prismatic_loop_small_q_limit,
    prismatic_loop_small_q_limit_total,
    q_dot_u_tilde,
    q_dot_u_tilde_per_loop,
    small_angle_amplitude,
    surface_form_factor,
    u_tilde,
)
from .network import DislocationNetwork, read_paradis, write_paradis
from .realspace import (
    SegmentDislocation,
    green_gradient_isotropic,
    lame_from_voigt,
    network_distortion,
    segment_dislocations,
)
from .validate import (
    BurgersConservationResult,
    Loop,
    ResolutionReport,
    check_burgers_conservation,
    find_loops,
    loop_area_vectors_um2,
    relaxation_volumes_um3,
    resolution_report,
    validate_network,
)

__all__ = [
    # elasticity
    "bcc_slip_systems",
    "cubic_stiffness",
    "fcc_slip_systems",
    "hexagonal_stiffness",
    # network
    "DislocationNetwork",
    "read_paradis",
    "write_paradis",
    # generators (ExaDiS-free equivalents; see midas_ddd.exadis for the real ones)
    "combine",
    "polygon_area_exact_um2",
    "prismatic_loop",
    "straight_line",
    # Fourier kernel
    "FourierResult",
    "acoustic_tensor",
    "isotropic_stiffness",
    "loop_small_q_limit",
    "prismatic_loop_small_q_limit",
    "prismatic_loop_small_q_limit_total",
    "q_dot_u_tilde",
    "q_dot_u_tilde_per_loop",
    "small_angle_amplitude",
    "surface_form_factor",
    "u_tilde",
    # real-space distortion (ELASTIC part; see the module docstring)
    "SegmentDislocation",
    "green_gradient_isotropic",
    "lame_from_voigt",
    "network_distortion",
    "segment_dislocations",
    # validation
    "BurgersConservationResult",
    "Loop",
    "ResolutionReport",
    "check_burgers_conservation",
    "find_loops",
    "loop_area_vectors_um2",
    "relaxation_volumes_um3",
    "resolution_report",
    "validate_network",
    "__version__",
]

# `midas_ddd.exadis` is NOT imported here: it pulls pyexadis, which needs a
# Kokkos build. Import it explicitly when you have one.
