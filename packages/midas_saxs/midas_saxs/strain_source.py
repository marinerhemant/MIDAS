"""Small-angle amplitude of a dislocation network's strain field.

The small-angle amplitude of a dislocation loop has TWO parts, and taking only
the first inverts the answer:

    A(q) ~ rho_e [ i q.u~(q)  +  b A~(q) ]        (distortion + LAUE)

Ehrhart, Trinkaus & Larson, *Diffuse scattering from dislocation loops*, Phys.
Rev. B **25** (1982) 834, Eq. (8a). The Laue term is the scattering from the
loop's own extra or missing atoms, and at small angle -- where the scattering
vector IS the deviation, ``K = q`` -- it is the same order as the distortion
term, not a correction to it.

Their Eq. (8b) recasts the sum so that it carries a factor ``q x A~(q)``, hence
**the total vanishes identically for q along the loop normal**, precisely where
the distortion term alone is at its maximum ``dV``. The small-q laws are

    distortion only :  dV [ kappa + (1 - kappa) cos^2(theta) ]   <- NOT the answer
    total           :  dV (1 - kappa) sin^2(theta)               <- the answer

which sum to ``dV`` identically. This module uses
:func:`midas_ddd.small_angle_amplitude` (the total). An earlier version used
``q_dot_u_tilde`` (distortion only) and therefore reported a loop as scattering
most strongly along its normal, when in fact it does not scatter there at all.

Near a Bragg peak the distortion term carries a factor ``G`` the Laue term does
not, so ``G/q >> 1`` and :mod:`midas_defect.huang` is correct as it stands --
this is a small-angle-specific correction.

What this does and does not include
-----------------------------------
**Included:** closed dislocation loops -- the irradiation population, with both
the distortion and the Laue term. Their amplitude at ``q -> 0`` is
``rho_e dV (1 - kappa) sin^2(theta)``: zero along the loop normal, maximal in the
loop plane.

**Included, under test:** dislocation lines, the deformation population, through
the line-integral form of the same amplitude (Seeger & Kroner 1959). A straight
edge line scatters into a sheet perpendicular to itself; an isotropic screw does
not scatter at all. Two kinds of line are treated differently:

- **Finite lines** (open lines, junction clusters that do not wind) have an
  amplitude at every q: :func:`midas_ddd.line_small_angle_amplitude`.
- **Lines that close only through a periodic cell's boundary** have one only on
  the cell's reciprocal lattice. An earlier version evaluated them at pixel q,
  where the result depended on node numbering (/verify claim 8305670629a7,
  refuted). They now come as an intensity at a stated resolution,
  :func:`midas_ddd.periodic_small_angle_intensity`, preregistered 2026-09-10 and
  not yet verified.

See :func:`network_intensities`. :func:`loop_amplitude` remains loops-only.

**Not included:** voids, bubbles and precipitates. Those are *density* contrast
rather than strain contrast and live in :mod:`midas_saxs.particles`. In an
irradiated material they typically dominate the small-angle image by two to
three orders of magnitude, so a strain-only image will not resemble a
measurement. Use :func:`midas_saxs.detector.simulate_frame` with both.

Units
-----
``q`` arrives in **inverse angstroms** (the SAXS convention, what
:mod:`midas_saxs.geometry` returns) and is converted to inverse micrometers for
the kernel, whose node positions are micrometers. ``q.u~`` comes back in µm^3
and is converted to A^3 so it multiplies an electron density in e/A^3. The
conversions are named calls, never bare literals.
"""
from __future__ import annotations

import math
import warnings as _warnings
from typing import Dict, List, NamedTuple, Optional, Sequence, Union

import torch

from .geometry import inv_A_to_inv_um, inv_um_to_inv_A

__all__ = [
    "A3_PER_UM3",
    "electron_density_per_A3",
    "loop_amplitude",
    "loop_intensity",
    "network_amplitudes",
    "network_intensities",
]

#: 1 um^3 = 1e12 A^3.
A3_PER_UM3 = 1.0e12

_DDD_HINT = (
    "midas_ddd is required for the dislocation source term but is not installed. "
    "It is an optional extra so that plain particle SAXS does not pull a "
    "dislocation-dynamics stack:  pip install 'midas-saxs[dislocations]'"
)


def _require_ddd():
    try:
        import midas_ddd  # noqa: F401
        return midas_ddd
    except ImportError as exc:                    # pragma: no cover - env dependent
        raise ImportError(_DDD_HINT) from exc


def electron_density_per_A3(
    elements: Sequence[str],
    counts: Sequence[float],
    cell_volume_A3: float,
) -> float:
    """Average electron density, e/A^3, from the unit-cell contents.

    ``Z`` per species comes from :mod:`midas_hkls.form_factors` at ``s = 0``,
    where the atomic form factor equals the electron count. Never hard-code a
    Z table; the one in midas-hkls is the single source.

    Examples
    --------
    Copper, FCC, a = 3.615 A, 4 atoms per cell::

        electron_density_per_A3(["Cu"], [4], 3.615 ** 3)   # -> 2.455 e/A^3
    """
    from midas_hkls.form_factors import form_factor

    if len(elements) != len(counts):
        raise ValueError(f"{len(elements)} elements but {len(counts)} counts")
    if cell_volume_A3 <= 0:
        raise ValueError(f"cell_volume_A3 must be positive, got {cell_volume_A3}")
    total_Z = 0.0
    for el, n in zip(elements, counts):
        total_Z += float(n) * float(form_factor(0.0, el))
    return total_Z / float(cell_volume_A3)


def loop_amplitude(
    net,
    q_inv_A: torch.Tensor,
    C6: torch.Tensor,
    *,
    electron_density_e_per_A3: float,
    incoherent: bool = False,
    return_result: bool = False,
):
    """Small-angle scattering amplitude of the closed loops in ``net``.

    Parameters
    ----------
    net : midas_ddd.DislocationNetwork
    q_inv_A : (Q, 3) tensor
        Scattering vectors, **inverse angstroms**.
    C6 : (6, 6) tensor
        Voigt stiffness (only ratios matter).
    electron_density_e_per_A3
        Matrix electron density; see :func:`electron_density_per_A3`.
    incoherent
        ``False`` (default): sum loop amplitudes **coherently**, keeping the
        positional phases. This is what a specific configuration scatters, and
        it produces speckle. ``True``: sum ``|A|^2`` loop by loop, the
        ensemble-averaged result appropriate to a dilute random population under
        a beam much larger than the inter-loop spacing -- which is what a
        conventional SAXS measurement actually reports.

    Returns
    -------
    Complex ``(Q,)`` amplitude in electrons when ``incoherent`` is False; real
    ``(Q,)`` intensity in electrons² when it is True. Pass ``return_result=True``
    to also get the :class:`midas_ddd.FourierResult`, which carries the ignored
    open-line length and the quadrature diagnostic.
    """
    ddd = _require_ddd()
    from midas_ddd.fourier import small_angle_amplitude, u_tilde
    from midas_ddd.validate import find_loops

    q_inv_A = torch.as_tensor(q_inv_A, dtype=net.nodes_um.dtype,
                              device=net.nodes_um.device)
    if q_inv_A.ndim == 1:
        q_inv_A = q_inv_A.unsqueeze(0)
    q_um = inv_A_to_inv_um(q_inv_A)

    loops = find_loops(net)
    scale = electron_density_e_per_A3 * A3_PER_UM3      # e/A^3 * A^3/um^3 -> e/um^3

    # ONE pass, whichever mode: `per_loop=True` returns both the coherent total
    # and the per-loop stack the incoherent sum needs. Asking for them
    # separately evaluated every loop's surface form factor twice, which
    # profiling showed was the entire cost of a frame.
    # `res` is kept only for its diagnostics (ignored open-line length,
    # quadrature qh, loop count); the AMPLITUDE comes from the total, which
    # includes the Laue term.
    res = u_tilde(net, q_um, C6, loops=loops)
    per = small_angle_amplitude(net, q_um, C6, loops=loops, per_loop=True)  # (L,Q)

    if incoherent:
        total = ((scale * per).abs() ** 2).sum(dim=0)
        return (total, res) if return_result else total

    if len(loops) > 1 and any(bool(p) for p in net.pbc):
        _warnings.warn(
            "loop_amplitude(incoherent=False) sums several loops coherently in a periodic "
            "cell. Each loop's position is fixed only up to a lattice vector, so off the "
            "cell's reciprocal lattice that coherent sum is not defined by the network "
            "alone. Use incoherent=True, or network_intensities(coherent=True).",
            stacklevel=2)
    amp = -1j * scale * per.sum(dim=0)
    return (amp, res) if return_result else amp


def loop_intensity(
    net,
    q_inv_A: torch.Tensor,
    C6: torch.Tensor,
    *,
    electron_density_e_per_A3: float,
    incoherent: bool = False,
) -> torch.Tensor:
    """``|A(q)|^2`` for the loop population. ``(Q,)`` real, electrons²."""
    out = loop_amplitude(net, q_inv_A, C6,
                         electron_density_e_per_A3=electron_density_e_per_A3,
                         incoherent=incoherent)
    return out if incoherent else out.abs() ** 2


class _Parts(NamedTuple):
    """Every segment of a network in exactly one of: a closed loop, a finite line, a winding line."""

    loops: list
    in_loop: torch.Tensor           # (M,) bool
    components: torch.Tensor        # (M,) int64 connected-component label
    finite: torch.Tensor            # (M,) bool: not a loop, and its component does not wind
    winding: torch.Tensor           # (M,) bool: its component closes through the periodic boundary
    n_finite_objects: int
    n_winding_components: int


def _network_parts(net) -> _Parts:
    from midas_ddd.fourier import winding_components
    from midas_ddd.validate import find_loops

    dev = net.nodes_um.device
    loops = find_loops(net)
    in_loop = torch.zeros(net.n_segments, dtype=torch.bool, device=dev)
    for lp in loops:
        in_loop[torch.as_tensor(lp.segment_indices, dtype=torch.int64, device=dev)] = True
    components, _, winds = winding_components(net)
    winds = winds.to(dev)
    winding = winds[components] & ~in_loop
    finite = ~in_loop & ~winding
    return _Parts(loops=loops, in_loop=in_loop, components=components, finite=finite,
                  winding=winding,
                  n_finite_objects=int(torch.unique(components[finite]).numel()),
                  n_winding_components=int(torch.unique(components[winding]).numel()))


def _group_by_component(components: torch.Tensor, mask: torch.Tensor):
    """Labels ``0..C-1`` for the components that ``mask`` touches, ``-1`` elsewhere."""
    ids = torch.unique(components[mask])
    size = int(components.max()) + 1 if components.numel() else 1
    remap = torch.full((size,), -1, dtype=torch.int64, device=components.device)
    remap[ids] = torch.arange(ids.numel(), dtype=torch.int64, device=components.device)
    return torch.where(mask, remap[components], torch.full_like(components, -1)), int(ids.numel())


def _line_end_warning(net) -> List[str]:
    degree = torch.bincount(net.segments.reshape(-1), minlength=net.n_nodes)
    n_ends = int((degree == 1).sum())
    if not n_ends:
        return []
    return [f"{n_ends} line end(s) terminate inside the medium. Burgers vector is not "
            f"conserved there, so the amplitude includes those ends' unphysical sources, "
            f"which scatter off the lines' reciprocal-space sheets."]


def _object_amplitudes(net, q_um, C6, parts: _Parts, *, scale, include_lines, loop_method,
                       warnings: List[str]):
    """``(amp_loops (L, Q), amp_lines (C, Q))`` in electrons, for the objects defined at every q."""
    from midas_ddd.fourier import line_small_angle_amplitude, loop_vertices, small_angle_amplitude

    dev = net.nodes_um.device
    Q = q_um.shape[0]
    cdt = torch.complex128 if net.nodes_um.dtype == torch.float64 else torch.complex64
    amp_loops = torch.zeros(0, Q, dtype=cdt, device=dev)
    if parts.loops:
        per = small_angle_amplitude(net, q_um, C6, loops=parts.loops, per_loop=True,
                                    method=loop_method)
        amp_loops = -1j * scale * per
        if loop_method == "surface":
            qmax = float(torch.linalg.vector_norm(q_um, dim=-1).max())
            rmax = max(float(torch.linalg.vector_norm(V - V.mean(dim=0), dim=-1).max())
                       for V in (loop_vertices(net, lp) for lp in parts.loops))
            if rmax * qmax > 0.4 * 64:
                warnings.append(
                    f"loop cut-surface quadrature is capped at 64 rings but qR reaches "
                    f"{rmax * qmax:.3g}; use loop_method='line', which needs no "
                    f"triangulation.")
    amp_lines = torch.zeros(0, Q, dtype=cdt, device=dev)
    if include_lines and bool(parts.finite.any()):
        labels, n_line = _group_by_component(parts.components, parts.finite)
        per = line_small_angle_amplitude(net, q_um, C6, per_component=True,
                                         labels=labels, n_groups=n_line)
        amp_lines = -1j * scale * per
    return amp_loops, amp_lines


def _left_out_warning(net, parts: _Parts) -> List[str]:
    mask = parts.finite | parts.winding
    if not bool(mask.any()):
        return []
    length = float(net.segment_lengths_um()[mask].sum().detach())
    return [f"{int(mask.sum())} segment(s) ({length:.3g} um of line) are not in any closed "
            f"loop and were left out (include_lines=False)."]


def network_amplitudes(
    net,
    q_inv_A: torch.Tensor,
    C6: torch.Tensor,
    *,
    electron_density_e_per_A3: float,
    include_lines: bool = True,
    loop_method: str = "line",
):
    """Small-angle amplitude, in electrons, of every object in ``net`` that HAS one at every q.

    Returns ``(amps, info)``:

    * ``amps["loops"]``: ``(L, Q)`` complex, one row per closed loop, from
      :func:`midas_ddd.small_angle_amplitude` (``loop_method``).
    * ``amps["lines"]``: ``(C, Q)`` complex, one row per FINITE connected component
      that is not a simple closed loop (open lines, junction clusters that do not
      wind), from :func:`midas_ddd.line_small_angle_amplitude`.
    * ``info``: counts, line lengths and warnings.

    **Not here:** lines that close only through the periodic boundary. They have no
    amplitude off the cell's reciprocal lattice (/verify claim 8305670629a7).
    :func:`network_intensities` gives their intensity at a stated resolution;
    ``info["n_winding_components"]`` says how many were left to it.

    **How to sum the rows.** Each row is the coherent amplitude of one object. Summing
    ``|row|^2`` gives the dilute random population under a beam much larger than the
    object spacing. Summing the rows first is the coherent case. In a periodic cell
    that coherent sum is not defined off the lattice, because each object's lattice
    translation is arbitrary; use ``network_intensities(coherent=True)`` instead.
    """
    _require_ddd()
    dev = net.nodes_um.device
    q_inv_A = torch.as_tensor(q_inv_A, dtype=net.nodes_um.dtype, device=dev)
    if q_inv_A.ndim == 1:
        q_inv_A = q_inv_A.unsqueeze(0)
    q_um = inv_A_to_inv_um(q_inv_A)
    scale = electron_density_e_per_A3 * A3_PER_UM3
    warnings: List[str] = []
    parts = _network_parts(net)
    amp_loops, amp_lines = _object_amplitudes(net, q_um, C6, parts, scale=scale,
                                              include_lines=include_lines,
                                              loop_method=loop_method, warnings=warnings)
    seg_len = net.segment_lengths_um().detach()
    if include_lines:
        if bool(parts.finite.any()):
            warnings.extend(_line_end_warning(net))
        if parts.n_winding_components:
            warnings.append(
                f"{parts.n_winding_components} line(s) close only through the periodic "
                f"boundary and have no amplitude off the cell's reciprocal lattice; they "
                f"are not in these rows. Use network_intensities.")
    else:
        warnings.extend(_left_out_warning(net, parts))
    info = dict(
        n_loops=len(parts.loops),
        n_line_objects=parts.n_finite_objects if include_lines else 0,
        n_winding_components=parts.n_winding_components,
        line_length_um=float(seg_len[parts.finite].sum()) if include_lines else 0.0,
        warnings=warnings,
    )
    return {"loops": amp_loops, "lines": amp_lines}, info


def network_intensities(
    net,
    q_inv_A: torch.Tensor,
    C6: torch.Tensor,
    *,
    electron_density_e_per_A3: float,
    include_lines: bool = True,
    coherent: bool = False,
    periodic_resolution_fwhm_inv_A: Optional[float] = None,
    loop_method: str = "line",
):
    """Small-angle intensity of a dislocation network, by component, in electrons² per cell.

    Returns ``(components, info)``. ``components`` maps a name to a ``(Q,)`` real
    intensity; the components add up to the total.

    **Incoherent** (default), what a beam much larger than the object spacing measures:

    * ``"loops"``: ``sum |A_loop(q)|^2`` over closed loops, exact at every q.
    * ``"lines"``: ``sum |A(q)|^2`` over finite line objects (open lines, junction
      clusters that do not wind).
    * ``"periodic_lines"``: lines that close only through the periodic boundary.
      - All of them are summed coherently together, because a dipole's screening
        lives in the interference.
      - Computed as the cell's exact reciprocal-lattice intensities averaged at the
        stated resolution (:func:`midas_ddd.periodic_small_angle_intensity`).
      - Preregistered 2026-09-10 and not yet verified.

    **Coherent** (``coherent=True``), the speckle of one specific configuration:

    * **Non-periodic cell, or a periodic cell holding one finite object:** the
      coherent sum at every q.
    * **Periodic cell with two or more objects or any winding line:** the coherent
      total is defined only on the reciprocal lattice, so every included dislocation
      goes through :func:`midas_ddd.periodic_small_angle_intensity`, with a warning.
    * **Key:** ``"dislocations"`` when loops and lines are both present, otherwise
      ``"loops"`` or ``"lines"``.
    * A cell periodic along only some axes, where the lattice would be needed, raises
      ``NotImplementedError``.

    ``info`` holds counts and lengths, ``warnings``, and ``periodic``. ``periodic``
    is ``None`` unless the lattice path ran; then it is a dict with
    ``fwhm_inv_A``, ``q_floor_inv_A``, ``n_lattice_points``, ``n_eff_median``,
    ``ripple_bound``, ``roundoff_floor_max``, ``above_roundoff`` and ``burgers``.
    Pixels below ``q_floor_inv_A`` are not representative. A component that never
    rises above ``1e6 x`` its float64 roundoff floor is reported as roundoff (a
    pure-screw network, for instance), and a winding network that does not conserve
    Burgers vector at a free node is reported as unphysical; both go into
    ``warnings``.
    """
    _require_ddd()
    from midas_ddd.fourier import periodic_small_angle_intensity

    dev = net.nodes_um.device
    q_inv_A = torch.as_tensor(q_inv_A, dtype=net.nodes_um.dtype, device=dev)
    if q_inv_A.ndim == 1:
        q_inv_A = q_inv_A.unsqueeze(0)
    q_um = inv_A_to_inv_um(q_inv_A)
    scale = electron_density_e_per_A3 * A3_PER_UM3
    warnings: List[str] = []
    parts = _network_parts(net)
    fully_periodic = all(bool(p) for p in net.pbc)
    any_periodic = any(bool(p) for p in net.pbc)
    fwhm_um = (None if periodic_resolution_fwhm_inv_A is None
               else float(inv_A_to_inv_um(float(periodic_resolution_fwhm_inv_A))))
    lines_mask = (parts.finite | parts.winding) if include_lines else torch.zeros_like(parts.finite)
    has_loops, has_lines = bool(parts.loops), bool(lines_mask.any())
    has_winding = include_lines and parts.n_winding_components > 0
    n_objects = (len(parts.loops) + (parts.n_finite_objects if include_lines else 0)
                 + (1 if has_winding else 0))
    periodic = None

    def through_lattice(mask, what):
        if not fully_periodic:
            raise NotImplementedError(
                f"{what} needs the cell's reciprocal lattice, which is used only for a cell "
                f"periodic along all three axes; got pbc={tuple(bool(p) for p in net.pbc)}.")
        labels = torch.where(mask, torch.zeros_like(parts.components),
                             torch.full_like(parts.components, -1))
        with _warnings.catch_warnings():
            _warnings.simplefilter("ignore")        # reported through info["warnings"] instead
            I, pinfo = periodic_small_angle_intensity(net, q_um, C6, fwhm_inv_um=fwhm_um,
                                                      labels=labels, n_groups=1)
        I = scale ** 2 * I
        floor = scale ** 2 * pinfo["roundoff_floor"]
        above = bool((I > 1e6 * floor).any())
        if not above:
            warnings.append(
                f"the {what.lower()} component peaks at {float(I.max()):.3g} electrons^2 "
                f"against a float64 roundoff floor of {float(floor.max()):.3g}: it is roundoff, "
                f"not signal. Pure screws have no dilatation in isotropic or cubic elasticity, "
                f"and a pure-screw network scatters nothing at small angle.")
        bg = pinfo["burgers"]
        if bg["n_violating"]:
            warnings.append(
                f"Burgers vector is not conserved at {bg['n_violating']} free node(s) of the "
                f"{what.lower()} (largest residual {bg['max_residual_b']:.3g} b); the network "
                f"is not an elastic solution there.")
        return I, dict(
            fwhm_inv_A=float(inv_um_to_inv_A(pinfo["fwhm_inv_um"])),
            q_floor_inv_A=float(inv_um_to_inv_A(pinfo["q_floor_inv_um"])),
            n_lattice_points=int(pinfo["n_lattice_points"]),
            n_eff_median=float(pinfo["n_eff"].median()) if pinfo["n_eff"].numel() else float("nan"),
            ripple_bound=float(pinfo["ripple_bound"]),
            roundoff_floor_max=float(floor.max()) if floor.numel() else 0.0,
            above_roundoff=above,
            burgers=bg)

    components: Dict[str, torch.Tensor] = {}
    if coherent:
        key = "dislocations" if (has_loops and has_lines) else "lines" if has_lines else "loops"
        if (has_loops or has_lines) and any_periodic and (has_winding or n_objects > 1):
            I, periodic = through_lattice(parts.in_loop | lines_mask,
                                          "Coherent dislocation sum")
            components[key] = I
            warnings.append(
                f"coherent sum in a periodic cell: it is defined only on the cell's "
                f"reciprocal lattice, so every dislocation, loops included, is shown at a "
                f"resolution of FWHM {periodic['fwhm_inv_A']:.3g} 1/A "
                f"(midas_ddd.periodic_small_angle_intensity).")
        elif has_loops or has_lines:
            amp_loops, amp_lines = _object_amplitudes(net, q_um, C6, parts, scale=scale,
                                                      include_lines=include_lines,
                                                      loop_method=loop_method,
                                                      warnings=warnings)
            A = amp_loops.sum(dim=0) + amp_lines.sum(dim=0)
            components[key] = A.real ** 2 + A.imag ** 2
    else:
        amp_loops, amp_lines = _object_amplitudes(net, q_um, C6, parts, scale=scale,
                                                  include_lines=include_lines,
                                                  loop_method=loop_method, warnings=warnings)
        if amp_loops.shape[0]:
            components["loops"] = (amp_loops.abs() ** 2).sum(dim=0)
        if amp_lines.shape[0]:
            components["lines"] = (amp_lines.abs() ** 2).sum(dim=0)
        if has_winding:
            components["periodic_lines"], periodic = through_lattice(
                parts.winding, "Periodic lines")

    if include_lines and has_lines:
        warnings.extend(_line_end_warning(net))
    if not include_lines:
        warnings.extend(_left_out_warning(net, parts))
    seg_len = net.segment_lengths_um().detach()
    info = dict(
        n_loops=len(parts.loops),
        n_line_objects=parts.n_finite_objects if include_lines else 0,
        n_winding_components=parts.n_winding_components if include_lines else 0,
        line_length_um=float(seg_len[parts.finite].sum()) if include_lines else 0.0,
        winding_length_um=float(seg_len[parts.winding].sum()) if include_lines else 0.0,
        periodic=periodic,
        warnings=warnings,
    )
    return components, info
