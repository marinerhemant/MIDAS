"""Build dislocation configurations without ExaDiS.

ExaDiS ships the real generators -- ``generate_prismatic_config``,
``generate_line_config``, ``insert_prismatic_loop``, ``insert_frank_read_src``
in ``pyexadis_utils`` -- and :mod:`midas_ddd.exadis` calls them when a build is
available. But ExaDiS needs Kokkos and CMake, so it can never be a pip
dependency, and a test suite that can only run where ExaDiS is installed is a
test suite that does not run.

These are small, exact, pure-torch equivalents for the two configurations this
project actually needs:

* :func:`prismatic_loop` -- an irradiation-type loop, the population whose
  relaxation volume carries the entire small-angle signal.
* :func:`straight_line` -- a deformation-type line, which has no relaxation
  volume and is the negative control for the loop gate.

They are deliberately analytic: a regular polygon's area vector is known in
closed form, so :func:`prismatic_loop` is what makes the ``dV = b . A`` gate
checkable against something other than the code being tested.
"""
from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple

import torch

from .network import DislocationNetwork

__all__ = [
    "prismatic_loop",
    "straight_line",
    "combine",
    "polygon_area_exact_um2",
]

_ANGSTROM_PER_UM = 1.0e4


def _orthonormal_basis(n: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Two unit vectors spanning the plane normal to ``n``."""
    n = n / torch.linalg.norm(n)
    seed = torch.tensor([1.0, 0.0, 0.0], dtype=n.dtype, device=n.device)
    if torch.abs(n @ seed) > 0.9:
        seed = torch.tensor([0.0, 1.0, 0.0], dtype=n.dtype, device=n.device)
    e1 = seed - (seed @ n) * n
    e1 = e1 / torch.linalg.norm(e1)
    e2 = torch.linalg.cross(n, e1)
    return e1, e2


def polygon_area_exact_um2(radius_um: float, n_segments: int) -> float:
    """Area of the regular ``n``-gon inscribed in a circle of ``radius_um``.

    ``(n/2) R^2 sin(2 pi / n)``, which tends to ``pi R^2`` as ``n -> inf``. The
    gate on relaxation volume must use *this*, not ``pi R^2``: a 12-gon is 3.4 %
    smaller than its circumscribing circle, which would masquerade as a 3.4 %
    error in the kernel.
    """
    return 0.5 * n_segments * radius_um ** 2 * math.sin(2.0 * math.pi / n_segments)


def prismatic_loop(
    *,
    radius_um: float,
    burgers: Sequence[float] = (1.0, 1.0, 1.0),
    center_um: Sequence[float] = (0.0, 0.0, 0.0),
    n_segments: int = 24,
    b_magnitude_A: float = 2.556,
    burgers_scale_b: float = 1.0,
    cell_size_um: Optional[float] = None,
    pbc: Tuple[bool, bool, bool] = (False, False, False),
    dtype: torch.dtype = torch.float64,
    device=None,
) -> DislocationNetwork:
    """A single closed prismatic loop: a regular polygon in the plane normal to ``b``.

    "Prismatic" means the Burgers vector is along the loop normal, so the loop
    is a disc of inserted (interstitial) or removed (vacancy) material. Its
    relaxation volume is ``dV = b . A = |b| * area``, with the sign set by the
    circulation sense relative to ``b`` -- flip ``burgers_scale_b`` to negative
    to turn an interstitial loop into a vacancy loop.

    Parameters
    ----------
    radius_um
        Circumradius of the polygon.
    burgers
        Burgers direction (need not be normalised); also the loop-plane normal.
    n_segments
        Polygon sides. The exact area is :func:`polygon_area_exact_um2`.
    burgers_scale_b
        ``|b|`` in units of ``b``. 1.0 for a perfect loop; negative flips the
        loop character (interstitial <-> vacancy). Passing a negative ``burgers``
        direction does the same thing -- the circulation is fixed by a canonical
        normal, so the sign of b is not absorbed by the geometry.
    cell_size_um
        Cubic cell edge. Defaults to 8x the loop diameter, so the loop is
        isolated and the default non-periodic convention is safe.
    """
    if n_segments < 3:
        raise ValueError(f"a loop needs at least 3 segments, got {n_segments}")
    if radius_um <= 0:
        raise ValueError(f"radius_um must be positive, got {radius_um}")

    bdir = torch.as_tensor(burgers, dtype=dtype, device=device)
    if float(torch.linalg.norm(bdir)) == 0.0:
        raise ValueError("burgers must be a non-zero vector")
    bdir = bdir / torch.linalg.norm(bdir)

    # The loop's CIRCULATION follows a canonical normal (first non-zero component
    # positive), not the Burgers direction itself. Without this the geometry
    # flipped along with b, so A flipped too and dV = b.A stayed positive:
    # `prismatic_loop(burgers=(0,0,-1))` silently produced an INTERSTITIAL loop,
    # making the vacancy character unreachable through the obvious entry point.
    # Now b keeps its own sign against a fixed circulation, so a negative Burgers
    # direction and `burgers_scale_b=-1` both give a vacancy loop, consistently.
    nrm = bdir.clone()
    nz = torch.nonzero(torch.abs(nrm) > 1e-12).flatten()
    if float(nrm[nz[0]]) < 0:
        nrm = -nrm
    e1, e2 = _orthonormal_basis(nrm)
    c = torch.as_tensor(center_um, dtype=dtype, device=device)

    ang = torch.arange(n_segments, dtype=dtype, device=device) * (2.0 * math.pi / n_segments)
    nodes = c + radius_um * (torch.cos(ang)[:, None] * e1 + torch.sin(ang)[:, None] * e2)

    idx = torch.arange(n_segments, dtype=torch.int64, device=device)
    segs = torch.stack([idx, torch.roll(idx, -1)], dim=1)

    # Every segment carries the same b, traversed in the circulation sense. With
    # two arms per node carrying +b and -b, Burgers conservation is exact.
    burg = (bdir * burgers_scale_b).expand(n_segments, 3).clone()
    # Glide plane of a prismatic segment: contains both the line and b. The loop
    # itself is sessile in its own plane, so record the segment-local plane.
    seg_dir = nodes[segs[:, 1]] - nodes[segs[:, 0]]
    pl = torch.linalg.cross(seg_dir, bdir.expand_as(seg_dir))
    pn = torch.linalg.vector_norm(pl, dim=-1, keepdim=True)
    normals = torch.where(pn > 1e-14, pl / pn.clamp(min=1e-30), torch.zeros_like(pl))

    half = (cell_size_um if cell_size_um is not None else 8.0 * radius_um) / 2.0
    return DislocationNetwork(
        nodes_um=nodes,
        segments=segs,
        burgers_b=burg,
        normals=normals,
        b_magnitude_A=float(b_magnitude_A),
        cell_min_um=c - half,
        cell_max_um=c + half,
        pbc=pbc,
        constraints=torch.zeros(n_segments, dtype=torch.int64, device=device),
        node_tags=[f"0,{i}" for i in range(n_segments)],
        source="midas_ddd.generate.prismatic_loop",
    )


def straight_line(
    *,
    length_um: float,
    burgers: Sequence[float] = (1.0, 1.0, 0.0),
    line: Sequence[float] = (1.0, -1.0, 0.0),
    slip_normal: Sequence[float] = (1.0, 1.0, 1.0),
    center_um: Sequence[float] = (0.0, 0.0, 0.0),
    n_segments: int = 20,
    b_magnitude_A: float = 2.556,
    cell_size_um: Optional[float] = None,
    dtype: torch.dtype = torch.float64,
    device=None,
) -> DislocationNetwork:
    """An open straight dislocation of finite length, pinned at both ends.

    The end nodes get ParaDiS constraint code 7 (pinned), because an open line
    genuinely does not conserve Burgers flux at its termini -- the line has to
    end somewhere, and :func:`check_burgers_conservation` skips constrained
    nodes for exactly this reason.

    This is the deformation-type population, and the negative control for the
    loop gate: it has no enclosed area, hence no relaxation volume, hence no
    ``q -> 0`` small-angle signal.
    """
    if n_segments < 1:
        raise ValueError(f"need at least 1 segment, got {n_segments}")

    t = torch.as_tensor(line, dtype=dtype, device=device)
    t = t / torch.linalg.norm(t)
    c = torch.as_tensor(center_um, dtype=dtype, device=device)
    s = torch.linspace(-0.5 * length_um, 0.5 * length_um, n_segments + 1,
                       dtype=dtype, device=device)
    nodes = c + s[:, None] * t

    idx = torch.arange(n_segments, dtype=torch.int64, device=device)
    segs = torch.stack([idx, idx + 1], dim=1)

    bdir = torch.as_tensor(burgers, dtype=dtype, device=device)
    bdir = bdir / torch.linalg.norm(bdir)
    burg = bdir.expand(n_segments, 3).clone()

    nvec = torch.as_tensor(slip_normal, dtype=dtype, device=device)
    nvec = nvec / torch.linalg.norm(nvec)
    normals = nvec.expand(n_segments, 3).clone()

    cons = torch.zeros(n_segments + 1, dtype=torch.int64, device=device)
    cons[0] = 7
    cons[-1] = 7

    half = (cell_size_um if cell_size_um is not None else 4.0 * length_um) / 2.0
    return DislocationNetwork(
        nodes_um=nodes,
        segments=segs,
        burgers_b=burg,
        normals=normals,
        b_magnitude_A=float(b_magnitude_A),
        cell_min_um=c - half,
        cell_max_um=c + half,
        pbc=(False, False, False),
        constraints=cons,
        node_tags=[f"0,{i}" for i in range(n_segments + 1)],
        source="midas_ddd.generate.straight_line",
    )


def combine(
    nets: Sequence[DislocationNetwork],
    *,
    cell_size_um: Optional[float] = None,
    cell_min_um: Optional[Sequence[float]] = None,
    cell_max_um: Optional[Sequence[float]] = None,
    warn_inflation: float = 1.5,
) -> DislocationNetwork:
    """Merge networks into one, re-indexing nodes.

    Used to build a mixed population -- deformation lines plus irradiation loops
    in one box, which is the configuration the whole project is aimed at.
    Requires a common ``b_magnitude_A``: two different Burgers scales in one
    network would make ``burgers_b`` ambiguous.

    **Pass the real box.** Without ``cell_size_um`` (or an explicit min/max) the
    cell is the UNION of the inputs' cells, and each generated loop carries its
    own cell centred on itself. Union those and you get a volume much larger than
    the box the loops were actually placed in -- measured: 14 loops placed in a
    3.2 um box gave a union of 123.7 um^3 against the true 32.8, so
    :meth:`DislocationNetwork.dislocation_density_um2` came out **3.8x too low**.
    Any density, and anything derived from one, is wrong by that factor. A
    warning fires when the union exceeds the largest input cell by more than
    ``warn_inflation``.
    """
    nets = list(nets)
    if not nets:
        raise ValueError("combine() needs at least one network")
    b0 = nets[0].b_magnitude_A
    for n in nets[1:]:
        if abs(n.b_magnitude_A - b0) > 1e-12:
            raise ValueError(
                f"cannot combine networks with different b_magnitude_A "
                f"({b0} vs {n.b_magnitude_A}); burgers_b would mean two things at once")

    nodes, segs, burg, norms, cons, tags = [], [], [], [], [], []
    offset = 0
    for k, n in enumerate(nets):
        nodes.append(n.nodes_um)
        segs.append(n.segments + offset)
        burg.append(n.burgers_b)
        norms.append(n.normals)
        cons.append(n.constraints if n.constraints is not None
                    else torch.zeros(n.n_nodes, dtype=torch.int64, device=n.nodes_um.device))
        tags.extend(f"{k},{i}" for i in range(n.n_nodes))
        offset += n.n_nodes

    dtype, device = nets[0].nodes_um.dtype, nets[0].nodes_um.device
    if cell_size_um is not None:
        if cell_min_um is not None or cell_max_um is not None:
            raise ValueError("pass cell_size_um OR cell_min_um/cell_max_um, not both")
        half = 0.5 * float(cell_size_um)
        cmin = torch.full((3,), -half, dtype=dtype, device=device)
        cmax = torch.full((3,), half, dtype=dtype, device=device)
    elif cell_min_um is not None and cell_max_um is not None:
        cmin = torch.as_tensor(cell_min_um, dtype=dtype, device=device)
        cmax = torch.as_tensor(cell_max_um, dtype=dtype, device=device)
    else:
        cmin = torch.stack([n.cell_min_um for n in nets]).min(dim=0).values
        cmax = torch.stack([n.cell_max_um for n in nets]).max(dim=0).values
        union_vol = float(torch.prod(cmax - cmin))
        biggest = max(float(torch.prod(n.cell_max_um - n.cell_min_um)) for n in nets)
        if biggest > 0 and union_vol > warn_inflation * biggest:
            import warnings
            warnings.warn(
                f"combine(): the unioned cell is {union_vol / biggest:.1f}x the "
                f"largest input cell ({union_vol:.3g} vs {biggest:.3g} um^3). Each "
                f"input carries its own cell centred on itself, so this is almost "
                f"certainly NOT the box you placed them in, and "
                f"dislocation_density_um2() will be too low by that factor. Pass "
                f"cell_size_um=<box> to fix it.", RuntimeWarning, stacklevel=2)
    pbc = tuple(all(n.pbc[a] for n in nets) for a in range(3))

    return DislocationNetwork(
        nodes_um=torch.cat(nodes),
        segments=torch.cat(segs),
        burgers_b=torch.cat(burg),
        normals=torch.cat(norms),
        b_magnitude_A=b0,
        cell_min_um=cmin,
        cell_max_um=cmax,
        pbc=pbc,  # type: ignore[arg-type]
        constraints=torch.cat(cons),
        node_tags=tags,
        source="midas_ddd.generate.combine(" + ", ".join(n.source for n in nets) + ")",
    )
