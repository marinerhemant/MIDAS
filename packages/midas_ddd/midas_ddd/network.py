"""Dislocation network container and ParaDiS / ExaDiS ``.data`` ingest.

The ``.data`` restart format is what ExaDiS's own ``read_paradis`` /
``write_data`` read and write, so it is the bridge that always works -- ExaDiS
needs a Kokkos/CMake build and can never be a pip dependency, but its output
files are plain text.

Units (MIDAS convention: micrometers, degrees, angstroms for lattice scale)
---------------------------------------------------------------------------
ParaDiS stores **everything in units of the Burgers magnitude b**: node
coordinates, the cell bounds, and the Burgers vectors (a perfect dislocation
has ``|b| = 1`` in file units). On read we convert:

* node positions and cell bounds  ->  **micrometers**
* Burgers vectors                 ->  kept **in units of b**, exactly as written,
  with the scale carried alongside as ``b_magnitude_A`` (angstroms)

Burgers vectors are deliberately **not** normalised. A perfect dislocation has
``|b| = 1`` in file units, but a junction formed by a dislocation reaction has
``|b| = sqrt(2)``, and that magnitude is exactly what balances Burgers
conservation at the junction node. The real 5055-node FCC-Cu network in this
repo has 108 such arms out of 11860; normalising them made 60 nodes fail the
conservation check with a residual of precisely ``sqrt(2) - 1``.

:meth:`DislocationNetwork.burgers_directions` and
:meth:`DislocationNetwork.burgers_magnitudes_b` split them back into the
``(burgers=..., burgers_length_A=...)`` form that
:func:`midas_dfxm.dislocation.stroh_dislocation` takes, so handing a network to
the DFXM forward still needs no unit surgery at the call site.

Sign convention
---------------
ParaDiS records one *arm* per node per neighbour, with the Burgers vector
pointing **outward from that node along that arm**. A segment ``(i, j)`` in this
container carries the Burgers vector recorded on node ``i``'s arm toward ``j``;
node ``j``'s arm toward ``i`` carries its negative. Burgers conservation at an
internal node is therefore ``sum of arm Burgers = 0`` -- see
:func:`midas_ddd.validate.check_burgers_conservation`, which the Fourier kernel
requires and which :func:`read_paradis` runs by default.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

__all__ = [
    "DislocationNetwork",
    "read_paradis",
    "write_paradis",
]

_ANGSTROM_PER_UM = 1.0e4


@dataclass
class DislocationNetwork:
    """A discrete-dislocation network: nodes, segments, Burgers vectors, cell.

    Attributes
    ----------
    nodes_um : (N, 3) tensor
        Node positions, micrometers.
    segments : (M, 2) int64 tensor
        Node-index pairs. Each physical segment appears **once**, oriented
        ``i -> j``; the reciprocal arm is not repeated.
    burgers_b : (M, 3) tensor
        Burgers vector of segment ``i -> j`` **in units of b** (crystal
        Cartesian). Multiply by ``b_magnitude_A`` for angstroms. NOT normalised:
        a perfect dislocation has ``|b| = 1`` but a junction formed by a
        dislocation reaction has ``|b| = sqrt(2)`` (or other values), and that
        magnitude is what makes Burgers conservation balance at the junction
        node. Normalising it here silently breaks the conservation check and,
        downstream, the relaxation volume of any loop closed through a junction.
    normals : (M, 3) tensor
        Glide-plane normal of each segment (unit). A zero row means the file did
        not record one (junction / sessile segments sometimes have none).
    b_magnitude_A : float
        Burgers magnitude in angstroms, e.g. 2.556 for Cu.
    cell_min_um, cell_max_um : (3,) tensor
        Simulation cell bounds, micrometers.
    pbc : tuple of 3 bool
        Periodicity per axis. Segment vectors use the minimum-image convention
        along periodic axes (:meth:`segment_vectors_um`).
    constraints : (N,) int64 tensor
        ParaDiS node constraint code (0 = free, non-zero = pinned/surface).
    node_tags : list of str
        Original ``domain,index`` tags, kept so a network can be written back
        out and diffed against its source.
    """

    nodes_um: torch.Tensor
    segments: torch.Tensor
    burgers_b: torch.Tensor
    normals: torch.Tensor
    b_magnitude_A: float
    cell_min_um: torch.Tensor
    cell_max_um: torch.Tensor
    pbc: Tuple[bool, bool, bool] = (True, True, True)
    constraints: Optional[torch.Tensor] = None
    node_tags: List[str] = field(default_factory=list)
    source: str = ""

    # ---------------------------------------------------------------- basics

    @property
    def n_nodes(self) -> int:
        return int(self.nodes_um.shape[0])

    @property
    def n_segments(self) -> int:
        return int(self.segments.shape[0])

    @property
    def cell_size_um(self) -> torch.Tensor:
        return self.cell_max_um - self.cell_min_um

    @property
    def cell_volume_um3(self) -> float:
        return float(torch.prod(self.cell_size_um))

    @property
    def b_magnitude_um(self) -> float:
        return self.b_magnitude_A / _ANGSTROM_PER_UM

    def burgers_um(self) -> torch.Tensor:
        """Physical Burgers vectors, micrometers. ``(M, 3)``."""
        return self.burgers_b * self.b_magnitude_um

    def burgers_magnitudes_b(self) -> torch.Tensor:
        """``|b|`` per segment in units of b. ``(M,)``.

        1 for a perfect dislocation, sqrt(2) for a typical FCC junction, and
        1/sqrt(3) for a Shockley partial. A census of this is the quickest way
        to see whether a network contains reactions.
        """
        return torch.linalg.vector_norm(self.burgers_b, dim=-1)

    def burgers_directions(self) -> torch.Tensor:
        """Unit Burgers directions. ``(M, 3)``.

        Paired with ``burgers_magnitudes_b() * b_magnitude_A`` this is the
        ``(burgers=..., burgers_length_A=...)`` split that
        :func:`midas_dfxm.dislocation.stroh_dislocation` takes.
        """
        m = self.burgers_magnitudes_b().unsqueeze(-1)
        return torch.where(m > 1e-12, self.burgers_b / m.clamp(min=1e-30),
                           torch.zeros_like(self.burgers_b))

    # ------------------------------------------------------------- geometry

    def segment_vectors_um(self) -> torch.Tensor:
        """``r_j - r_i`` per segment, minimum-image along periodic axes. ``(M, 3)``.

        The minimum-image step is what keeps a segment that crosses a periodic
        boundary from being read as a spurious box-length-long dislocation. It is
        the same "closest image convention" ExaDiS's ``get_segments_end_points``
        applies.
        """
        ra = self.nodes_um[self.segments[:, 0]]
        rb = self.nodes_um[self.segments[:, 1]]
        d = rb - ra
        L = self.cell_size_um
        for ax in range(3):
            if self.pbc[ax] and float(L[ax]) > 0:
                d[:, ax] = d[:, ax] - L[ax] * torch.round(d[:, ax] / L[ax])
        return d

    def segment_lengths_um(self) -> torch.Tensor:
        """Length of each segment, micrometers. ``(M,)``."""
        return torch.linalg.vector_norm(self.segment_vectors_um(), dim=-1)

    def total_line_length_um(self) -> float:
        return float(self.segment_lengths_um().sum())

    def dislocation_density_um2(self) -> float:
        """Total line length per unit volume, µm^-2 (i.e. m^-2 x 1e-12)."""
        vol = self.cell_volume_um3
        if vol <= 0:
            raise ValueError("cell volume is zero; cannot form a density")
        return self.total_line_length_um() / vol

    def node_arms(self) -> Dict[int, List[Tuple[int, int]]]:
        """``node -> [(segment index, +1 if node is the tail, -1 if the head)]``.

        The sign says how to orient that segment's stored Burgers vector to get
        the arm pointing *outward* from this node, which is what Burgers
        conservation sums.
        """
        arms: Dict[int, List[Tuple[int, int]]] = {}
        seg = self.segments.tolist()
        for s, (i, j) in enumerate(seg):
            arms.setdefault(int(i), []).append((s, +1))
            arms.setdefault(int(j), []).append((s, -1))
        return arms

    def to(self, *, dtype=None, device=None) -> "DislocationNetwork":
        """Move/cast the tensor fields, leaving metadata alone."""
        def _c(t):
            if t is None:
                return None
            if t.is_floating_point():
                return t.to(dtype=dtype, device=device)
            return t.to(device=device)
        return DislocationNetwork(
            nodes_um=_c(self.nodes_um),
            segments=_c(self.segments),
            burgers_b=_c(self.burgers_b),
            normals=_c(self.normals),
            b_magnitude_A=self.b_magnitude_A,
            cell_min_um=_c(self.cell_min_um),
            cell_max_um=_c(self.cell_max_um),
            pbc=self.pbc,
            constraints=_c(self.constraints),
            node_tags=list(self.node_tags),
            source=self.source,
        )

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (f"DislocationNetwork({self.n_nodes} nodes, {self.n_segments} segments, "
                f"b={self.b_magnitude_A} A, cell={self.cell_size_um.tolist()} um)")


# ---------------------------------------------------------------------------
# ParaDiS .data reader
# ---------------------------------------------------------------------------

_VEC_RE = re.compile(r"^\s*(\w+)\s*=\s*\[\s*$")
#: "0,    3" -> "0,3". ExaDiS pads its tags; other writers do not.
_TAG_WS = re.compile(r",\s+")


def _parse_header(lines: Sequence[str]) -> Tuple[Dict[str, object], int]:
    """Read the ``key = value`` / ``key = [ ... ]`` preamble up to ``nodalData``."""
    params: Dict[str, object] = {}
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        stripped = line.strip()
        if stripped.startswith("nodalData"):
            return params, i + 1
        if not stripped or stripped.startswith("#"):
            i += 1
            continue
        m = _VEC_RE.match(line)
        if m:                                   # multi-line bracketed vector
            key = m.group(1)
            vals: List[float] = []
            i += 1
            while i < n and "]" not in lines[i]:
                tok = lines[i].strip()
                if tok and not tok.startswith("#"):
                    vals.extend(float(t) for t in tok.split())
                i += 1
            i += 1                              # consume the closing ']'
            params[key] = vals
            continue
        if "=" in stripped:
            key, _, val = stripped.partition("=")
            params[key.strip()] = val.strip()
        i += 1
    raise ValueError("no 'nodalData' section found -- is this a ParaDiS .data file?")


def read_paradis(
    path,
    *,
    b_magnitude_A: float = 2.556,
    pbc: Tuple[bool, bool, bool] = (True, True, True),
    validate: bool = True,
    burgers_tol: float = 1e-6,
    dtype: torch.dtype = torch.float64,
    device=None,
) -> DislocationNetwork:
    """Read a ParaDiS / ExaDiS ``.data`` restart file into a network.

    Parameters
    ----------
    path
        The ``.data`` file.
    b_magnitude_A
        Burgers magnitude in angstroms. **The file does not record it** -- it
        stores everything in units of ``b`` -- so it must come from the material.
        Default 2.556 (Cu). Getting this wrong rescales the whole network.
    pbc
        Periodicity per axis, used by the minimum-image segment convention.
    validate
        Run :func:`midas_ddd.validate.check_burgers_conservation` and raise on
        failure. The Fourier kernel is undefined for a non-conserving network,
        so this defaults on; pass ``False`` only to inspect a broken file.
    burgers_tol
        Tolerance for that check, in units of ``b``.

    Returns
    -------
    DislocationNetwork
        Each physical segment appears once, oriented from the lower node index
        to the higher.
    """
    path = Path(path)
    text = path.read_text()
    lines = text.splitlines()
    params, start = _parse_header(lines)

    tag_to_idx: Dict[str, int] = {}
    positions: List[Tuple[float, float, float]] = []
    constraints: List[int] = []
    tags: List[str] = []
    # arm records as (tail tag, head tag, burgers, normal), both directions present
    arms: List[Tuple[str, str, Tuple[float, float, float], Tuple[float, float, float]]] = []

    # Indices are assigned ONLY on a primary line, so node ordering follows the
    # file rather than arm-traversal order. That makes read(write(net)) an
    # identity on node indices instead of a permutation of them -- without it,
    # "node 37" means a different node after a round trip.
    def _declare(tag: str) -> int:
        if tag in tag_to_idx:
            return tag_to_idx[tag]
        tag_to_idx[tag] = len(tags)
        tags.append(tag)
        positions.append((math.nan, math.nan, math.nan))
        constraints.append(0)
        return tag_to_idx[tag]

    # ExaDiS writes node tags as "0,    3" -- whitespace AFTER the comma -- while
    # ParaDiS restart files written elsewhere use the compact "0,3". Splitting on
    # whitespace without normalising shifts every field by one and the arm count
    # is read as a coordinate ("invalid literal for int(): '1602.0005'").
    # Collapse the space so both dialects tokenise identically.
    def _norm(line: str) -> str:
        return _TAG_WS.sub(",", line)

    k = start
    n = len(lines)
    while k < n:
        raw = _norm(lines[k])
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            k += 1
            continue
        parts = stripped.split()
        # A primary line is: node_tag x y z num_arms [constraint]
        if "," not in parts[0] or len(parts) < 5:
            k += 1
            continue
        tag = parts[0]
        idx = _declare(tag)
        positions[idx] = (float(parts[1]), float(parts[2]), float(parts[3]))
        n_arms = int(parts[4])
        constraints[idx] = int(parts[5]) if len(parts) > 5 else 0
        tags[idx] = tag
        k += 1
        for _ in range(n_arms):
            if k + 1 >= n:
                raise ValueError(
                    f"{path.name}: truncated arm record for node {tag} at line {k + 1}")
            a = _norm(lines[k]).split()
            nbr = a[0]
            burg = (float(a[1]), float(a[2]), float(a[3]))
            plane_tokens = _norm(lines[k + 1]).split()
            normal = (float(plane_tokens[0]), float(plane_tokens[1]), float(plane_tokens[2]))
            arms.append((tag, nbr, burg, normal))
            k += 2

    if not tags:
        raise ValueError(f"{path.name}: nodalData section contained no nodes")

    unknown = sorted({nbr for _, nbr, _, _ in arms if nbr not in tag_to_idx})
    if unknown:
        raise ValueError(
            f"{path.name}: {len(unknown)} node(s) referenced by an arm but never "
            f"declared with a primary line, e.g. {unknown[:5]}")

    pos = np.asarray(positions, dtype=float)
    if np.isnan(pos).any():
        missing = [tags[i] for i in np.where(np.isnan(pos).any(axis=1))[0]]
        raise ValueError(
            f"{path.name}: {len(missing)} node(s) referenced by an arm but never "
            f"declared with a primary line, e.g. {missing[:5]}")

    # Collapse the two reciprocal arms of each segment into one oriented record.
    seen: Dict[Tuple[int, int], int] = {}
    seg_i: List[int] = []
    seg_j: List[int] = []
    seg_b: List[Tuple[float, float, float]] = []
    seg_n: List[Tuple[float, float, float]] = []
    for tail, head, burg, normal in arms:
        i, j = tag_to_idx[tail], tag_to_idx[head]
        key = (min(i, j), max(i, j))
        if key in seen:
            continue
        seen[key] = len(seg_i)
        # Orient low -> high, flipping the Burgers vector when the arm ran the
        # other way, so the stored b always belongs to the stored direction.
        if i <= j:
            seg_i.append(i); seg_j.append(j); seg_b.append(burg)
        else:
            seg_i.append(j); seg_j.append(i)
            seg_b.append((-burg[0], -burg[1], -burg[2]))
        seg_n.append(normal)

    def _t(a, dt=dtype):
        return torch.as_tensor(np.asarray(a, dtype=float), dtype=dt, device=device)

    b_um = b_magnitude_A / _ANGSTROM_PER_UM
    nodes_um = _t(pos) * b_um

    normals = _t(seg_n)
    nrm = torch.linalg.vector_norm(normals, dim=-1, keepdim=True)
    normals = torch.where(nrm > 1e-12, normals / nrm.clamp(min=1e-30), torch.zeros_like(normals))

    # File Burgers vectors are already in units of b. Keep them EXACTLY as
    # written -- do not normalise. A perfect dislocation has |b| = 1, but a
    # junction from a dislocation reaction has |b| = sqrt(2), and that magnitude
    # is precisely what balances Burgers conservation at the junction node.
    # (Normalising here made 60 of this file's 5055 nodes fail the check with a
    # residual of exactly sqrt(2) - 1.)
    burgers_b = _t(seg_b)

    cmin = params.get("minCoordinates")
    cmax = params.get("maxCoordinates")
    if isinstance(cmin, list) and isinstance(cmax, list) and len(cmin) == 3 == len(cmax):
        cell_min = _t(cmin) * b_um
        cell_max = _t(cmax) * b_um
    else:
        # No bounds in the header: fall back to the node bounding box, and turn
        # PBC off, because a minimum image against a guessed cell is worse than
        # no minimum image at all.
        cell_min = nodes_um.min(dim=0).values
        cell_max = nodes_um.max(dim=0).values
        pbc = (False, False, False)

    net = DislocationNetwork(
        nodes_um=nodes_um,
        segments=torch.tensor(list(zip(seg_i, seg_j)), dtype=torch.int64, device=device),
        burgers_b=burgers_b,
        normals=normals,
        b_magnitude_A=float(b_magnitude_A),
        cell_min_um=cell_min,
        cell_max_um=cell_max,
        pbc=tuple(bool(x) for x in pbc),  # type: ignore[arg-type]
        constraints=torch.tensor(constraints, dtype=torch.int64, device=device),
        node_tags=tags,
        source=str(path),
    )

    if validate:
        from .validate import check_burgers_conservation
        check_burgers_conservation(net, tol=burgers_tol, raise_on_fail=True)
    return net


def write_paradis(net: DislocationNetwork, path) -> None:
    """Write a network back out in ParaDiS ``.data`` format.

    Round-trips through :func:`read_paradis`. Emits both reciprocal arms per
    segment, as the format requires. Written primarily so a network built or
    modified here can be handed back to ExaDiS.
    """
    path = Path(path)
    b_um = net.b_magnitude_um
    pos_b = (net.nodes_um / b_um).tolist()
    cmin = (net.cell_min_um / b_um).tolist()
    cmax = (net.cell_max_um / b_um).tolist()
    burg_b = net.burgers_b.tolist()          # already in units of b, magnitude preserved
    normals = net.normals.tolist()

    arms: Dict[int, List[Tuple[int, List[float], List[float]]]] = {}
    for s, (i, j) in enumerate(net.segments.tolist()):
        b = burg_b[s]
        arms.setdefault(int(i), []).append((int(j), b, normals[s]))
        arms.setdefault(int(j), []).append((int(i), [-b[0], -b[1], -b[2]], normals[s]))

    tags = net.node_tags or [f"0,{i}" for i in range(net.n_nodes)]
    cons = net.constraints.tolist() if net.constraints is not None else [0] * net.n_nodes

    out: List[str] = []
    out.append("dataFileVersion =   4  ")
    out.append("numFileSegments =   1  ")
    out.append("minCoordinates = [")
    out.extend(f"  {v:.6e}" for v in cmin)
    out.append("  ]")
    out.append("maxCoordinates = [")
    out.extend(f"  {v:.6e}" for v in cmax)
    out.append("  ]")
    out.append(f"nodeCount =   {net.n_nodes}  ")
    out.append("dataDecompType =   2  ")
    out.append("#")
    out.append("#  END OF DATA FILE PARAMETERS")
    out.append("#")
    out.append("")
    out.append("nodalData = ")
    out.append("#  Primary lines: node_tag, x, y, z, num_arms, constraint")
    out.append("#  Secondary lines: arm_tag, burgx, burgy, burgz, nx, ny, nz")
    for i in range(net.n_nodes):
        a = arms.get(i, [])
        x, y, z = pos_b[i]
        out.append(f" {tags[i]} {x:.8f} {y:.8f} {z:.8f} {len(a)} {cons[i]}")
        for nbr, b, nn in a:
            out.append(f"   {tags[nbr]} {b[0]:.10e} {b[1]:.10e} {b[2]:.10e}")
            out.append(f"       {nn[0]:.10e} {nn[1]:.10e} {nn[2]:.10e}")
    path.write_text("\n".join(out) + "\n")
