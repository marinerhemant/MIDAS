"""Network validation: Burgers conservation, loop inventory, resolution limits.

Three checks, each guarding a different way the downstream physics goes wrong
silently.

**Burgers conservation** (:func:`check_burgers_conservation`). The Fourier kernel
turns each closed loop's surface integral into a line integral by Stokes. That
conversion is only reference-point independent when ``sum b = 0`` at every
internal node. A network that violates it still *runs* and returns numbers that
depend on where you put the cap point -- which is the worst failure mode there
is. So this is a hard gate, on by default in :func:`read_paradis`.

**Loop inventory** (:func:`find_loops`). A prismatic loop's entire small-angle
signature is its relaxation volume ``dV = b . A``, and ``A`` only exists for a
*closed* circuit. Open lines contribute nothing at ``q -> 0``. Separating the two
populations is therefore not bookkeeping, it is the physics -- and it is what
lets the ``dV`` gate be written at all.

**Resolution report** (:func:`resolution_report`). A nodal DDD code discretises a
2 nm loop into a handful of segments, and SAXS at ``q ~ 1/R`` probes exactly that
scale. The elastic field within a few core radii of a segment is not faithful.
This does not fix that; it makes the limit visible with a number attached
instead of letting a plausible-looking curve come back from a q range the
network cannot support.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import math

import torch

__all__ = [
    "BurgersConservationResult",
    "Loop",
    "ResolutionReport",
    "check_burgers_conservation",
    "find_loops",
    "loop_area_vectors_um2",
    "relaxation_volumes_um3",
    "resolution_report",
    "validate_network",
]


# ---------------------------------------------------------------------------
# Burgers conservation
# ---------------------------------------------------------------------------

@dataclass
class BurgersConservationResult:
    """Per-node Burgers-sum residuals, in units of ``b``."""

    ok: bool
    max_residual: float
    n_violating: int
    violating_nodes: List[int]
    residuals: torch.Tensor          # (N, 3), units of b
    ignored_pinned: int = 0

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (f"BurgersConservationResult(ok={self.ok}, "
                f"max_residual={self.max_residual:.3e} b, "
                f"n_violating={self.n_violating})")


def check_burgers_conservation(
    net,
    *,
    tol: float = 1e-6,
    raise_on_fail: bool = False,
    ignore_constrained: bool = True,
) -> BurgersConservationResult:
    """Sum the outward arm Burgers vectors at every node; they must vanish.

    Parameters
    ----------
    tol
        Tolerance on ``|sum b|`` in units of ``b``.
    raise_on_fail
        Raise :class:`ValueError` instead of returning a failing result.
    ignore_constrained
        Skip nodes with a non-zero ParaDiS constraint code. A pinned or surface
        node legitimately terminates a line -- flux leaves through the boundary
        -- so it is not required to balance. Free nodes always are.

    Notes
    -----
    Segments are stored oriented ``i -> j`` with the Burgers vector belonging to
    that direction, so node ``i`` contributes ``+b`` and node ``j`` contributes
    ``-b``. That is the same convention ParaDiS writes to file.
    """
    n = net.n_nodes
    resid = torch.zeros((n, 3), dtype=net.burgers_b.dtype, device=net.burgers_b.device)
    i_idx = net.segments[:, 0]
    j_idx = net.segments[:, 1]
    resid.index_add_(0, i_idx, net.burgers_b)
    resid.index_add_(0, j_idx, -net.burgers_b)

    mag = torch.linalg.vector_norm(resid, dim=-1)

    skip = torch.zeros(n, dtype=torch.bool, device=mag.device)
    n_ignored = 0
    if ignore_constrained and net.constraints is not None:
        skip = net.constraints.to(mag.device) != 0
        n_ignored = int(skip.sum())

    bad = (mag > tol) & (~skip)
    idx = torch.nonzero(bad, as_tuple=False).flatten().tolist()
    checked = mag[~skip]
    result = BurgersConservationResult(
        ok=len(idx) == 0,
        max_residual=float(checked.max()) if checked.numel() else 0.0,
        n_violating=len(idx),
        violating_nodes=idx[:64],
        residuals=resid,
        ignored_pinned=n_ignored,
    )
    if raise_on_fail and not result.ok:
        tags = net.node_tags or [str(i) for i in range(n)]
        worst = sorted(idx, key=lambda i: -float(mag[i]))[:5]
        detail = ", ".join(f"{tags[i]} (|sum b|={float(mag[i]):.3e})" for i in worst)
        raise ValueError(
            f"Burgers vector is not conserved at {len(idx)} node(s); worst: {detail}. "
            f"Tolerance {tol:g} b. The Fourier kernel converts each loop's surface "
            f"integral to a line integral by Stokes, which is only reference-point "
            f"independent when sum(b) = 0 at every free node -- a non-conserving "
            f"network returns cap-point-dependent numbers rather than an error. "
            f"Pass validate=False to inspect the file anyway."
        )
    return result


# ---------------------------------------------------------------------------
# Loop inventory
# ---------------------------------------------------------------------------

@dataclass
class Loop:
    """A closed circuit in the network.

    Attributes
    ----------
    node_indices : list of int
        Nodes in circuit order. The circuit closes from the last back to the
        first; the first node is not repeated.
    segment_indices : list of int
        Segments in circuit order.
    orientations : list of int
        ``+1`` where the stored segment direction matches the traversal
        direction, ``-1`` where it opposes it.
    burgers_b : (3,) tensor
        The loop's Burgers vector, units of b, as seen along the traversal.
        Only meaningful when ``uniform_burgers`` is True.
    uniform_burgers : bool
        Whether every segment in the circuit carries the same Burgers vector.
        A prismatic loop does; a circuit closed through junctions may not, and
        its relaxation volume is then not a single ``b . A``.
    """

    node_indices: List[int]
    segment_indices: List[int]
    orientations: List[int]
    burgers_b: torch.Tensor
    uniform_burgers: bool


def find_loops(net, *, burgers_tol: float = 1e-6) -> List[Loop]:
    """Find closed circuits made of degree-2 nodes.

    This deliberately finds only the *simple* loops -- circuits every one of
    whose nodes has exactly two arms. That is what a prismatic / Frank loop from
    ``generate_prismatic_config`` looks like, and it is the population whose
    relaxation volume carries the small-angle signal. Circuits that pass through
    a junction node (degree > 2) are not enumerated: their decomposition into
    loops is not unique, so silently picking one would be inventing a number.
    """
    deg: Dict[int, int] = {}
    adj: Dict[int, List[Tuple[int, int]]] = {}     # node -> [(segment, other node)]
    for s, (i, j) in enumerate(net.segments.tolist()):
        i, j = int(i), int(j)
        deg[i] = deg.get(i, 0) + 1
        deg[j] = deg.get(j, 0) + 1
        adj.setdefault(i, []).append((s, j))
        adj.setdefault(j, []).append((s, i))

    loops: List[Loop] = []
    visited_seg: set = set()
    for start in sorted(adj):
        if deg.get(start, 0) != 2:
            continue
        for seg0, _ in adj[start]:
            if seg0 in visited_seg:
                continue
            # Walk the chain from `start` through degree-2 nodes.
            nodes = [start]
            segs: List[int] = []
            ors: List[int] = []
            prev_seg, cur = seg0, start
            closed = False
            while True:
                nxt = None
                for s, other in adj[cur]:
                    if s == prev_seg and segs:
                        continue
                    if s == seg0 and not segs:
                        nxt = (s, other)
                        break
                    if s != prev_seg:
                        nxt = (s, other)
                        break
                if nxt is None:
                    break
                s, other = nxt
                stored_i = int(net.segments[s, 0])
                ors.append(+1 if stored_i == cur else -1)
                segs.append(s)
                prev_seg, cur = s, other
                if cur == start:
                    closed = True
                    break
                if deg.get(cur, 0) != 2:
                    break
                nodes.append(cur)
                if len(segs) > net.n_segments:      # pathological safety net
                    break
            if closed and len(segs) >= 3 and not (set(segs) & visited_seg):
                visited_seg.update(segs)
                bvecs = torch.stack([net.burgers_b[s] * o for s, o in zip(segs, ors)])
                b0 = bvecs[0]
                uniform = bool(torch.all(
                    torch.linalg.vector_norm(bvecs - b0, dim=-1) <= burgers_tol))
                loops.append(Loop(node_indices=nodes, segment_indices=segs,
                                  orientations=ors, burgers_b=b0,
                                  uniform_burgers=uniform))
    return loops


def loop_area_vectors_um2(net, loops: Sequence[Loop]) -> torch.Tensor:
    """Signed area vector ``A = 1/2 sum r_k x r_{k+1}`` per loop. ``(L, 3)`` µm².

    Computed about the loop centroid, which makes it independent of the origin
    for a closed circuit and numerically better behaved for a small loop far
    from the box centre. Segment vectors use the minimum-image convention, so a
    loop straddling a periodic boundary still closes.
    """
    if not loops:
        return torch.zeros((0, 3), dtype=net.nodes_um.dtype, device=net.nodes_um.device)

    segvec = net.segment_vectors_um()
    out = []
    for lp in loops:
        # Rebuild the circuit's vertices by walking the (minimum-image) segment
        # vectors, so a loop crossing a boundary is reconstructed unwrapped.
        r = torch.zeros(3, dtype=net.nodes_um.dtype, device=net.nodes_um.device)
        verts = [r]
        for s, o in zip(lp.segment_indices, lp.orientations):
            r = r + segvec[s] * o
            verts.append(r)
        V = torch.stack(verts[:-1])                     # drop the repeated closure point
        V = V - V.mean(dim=0, keepdim=True)
        A = 0.5 * torch.linalg.cross(V, torch.roll(V, -1, dims=0)).sum(dim=0)
        out.append(A)
    return torch.stack(out)


def relaxation_volumes_um3(net, loops: Sequence[Loop]) -> torch.Tensor:
    """``dV = b . A`` per loop, µm³. ``(L,)``.

    This is *the* quantity the small-angle signal is made of -- but note that a
    loop does NOT simply scatter like a compact particle of volume ``dV``: its
    ``q -> 0`` amplitude is direction dependent, running from ``kappa dV`` in the
    loop plane to ``dV`` along the normal. ``dV`` sets the SCALE; the angular
    dependence is in :mod:`midas_ddd.fourier`. For a circular prismatic loop of
    radius R the magnitude reduces to ``pi R^2 |b|``.
    """
    A = loop_area_vectors_um2(net, loops)
    if A.shape[0] == 0:
        return torch.zeros((0,), dtype=net.nodes_um.dtype, device=net.nodes_um.device)
    b = torch.stack([lp.burgers_b for lp in loops]) * net.b_magnitude_um
    return (b * A).sum(dim=-1)


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------

@dataclass
class ResolutionReport:
    """Segment-length statistics against the q range a caller wants to use."""

    n_segments: int
    min_length_um: float
    median_length_um: float
    max_length_um: float
    core_radius_um: float
    n_below_core: int
    q_max_supported_inv_A: float
    warnings: List[str]

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (f"ResolutionReport(n={self.n_segments}, "
                f"median={self.median_length_um * 1e3:.2f} nm, "
                f"q_max_supported={self.q_max_supported_inv_A:.3e} 1/A, "
                f"{len(self.warnings)} warning(s))")


def resolution_report(
    net,
    *,
    q_max_inv_A: Optional[float] = None,
    core_radius_b: float = 1.0,
) -> ResolutionReport:
    """What q range this discretisation can actually support.

    A segment of length ``L`` cannot represent structure finer than ``L``, so the
    honest ceiling is ``q_max ~ 2 pi / L_median``. Ask for more and the answer is
    controlled by the polyline, not the physics.
    """
    L = net.segment_lengths_um()
    core_um = core_radius_b * net.b_magnitude_um
    med = float(L.median()) if L.numel() else 0.0
    q_sup = (2.0 * math.pi / (med * _UM_TO_A)) if med > 0 else 0.0

    warnings: List[str] = []
    n_below = int((L < core_um).sum()) if L.numel() else 0
    if n_below:
        warnings.append(
            f"{n_below} segment(s) are shorter than the core radius "
            f"({core_um * 1e4:.2f} A); the elastic field there is not faithful.")
    if q_max_inv_A is not None and q_sup > 0 and q_max_inv_A > q_sup:
        warnings.append(
            f"requested q_max = {q_max_inv_A:.3e} 1/A exceeds what this "
            f"discretisation supports ({q_sup:.3e} 1/A, set by the median segment "
            f"length of {med * 1e3:.2f} nm). Intensity above that q is a property "
            f"of the polyline, not of the dislocation.")
    return ResolutionReport(
        n_segments=int(L.numel()),
        min_length_um=float(L.min()) if L.numel() else 0.0,
        median_length_um=med,
        max_length_um=float(L.max()) if L.numel() else 0.0,
        core_radius_um=core_um,
        n_below_core=n_below,
        q_max_supported_inv_A=q_sup,
        warnings=warnings,
    )


_UM_TO_A = 1.0e4


def validate_network(net, *, q_max_inv_A: Optional[float] = None) -> Dict[str, object]:
    """Run every check and return a summary dict. Does not raise.

    The one-call survey for "what did I just load".
    """
    cons = check_burgers_conservation(net, raise_on_fail=False)
    loops = find_loops(net)
    dV = relaxation_volumes_um3(net, loops)
    res = resolution_report(net, q_max_inv_A=q_max_inv_A)
    n_uniform = sum(1 for lp in loops if lp.uniform_burgers)
    return {
        "n_nodes": net.n_nodes,
        "n_segments": net.n_segments,
        "total_line_length_um": net.total_line_length_um(),
        "dislocation_density_um2": net.dislocation_density_um2(),
        "burgers_conserved": cons.ok,
        "burgers_max_residual_b": cons.max_residual,
        "n_loops": len(loops),
        "n_loops_uniform_b": n_uniform,
        "total_relaxation_volume_um3": float(dV.sum()) if dV.numel() else 0.0,
        "resolution": res,
        "warnings": res.warnings + ([] if cons.ok else
                                    [f"Burgers not conserved at {cons.n_violating} node(s)"]),
    }
