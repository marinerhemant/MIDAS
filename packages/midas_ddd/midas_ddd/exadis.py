"""Optional in-process bridge to ExaDiS (``pyexadis``).

ExaDiS needs a Kokkos/CMake build, so it can never be a pip dependency and is
imported lazily here. Everything in this module has a file-based equivalent that
always works: ExaDiS's own ``write_data`` emits ParaDiS ``.data``, and
:func:`midas_ddd.read_paradis` reads it. Use this module when ExaDiS is on the
same machine and you want to skip the file; use the file bridge otherwise.

Generators wrapped
------------------
``generate_prismatic_config`` -- irradiation-type loop populations, with
``radius`` accepting ``[min, max]`` for a size distribution.
``generate_line_config`` -- deformation-type straight infinite lines.

Both are ``pyexadis_utils`` functions; we convert their output into a
:class:`~midas_ddd.network.DislocationNetwork` and run the standard validation
on it, so a network built in-process is held to the same Burgers-conservation
gate as one read from disk.

Units
-----
ExaDiS, like ParaDiS, works in **units of b**. The conversion to micrometers
happens here, and ``b_magnitude_A`` must be supplied because neither the library
nor the file records it.
"""
from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np
import torch

from .network import DislocationNetwork

__all__ = [
    "have_pyexadis",
    "initialize_kokkos",
    "require_pyexadis",
    "from_pyexadis",
    "generate_prismatic_config",
    "generate_line_config",
]

_ANGSTROM_PER_UM = 1.0e4

_INSTALL_HINT = (
    "pyexadis is not importable. ExaDiS needs a Kokkos/CMake build and cannot be "
    "pip-installed, so it is an optional dependency here.\n"
    "  Build it (on a host with internet -- chiltepin is the only one at the "
    "beamline):\n"
    "    git clone --recursive https://github.com/LLNL/exadis.git\n"
    "    cd exadis && mkdir build && cd build\n"
    "    cmake .. -DEXADIS_PYTHON_BINDING=On && make -j\n"
    "  then put the build's python/ directory on PYTHONPATH.\n"
    "Alternatively, skip this module entirely: run ExaDiS anywhere, call its\n"
    "`write_data(N, 'net.data')`, and read the result with\n"
    "`midas_ddd.read_paradis('net.data', b_magnitude_A=...)`."
)


def initialize_kokkos():
    """Public alias for the one-shot Kokkos start-up. Safe to call repeatedly."""
    _ensure_kokkos()


def have_pyexadis() -> bool:
    """Whether ``pyexadis`` can be imported in this interpreter."""
    try:
        import pyexadis  # noqa: F401
        return True
    except Exception:
        return False


_KOKKOS_STARTED = False


def _ensure_kokkos():
    """Initialise Kokkos exactly once, and arrange to finalise at exit.

    ExaDiS is Kokkos-backed, and **every** entry point segfaults if Kokkos has
    not been initialised -- not an exception, a hard crash that takes the
    interpreter with it. `pyexadis.initialize()` is not optional and it is not
    idempotent, hence the module-level guard.

    This was the first thing to go wrong when the bridge was finally run against
    a real build: the very first `generate_prismatic_config` call aborted the
    process.
    """
    global _KOKKOS_STARTED
    if _KOKKOS_STARTED:
        return
    import atexit

    import pyexadis
    pyexadis.initialize()
    _KOKKOS_STARTED = True

    def _shutdown():
        try:
            pyexadis.finalize()
        except Exception:                        # pragma: no cover - interpreter teardown
            pass
    atexit.register(_shutdown)


def require_pyexadis():
    """Import ExaDiS, initialise Kokkos, return ``(pyexadis, pyexadis_utils)``.

    Raises :class:`ImportError` with build instructions when ExaDiS is absent.
    """
    try:
        import pyexadis
        import pyexadis_utils
    except Exception as exc:                     # pragma: no cover - env dependent
        raise ImportError(f"{_INSTALL_HINT}\n\nUnderlying error: {exc!r}") from exc
    _ensure_kokkos()
    return pyexadis, pyexadis_utils


def _cell_bounds_b(cell) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Best-effort ``(min, max)`` in units of b from a pyexadis Cell.

    The Cell API has moved between ExaDiS revisions, so this probes the shapes
    it has carried rather than pinning one. Returns None when it cannot tell,
    and the caller falls back to the node bounding box with PBC off -- a
    minimum-image convention against a *guessed* cell is worse than none.
    """
    if cell is None:
        return None
    origin = None
    for name in ("origin", "get_origin"):
        if hasattr(cell, name):
            v = getattr(cell, name)
            origin = np.asarray(v() if callable(v) else v, dtype=float)
            break
    h = None
    for name in ("h", "H", "get_h"):
        if hasattr(cell, name):
            v = getattr(cell, name)
            h = np.asarray(v() if callable(v) else v, dtype=float)
            break
    if h is None or h.size != 9:
        return None
    h = h.reshape(3, 3)
    if not np.allclose(h, np.diag(np.diag(h)), atol=1e-8 * max(1.0, float(np.abs(h).max()))):
        # A non-orthogonal cell has no axis-aligned min/max, and the
        # minimum-image convention in DislocationNetwork assumes one.
        return None
    if origin is None or origin.size != 3:
        origin = -0.5 * np.diag(h)
    return origin.reshape(3), origin.reshape(3) + np.diag(h)


def _cell_pbc(cell) -> Tuple[bool, bool, bool]:
    for name in ("is_periodic", "pbc", "get_pbc"):
        if hasattr(cell, name):
            v = getattr(cell, name)
            v = v() if callable(v) else v
            try:
                arr = [bool(x) for x in np.asarray(v).reshape(-1)[:3]]
                if len(arr) == 3:
                    return (arr[0], arr[1], arr[2])
            except Exception:
                pass
    return (True, True, True)


def from_pyexadis(
    N,
    *,
    b_magnitude_A: float,
    validate: bool = True,
    dtype: torch.dtype = torch.float64,
    device=None,
) -> DislocationNetwork:
    """Convert an ExaDiS ``DisNetManager`` / ``ExaDisNet`` to a network.

    Reads ``get_nodes_data()`` (``tags``, ``positions``, ``constraints``) and
    ``get_segs_data()`` (``nodeids``, ``burgers``, ``planes``). ExaDiS already
    stores one record per *segment* rather than per arm, so no arm collapsing is
    needed -- unlike the ``.data`` file path.

    Burgers vectors are taken **as given**, not normalised: a junction from a
    dislocation reaction has ``|b| != 1`` and that magnitude is what balances
    Burgers conservation at the junction node.
    """
    disnet = N.get_disnet() if hasattr(N, "get_disnet") else N
    nodes = disnet.get_nodes_data()
    segs = disnet.get_segs_data()

    pos_b = np.asarray(nodes["positions"], dtype=float)
    cons = np.asarray(nodes.get("constraints", np.zeros((len(pos_b), 1))), dtype=int).reshape(-1)
    tags_arr = np.asarray(nodes.get("tags", np.zeros((len(pos_b), 2))), dtype=int)
    tags = [f"{int(a)},{int(b)}" for a, b in tags_arr] if tags_arr.size else \
           [f"0,{i}" for i in range(len(pos_b))]

    nodeids = np.asarray(segs["nodeids"], dtype=np.int64).reshape(-1, 2)
    burg_b = np.asarray(segs["burgers"], dtype=float).reshape(-1, 3)
    planes = np.asarray(segs["planes"], dtype=float).reshape(-1, 3)

    b_um = b_magnitude_A / _ANGSTROM_PER_UM

    def _t(a, dt=dtype):
        return torch.as_tensor(np.ascontiguousarray(a), dtype=dt, device=device)

    nodes_um = _t(pos_b) * b_um
    normals = _t(planes)
    nn = torch.linalg.vector_norm(normals, dim=-1, keepdim=True)
    normals = torch.where(nn > 1e-12, normals / nn.clamp(min=1e-30), torch.zeros_like(normals))

    cell = getattr(disnet, "cell", None)
    bounds = _cell_bounds_b(cell)
    if bounds is not None:
        cell_min = _t(bounds[0]) * b_um
        cell_max = _t(bounds[1]) * b_um
        pbc = _cell_pbc(cell)
    else:
        cell_min = nodes_um.min(dim=0).values
        cell_max = nodes_um.max(dim=0).values
        pbc = (False, False, False)

    net = DislocationNetwork(
        nodes_um=nodes_um,
        segments=torch.as_tensor(nodeids, dtype=torch.int64, device=device),
        burgers_b=_t(burg_b),
        normals=normals,
        b_magnitude_A=float(b_magnitude_A),
        cell_min_um=cell_min,
        cell_max_um=cell_max,
        pbc=pbc,
        constraints=torch.as_tensor(cons, dtype=torch.int64, device=device),
        node_tags=tags,
        source="pyexadis",
    )
    if validate:
        from .validate import check_burgers_conservation
        check_burgers_conservation(net, raise_on_fail=True)
    return net


def generate_prismatic_config(
    *,
    crystal: str,
    box_size_b: float,
    num_loops: int,
    radius_b,
    b_magnitude_A: float,
    maxseg_b: int = -1,
    seed: int = -1,
    uniform: bool = False,
    Rorient=None,
    validate: bool = True,
    dtype: torch.dtype = torch.float64,
    device=None,
) -> DislocationNetwork:
    """Irradiation-type loop population, via ExaDiS ``generate_prismatic_config``.

    Parameters
    ----------
    crystal
        ``"fcc"`` or ``"bcc"``. Sets the allowed Burgers family (1/2<110> for
        FCC, 1/2<111> for BCC).
    box_size_b, radius_b, maxseg_b
        In **units of b**, matching ExaDiS's own convention. ``radius_b`` may be
        ``[min, max]`` for a size distribution.
    b_magnitude_A
        Needed to convert to micrometers; ExaDiS does not carry it.

    Notes
    -----
    ``maxseg_b`` sets the discretisation, and therefore the maximum q at which
    the resulting network means anything -- see
    :func:`midas_ddd.validate.resolution_report`. For SAXS at ``q ~ 1/R`` you
    want several segments across the loop, not the default coarse polygon.
    """
    _, utils = require_pyexadis()
    N = utils.generate_prismatic_config(
        crystal, box_size_b, num_loops, radius_b,
        maxseg=maxseg_b, Rorient=Rorient, seed=seed, uniform=uniform)
    return from_pyexadis(N, b_magnitude_A=b_magnitude_A, validate=validate,
                         dtype=dtype, device=device)


def generate_line_config(
    *,
    crystal: str,
    box_size_b: float,
    num_lines: int,
    b_magnitude_A: float,
    theta_deg: Optional[Sequence[float]] = None,
    maxseg_b: int = -1,
    seed: int = -1,
    Rorient=None,
    validate: bool = True,
    dtype: torch.dtype = torch.float64,
    device=None,
) -> DislocationNetwork:
    """Deformation-type straight infinite lines, via ExaDiS ``generate_line_config``.

    ``theta_deg`` is the list of allowed character angles (0 = screw, 90 = edge).

    These lines are periodic and open -- they enclose no area, so they carry no
    relaxation volume and contribute nothing at ``q -> 0``. That is physics, not
    a limitation: it is the reason the loop population and the line population
    are separable at small angle in the first place.
    """
    _, utils = require_pyexadis()
    N = utils.generate_line_config(
        crystal, box_size_b, num_lines, theta=theta_deg,
        maxseg=maxseg_b, Rorient=Rorient, seed=seed, verbose=False)
    return from_pyexadis(N, b_magnitude_A=b_magnitude_A, validate=validate,
                         dtype=dtype, device=device)
