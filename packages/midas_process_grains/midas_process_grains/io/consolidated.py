"""HDF5 archive writers.

Two outputs:

  ``data_consolidated.h5`` — backwards-compatible per-grain archive that
  mirrors the schema produced by the upstream MIDAS pipeline. Existing
  post-processing scripts (DREAM.3D bridges, paraview pipelines) read this.

  ``processgrains_diagnostics.h5`` — *new* aux file with the spot-aware merge
  metadata: cluster sizes, edge weights, conflict-resolution policies used,
  per-member symmetry op chosen, etc. Optional; on by default in the new
  pipeline so users can inspect what changed vs the C output.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Mapping, Union

import numpy as np

if TYPE_CHECKING:
    from ..result import ProcessGrainsResult

# Integer per-grain diagnostic arrays written by the spot-aware / legacy
# pipeline. ``c_parity`` supplies only the subset it actually measures — see
# write_diagnostics_h5's ``int_keys`` note.
_SPOT_AWARE_INT_KEYS = (
    "cluster_sizes",
    "n_resolved_hkls",
    "n_majority_hkls",
    "n_residual_tie_hkls",
    "n_forward_sim_hkls",
)


def write_consolidated_h5(
    path: Union[str, Path],
    result: "ProcessGrainsResult",
) -> None:
    """Write the legacy-compatible per-grain HDF5 archive.

    Schema chosen to match what existing midas downstream tools expect:

        /grains/ids           : int32   (N,)
        /grains/orient_mat    : float64 (N, 3, 3)
        /grains/positions     : float64 (N, 3)
        /grains/lattice       : float64 (N, 6)
        /grains/grain_radius  : float64 (N,)
        /grains/confidence    : float64 (N,)
        /grains/strain_lab    : float64 (N, 3, 3)
        /grains/strain_grain  : float64 (N, 3, 3)
        /grains/stress_lab    : float64 (N, 3, 3) (only if computed)
        /grains/stress_grain  : float64 (N, 3, 3) (only if computed)
        /attrs/sg_nr          : int
        /attrs/lattice_ref    : float64 (6,)
        /attrs/mode           : str
        /attrs/midas_pg_version : str
    """
    import h5py
    from .. import __version__

    p = Path(path)
    with h5py.File(p, "w") as f:
        g = f.create_group("grains")
        g.create_dataset("ids", data=result.ids.detach().cpu().numpy().astype(np.int32))
        g.create_dataset(
            "orient_mat",
            data=result.orient_mat.detach().cpu().numpy().astype(np.float64),
        )
        g.create_dataset(
            "positions",
            data=result.positions.detach().cpu().numpy().astype(np.float64),
        )
        g.create_dataset(
            "lattice",
            data=result.lattice.detach().cpu().numpy().astype(np.float64),
        )
        g.create_dataset(
            "grain_radius",
            data=result.grain_radius.detach().cpu().numpy().astype(np.float64),
        )
        g.create_dataset(
            "confidence",
            data=result.confidence.detach().cpu().numpy().astype(np.float64),
        )
        g.create_dataset(
            "strain_lab",
            data=result.strain_lab.detach().cpu().numpy().astype(np.float64),
        )
        g.create_dataset(
            "strain_grain",
            data=result.strain_grain.detach().cpu().numpy().astype(np.float64),
        )
        if result.stress_lab is not None:
            g.create_dataset(
                "stress_lab",
                data=result.stress_lab.detach().cpu().numpy().astype(np.float64),
            )
        if result.stress_grain is not None:
            g.create_dataset(
                "stress_grain",
                data=result.stress_grain.detach().cpu().numpy().astype(np.float64),
            )
        a = f.create_group("attrs")
        a.attrs["sg_nr"] = int(result.sg_nr)
        a.attrs["lattice_ref"] = np.asarray(result.lattice_reference, dtype=np.float64)
        a.attrs["mode"] = result.mode
        a.attrs["midas_pg_version"] = __version__


def write_diagnostics_h5(
    path: Union[str, Path],
    result: "ProcessGrainsResult",
) -> None:
    """Write the new diagnostics archive with Phase-2/3 metadata.

    Schema (per-grain padded arrays where appropriate):

        /diagnostics/cluster_sizes       : int32 (N,)
        /diagnostics/n_resolved_hkls     : int32 (N,)
        /diagnostics/n_majority_hkls     : int32 (N,)
        /diagnostics/n_residual_tie_hkls : int32 (N,)
        /diagnostics/n_forward_sim_hkls  : int32 (N,)
        /residuals/...                   : see :func:`write_diagnostics_arrays`
        /attrs/...

    Thin adapter over :func:`write_diagnostics_arrays`, which ``c_parity``
    calls directly — it has no ``ProcessGrainsResult`` to hand.
    """
    diag = result.diagnostics or {}
    write_diagnostics_arrays(
        path,
        diagnostics=diag,
        n_grains=result.n_grains,
        mode=result.mode,
        int_keys=_SPOT_AWARE_INT_KEYS,
    )


def write_diagnostics_arrays(
    path: Union[str, Path],
    *,
    diagnostics: Mapping[str, object],
    n_grains: int,
    mode: str,
    int_keys: tuple = _SPOT_AWARE_INT_KEYS,
) -> None:
    """Write ``processgrains_diagnostics.h5`` from plain arrays.

    The single implementation of the sidecar schema, shared by every mode so
    downstream readers (``utils/midas_ff_report.py``,
    ``utils/midas_ff_report_beamreport.py``) see one layout regardless of
    which mode produced the run.

    Parameters
    ----------
    diagnostics
        Mapping that may carry any of ``int_keys`` (per-grain int32 arrays),
        ``"residuals"`` (the :func:`decompose_residuals` output dict),
        ``"residuals_spot_table"`` (the ``(n_spots, 11)`` per-spot table) and
        ``"edge_weights_per_cluster"``.
    n_grains
        Length of the per-grain arrays; a key in ``int_keys`` that is absent
        from ``diagnostics`` is written as zeros of this length.
    mode
        Stamped on ``/attrs/mode`` so a reader can tell which pipeline wrote
        the file.
    int_keys
        Which per-grain integer arrays to write. **Pass only the keys the
        calling mode actually measures.** The zero-fill above is a padding
        convenience for the spot-aware pipeline, where every key is genuinely
        computed; a mode that does not compute one must leave it out rather
        than emit zeros, which a reader cannot distinguish from a measured
        count of zero.
    """
    import h5py

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    diag = diagnostics or {}
    with h5py.File(p, "w") as f:
        g = f.create_group("diagnostics")
        for key in int_keys:
            arr = np.asarray(diag.get(key, np.zeros(n_grains)), dtype=np.int32)
            g.create_dataset(key, data=arr)

        # Signed residual decomposition (see compute/residual_decomposition).
        #   /residuals/<aggregate arrays + scalars>
        #   /residuals/spot_table : float32 (n_spots, 11), gzip — layout in
        #       the ``columns`` attribute (SPOT_RESIDUAL_COLS).
        if "residuals" in diag:
            r = f.create_group("residuals")
            for key, arr in diag["residuals"].items():   # type: ignore[union-attr]
                r.create_dataset(key, data=np.asarray(arr))
            tbl = diag.get("residuals_spot_table")
            if tbl is not None and np.asarray(tbl).size:
                from ..compute.residual_decomposition import SPOT_RESIDUAL_COLS
                ds = r.create_dataset(
                    "spot_table",
                    data=np.asarray(tbl, dtype=np.float32),
                    compression="gzip", compression_opts=4,
                )
                ds.attrs["columns"] = ",".join(SPOT_RESIDUAL_COLS)

        # Optional richer per-grain blobs (variable-length).
        if "edge_weights_per_cluster" in diag:
            ew = diag["edge_weights_per_cluster"]
            ew_grp = g.create_group("edge_weights_per_cluster")
            for i, arr in enumerate(ew):                  # type: ignore[arg-type]
                ew_grp.create_dataset(
                    str(i), data=np.asarray(arr, dtype=np.float64),
                )

        a = f.create_group("attrs")
        a.attrs["mode"] = mode
