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
from typing import TYPE_CHECKING, Union

import numpy as np

if TYPE_CHECKING:
    from ..result import ProcessGrainsResult


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
        /attrs/...
    """
    import h5py

    p = Path(path)
    with h5py.File(p, "w") as f:
        g = f.create_group("diagnostics")
        diag = result.diagnostics or {}
        for key in (
            "cluster_sizes",
            "n_resolved_hkls",
            "n_majority_hkls",
            "n_residual_tie_hkls",
            "n_forward_sim_hkls",
        ):
            arr = np.asarray(diag.get(key, np.zeros(result.n_grains)), dtype=np.int32)
            g.create_dataset(key, data=arr)

        # Signed residual decomposition (see compute/residual_decomposition).
        #   /residuals/<aggregate arrays + scalars>
        #   /residuals/spot_table : float32 (n_spots, 11), gzip — layout in
        #       the ``columns`` attribute (SPOT_RESIDUAL_COLS).
        if "residuals" in diag:
            r = f.create_group("residuals")
            for key, arr in diag["residuals"].items():
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
            for i, arr in enumerate(ew):
                ew_grp.create_dataset(
                    str(i), data=np.asarray(arr, dtype=np.float64),
                )

        a = f.create_group("attrs")
        a.attrs["mode"] = result.mode
