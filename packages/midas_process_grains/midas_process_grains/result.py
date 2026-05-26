"""ProcessGrainsResult — the in-memory output of one pipeline run.

Mirrors the pattern of ``IndexerResult``, ``MergeResult`` etc.: tensors live
on the user's device until ``.write()`` is called, which materialises CSVs
(and optionally an HDF5 archive) on disk.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch


@dataclass
class ProcessGrainsResult:
    """All per-grain tensors plus the cluster-mapping table.

    Attributes
    ----------
    ids : (n_grains,) int64
        Per-grain ID (1-indexed, matches the rep's SpotID by convention).
    rep_pos : (n_grains,) int64
        Row index in OrientPosFit.bin of each grain's representative seed.
    orient_mat : (n_grains, 3, 3) float64
        Orientation matrices in lab frame.
    positions : (n_grains, 3) float64
        Centre-of-mass positions in lab frame (µm).
    lattice : (n_grains, 6) float64
        Refined lattice parameters (a, b, c, α, β, γ).
    grain_radius : (n_grains,) float64
        Grain radius (µm) from `meanRadius` column of OrientPosFit.bin.
    confidence : (n_grains,) float64
        Confidence (matched / expected reflections).
    strain_lab : (n_grains, 3, 3) float64
        Strain tensor in lab frame, computed via the chosen `StrainMethod`.
    strain_grain : (n_grains, 3, 3) float64
        Strain tensor in grain frame.
    stress_lab : (n_grains, 3, 3) float64, optional
        Cauchy stress in lab frame; populated only when `MaterialName` was set.
    stress_grain : (n_grains, 3, 3) float64, optional
    spot_matrix_rows : (n_rows, 12) float64
        Per-spot rows in the canonical 12-column SpotMatrix.csv layout.
    grain_ids_key : list[(rep_id, rep_pos, [(other_id, other_pos), ...])]
        Cluster-mapping for GrainIDsKey.csv.
    diagnostics : dict
        Per-grain diagnostic blobs from Phase 2/3: cluster size, edge weights,
        n_supporters per resolved hkl, conflict-resolution policy used, etc.
    """

    ids: torch.Tensor
    rep_pos: torch.Tensor
    orient_mat: torch.Tensor
    positions: torch.Tensor
    lattice: torch.Tensor
    grain_radius: torch.Tensor
    confidence: torch.Tensor
    strain_lab: torch.Tensor
    strain_grain: torch.Tensor
    # Per-grain refiner residuals (legacy Grains.csv DiffPos/Ome/Angle cols).
    # Sourced from OrientPosFit[rep_pos, 22:25] (ErrorsFin from the refiner).
    diff_pos_um: torch.Tensor = field(default_factory=lambda: torch.zeros(0))
    diff_ome_deg: torch.Tensor = field(default_factory=lambda: torch.zeros(0))
    diff_angle_deg: torch.Tensor = field(default_factory=lambda: torch.zeros(0))
    # Per-grain strain-solver L2 residual (µε). Legacy RMSErrorStrain column.
    rms_error_strain: torch.Tensor = field(default_factory=lambda: torch.zeros(0))
    # Per-grain phase index (1-based). Currently single-phase.
    phase_nr: torch.Tensor = field(default_factory=lambda: torch.zeros(0, dtype=torch.int32))
    stress_lab: Optional[torch.Tensor] = None
    stress_grain: Optional[torch.Tensor] = None
    spot_matrix_rows: torch.Tensor = field(default_factory=lambda: torch.zeros((0, 12)))
    grain_ids_key: List[Tuple[int, int, List[Tuple[int, int]]]] = field(default_factory=list)
    diagnostics: Dict = field(default_factory=dict)
    # Snapshot of what the run used.
    sg_nr: int = 225
    lattice_reference: Tuple[float, ...] = (0.0,) * 6
    mode: str = "spot_aware"

    @property
    def n_grains(self) -> int:
        return int(self.ids.shape[0])

    # -----------------------------------------------------------------------
    # Serialisation
    # -----------------------------------------------------------------------

    def write(
        self,
        out_dir: Union[str, Path],
        *,
        h5: bool = True,
        diagnostics_h5: bool = True,
    ) -> Path:
        """Materialise the standard CSV outputs (+ optional HDF5 archives).

        Parameters
        ----------
        out_dir : path
            Destination directory. Created if missing.
        h5 : bool
            Also write a ``data_consolidated.h5`` mirroring the legacy
            schema for backwards compatibility.
        diagnostics_h5 : bool
            Also write a ``processgrains_diagnostics.h5`` with the Phase-2 /
            Phase-3 metadata (cluster sizes, conflict-resolution traces,
            symmetry op chosen per member, edge weights).

        Returns
        -------
        out_dir : Path
        """
        from .io.csv import (
            write_grain_ids_key_csv,
            write_grains_csv,
            write_spot_matrix_csv,
        )
        from .io.consolidated import write_consolidated_h5, write_diagnostics_h5

        d = Path(out_dir)
        d.mkdir(parents=True, exist_ok=True)

        # Detach + cpu + numpy views for the writers.
        n = self.n_grains
        om_flat = self.orient_mat.detach().cpu().reshape(n, 9).numpy().astype(np.float64)
        # Per-grain Bunge ZXZ Euler in radians (cols 44-46 of legacy schema).
        from midas_stress.orientation import orient_mat_to_euler
        eul = np.zeros((n, 3), dtype=np.float64)
        for i in range(n):
            eul[i] = np.asarray(orient_mat_to_euler(om_flat[i].tolist()), dtype=np.float64)
        # 3x3 strain tensors → flat 9-vector in µε (rows 0..8 = row-major 3x3).
        def _to_9mu(eps):
            arr = eps.detach().cpu().numpy().astype(np.float64).reshape(n, 9)
            return (arr * 1e6).astype(np.float64)
        # Fill diff_pos/ome/angle and rms with zeros if absent (older runs).
        def _arr(t, dtype=np.float64):
            if t is None or (hasattr(t, "numel") and t.numel() == 0):
                return np.zeros(n, dtype=dtype)
            return t.detach().cpu().numpy().astype(dtype)
        grains = {
            "ids": self.ids.detach().cpu().numpy().astype(np.int32),
            "orient_mat": om_flat,
            "positions": self.positions.detach().cpu().numpy().astype(np.float64),
            "lattices": self.lattice.detach().cpu().numpy().astype(np.float64),
            "diff_pos_um": _arr(self.diff_pos_um),
            "diff_ome_deg": _arr(self.diff_ome_deg),
            "diff_angle_deg": _arr(self.diff_angle_deg),
            "grain_radius": self.grain_radius.detach().cpu().numpy().astype(np.float64),
            "confidence": self.confidence.detach().cpu().numpy().astype(np.float64),
            "strain_fab_3x3": _to_9mu(self.strain_grain),
            "strain_ken_3x3": _to_9mu(self.strain_lab),
            "rms_error_strain": _arr(self.rms_error_strain),
            "phase_nr": _arr(self.phase_nr, dtype=np.int32) if self.phase_nr.numel() > 0
                          else np.ones(n, dtype=np.int32),
            "eul_rad": eul,
        }
        write_grains_csv(
            d / "Grains.csv", grains,
            sg_nr=self.sg_nr, lattice=self.lattice_reference,
        )

        sm = self.spot_matrix_rows.detach().cpu().numpy().astype(np.float64)
        if sm.size > 0:
            write_spot_matrix_csv(d / "SpotMatrix.csv", sm)
        write_grain_ids_key_csv(d / "GrainIDsKey.csv", self.grain_ids_key)

        if h5:
            write_consolidated_h5(d / "data_consolidated.h5", self)
        if diagnostics_h5:
            write_diagnostics_h5(d / "processgrains_diagnostics.h5", self)

        return d


def _strain_tensor_to_voigt6(eps: torch.Tensor) -> torch.Tensor:
    """Convert ``(n, 3, 3)`` symmetric strain tensor to (n, 6) Voigt."""
    n = eps.shape[0]
    out = torch.zeros((n, 6), dtype=eps.dtype, device=eps.device)
    out[:, 0] = eps[:, 0, 0]
    out[:, 1] = eps[:, 1, 1]
    out[:, 2] = eps[:, 2, 2]
    out[:, 3] = eps[:, 0, 1]
    out[:, 4] = eps[:, 0, 2]
    out[:, 5] = eps[:, 1, 2]
    return out.detach().cpu()
