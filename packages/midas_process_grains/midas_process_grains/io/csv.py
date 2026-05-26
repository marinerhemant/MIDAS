"""CSV writers for ``Grains.csv``, ``SpotMatrix.csv``, ``GrainIDsKey.csv``.

These mirror what ``FF_HEDM/src/ProcessGrains.c`` emits so the new pipeline
is a drop-in for downstream MIDAS tooling (DREAM.3D bridges, paraview scripts,
midas_stress consumers, etc.).

Conventions
-----------

``Grains.csv``: column-major MIDAS format; header lines + 8 skip lines + N rows
of 23 columns. We mirror the legacy column order:

  GrainID, O11..O33 (9), X, Y, Z,
  ε11_lab, ε22_lab, ε33_lab, ε12_lab, ε13_lab, ε23_lab,
  GrainRadius, Confidence

The "23-column legacy" form has more fields (lattice parameters, lab-frame
strain, additional crystal-frame strain, etc.); we expose them as part of
``Grains_extended.csv`` to avoid breaking existing consumers.

``SpotMatrix.csv``: tab-separated, 12 columns:

  GrainID, SpotID, Omega, DetectorHor, DetectorVert, OmeRaw, Eta,
  RingNr, YLab, ZLab, Theta, StrainError

``GrainIDsKey.csv``: per-cluster mapping. Each line:

  bestGrainID bestPos otherID otherPos otherID otherPos ...

(see ``ProcessGrains.c:703-710``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Sequence, Tuple, Union

import numpy as np


# ---------------------------------------------------------------------------
# Grains.csv
# ---------------------------------------------------------------------------

# Legacy ProcessGrains.c Grains.csv schema (47 data columns + 1 ID).
# Every downstream MIDAS tool (and Indrajeet's notebook) consumes this format.
# Columns 0..46:
#   0          GrainID
#   1..9       O11..O33        (orientation matrix, row-major)
#   10..12     X Y Z           (position, µm)
#   13..18     a b c α β γ     (lattice parameters; Å and degrees)
#   19         DiffPos         (refiner residual: position error in µm)
#   20         DiffOme         (refiner residual: ω error in degrees)
#   21         DiffAngle       (refiner residual: internal angle in degrees)
#   22         GrainRadius     (mean of per-spot grain radii, µm)
#   23         Confidence      (NrObserved / NrExpected)
#   24..32     eFab11..eFab33  (Fable-Beaudoin grain-frame strain, µε)
#   33..41     eKen11..eKen33  (Kenesei lab/sample-frame strain,    µε)
#   42         RMSErrorStrain  (strain-solver L2 residual, µε)
#   43         PhaseNr         (1-based phase index)
#   44..46     Eul0 Eul1 Eul2  (Euler angles in RADIANS, ZXZ Bunge)
GRAINS_HEADER_COLS = (
    "%ID\tO11\tO12\tO13\tO21\tO22\tO23\tO31\tO32\tO33\tX\tY\tZ\t"
    "a\tb\tc\talpha\tbeta\tgamma\tDiffPos\tDiffOme\tDiffAngle\tGrainRadius\tConfidence\t"
    "eFab11\teFab12\teFab13\teFab21\teFab22\teFab23\teFab31\teFab32\teFab33\t"
    "eKen11\teKen12\teKen13\teKen21\teKen22\teKen23\teKen31\teKen32\teKen33\t"
    "RMSErrorStrain\tPhaseNr\tEul0\tEul1\tEul2\n"
)
GRAINS_HEADER_LINES_LEGACY = [
    "%NumGrains 0\n",  # placeholder; first writer pass replaces this
    "%BeamCenter 0 0\n",
    "%BeamThickness 0\n",
    "%GlobalPosition 0\n",
    "%NumPhases 1\n",
    "%PhaseInfo\n",
    "%\tSpaceGroup:225\n",
    "%\tLattice Parameter:0 0 0 0 0 0\n",
    GRAINS_HEADER_COLS,
]


def write_grains_csv(
    path: Union[str, Path],
    grains: dict,
    *,
    sg_nr: int = 225,
    lattice: Sequence[float] = (0.0,) * 6,
    beam_center: Tuple[float, float] = (0.0, 0.0),
    beam_thickness: float = 0.0,
    global_position: float = 0.0,
) -> None:
    """Write the full 47-column legacy ``Grains.csv``.

    Required ``grains`` keys (each an (n,) or (n, …) array):
      ids                int32   (n,)
      orient_mat         float64 (n, 9)
      positions          float64 (n, 3)
      lattices           float64 (n, 6)   a, b, c, α, β, γ
      diff_pos_um        float64 (n,)
      diff_ome_deg       float64 (n,)
      diff_angle_deg     float64 (n,)
      grain_radius       float64 (n,)
      confidence         float64 (n,)
      strain_fab_3x3     float64 (n, 9)   row-major 3x3 grain-frame, µε
      strain_ken_3x3     float64 (n, 9)   row-major 3x3 sample/lab,   µε
      rms_error_strain   float64 (n,)     µε
      phase_nr           int32   (n,)
      eul_rad            float64 (n, 3)   Euler ZXZ Bunge, radians
    """
    n = len(grains["ids"])
    # Required keys (fail loud rather than silently zero-fill).
    required = (
        "ids", "orient_mat", "positions", "lattices",
        "diff_pos_um", "diff_ome_deg", "diff_angle_deg",
        "grain_radius", "confidence",
        "strain_fab_3x3", "strain_ken_3x3", "rms_error_strain",
        "phase_nr", "eul_rad",
    )
    missing = [k for k in required if k not in grains]
    if missing:
        raise KeyError(f"write_grains_csv: missing required keys: {missing}")
    p = Path(path)
    with open(p, "w") as fp:
        fp.write(f"%NumGrains {n}\n")
        fp.write(f"%BeamCenter {beam_center[0]} {beam_center[1]}\n")
        fp.write(f"%BeamThickness {beam_thickness}\n")
        fp.write(f"%GlobalPosition {global_position}\n")
        fp.write("%NumPhases 1\n")
        fp.write("%PhaseInfo\n")
        fp.write(f"%\tSpaceGroup:{sg_nr}\n")
        latstr = "\t".join(f"{x:.6f}" for x in lattice)
        fp.write(f"%\tLattice Parameter:{latstr}\n")
        fp.write(GRAINS_HEADER_COLS)
        for i in range(n):
            row: List[str] = [str(int(grains["ids"][i]))]
            row.extend(f"{grains['orient_mat'][i, k]:.9f}" for k in range(9))
            row.extend(f"{grains['positions'][i, k]:.6f}" for k in range(3))
            row.extend(f"{grains['lattices'][i, k]:.6f}" for k in range(6))
            row.append(f"{grains['diff_pos_um'][i]:.6f}")
            row.append(f"{grains['diff_ome_deg'][i]:.6f}")
            row.append(f"{grains['diff_angle_deg'][i]:.6f}")
            row.append(f"{grains['grain_radius'][i]:.6f}")
            row.append(f"{grains['confidence'][i]:.6f}")
            row.extend(f"{grains['strain_fab_3x3'][i, k]:.6e}" for k in range(9))
            row.extend(f"{grains['strain_ken_3x3'][i, k]:.6e}" for k in range(9))
            row.append(f"{grains['rms_error_strain'][i]:.6e}")
            row.append(str(int(grains["phase_nr"][i])))
            row.extend(f"{grains['eul_rad'][i, k]:.9f}" for k in range(3))
            fp.write("\t".join(row) + "\n")


# ---------------------------------------------------------------------------
# SpotMatrix.csv
# ---------------------------------------------------------------------------

SPOT_MATRIX_HEADER = (
    "%GrainID\tSpotID\tOmega\tDetectorHor\tDetectorVert\tOmeRaw\tEta\t"
    "RingNr\tYLab\tZLab\tTheta\tStrainError\n"
)


def write_spot_matrix_csv(
    path: Union[str, Path],
    rows: np.ndarray,
) -> None:
    """Write a ``SpotMatrix.csv`` from an (n_rows, 12) array.

    Column order matches ``ProcessGrains.c::SpotMatrix_l`` exactly:
    GrainID, SpotID, Omega, DetectorHor, DetectorVert, OmeRaw, Eta, RingNr,
    YLab, ZLab, Theta, StrainError.
    """
    if rows.ndim != 2 or rows.shape[1] != 12:
        raise ValueError(
            f"rows must have shape (n, 12); got {rows.shape}"
        )
    p = Path(path)
    with open(p, "w") as fp:
        fp.write(SPOT_MATRIX_HEADER)
        for r in range(rows.shape[0]):
            fp.write(
                "\t".join((
                    f"{int(rows[r, 0])}",                # GrainID
                    f"{int(rows[r, 1])}",                # SpotID
                    f"{rows[r, 2]:.6f}",                 # Omega
                    f"{rows[r, 3]:.6f}",                 # DetectorHor
                    f"{rows[r, 4]:.6f}",                 # DetectorVert
                    f"{rows[r, 5]:.6f}",                 # OmeRaw
                    f"{rows[r, 6]:.6f}",                 # Eta
                    f"{int(rows[r, 7])}",                # RingNr
                    f"{rows[r, 8]:.6f}",                 # YLab
                    f"{rows[r, 9]:.6f}",                 # ZLab
                    f"{rows[r, 10]:.6f}",                # Theta
                    f"{rows[r, 11]:.6e}",                # StrainError
                )) + "\n"
            )


# ---------------------------------------------------------------------------
# GrainIDsKey.csv
# ---------------------------------------------------------------------------


def write_grain_ids_key_csv(
    path: Union[str, Path],
    clusters: Iterable[Tuple[int, int, Sequence[Tuple[int, int]]]],
) -> None:
    """Write a ``GrainIDsKey.csv`` describing the cluster mapping.

    Each cluster yields a single line::

        bestGrainID bestPos otherID otherPos otherID otherPos ...

    where ``bestPos`` is the row index in ``OrientPosFit.bin`` of the cluster
    representative and ``other(ID, Pos)`` are the same for the cluster's
    non-representative members.
    """
    p = Path(path)
    with open(p, "w") as fp:
        for best_id, best_pos, others in clusters:
            tokens = [str(int(best_id)), str(int(best_pos))]
            for oid, opos in others:
                tokens.append(str(int(oid)))
                tokens.append(str(int(opos)))
            fp.write(" ".join(tokens) + "\n")
