"""Real-data joint calibration runner — paired Pilatus calibrant + HEDM scan.

Paper figure 2 (the headline).  Compare deviatoric strain on a downstream
grain map against (a) paper-3 single-image powder baseline, and (b) Wright
multi-distance baseline if multi-distance data is available.

Status
------
GATED ON DATA.  Internal network is down as of 2026-05-09; user expects
real paired data within a week.  Fill in:

  1. Read calibrant zarr → run paper-3 four-stage to get a powder-only
     ``CalibrationSpec`` MAP plus per-panel σ baseline.

  2. Read HEDM scan zarr → run ``midas_ff_pipeline`` through the index
     stage to get a Grains.csv with N_g ~ 100 grains.

  3. Build joint spec via ``build_joint_spec`` initialised from the
     paper-3 MAP + the Grains.csv.

  4. Build powder_residual_fn via the paper-3 ``pseudo_strain_residual``
     closure.

  5. Build hedm_residual_fn via
     ``midas_fit_grain.spec_residual.hedm_spot_residual`` plus a
     ``HEDMResidualBundle`` whose model/observations/matches come from
     the HEDM-stage outputs.

  6. Run ``AlternatingDriver`` to convergence.  Report joint cost
     trajectory, MAP, and ``fisher_block_rank`` on
     ``panel_delta_yz`` + ``panel_delta_theta`` + ``panel_delta_lsd`` +
     ``panel_delta_p2``.

  7. Run a downstream grain-fit using the joint MAP as the calibration;
     compute deviatoric strain distribution.  Compare against the
     paper-3 baseline distribution.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibrant-zarr", type=Path, required=False,
                        help="Path to powder calibrant Zarr (single image at HEDM detector pose)")
    parser.add_argument("--hedm-zarr", type=Path, required=False,
                        help="Path to HEDM scan Zarr (full omega sweep)")
    parser.add_argument("--paramstest", type=Path, required=False,
                        help="paper-3 paramstest.txt with single-image MAP as initial guess")
    parser.add_argument("--grains-csv", type=Path, required=False,
                        help="Grains.csv from a prior MIDAS grain-fit (N_g ~ 100)")
    parser.add_argument("--output", type=Path, default=Path("runs/joint_pilatus_real"))
    args = parser.parse_args(argv)

    print(f"[runner] GATED ON DATA — see TODO markers in source.  Args: {args}")
    print("[runner] Steps:")
    for i, line in enumerate(__doc__.split("Fill in:")[1].strip().splitlines(), start=1):
        if line.strip().startswith(tuple(str(d) + "." for d in range(1, 10))):
            print(f"  [{i}] {line.strip()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
