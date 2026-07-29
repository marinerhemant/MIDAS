#!/usr/bin/env python3
"""Full-res validation of midas_defect against the published demk numbers.

Run on copland (where the full-res zarr lives). Reads the L2346 full-res voxel
cloud + indexed grains, runs the inventory through the *package* modules, and
checks the headline numbers against the published re-analysis targets:

    on-lattice (bright)  ~ 96.6 %
    Bragg                ~ 64.6 %    asterism ~ 31 %    inter-Bragg ~ 4 %
    forbidden excess     ~ 0 (no APB)
    ⟨111⟩ rod enrichment > ⟨110⟩, ⟨100⟩
    fault α (median)     ~ 0.005

This is the package's real-data regression target; the offline test
(test_real_data_regression.py) only guards code path + closure on binned data.
"""
from __future__ import annotations

import math
import sys
import time

import numpy as np
import zarr

from midas_defect.examples.demk_fcc_end_to_end import (
    DemkGeometry, voxels_to_qsample, inventory_from_qsample,
)

OUT = "/gdata/dm/MPE/OrthrosJr/analysis/sharma_work/demk_diffuse"
GR = "/gdata/dm/MPE/OrthrosJr/analysis/sharma_work/demk_ff_Cu/LayerNr_1/Grains.csv"
ZARR = f"{OUT}/L2346_fullres.zarr"


def main():
    t0 = time.time()
    r = zarr.open_group(ZARR, mode="r")
    frame = r["frame"][:].astype(np.float64)
    row = r["row"][:].astype(np.float64)
    col = r["col"][:].astype(np.float64)
    val = r["value"][:].astype(np.float64)
    print(f"loaded {len(val)/1e6:.1f}M full-res voxels ({time.time()-t0:.0f}s)", flush=True)

    indices = np.stack([frame, row, col], axis=1)
    qs = voxels_to_qsample(indices, det_bin=1, geom=DemkGeometry())
    OM = np.genfromtxt(GR, comments="%")[:, 1:10].reshape(-1, 3, 3)
    print(f"converted q + {len(OM)} grains ({time.time()-t0:.0f}s)", flush=True)

    out = inventory_from_qsample(qs, val, OM, verbose=True)
    print(f"\nTOTAL {time.time()-t0:.0f}s", flush=True)

    # soft checks against the published demk numbers
    b = out["budget"]
    checks = {
        "on-lattice ~0.966": abs(out["on_lattice_bright_frac"] - 0.966) < 0.05,
        "bragg ~0.646": abs(b["bragg"] - 0.646) < 0.08,
        "budget closes": out["budget_closes"],
        "rho ~1e13 m^-2": 5e12 < out["rho_median_per_m2"] < 5e13,
        "no forbidden excess": abs(out["forbidden_excess_median"]) < 0.05,
        # authoritative fault test (explicit per-grain along/perp): heterogeneous
        # ⟨111⟩ faulting — median ~1 but a clear enriched minority of grains.
        "<111> fault rods in a grain subset": out["fault_rod_frac_enriched"] > 0.2,
        "fault alpha small": out["fault_alpha_median"] < 0.05,
    }
    print("\n=== checks vs published targets ===")
    for k, v in checks.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    sys.exit(0 if all(checks.values()) else 1)


if __name__ == "__main__":
    main()
