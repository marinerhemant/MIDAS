"""Real-data regression on the binned demk L1 (env-gated, no repo fixture).

Run with:  MIDAS_DEFECT_REAL_DATA=1 pytest tests/test_real_data_regression.py

The binned (det_bin=4) data is too coarse to reproduce the full-res *physics*
numbers (budget %, ρ, 96.6 % on-lattice), which are pinned by a separate
copland full-res run. What this test guards is the **code path + closure +
geometry sanity**: the inventory runs end-to-end, the budget closes to 100 %,
and the on-lattice fraction sits far above the wrong-geometry chance floor
(~5 %) — i.e. the validated detector model is wired correctly.
"""
import numpy as np

from midas_defect.examples.demk_fcc_end_to_end import run_inventory


def test_demk_bin4_inventory_runs_and_closes(demk_bin4_paths):
    voxels, grains = demk_bin4_paths
    out = run_inventory(voxels, grains, verbose=False)

    # closure: the 4-bin budget sums to 1.0
    assert out["budget_closes"]
    assert abs(sum(out["budget"].values()) - 1.0) < 1e-9

    # sane scale
    assert out["n_voxels"] > 1_000_000
    assert out["n_grains"] == 248

    # geometry sanity: bright voxels land on the lattice far above the
    # ~5 % wrong-geometry chance floor (binning caps this below the full-res
    # 96.6 %, but it must clearly beat a scrambled geometry).
    assert out["on_lattice_bright_frac"] > 0.20

    # all reported numbers finite (run_inventory return-key surface)
    assert np.isfinite(out["forbidden_excess_median"])
    assert np.isfinite(out["fault_alpha_median"])
    assert np.isfinite(out["fault_rod_along_perp_median"])
    assert np.isfinite(out["fault_rod_frac_enriched"])
    assert np.isfinite(out["rho_median_per_m2"])
    assert np.isfinite(out["domain_size_A_median"])
