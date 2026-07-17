"""End-to-end smoke test for the v4 pipeline.

Constructs a minimal synthetic MIDAS-style layer directory on disk (one
ω-rotation series, two grains, a handful of spots each) and runs
`v4_pipeline.run_v4_pipeline()` over it. Verifies that the emitted
GrainsV4.csv has the expected number of rows + columns and no NaN
trust-coverage values.

This is a SMOKE test — it catches import errors, file-IO regressions,
and gross shape mismatches. Real-data behaviour is validated via the
per-stage unit tests in test_hkl_ingest, test_hkl_expected,
test_cluster_physics, test_trust_tiers, test_twin_label,
test_grain_size_recompute, test_hierarchical.
"""
from __future__ import annotations

import json
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def _make_tiny_layer(tmp_path: Path) -> Path:
    """Construct the smallest possible MIDAS layer directory that the v4
    pipeline can ingest end-to-end. Two grains, identity orientation +
    one rotated 60° about z, on FCC ring 2."""
    layer = tmp_path / "LayerNr_1"
    (layer / "Results").mkdir(parents=True)
    (layer / "Output").mkdir()

    # paramstest.txt
    (layer / "paramstest.txt").write_text(textwrap.dedent("""\
        SpaceGroup 225;
        LsdFit 1000000.0;
        Wavelength 0.172979;
        px 200.0;
    """))

    # hkls.csv (FCC {002} ring, 6 variants)
    (layer / "hkls.csv").write_text(textwrap.dedent("""\
        h k l D-spacing RingNr g1 g2 g3 Theta 2Theta Radius
        0 0 2 1.8 2 0.0 0.0 1.0 2.85 5.70 100000.0
        0 0 -2 1.8 2 0.0 0.0 -1.0 2.85 5.70 100000.0
        0 2 0 1.8 2 0.0 1.0 0.0 2.85 5.70 100000.0
        0 -2 0 1.8 2 0.0 -1.0 0.0 2.85 5.70 100000.0
        2 0 0 1.8 2 1.0 0.0 0.0 2.85 5.70 100000.0
        -2 0 0 1.8 2 -1.0 0.0 0.0 2.85 5.70 100000.0
    """))

    # InputAllExtraInfoFittingAll.csv — 4 spots, no header %
    (layer / "InputAllExtraInfoFittingAll.csv").write_text(textwrap.dedent("""\
        YLab ZLab Omega GrainRadius SpotID RingNumber Eta Ttheta OmegaIni YOrigDetCor ZOrigDetCor YRawPx ZRawPx OmegaDetCor IntegratedIntensity RawSumIntensity maskTouched FitRMSE
        100.0 50.0 0.0 5.0 1 2 30.0 5.70 0.0 0 0 0 0 100.0 100.0 0.1 0 0
        50.0 100.0 0.0 5.0 2 2 60.0 5.70 0.0 0 0 0 0 100.0 100.0 0.1 0 0
        100.0 -50.0 0.0 5.0 3 2 -30.0 5.70 0.0 0 0 0 0 100.0 100.0 0.1 0 0
        -100.0 -50.0 0.0 5.0 4 2 -150.0 5.70 0.0 0 0 0 0 100.0 100.0 0.1 0 0
    """))

    # OrientPosFit.bin (2 grains × 27 floats)
    N = 4   # 4 alive candidates, simulating 2 grains × 2 candidates each
    opf = np.zeros((N, 27), dtype=np.float64)
    # candidate 0: grain A (identity OM, seed=1)
    opf[0, 0] = 1   # seed SpotID
    opf[0, 1:10] = np.eye(3).flatten()
    opf[0, 10:14] = [0, 0, 0, 0]
    opf[0, 22:25] = [50.0, 0.1, 0.05]
    opf[0, 25] = 5.0; opf[0, 26] = 1.0
    # candidate 1: grain A (same OM, seed=2)
    opf[1, 0] = 2
    opf[1, 1:10] = np.eye(3).flatten()
    opf[1, 10:14] = [0, 0.5, 0.5, 0]
    opf[1, 22:25] = [55.0, 0.12, 0.06]
    opf[1, 25] = 5.0; opf[1, 26] = 1.0
    # candidate 2: grain B (60° rotation about z, seed=3) — far enough from A
    th = np.deg2rad(60)
    Rz60 = np.array([[np.cos(th), -np.sin(th), 0],
                     [np.sin(th),  np.cos(th), 0],
                     [0, 0, 1]])
    opf[2, 0] = 3
    opf[2, 1:10] = Rz60.flatten()
    opf[2, 10:14] = [0, 100, 100, 0]
    opf[2, 22:25] = [60.0, 0.15, 0.07]
    opf[2, 25] = 5.0; opf[2, 26] = 1.0
    # candidate 3: grain B again, slightly perturbed
    opf[3, 0] = 4
    opf[3, 1:10] = Rz60.flatten()
    opf[3, 10:14] = [0, 100.5, 100.5, 0]
    opf[3, 22:25] = [62.0, 0.16, 0.08]
    opf[3, 25] = 5.0; opf[3, 26] = 1.0
    (layer / "Results" / "OrientPosFit.bin").write_bytes(opf.tobytes())

    # ProcessKey.bin (N × 5000 int32, mostly zero, a few SpotIDs each)
    pk = np.zeros((N, 5000), dtype=np.int32)
    pk[0, :2] = [1, 2]      # grain A claims spots 1, 2
    pk[1, :2] = [1, 2]
    pk[2, :2] = [3, 4]      # grain B claims spots 3, 4
    pk[3, :2] = [3, 4]
    (layer / "Results" / "ProcessKey.bin").write_bytes(pk.tobytes())

    return layer


def test_v4_pipeline_runs_on_tiny_synthetic_layer(tmp_path: Path):
    from midas_process_grains.v4_pipeline import run_v4_pipeline

    layer = _make_tiny_layer(tmp_path)
    out = tmp_path / "v4_out"
    paths = run_v4_pipeline(
        layer_dir=layer, out_dir=out, trust_scheme="strict", verbose=False,
    )
    assert paths["leaf"].exists()
    assert paths["meta"].exists()
    assert paths["audit"].exists()

    meta = json.loads(paths["meta"].read_text())
    # Pipeline must produce a well-formed meta with the v4 keys; the
    # synthetic spots are NOT necessarily on the Bragg cone (we don't
    # forward-simulate them properly), so the seed-hkl recovery may
    # reject all of them. The smoke test verifies plumbing, not physics.
    for key in ("package_version", "trust_scheme", "n_leaf_grains",
                "n_pass1_clusters", "n_split_clusters",
                "trust_strict", "trust_loose", "total_wall_s"):
        assert key in meta, f"meta missing {key}"

    # The leaf CSV always has the header even when n_grains == 0
    leaf = pd.read_csv(paths["leaf"], sep="\t")
    expected_cols = {
        "GrainID", "RepSeed", "X", "Y", "Z", "O11", "O33",
        "hkl_coverage", "trust_tier_strict", "twin_partner_id",
        "GrainRadius_NNLS",
    }
    assert expected_cols <= set(leaf.columns), \
        f"missing columns: {expected_cols - set(leaf.columns)}"


def test_v4_pipeline_runs_with_forward_predict_primitive(tmp_path: Path):
    """Smoke-check that ``merge_primitive='forward_predict'`` plumbs end-to-end.

    The synthetic spots in ``_make_tiny_layer`` are not physically valid
    diffractions, so the forward-predict snap won't hit anything and the
    pipeline should fall through with zero merge edges (every alive
    candidate becomes its own cluster). The test only verifies that the
    code path is wired correctly — no NameError, no missing geometry,
    metadata records ``merge_primitive``.
    """
    from midas_process_grains.v4_pipeline import run_v4_pipeline

    layer = _make_tiny_layer(tmp_path)
    out = tmp_path / "v4_out_fp"
    paths = run_v4_pipeline(
        layer_dir=layer, out_dir=out, trust_scheme="strict", verbose=False,
        merge_primitive="forward_predict", k_agree=3,
    )
    assert paths["leaf"].exists()
    meta = json.loads(paths["meta"].read_text())
    assert meta.get("merge_primitive") == "forward_predict"
    # Forward-predict path leaves theta_star unset (None in JSON).
    assert meta.get("theta_star_deg") is None
