"""Unit tests for midas_pipeline.config — the interface contracts.

These tests pin field names, default values, and post-init invariants
so that parallel-stream developers (P2–P8) have a stable target. Any
breaking change here should fail loudly.
"""

from __future__ import annotations

import numpy as np
import pytest

from midas_pipeline.config import (
    EMConfig,
    FusionConfig,
    LayerSelection,
    MachineConfig,
    PipelineConfig,
    ReconConfig,
    RefinementConfig,
    ScanGeometry,
    SeedingConfig,
    read_scan_geometry_from_paramfile,
    sniff_scan_mode_from_paramfile,
)


# ---------------------------------------------------------------------------
# ScanGeometry
# ---------------------------------------------------------------------------


class TestScanGeometry:
    def test_ff_classmethod(self):
        scan = ScanGeometry.ff()
        assert scan.scan_mode == "ff"
        assert scan.n_scans == 1
        assert scan.scan_positions.shape == (1,)
        assert scan.scan_positions[0] == 0.0
        assert scan.scan_pos_tol_um == 0.0
        assert scan.is_ff and not scan.is_pf

    def test_pf_uniform(self):
        scan = ScanGeometry.pf_uniform(n_scans=5, scan_step_um=2.0, beam_size_um=4.0)
        assert scan.scan_mode == "pf"
        assert scan.n_scans == 5
        np.testing.assert_allclose(scan.scan_positions, [-4.0, -2.0, 0.0, 2.0, 4.0])
        assert scan.beam_size_um == 4.0
        # Default is single-sided (matches C + correct physics — see
        # discussion in ScanGeometry.friedel_symmetric_scan_filter docstring).
        assert scan.friedel_symmetric_scan_filter is False
        assert scan.is_pf and not scan.is_ff

    def test_pf_with_explicit_start(self):
        scan = ScanGeometry.pf_uniform(
            n_scans=3, scan_step_um=5.0, beam_size_um=5.0, start_um=10.0,
        )
        np.testing.assert_allclose(scan.scan_positions, [10.0, 15.0, 20.0])

    def test_ff_with_nscans_gt_1_raises(self):
        with pytest.raises(ValueError, match="n_scans=1"):
            ScanGeometry(scan_mode="ff", n_scans=5,
                         scan_positions=np.zeros(5), beam_size_um=1.0)

    def test_pf_with_nscans_lt_2_raises(self):
        with pytest.raises(ValueError, match="n_scans>=2"):
            ScanGeometry(scan_mode="pf", n_scans=1,
                         scan_positions=np.zeros(1), beam_size_um=1.0)

    def test_negative_beam_size_raises(self):
        with pytest.raises(ValueError, match="beam_size_um"):
            ScanGeometry.ff(beam_size_um=-1.0)

    def test_mismatched_n_scans_raises(self):
        with pytest.raises(ValueError, match="entries but n_scans"):
            ScanGeometry(scan_mode="pf", n_scans=3,
                         scan_positions=np.zeros(5), beam_size_um=1.0)


# ---------------------------------------------------------------------------
# RefinementConfig
# ---------------------------------------------------------------------------


class TestRefinementConfig:
    def test_defaults(self):
        rc = RefinementConfig()
        assert rc.position_mode == "fixed"
        assert rc.solver == "lbfgs"
        assert rc.loss == "full3d"
        assert rc.mode == "all_at_once"

    def test_voxel_bounded_mode(self):
        rc = RefinementConfig(position_mode="voxel_bounded")
        assert rc.position_mode == "voxel_bounded"


# ---------------------------------------------------------------------------
# ReconConfig + FusionConfig + EMConfig + SeedingConfig
# ---------------------------------------------------------------------------


class TestReconConfig:
    def test_defaults(self):
        rc = ReconConfig()
        assert rc.method == "fbp"
        assert rc.sino_conf_min == 0.5            # MIDAS_PF_SINO_CONF_MIN default
        assert rc.sino_scan_tol_um == 1.5         # MIDAS_PF_SINO_SCAN_TOL default
        assert rc.cull_min_size == 0
        assert rc.do_tomo is True


class TestSeedingConfig:
    def test_defaults(self):
        sc = SeedingConfig()
        assert sc.mode == "unseeded"
        assert sc.merged_ref_scan == -1


# ---------------------------------------------------------------------------
# PipelineConfig
# ---------------------------------------------------------------------------


def _ff_config(tmp_path) -> PipelineConfig:
    params = tmp_path / "Parameters.txt"
    params.write_text("RingThresh 1 100\n")
    return PipelineConfig(
        result_dir=str(tmp_path / "run"),
        params_file=str(params),
        scan=ScanGeometry.ff(),
    )


def _pf_config(tmp_path) -> PipelineConfig:
    params = tmp_path / "Parameters.txt"
    params.write_text("nScans 5\nBeamSize 5.0\n")
    return PipelineConfig(
        result_dir=str(tmp_path / "run"),
        params_file=str(params),
        scan=ScanGeometry.pf_uniform(n_scans=5, scan_step_um=2.0, beam_size_um=5.0),
    )


class TestPipelineConfig:
    def test_ff_smoke(self, tmp_path):
        cfg = _ff_config(tmp_path)
        assert cfg.is_ff
        assert not cfg.is_pf
        assert cfg.scan.n_scans == 1
        assert cfg.process_grains_mode == "spot_aware"     # FF-default

    def test_pf_smoke(self, tmp_path):
        cfg = _pf_config(tmp_path)
        assert cfg.is_pf
        assert not cfg.is_ff
        assert cfg.scan.n_scans == 5
        assert cfg.recon.method == "fbp"

    def test_resume_from_requires_stage(self, tmp_path):
        params = tmp_path / "P.txt"
        params.write_text("")
        with pytest.raises(ValueError, match="resume_from_stage"):
            PipelineConfig(
                result_dir=str(tmp_path / "r"), params_file=str(params),
                scan=ScanGeometry.ff(), resume="from",
            )

    def test_resume_from_stage_promotes_resume(self, tmp_path):
        params = tmp_path / "P.txt"
        params.write_text("")
        cfg = PipelineConfig(
            result_dir=str(tmp_path / "r"), params_file=str(params),
            scan=ScanGeometry.ff(), resume_from_stage="indexing",
        )
        assert cfg.resume == "from"

    def test_layer_dir(self, tmp_path):
        cfg = _ff_config(tmp_path)
        assert cfg.layer_dir(3).name == "LayerNr_3"

    def test_merged_ref_scan_resolves_sentinel(self, tmp_path):
        cfg = _pf_config(tmp_path)
        # Default seeding.merged_ref_scan = -1; n_scans = 5 → 2.
        assert cfg.merged_ref_scan() == 2

    def test_ff_with_do_tomo_silently_disabled(self, tmp_path):
        """FF mode + do_tomo=True should silently flip to do_tomo=False (tomo is PF-only)."""
        params = tmp_path / "P.txt"
        params.write_text("")
        cfg = PipelineConfig(
            result_dir=str(tmp_path / "r"), params_file=str(params),
            scan=ScanGeometry.ff(),
            recon=ReconConfig(do_tomo=True),
        )
        assert cfg.recon.do_tomo is False


# ---------------------------------------------------------------------------
# sniff_scan_mode_from_paramfile
# ---------------------------------------------------------------------------


class TestSniffScanMode:
    def test_nscans_gt_1_sniffs_pf(self, tmp_path):
        p = tmp_path / "P.txt"
        p.write_text("nScans 5\nBeamSize 5.0\n")
        assert sniff_scan_mode_from_paramfile(p) == "pf"

    def test_default_ff(self, tmp_path):
        p = tmp_path / "P.txt"
        p.write_text("RingThresh 1 100\n")
        assert sniff_scan_mode_from_paramfile(p) == "ff"

    def test_missing_file_defaults_ff(self, tmp_path):
        assert sniff_scan_mode_from_paramfile(tmp_path / "nope.txt") == "ff"


# ---------------------------------------------------------------------------
# read_scan_geometry_from_paramfile
# ---------------------------------------------------------------------------


class TestReadScanGeometry:
    def test_missing_returns_none(self, tmp_path):
        assert read_scan_geometry_from_paramfile(tmp_path / "nope.txt") is None

    def test_no_scanning_keys_returns_none(self, tmp_path):
        p = tmp_path / "P.txt"
        p.write_text("RingThresh 1 100\n")
        assert read_scan_geometry_from_paramfile(p) is None

    def test_full_geometry(self, tmp_path):
        p = tmp_path / "P.txt"
        p.write_text(
            "nScans 15\n"
            "ScanStep 5.0\n"
            "BeamSize 4.0\n"
            "ScanPosTol 2.5\n"
        )
        out = read_scan_geometry_from_paramfile(p)
        assert out == {
            "n_scans": 15,
            "scan_step_um": 5.0,
            "beam_size_um": 4.0,
            "scan_pos_tol_um": 2.5,
        }

    def test_px_is_not_scan_step(self, tmp_path):
        """``px`` is detector pixel pitch; it must NOT be sniffed as
        scan_step. Only ``ScanStep`` keys the scan step."""
        p = tmp_path / "P.txt"
        p.write_text("nScans 15\npx 200.0;\n")
        out = read_scan_geometry_from_paramfile(p)
        assert out == {"n_scans": 15}
        # And when ScanStep is present alongside px, only ScanStep wins.
        p.write_text("nScans 15\npx 200.0;\nScanStep 5.0;\n")
        out = read_scan_geometry_from_paramfile(p)
        assert out == {"n_scans": 15, "scan_step_um": 5.0}

    def test_partial_returns_what_it_finds(self, tmp_path):
        p = tmp_path / "P.txt"
        p.write_text("nScans 7\n")
        out = read_scan_geometry_from_paramfile(p)
        assert out == {"n_scans": 7}
