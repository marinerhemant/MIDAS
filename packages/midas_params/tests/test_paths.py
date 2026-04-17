"""Tests for path-specific rules: RI and PF."""

from __future__ import annotations

import textwrap

import pytest

from midas_params import Path
from midas_params.validator import validate


# ─── RI cross-field rules ────────────────────────────────────────────────────


def test_ri_rmax_le_rmin(tmp_path):
    fn = tmp_path / "p.txt"
    fn.write_text("RMin 100\nRMax 50\n")
    r = validate(str(fn), Path.RI)
    assert any(i.rule == "ri_rmax_gt_rmin" and i.key == "RMax" for i in r.errors)


def test_ri_rmax_exceeds_rhod_warns(tmp_path):
    fn = tmp_path / "p.txt"
    fn.write_text("RMin 100\nRMax 500000\nRhoD 200000\n")
    r = validate(str(fn), Path.RI)
    warns = [i for i in r.warnings if i.rule == "ri_rmax_gt_rmin"]
    assert warns


def test_ri_etamax_le_etamin(tmp_path):
    fn = tmp_path / "p.txt"
    fn.write_text("EtaMin 180\nEtaMax 0\n")
    r = validate(str(fn), Path.RI)
    assert any(i.rule == "ri_etamax_gt_etamin" for i in r.errors)


def test_ri_eta_range_over_360_warns(tmp_path):
    fn = tmp_path / "p.txt"
    fn.write_text("EtaMin -200\nEtaMax 200\n")
    r = validate(str(fn), Path.RI)
    assert any(i.rule == "ri_etamax_gt_etamin" and i.severity.value == "warning"
               for i in r.issues)


def test_ri_no_omega_required(tmp_path):
    """RI should not require OmegaStart/OmegaRange — only RMin/RMax."""
    fn = tmp_path / "p.txt"
    fn.write_text(textwrap.dedent("""
        RMin 10
        RMax 1000
        RBinSize 0.25
    """).strip())
    r = validate(str(fn), Path.RI)
    missing_required = [i.key for i in r.errors if i.rule == "required_key_missing"]
    # OmegaRange / BoxSize should NOT be in the missing list for RI
    assert "OmegaRange" not in missing_required
    assert "BoxSize" not in missing_required
    assert "OmegaStart" not in missing_required


def test_ri_requires_rmin_rmax(tmp_path):
    """RI must flag missing RMin and RMax."""
    fn = tmp_path / "p.txt"
    fn.write_text("# empty RI config\n")
    r = validate(str(fn), Path.RI)
    missing = {i.key for i in r.errors if i.rule == "required_key_missing"}
    assert "RMin" in missing
    assert "RMax" in missing


# ─── PF cross-field rules ────────────────────────────────────────────────────


def test_pf_nscans_without_scanstep_warns(tmp_path):
    fn = tmp_path / "p.txt"
    fn.write_text("nScans 5\n")
    r = validate(str(fn), Path.PF)
    assert any(i.rule == "pf_nscans_implies_scanstep" for i in r.warnings)


def test_pf_nscans_with_scanstep_ok(tmp_path):
    fn = tmp_path / "p.txt"
    fn.write_text("nScans 5\nScanStep 10\nBeamSize 5\n")
    r = validate(str(fn), Path.PF)
    assert not any(i.rule == "pf_nscans_implies_scanstep" for i in r.warnings)


def test_pf_nscans_1_no_warning(tmp_path):
    """nScans=1 is FF-style, no need for ScanStep."""
    fn = tmp_path / "p.txt"
    fn.write_text("nScans 1\n")
    r = validate(str(fn), Path.PF)
    assert not any(i.rule == "pf_nscans_implies_scanstep" for i in r.warnings)


# ─── Path-scoped registry counts ─────────────────────────────────────────────


def test_ri_shrunk():
    """After the path cleanup, RI should have far fewer applicable keys
    than FF — proof that we're not over-including indexing-only params."""
    from midas_params.registry import for_path
    ff_n = len(for_path(Path.FF))
    ri_n = len(for_path(Path.RI))
    assert ri_n < ff_n - 30, \
        f"Expected RI ({ri_n}) to be much smaller than FF ({ff_n})"


def test_ri_has_integration_keys():
    """RI must know about RMin, RMax, RBinSize, EtaMin, EtaMax."""
    from midas_params.registry import for_path
    ri_names = {p.name for p in for_path(Path.RI)}
    assert {"RMin", "RMax", "RBinSize", "EtaMin", "EtaMax"} <= ri_names


def test_ff_does_not_include_ri_only_keys():
    """RI-only integration bounds shouldn't appear in FF."""
    from midas_params.registry import for_path
    ff_names = {p.name for p in for_path(Path.FF)}
    assert "RMin" not in ff_names
    assert "RMax" not in ff_names


def test_ff_no_longer_includes_grain_keys_in_ri():
    """Grain-analysis keys should be absent from RI."""
    from midas_params.registry import for_path
    ri_names = {p.name for p in for_path(Path.RI)}
    assert "Rsample" not in ri_names
    assert "MinNrSpots" not in ri_names
    assert "Completeness" not in ri_names
    assert "GrainsFile" not in ri_names
