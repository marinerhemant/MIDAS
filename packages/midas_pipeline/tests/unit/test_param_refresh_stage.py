"""zip_convert must re-read Parameters.txt into an archive it reuses.

Regression: zip_convert reused any existing ``*.MIDAS.zip`` unconditionally and
never compared it against ``Parameters.txt``, while ``transforms`` reads the
geometry from the **zarr**. Editing ``tx``/``Lsd``/``BC`` and re-running into the
same result folder therefore kept the OLD value, silently, and the run reported
success -- which reads as "changing tx does nothing" and sends people looking
for a ``tx`` in the refiner that was never supposed to be there.

Refreshing the zarr alone is not enough, though: the stages downstream skip when
their own output exists, so a changed ``RingThresh`` with ``AllPeaks_PS.bin``
still on disk would just move the staleness. That is refused, not warned about.
"""

from __future__ import annotations

import types

import pytest

from midas_pipeline.stages._param_refresh import (KEY_EARLIEST_STAGE,
                                                  STAGE_ORDER, earliest_stage,
                                                  refresh_zip_params,
                                                  stale_outputs)


# ── the key -> stage map ─────────────────────────────────────────────────────
def test_geometry_keys_enter_at_transforms():
    for key in ("tx", "ty", "tz", "BC", "Wedge", "p3", "a4", "iso_R2"):
        assert KEY_EARLIEST_STAGE[key] == "transforms", key


def test_peak_search_keys_enter_at_peakfit():
    for key in ("RingThresh", "MinPeakSNR", "BgSubtract", "ImTransOpt"):
        assert KEY_EARLIEST_STAGE[key] == "peakfit", key


def test_lattice_and_wavelength_enter_at_hkl():
    for key in ("LatticeConstant", "SpaceGroup", "Wavelength", "Lsd"):
        assert KEY_EARLIEST_STAGE[key] == "hkl", key


def test_unmapped_key_is_treated_as_invalidating_everything():
    """An unknown key has an unknown blast radius; assume the worst."""
    assert earliest_stage(["SomeKeyNobodyMapped"]) == STAGE_ORDER[0]


def test_earliest_wins_across_several_keys():
    assert earliest_stage(["MarginRadial", "tx"]) == "transforms"
    assert earliest_stage(["tx", "RingThresh"]) == "peakfit"


# ── the stale-output guard ───────────────────────────────────────────────────
def test_no_outputs_means_nothing_stale(tmp_path):
    assert stale_outputs(["tx"], tmp_path) == []


def test_transforms_output_is_stale_after_a_tx_edit(tmp_path):
    (tmp_path / "InputAll.csv").write_text("")
    found = stale_outputs(["tx"], tmp_path)
    assert [f[1] for f in found] == ["transforms"]


def test_a_tx_edit_ignores_an_upstream_peakfit_output(tmp_path):
    """tx enters at transforms, so the peak search stays valid."""
    (tmp_path / "Temp").mkdir()
    (tmp_path / "Temp" / "AllPeaks_PS.bin").write_bytes(b"")
    assert stale_outputs(["tx"], tmp_path) == []


def test_a_ringthresh_edit_does_invalidate_the_peak_search(tmp_path):
    (tmp_path / "Temp").mkdir()
    (tmp_path / "Temp" / "AllPeaks_PS.bin").write_bytes(b"")
    found = stale_outputs(["RingThresh"], tmp_path)
    assert [f[1] for f in found] == ["peakfit"]


def test_later_stages_are_invalidated_transitively(tmp_path):
    (tmp_path / "InputAll.csv").write_text("")
    (tmp_path / "Grains.csv").write_text("")
    stages = {f[1] for f in stale_outputs(["tx"], tmp_path)}
    assert stages == {"transforms", "process_grains"}


# ── the stage entry point ────────────────────────────────────────────────────
class _FakeReport:
    def __init__(self, keys):
        self.applied = list(keys)
        self.changed_keys = list(keys)

    def summary(self):
        return "refreshed " + ",".join(self.changed_keys)

    def to_metrics(self):
        return {"n_params_refreshed": len(self.applied)}


def _patch_zipper(monkeypatch, changed, applied=None):
    """Stand in for midas_zipper.param_refresh without building an archive."""
    import midas_zipper.param_refresh as pr

    changes = [types.SimpleNamespace(source_key=k) for k in changed]
    monkeypatch.setattr(pr, "diff_analysis_params",
                        lambda *a, **k: (changes, [], []))
    seen = {}

    def _apply(zip_path, param_file, **kw):
        seen["called"] = True
        return _FakeReport(applied if applied is not None else changed)

    monkeypatch.setattr(pr, "refresh_analysis_params", _apply)
    return seen


def test_refresh_refuses_when_the_edit_invalidates_an_existing_output(
        tmp_path, monkeypatch):
    pytest.importorskip("midas_zipper.param_refresh")
    seen = _patch_zipper(monkeypatch, ["RingThresh"])
    (tmp_path / "Temp").mkdir()
    (tmp_path / "Temp" / "AllPeaks_PS.bin").write_bytes(b"")

    with pytest.raises(RuntimeError, match="still on disk"):
        refresh_zip_params(zip_path=tmp_path / "d.MIDAS.zip",
                           param_file=str(tmp_path / "Parameters.txt"),
                           work_dir=tmp_path)
    assert "called" not in seen           # nothing was written


def test_force_proceeds_past_stale_outputs(tmp_path, monkeypatch):
    pytest.importorskip("midas_zipper.param_refresh")
    seen = _patch_zipper(monkeypatch, ["RingThresh"])
    (tmp_path / "Temp").mkdir()
    (tmp_path / "Temp" / "AllPeaks_PS.bin").write_bytes(b"")

    report = refresh_zip_params(zip_path=tmp_path / "d.MIDAS.zip",
                                param_file=str(tmp_path / "Parameters.txt"),
                                work_dir=tmp_path, force=True)
    assert seen.get("called") is True
    assert report.changed_keys == ["RingThresh"]


def test_clean_tree_refreshes_without_force(tmp_path, monkeypatch):
    pytest.importorskip("midas_zipper.param_refresh")
    seen = _patch_zipper(monkeypatch, ["tx"])
    report = refresh_zip_params(zip_path=tmp_path / "d.MIDAS.zip",
                                param_file=str(tmp_path / "Parameters.txt"),
                                work_dir=tmp_path)
    assert seen.get("called") is True
    assert report.changed_keys == ["tx"]


# ── config plumbing ──────────────────────────────────────────────────────────
def test_refresh_defaults_on_and_can_be_turned_off():
    from midas_pipeline.config import PipelineConfig
    import dataclasses

    fields = {f.name: f for f in dataclasses.fields(PipelineConfig)}
    assert fields["refresh_params"].default is True
    assert fields["force_param_refresh"].default is False


def test_cli_exposes_the_flags():
    from midas_pipeline.cli import _build_parser

    p = _build_parser()
    ns = p.parse_args(["run", "--params", "P.txt", "--result", "r/", "--no-refresh-params"])
    assert ns.refresh_params is False
    ns = p.parse_args(["run", "--params", "P.txt", "--result", "r/", "--force-param-refresh"])
    assert ns.refresh_params is True
    assert ns.force_param_refresh is True


def test_cli_flags_reach_the_config():
    """A flag the config never reads is a flag that does nothing."""
    from midas_pipeline.cli import _build_parser, build_config

    p = _build_parser()
    cfg = build_config(p.parse_args(["run", "--params", "P.txt", "--result", "r/",
                                     "--no-refresh-params"]))
    assert cfg.refresh_params is False


# ── end to end through the real stage ────────────────────────────────────────
def _real_archive(tmp_path, params_text):
    """Build a .MIDAS.zip through the actual create path."""
    zarr = pytest.importorskip("zarr")
    import numpy as np
    from midas_zipper.ff_zip import (create_zarr_structure,
                                     parse_parameter_file,
                                     write_analysis_parameters)

    pf = tmp_path / "Parameters.txt"
    pf.write_text(params_text)
    fn = tmp_path / "scan_000001.MIDAS.zip"
    store = zarr.ZipStore(str(fn), mode="w")
    root = zarr.group(store=store, overwrite=True)
    write_analysis_parameters(create_zarr_structure(root),
                              parse_parameter_file(str(pf)))
    root["exchange"].create_dataset("data", data=np.zeros((4, 8, 8), "uint16"))
    store.close()
    return pf, fn


_PARAMS = "tx 0.0\nty -0.15\nLsd 1000000.0\nSkipFrame 1\nRingThresh 1 100\n"


def _stage_ctx(tmp_path, pf, *, refresh=True, force=False):
    return types.SimpleNamespace(
        is_pf=False,
        layer_nr=1,
        layer_dir=tmp_path,
        config=types.SimpleNamespace(
            params_file=str(pf), zarr_path=None, convert_files=False,
            refresh_params=refresh, force_param_refresh=force,
        ),
    )


def test_zip_convert_refreshes_a_reused_archive(tmp_path):
    """The end-to-end fix: edit tx, re-run, the zarr must carry the new value."""
    import shutil
    if shutil.which("zip") is None:
        pytest.skip("Info-ZIP 'zip' not on PATH")
    zarr = pytest.importorskip("zarr")
    from midas_pipeline.stages import zip_convert

    pf, fn = _real_archive(tmp_path, _PARAMS)
    ap = "analysis/process/analysis_parameters"
    assert zarr.open(str(fn), "r")[f"{ap}/tx"][0] == 0.0

    pf.write_text(_PARAMS.replace("tx 0.0", "tx -0.2670"))
    result = zip_convert.run(_stage_ctx(tmp_path, pf))

    assert zarr.open(str(fn), "r")[f"{ap}/tx"][0] == pytest.approx(-0.2670)
    assert result.metrics["n_params_refreshed"] == 1
    assert "tx" in result.metrics["params_refreshed"]


def test_no_refresh_params_leaves_the_archive_alone(tmp_path):
    zarr = pytest.importorskip("zarr")
    from midas_pipeline.stages import zip_convert

    pf, fn = _real_archive(tmp_path, _PARAMS)
    pf.write_text(_PARAMS.replace("tx 0.0", "tx -0.2670"))
    zip_convert.run(_stage_ctx(tmp_path, pf, refresh=False))

    ap = "analysis/process/analysis_parameters"
    assert zarr.open(str(fn), "r")[f"{ap}/tx"][0] == 0.0


def test_zip_convert_refuses_a_baked_in_edit(tmp_path):
    """SkipFrame changed => the stored frames are wrong; rebuild, do not patch."""
    from midas_pipeline.stages import zip_convert

    pf, _fn = _real_archive(tmp_path, _PARAMS)
    pf.write_text(_PARAMS.replace("SkipFrame 1", "SkipFrame 2"))
    with pytest.raises(Exception, match="frames"):
        zip_convert.run(_stage_ctx(tmp_path, pf))
