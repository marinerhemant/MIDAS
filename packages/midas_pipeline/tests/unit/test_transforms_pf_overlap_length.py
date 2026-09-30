"""Issue #12: the PF transforms stage must hand the archive's OverlapLength to
the merge. It passed SkipFrame / UseMaximaPositions / UsePixelOverlap but not
OverlapLength, so every PF scan merged at the 2.0 px default."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("midas_transforms")

from midas_pipeline.stages import transforms as T


class _Stop(Exception):
    pass


def test_pf_merge_receives_archive_merge_keys(tmp_path: Path, monkeypatch):
    import midas_pipeline._pf_scans as pfs
    import midas_transforms.merge as M
    import midas_transforms.params as P

    scan_dir = tmp_path / "scan1"
    scan_dir.mkdir()
    zip_path = scan_dir / "s.MIDAS.zip"
    zip_path.write_bytes(b"")
    ps = scan_dir / "AllPeaks_PS.bin"
    ps.write_bytes(b"")
    scan = SimpleNamespace(scan_nr=1, zip_path=zip_path, allpeaks_ps_bin=ps,
                           allpeaks_px_bin=scan_dir / "AllPeaks_PX.bin",
                           scan_dir=scan_dir)

    zp = P.ZarrParams(OverlapLength=3.0, UseMaximaPositions=1,
                      UsePixelOverlap=0, SkipFrame=0)
    monkeypatch.setattr(P, "read_zarr_params", lambda _p: zp)
    monkeypatch.setattr(pfs, "iter_pf_scans", lambda **kw: [scan])
    monkeypatch.setattr(
        pfs, "fan_out_scans",
        lambda scans, fn, **kw: [(s, _safe(fn, s)) for s in scans])

    seen = {}

    def _fake_merge(**kw):
        seen.update(kw)
        raise _Stop

    monkeypatch.setattr(M, "merge_overlapping_peaks", _fake_merge)

    cfg = SimpleNamespace(params_file="p.txt", raw_dir=str(tmp_path),
                          scan=SimpleNamespace(n_scans=1), device="cpu",
                          dtype="float64", scan_workers=1)
    ctx = SimpleNamespace(config=cfg, layer_dir=tmp_path, layer_nr=1, is_pf=True)
    res = T._run_pf(ctx, started=0.0)

    assert seen["overlap_length"] == 3.0
    assert seen["use_maxima_positions"] is True
    assert res.metrics["n_scans_failed"] == 1   # the sentinel stopped it


def _safe(fn, s):
    try:
        return fn(s)
    except Exception as e:  # noqa: BLE001 - mirror fan_out_scans
        return e
