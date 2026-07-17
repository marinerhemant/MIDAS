"""N5/N6 tests: per-scan claims + fan-out helper.

Two independent runners over one scan list must never double-process a
scan (the Ni run's external 2-worker helper raced benignly — outputs
identical — but two owners per scan is still wrong).
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from midas_pipeline._pf_scans import (
    PFScanInfo, claim_scan, fan_out_scans, release_scan,
)


def _scan(tmp_path: Path, nr: int) -> PFScanInfo:
    return PFScanInfo(
        layer_nr=1, scan_nr=nr, abs_scan_nr=100 + nr,
        scan_dir=tmp_path / f"scan{nr}",
        zip_path=tmp_path / f"scan{nr}" / "x.MIDAS.zip",
        y_position_um=float(nr),
    )


def test_claim_is_exclusive_and_releasable(tmp_path: Path):
    assert claim_scan(tmp_path, "peakfit", 3)
    assert not claim_scan(tmp_path, "peakfit", 3)      # second owner refused
    assert claim_scan(tmp_path, "transforms", 3)       # other stage independent
    release_scan(tmp_path, "peakfit", 3)
    assert claim_scan(tmp_path, "peakfit", 3)          # reclaimable


def test_stale_claim_from_dead_local_pid_is_broken(tmp_path: Path):
    import socket
    claims = tmp_path / "midas_log" / "claims"
    claims.mkdir(parents=True)
    # A pid that cannot exist keeps the claim stale.
    (claims / "peakfit.scan7.claim").write_text(
        f"{socket.gethostname()} 99999999\n")
    assert claim_scan(tmp_path, "peakfit", 7), "stale claim must be broken"


def test_foreign_host_claim_is_honoured(tmp_path: Path):
    claims = tmp_path / "midas_log" / "claims"
    claims.mkdir(parents=True)
    (claims / "peakfit.scan7.claim").write_text("some-other-host 1234\n")
    assert not claim_scan(tmp_path, "peakfit", 7)


@pytest.mark.parametrize("n_workers", [1, 3])
def test_fan_out_runs_every_scan_once(tmp_path: Path, n_workers: int):
    scans = [_scan(tmp_path, i + 1) for i in range(6)]
    seen = []

    def worker(s):
        seen.append(s.scan_nr)
        return f"done-{s.scan_nr}"

    out = fan_out_scans(scans, worker, layer_dir=tmp_path,
                        stage="peakfit", n_workers=n_workers)
    assert sorted(seen) == [1, 2, 3, 4, 5, 6]
    assert [r for _s, r in out] == [f"done-{i}" for i in range(1, 7)]
    # All claims released.
    assert not list((tmp_path / "midas_log" / "claims").glob("*.claim"))


def test_fan_out_skips_scan_claimed_elsewhere(tmp_path: Path):
    scans = [_scan(tmp_path, i + 1) for i in range(3)]
    claims = tmp_path / "midas_log" / "claims"
    claims.mkdir(parents=True)
    (claims / "peakfit.scan2.claim").write_text("some-other-host 1\n")

    out = dict((s.scan_nr, r) for s, r in fan_out_scans(
        scans, lambda s: "ok", layer_dir=tmp_path,
        stage="peakfit", n_workers=2))
    assert out[1] == "ok" and out[3] == "ok"
    assert out[2] == "claimed-elsewhere"
    # The foreign claim must NOT be released by us.
    assert (claims / "peakfit.scan2.claim").exists()


def test_fan_out_captures_worker_exception(tmp_path: Path):
    scans = [_scan(tmp_path, 1)]

    def boom(s):
        raise RuntimeError("scan exploded")

    (_s, r), = fan_out_scans(scans, boom, layer_dir=tmp_path,
                             stage="transforms", n_workers=1)
    assert isinstance(r, RuntimeError)
    # Claim released even on failure.
    assert claim_scan(tmp_path, "transforms", 1)
