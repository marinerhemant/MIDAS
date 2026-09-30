"""Candidate brightness: per-candidate mean ln(I / ring median) over matched
spots, the per-voxel contender, and the opt-in tie-break."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from midas_pipeline.diagnostics.candidate_brightness import (
    brightness_tiebreak, candidate_brightness, spot_log_intensity)
from midas_pipeline.find_grains._consolidation_io import (
    write_ids_bin, write_keys_bin, write_vals_bin)

HDR = ("YLab ZLab Omega GrainRadius SpotID RingNumber Eta Ttheta OmegaIni "
       "YOrigDetCor ZOrigDetCor YRawPx ZRawPx OmegaDetCor IntegratedIntensity "
       "RawSumIntensity maskTouched FitRMSE")


def _layer(tmp: Path):
    """2 scans, ring 1 only. Spot intensities (NewID: I) chosen so the ring
    median is 100: bright 400 (ids 1-6), median 100 (ids 7-12), dim 25 (13-18)."""
    I = {**{i: 400.0 for i in range(1, 7)}, **{i: 100.0 for i in range(7, 13)},
         **{i: 25.0 for i in range(13, 19)}}
    rows = []
    for nid in range(1, 19):
        scan, orig = (nid - 1) % 2, 100 + nid
        rows.append((nid, orig, scan))
    (tmp / "IDsMergedScanning.csv").write_text(
        "NewID,OrigID,ScanNr\n" + "".join(f"{a},{b},{c}\n" for a, b, c in rows))
    for s in range(2):
        lines = [HDR]
        for nid, orig, scan in rows:
            if scan == s:
                v = np.zeros(18); v[4] = orig; v[5] = 1; v[14] = I[nid]
                lines.append(" ".join(f"{x:g}" for x in v))
        (tmp / f"InputAllExtraInfoFittingAll{s}.csv").write_text("\n".join(lines) + "\n")
    out = tmp / "Output"; out.mkdir()

    def cand(nobs, nexp, ia):
        r = np.zeros(16); r[1] = ia; r[2:11] = np.eye(3).ravel(); r[14] = nexp; r[15] = nobs
        return r
    # voxel 0: winner bright (6/6), contender dim (6/7 -> gap 0.143 > margin 0.05? use 0.9)
    # voxel 1: winner median (6/6), contender bright at equal completeness
    vals = [np.stack([cand(6, 6, 0.1), cand(6, 6.5, 0.2)]),
            np.stack([cand(6, 6, 0.1), cand(6, 6, 0.3)])]
    ids = [np.r_[1:7, 13:19].astype(np.int32), np.r_[7:13, 1:7].astype(np.int32)]
    keys = [np.array([[1, 6, 6, 0], [2, 6, 6, 0]], np.uint64),
            np.array([[3, 6, 6, 0], [4, 6, 6, 0]], np.uint64)]
    write_vals_bin(out / "IndexBest_all.bin", vals)
    write_keys_bin(out / "IndexKey_all.bin", keys)
    write_ids_bin(out / "IndexBest_IDs_all.bin", ids)
    return tmp


def test_spot_log_intensity_is_ring_median_normalised(tmp_path):
    ln = spot_log_intensity(_layer(tmp_path), 2)
    assert np.isnan(ln[0])
    np.testing.assert_allclose(ln[1:7], np.log(4.0))
    np.testing.assert_allclose(ln[7:13], 0.0)
    np.testing.assert_allclose(ln[13:19], np.log(0.25))


def test_candidate_brightness_and_contender(tmp_path):
    r = candidate_brightness(_layer(tmp_path), 2, margin=0.1, conf_min=0.8)
    np.testing.assert_allclose(r["brightness"], [np.log(4), np.log(.25), 0.0, np.log(4)], rtol=1e-6)
    assert list(r["winner"]) == [0, 0]            # voxel 1 tie -> lower IA
    assert list(r["contender"]) == [1, 1]
    np.testing.assert_allclose(r["contender_completeness_gap"], [1 - 6 / 6.5, 0.0], rtol=1e-6)
    assert (tmp_path / "Output" / "CandidateBrightness.npz").is_file()


def test_min_hits_gives_nan(tmp_path):
    r = candidate_brightness(_layer(tmp_path), 2, min_hits=7, write=False)
    assert np.isnan(r["brightness"]).all()


def test_tiebreak_off_is_the_indexer_rule():
    comp = np.array([0.9, 0.9, 0.95]); ia = np.array([0.2, 0.1, 0.3])
    assert brightness_tiebreak(comp, ia, np.array([5., 5., -5.]), 0.0) == 2


def test_tiebreak_prefers_brighter_within_margin_only():
    comp = np.array([0.95, 0.93, 0.80]); ia = np.zeros(3)
    b = np.array([-1.0, 1.0, 3.0])
    assert brightness_tiebreak(comp, ia, b, 0.05) == 1     # 0.80 is outside the margin
    assert brightness_tiebreak(comp, ia, np.array([1.0, np.nan, 3.0]), 0.05) == 0


def test_unexpected_error_in_the_default_on_diagnostic_does_not_abort_the_stage(monkeypatch, tmp_path):
    """candidate_brightness is on by default; any failure in it must degrade to a warning, and only the
    tie-break (which needs the file) may turn it into an error."""
    import midas_pipeline.diagnostics.candidate_brightness as cb
    from midas_pipeline.stages.find_grains_stage import _write_candidate_brightness

    def boom(*a, **k):
        raise RuntimeError("something unexpected")

    monkeypatch.setattr(cb, "candidate_brightness", boom)
    _write_candidate_brightness(tmp_path, 3, required=False)          # warns, does not raise
    with pytest.raises(RuntimeError, match="needs CandidateBrightness.npz"):
        _write_candidate_brightness(tmp_path, 3, required=True)
