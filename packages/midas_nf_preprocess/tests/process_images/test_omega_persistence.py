"""The omega-persistence diagnostic behind the SumFrames recommendation.

Measures the overlap of detection masks at increasing frame lag, pooled over
every lit pixel, against a non-adjacent-pairing control. Validated on real data
(Ce-5%Y, 0.1 deg/frame): excess over control +0.30, +0.15, +0.02 -> spans 3
frames, which is the SumFrames value that won a 14-configuration catalog.

The control is not optional. NF frames share lit regions simply because the same
rings are lit throughout, which put the raw control at 0.17-0.22 on real data --
against the ~0.008 that chance overlap of 0.8%-lit frames would predict. Without
subtracting it, every dataset looks maximally persistent.
"""
import logging

import pytest
import torch

from midas_nf_preprocess.process_images.params import ProcessParams
from midas_nf_preprocess.process_images.pipeline import ProcessImagesPipeline


def _pipe(**over):
    p = ProcessParams.__new__(ProcessParams)
    for f in p.__dataclass_fields__.values():
        setattr(p, f.name, f.default if f.default is not None else None)
    p.nr_pixels_y = p.nr_pixels_z = 64
    p.nr_files_per_distance = 40
    p.n_distances = 1
    p.sum_frames = 1
    p.omega_step = 0.1
    for k, v in over.items():
        setattr(p, k, v)
    pipe = ProcessImagesPipeline(p, device="cpu")
    pipe._persist_inter, pipe._persist_base, pipe._persist_ctrl = {}, {}, {}
    pipe._persist_max_lag = 6
    return pipe


def _masks(span, n_frames=40, size=64, n_spots=60, seed=0):
    """Frames where each spot stays lit for exactly ``span`` frames."""
    g = torch.Generator().manual_seed(seed)
    m = torch.zeros(n_frames, size, size, dtype=torch.bool)
    if n_frames <= span:                       # too short to plant anything
        return [m[i] for i in range(n_frames)]
    for s in range(n_spots):
        r = int(torch.randint(1, size - 1, (1,), generator=g))
        c = int(torch.randint(1, size - 1, (1,), generator=g))
        f0 = int(torch.randint(0, n_frames - span, (1,), generator=g))
        for f in range(f0, f0 + span):
            m[f, r - 1:r + 2, c - 1:c + 2] = True
    return [m[i] for i in range(n_frames)]


@pytest.mark.parametrize("span", [1, 2, 3, 4])
def test_recovers_the_planted_span(span, caplog):
    pipe = _pipe()
    pipe._accumulate_persistence(_masks(span))
    with caplog.at_level(logging.INFO):
        pipe._log_sumframes_recommendation(1)
    text = " ".join(r.getMessage() for r in caplog.records)
    assert f"span ~{span} frames" in text or (span == 1 and "span 1 frame" in text)


def test_single_frame_spots_recommend_no_summing(caplog):
    pipe = _pipe()
    pipe._accumulate_persistence(_masks(1))
    with caplog.at_level(logging.INFO):
        pipe._log_sumframes_recommendation(1)
    text = " ".join(r.getMessage() for r in caplog.records)
    assert "SumFrames 1 is correct" in text


def test_recommends_raising_sum_frames_when_undersummed(caplog):
    pipe = _pipe(sum_frames=1)
    pipe._accumulate_persistence(_masks(3))
    with caplog.at_level(logging.INFO):
        pipe._log_sumframes_recommendation(1)
    text = " ".join(r.getMessage() for r in caplog.records)
    assert "SumFrames is 1" in text and "consider summing" in text


def test_quiet_when_already_matched(caplog):
    pipe = _pipe(sum_frames=3)
    pipe._accumulate_persistence(_masks(3))
    with caplog.at_level(logging.INFO):
        pipe._log_sumframes_recommendation(1)
    text = " ".join(r.getMessage() for r in caplog.records)
    assert "is matched" in text
    assert "consider summing" not in text


def test_control_is_subtracted():
    """A static pattern lit in EVERY frame must not read as persistence."""
    pipe = _pipe()
    static = torch.zeros(64, 64, dtype=torch.bool)
    static[10:40, 10:40] = True
    pipe._accumulate_persistence([static.clone() for _ in range(40)])
    excess = {k: (pipe._persist_inter[k] - pipe._persist_ctrl[k]) / pipe._persist_base[k]
              for k in pipe._persist_base}
    assert max(excess.values()) < pipe.PERSIST_FLOOR, (
        f"an all-frames-identical pattern reported persistence: {excess}")


def test_too_few_frames_is_a_noop():
    pipe = _pipe()
    pipe._accumulate_persistence(_masks(3, n_frames=2))
    assert not pipe._persist_base
