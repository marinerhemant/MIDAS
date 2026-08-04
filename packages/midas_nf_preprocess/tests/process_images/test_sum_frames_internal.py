"""SumFrames is internal: it must not change a single value the user supplies.

The parameter file describes the experiment as performed --
``NrFilesPerDistance`` is the raw image count per distance, ``OmegaStep`` is the
rotation between raw images, ``EndNr`` is optional. ``SumFrames`` is a reduction
choice, so every quantity it affects is derived inside the code.

It used to be the opposite: the user restated NrFilesPerDistance, EndNr and
OmegaStep in post-sum units and the pipeline rewrote the file. Three coupled
edits for one setting is three chances to be wrong, and each way of being wrong
failed late and unhelpfully -- a missing file hundreds past the end of the scan,
or ``SpotsInfo.bin ... has N bits, need 3N`` after the whole reduction had run.

The invariants below are what make that impossible, so they are asserted
directly rather than through any single code path.
"""
import pytest

from midas_nf_preprocess.process_images.io import frame_paths
from midas_nf_preprocess.process_images.params import ProcessParams

RAW_PER_DISTANCE = 1800
RAW_STEP = -0.1
START = 5043


def _params(n_sum, *, n_dist=2, nfd=RAW_PER_DISTANCE, wf=0):
    return ProcessParams(
        data_directory="/d", orig_filename="s", raw_start_nr=START,
        nr_files_per_distance=nfd, n_distances=n_dist, wf_images=wf,
        sum_frames=n_sum)


def _files(p):
    out = []
    for d in range(1, p.n_distances + 1):
        out += [int(f.split("_")[-1].split(".")[0]) for f in frame_paths(p, d)]
    return out


@pytest.mark.parametrize("n_sum", [1, 2, 3, 4, 6, 9])
def test_files_read_are_independent_of_sum_frames(n_sum):
    """THE invariant: summing regroups frames, it does not read different ones."""
    assert _files(_params(n_sum)) == _files(_params(1))


@pytest.mark.parametrize("n_sum", [1, 2, 3, 4, 6, 9])
def test_output_frames_scale_with_sum_frames(n_sum):
    p = _params(n_sum)
    assert p.n_raw_per_distance == RAW_PER_DISTANCE
    assert p.n_frames_per_distance == RAW_PER_DISTANCE // n_sum


@pytest.mark.parametrize("n_sum", [1, 2, 3, 4, 6, 9])
def test_reduction_and_fit_agree_on_the_frame_count(n_sum):
    """The disagreement that produced 'has N bits, need 3N' after a full run.

    The reduction sizes SpotsInfo.bin by its frame count and the fit reads the
    file back expecting its own. Both now derive from the same raw count and the
    same SumFrames, from the same unmodified parameter file.
    """
    from midas_nf_fitorientation.params import FitParams
    fp = FitParams()
    fp.nr_files_per_distance = RAW_PER_DISTANCE
    fp.sum_frames = n_sum
    assert fp.n_frames_per_distance == _params(n_sum).n_frames_per_distance


@pytest.mark.parametrize("n_sum", [1, 2, 3, 4, 6, 9])
def test_total_rotation_is_invariant(n_sum):
    """Physics check: the sample still turns through the same angle.

    The fit's step must scale with SumFrames -- a summed frame genuinely spans
    that much rotation -- while the file keeps the recorded raw step.
    """
    from midas_nf_fitorientation.params import FitParams
    fp = FitParams()
    fp.nr_files_per_distance = RAW_PER_DISTANCE
    fp.sum_frames = n_sum
    fp.omega_step_raw = RAW_STEP
    assert fp.omega_step == pytest.approx(RAW_STEP * n_sum)
    assert (fp.omega_step * fp.n_frames_per_distance
            == pytest.approx(RAW_STEP * RAW_PER_DISTANCE))


def test_frame_grouping_is_exact_under_the_floor_convention():
    """Summed frame k must cover exactly raw frames k*n .. k*n+n-1.

    The fit computes floor((omega - omega_start) / omega_step), so
    omega_start + j*step is the LEADING EDGE of frame j. That makes scaling the
    step exact with omega_start untouched. Under a centre-of-bin convention
    omega_start would additionally need shifting by (n-1)/2 * raw_step, and
    nothing in the code would have said so.
    """
    import numpy as np
    n, w0 = 3, 180.0
    rng = np.random.default_rng(0)
    w = w0 + rng.uniform(-180, 0, 100_000)
    raw_idx = np.floor((w - w0) / RAW_STEP).astype(int)
    direct = np.floor((w - w0) / (RAW_STEP * n)).astype(int)
    assert np.array_equal(raw_idx // n, direct)


def test_indivisible_sum_frames_is_rejected_at_parse_time():
    with pytest.raises(ValueError, match="does not divide"):
        _params(7, nfd=1800)


def test_single_distance_and_white_field_offsets_still_hold():
    for n_sum in (1, 3):
        assert _files(_params(n_sum, n_dist=1)) == _files(_params(1, n_dist=1))
        assert _files(_params(n_sum, wf=5)) == _files(_params(1, wf=5))


def test_parameter_file_is_never_rewritten_for_sum_frames():
    """The conversion is in-memory only; nothing edits the user's file.

    workflows._normalise_sum_frames used to rewrite three keys on disk, so a
    file no longer said what the experiment did after one run.
    """
    from midas_nf_pipeline import workflows
    assert not hasattr(workflows, "_normalise_sum_frames")
