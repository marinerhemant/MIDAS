"""The NF frame-numbering contract, pinned against the reduction itself.

There are TWO different quantities here and conflating them caused a string of
misdiagnoses:

  RAW FILES ON DISK  RawStartNr .. RawStartNr
                       + nDistances*(NrFilesPerDistance*SumFrames + WFImages) - 1
                     This is what process_images.io.frame_paths actually opens.

  EndNr              StartNr + NrFilesPerDistance - 1, a PER-DISTANCE, POST-SUM
                     marker. It is NOT the last file on disk, and with two
                     distances it names a file barely halfway through the scan.

``test_formula_matches_the_reduction`` is the anchor: it asks frame_paths what
it opens rather than restating the formula, so if the loader's arithmetic ever
changes this fails instead of the validator silently drifting out of agreement.
"""
import pytest

pytest.importorskip("midas_nf_preprocess")

from midas_nf_preprocess.process_images.io import frame_paths
from midas_nf_preprocess.process_images.params import ProcessParams
from midas_params.schema import Path as MidasPath
from midas_params.validator import validate


def _files_the_reduction_opens(*, nfd, n_sum, n_dist, wf=0, raw_start=5043):
    p = ProcessParams(
        data_directory="/d", orig_filename="s", raw_start_nr=raw_start,
        nr_files_per_distance=nfd, n_distances=n_dist, wf_images=wf,
        sum_frames=n_sum)
    nums = []
    for d in range(1, n_dist + 1):
        nums += [int(f.split("_")[-1].split(".")[0]) for f in frame_paths(p, d)]
    return nums


@pytest.mark.parametrize("nfd,n_sum,n_dist", [
    (1800, 1, 2), (600, 3, 2), (1800, 1, 1), (900, 2, 3), (600, 3, 1),
])
def test_formula_matches_the_reduction(nfd, n_sum, n_dist):
    """The validator's range must be exactly what frame_paths opens."""
    nums = _files_the_reduction_opens(nfd=nfd, n_sum=n_sum, n_dist=n_dist)
    predicted_count = n_dist * (nfd * n_sum + 0)
    assert len(nums) == predicted_count
    assert min(nums) == 5043
    assert max(nums) == 5043 + predicted_count - 1


def test_same_raw_files_whichever_convention(tmp_path):
    """SumFrames must not change WHICH raw files are read, only their grouping.

    1800 unsummed and 600 summed-by-3 describe the same acquisition, so both
    must open the identical 3600 files. If they diverge, one of the two
    conventions is being mis-scaled.
    """
    a = _files_the_reduction_opens(nfd=1800, n_sum=1, n_dist=2)
    b = _files_the_reduction_opens(nfd=600, n_sum=3, n_dist=2)
    assert a == b


def _write(tmp_path, *, nfd, end, step, n_sum, n_files=3600, start=5043):
    scan = tmp_path / "scan"
    scan.mkdir(exist_ok=True)
    for i in range(n_files):
        (scan / f"scan_{start + i:06d}.tif").write_bytes(b"")
    f = tmp_path / "p.txt"
    f.write_text(
        f"DataDirectory {scan}\nOrigFileName scan\nextOrig tif\n"
        f"StartNr {start}\nEndNr {end}\nRawStartNr {start}\n"
        f"NrFilesPerDistance {nfd}\nnDistances 2\nSumFrames {n_sum}\n"
        f"OmegaStart 180\nOmegaStep {step}\nOmegaRange 0 180\n"
        f"Lsd 7228\nLsd 9229\nBC 996 37\nBC 1013 41\nNrPixels 2048\npx 1.48\n"
        f"Wavelength 0.1305\nSpaceGroup 194\n"
        f"LatticeParameter 3.6671 3.6671 11.805 90 90 120\n")
    return f


#: The rules this module is about. Filtering to them keeps the tests focused on
#: frame arithmetic instead of failing on unrelated required keys that a minimal
#: fixture paramfile omits.
FRAME_RULES = {"frames_exist_on_disk", "nf_frames_match_files_per_distance",
               "omega_range_within_scan", "startnr_le_endnr"}


def _issues(f, errors_only=True):
    rep = validate(str(f), MidasPath.NF)
    issues = getattr(rep, "issues", None) or getattr(rep, "messages", [])
    out = [m for m in issues if getattr(m, "rule", None) in FRAME_RULES]
    if errors_only:
        out = [m for m in out
               if str(getattr(m, "severity", "")).lower().endswith("error")]
    return out


def _errors(f):
    return _issues(f)


def test_valid_unsummed_passes(tmp_path):
    assert not _errors(_write(tmp_path, nfd=1800, end=6842, step=-0.1, n_sum=1))


def test_valid_summed_passes(tmp_path):
    assert not _errors(_write(tmp_path, nfd=600, end=5642, step=-0.3, n_sum=3))


def test_sum_frames_raised_without_dividing_is_caught(tmp_path):
    """The template bug: SumFrames 3 left beside the unsummed NrFilesPerDistance.

    Needs 10,800 raw files where 3,600 exist. Previously the disk check looked
    at one distance and ignored SumFrames, so this passed validation and failed
    hours later inside the reduction.
    """
    errs = _errors(_write(tmp_path, nfd=1800, end=6842, step=-0.1, n_sum=3))
    assert any(e.rule == "frames_exist_on_disk" for e in errs)
    msg = " ".join(e.suggestion or "" for e in errs)
    assert "10800 raw files" in msg and "NrFilesPerDistance" in msg


def test_mixed_convention_is_caught(tmp_path):
    """Post-sum NrFilesPerDistance beside a raw EndNr.

    The two stages then size SpotsInfo.bin differently -- reduction from
    NrFilesPerDistance, fit from the span -- and it dies AFTER the reduction
    with "has 5033164800 bits, need 15099494400".
    """
    errs = _errors(_write(tmp_path, nfd=600, end=6842, step=-0.1, n_sum=3))
    rules = {e.rule for e in errs}
    assert "nf_frames_match_files_per_distance" in rules
    msg = " ".join(e.message for e in errs)
    assert "EndNr=5642" in msg and "NrFilesPerDistance=1800" in msg, (
        "the error must name BOTH valid repairs")


def test_end_nr_is_not_the_last_file_on_disk(tmp_path):
    """Guard the distinction itself.

    A file whose EndNr equals the last raw file (StartNr + total - 1) is the
    'EndNr = StartNr + NrFilesPerDistance*nDistances' misreading, and it must
    NOT validate clean -- that number describes the disk range, not EndNr.
    """
    # A warning, not an error, now that the fit reads NrFilesPerDistance and
    # ignores EndNr -- but it must still be reported, because the file is
    # self-inconsistent and any reader of EndNr would be misled.
    flagged = _issues(_write(tmp_path, nfd=1800, end=5043 + 3600 - 1,
                             step=-0.1, n_sum=1), errors_only=False)
    assert any(e.rule == "nf_frames_match_files_per_distance" for e in flagged)
