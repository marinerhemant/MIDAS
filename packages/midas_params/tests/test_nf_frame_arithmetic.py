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
    predicted_count = n_dist * nfd   # RAW count; SumFrames does not scale it
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
    b = _files_the_reduction_opens(nfd=1800, n_sum=3, n_dist=2)
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
    assert not _errors(_write(tmp_path, nfd=1800, end=6842, step=-0.1, n_sum=3))


def test_raising_sum_frames_alone_is_ACCEPTED(tmp_path):
    """Raising SumFrames without touching anything else must validate.

    NrFilesPerDistance 1800 beside SumFrames 3 reads as the RAW count, which
    workflows._normalise_sum_frames converts to 600 (and OmegaStep to -0.3).
    That is the supported way to enable summing -- change one key -- so the
    validator has to accept exactly what the pipeline accepts. Demanding only
    the post-sum reading rejected a file that runs perfectly.
    """
    assert not _errors(_write(tmp_path, nfd=1800, end=6842, step=-0.1, n_sum=3))




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


# --- HDF5 (20-ID-D): one container per DISTANCE, not one file per frame ------

def _write_h5(tmp_path, *, n_dist, present, start=722):
    scan = tmp_path / "h5scan"
    scan.mkdir(exist_ok=True)
    for n in present:
        (scan / f"nf_{n:06d}.h5").write_bytes(b"")
    f = tmp_path / "p_h5.txt"
    f.write_text(
        f"DataDirectory {scan}\nOrigFileName nf\nextOrig h5\nDataLoc exchange/data\n"
        f"StartNr 0\nEndNr 1439\nRawStartNr {start}\n"
        f"NrFilesPerDistance 1440\nnDistances {n_dist}\nSumFrames 1\n"
        f"OmegaStart 179.75\nOmegaStep -0.25\nOmegaRange -180 180\n"
        + "".join(f"Lsd {8138.7 + 2000 * i}\nBC 2450 63\n" for i in range(n_dist))
        + "NrPixelsY 5120\nNrPixelsZ 4600\npx 0.548\n"
        f"Wavelength 0.19582415\nSpaceGroup 225\n"
        f"LatticeParameter 3.5954 3.5954 3.5954 90 90 90\n")
    return f


def test_h5_one_file_per_distance_passes(tmp_path):
    """The bt_20id_jul26b nf_sampleF layout: files 722, 723 for two distances."""
    assert not _errors(_write_h5(tmp_path, n_dist=2, present=[722, 723]))


def test_h5_missing_distance_file_is_reported(tmp_path):
    errs = _errors(_write_h5(tmp_path, n_dist=2, present=[722]))
    assert any(e.rule == "frames_exist_on_disk" and "000723.h5" in e.message for e in errs)


def test_h5_matches_the_reader(tmp_path):
    """Anchor on the loader: the validator's range is what layer_file opens."""
    from midas_nf_preprocess.process_images.io import layer_file
    p = ProcessParams(data_directory="/d", orig_filename="nf", raw_start_nr=722,
                      n_distances=3, ext_orig="h5")
    opened = [int(layer_file(p, d).split("_")[-1].split(".")[0]) for d in (1, 2, 3)]
    assert opened == [722, 723, 724]
