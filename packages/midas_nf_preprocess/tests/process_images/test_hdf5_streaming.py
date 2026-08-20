"""HDF5 frame sources, streaming, and the TIFF/HDF5 parity that justifies both.

The point of these tests is that adding a second acquisition layout did not add
a second reduction. Same frames in, same SpotsInfo.bin out, whichever container
they arrived in and whether or not the layer was ever resident in memory.
"""

import numpy as np
import pytest
import tifffile
import torch

from midas_nf_preprocess.process_images import (
    Hdf5FrameSource,
    ProcessParams,
    ProcessImagesPipeline,
    TiffFrameSource,
    is_hdf5,
    layer_file,
    load_stack,
    open_source,
    streaming_temporal_median,
    temporal_median,
)
from midas_nf_preprocess.process_images.io import check_pixel_scale

h5py = pytest.importorskip("h5py")

NZ, NY, NFRAMES = 12, 10, 8


def _frames(seed=0, n=NFRAMES, scale=1):
    """A stack with a moving bright spot on a low pedestal.

    The spot is bright enough that ``scale=64`` pushes the maximum past 12-bit
    full scale, which is the condition ``check_pixel_scale`` keys on: values
    being multiples of 64 is not on its own suspicious, since unscaled data can
    hit one by chance.
    """
    rng = np.random.default_rng(seed)
    a = rng.integers(0, 3, size=(n, NZ, NY)).astype(np.uint16)
    for j in range(n):
        a[j, 3 + (j % 4), 4 + (j % 3)] += 1000
    return (a * scale).astype(np.uint16)


def _write_h5(path, data, loc="exchange/data"):
    with h5py.File(path, "w") as f:
        f.create_dataset(loc, data=data)


def _write_tiffs(tmp_path, data, stem="img", start=0):
    for j in range(data.shape[0]):
        tifffile.imwrite(tmp_path / f"{stem}_{start + j:06d}.tif", data[j])


def _params(tmp_path, **kw):
    base = dict(
        data_directory=str(tmp_path),
        orig_filename="img",
        nr_pixels_y=NY,
        nr_pixels_z=NZ,
        nr_files_per_distance=NFRAMES,
        n_distances=1,
    )
    base.update(kw)
    return ProcessParams(**base)


# ----------------------------------------------------------------------
# layout selection
# ----------------------------------------------------------------------

def test_is_hdf5_covers_the_extensions_seen_in_the_wild():
    for ext in ("h5", "hdf5", "hdf", "nxs", "H5", ".h5"):
        assert is_hdf5(ProcessParams(ext_orig=ext)), ext
    for ext in ("tif", "tiff", "bin"):
        assert not is_hdf5(ProcessParams(ext_orig=ext))


def test_layer_file_index_advances_per_distance_not_per_frame(tmp_path):
    """One HDF5 holds a whole distance, so layer 2 is the NEXT file."""
    p = _params(tmp_path, ext_orig="h5", raw_start_nr=708, n_distances=3)
    assert layer_file(p, 1).endswith("img_000708.h5")
    assert layer_file(p, 2).endswith("img_000709.h5")
    assert layer_file(p, 3).endswith("img_000710.h5")


def test_open_source_dispatches_on_ext(tmp_path):
    data = _frames()
    _write_h5(tmp_path / "img_000000.h5", data)
    _write_tiffs(tmp_path, data)
    assert isinstance(open_source(_params(tmp_path, ext_orig="h5"), 1), Hdf5FrameSource)
    assert isinstance(open_source(_params(tmp_path, ext_orig="tif"), 1), TiffFrameSource)


# ----------------------------------------------------------------------
# the parity that matters
# ----------------------------------------------------------------------

def test_hdf5_and_tiff_give_the_same_stack(tmp_path):
    data = _frames()
    _write_h5(tmp_path / "img_000000.h5", data)
    _write_tiffs(tmp_path, data)
    a = load_stack(_params(tmp_path, ext_orig="h5"), 1, dtype=torch.float64)
    b = load_stack(_params(tmp_path, ext_orig="tif"), 1, dtype=torch.float64)
    assert torch.equal(a, b)


def test_streaming_and_materialised_give_an_identical_bitmask(tmp_path):
    """The whole justification for the streaming path: it changes nothing.

    Same frames, same median, same threshold -- so the only thing streaming
    buys is that the layer is never resident. If these two ever diverge, the
    memory fix has become a reduction change and the 20-ID results are not
    comparable with the 1-ID ones.
    """
    data = _frames()
    _write_h5(tmp_path / "img_000000.h5", data)

    common = dict(ext_orig="h5", blanket_subtraction=1.0, do_log_filter=0)
    streamed = ProcessImagesPipeline(
        _params(tmp_path, stream_frames=1, **common), device="cpu"
    ).process_layer(1)
    materialised = ProcessImagesPipeline(
        _params(tmp_path, stream_frames=-1, **common), device="cpu"
    ).process_layer(1)
    assert np.array_equal(streamed.buffer, materialised.buffer)
    assert streamed.count_bits() > 0        # and it actually found something


def test_hdf5_defaults_to_streaming_and_tiff_does_not(tmp_path):
    data = _frames()
    _write_h5(tmp_path / "img_000000.h5", data)
    _write_tiffs(tmp_path, data)
    h5_pipe = ProcessImagesPipeline(_params(tmp_path, ext_orig="h5"), device="cpu")
    tif_pipe = ProcessImagesPipeline(_params(tmp_path, ext_orig="tif"), device="cpu")
    assert h5_pipe._should_stream()
    assert not tif_pipe._should_stream()


# ----------------------------------------------------------------------
# streaming median
# ----------------------------------------------------------------------

def test_streaming_median_matches_temporal_median_exactly(tmp_path):
    """Element-identical, not close: torch.median picks the lower middle
    element on an even count where np.median would average, and a median that
    moves by half a count when the code path changes is a silent difference
    between reductions."""
    data = _frames()
    _write_h5(tmp_path / "img_000000.h5", data)
    p = _params(tmp_path, ext_orig="h5")
    stack = load_stack(p, 1, dtype=torch.float32)
    want = temporal_median(stack)
    with open_source(p, 1) as src:
        got = streaming_temporal_median(src, dtype=torch.float32)
    assert torch.equal(got, want)


@pytest.mark.parametrize("row_block", [1, 5, NZ, NZ + 3])
def test_row_block_does_not_change_the_median(tmp_path, row_block):
    data = _frames()
    _write_h5(tmp_path / "img_000000.h5", data)
    p = _params(tmp_path, ext_orig="h5")
    with open_source(p, 1) as src:
        want = streaming_temporal_median(src, dtype=torch.float32)
        got = streaming_temporal_median(src, row_block=row_block, dtype=torch.float32)
    assert torch.equal(got, want)


def test_median_frames_subsamples_evenly(tmp_path):
    """Opt-in subsampling: fewer frames, still spanning the layer."""
    data = _frames()
    _write_h5(tmp_path / "img_000000.h5", data)
    p = _params(tmp_path, ext_orig="h5")
    with open_source(p, 1) as src:
        sub = streaming_temporal_median(src, n_frames=3, dtype=torch.float32)
        full = streaming_temporal_median(src, dtype=torch.float32)
    assert sub.shape == full.shape
    # A subsample is a DIFFERENT estimator, so this asserts only that it ran
    # and stayed in range -- not that it agrees. On real 20-ID data it did
    # agree (identical blob counts; see params.median_frames for the numbers),
    # but on one band of one scan, which is why MedianFrames still defaults
    # to 0 and why this test does not assert agreement.
    assert float(sub.min()) >= 0.0


def test_median_frames_at_or_above_the_count_is_the_full_median(tmp_path):
    data = _frames()
    _write_h5(tmp_path / "img_000000.h5", data)
    p = _params(tmp_path, ext_orig="h5")
    with open_source(p, 1) as src:
        want = streaming_temporal_median(src, dtype=torch.float32)
        got = streaming_temporal_median(src, n_frames=NFRAMES, dtype=torch.float32)
    assert torch.equal(got, want)


# ----------------------------------------------------------------------
# SumFrames
# ----------------------------------------------------------------------

def test_sum_frames_groups_consecutive_raw_frames(tmp_path):
    data = _frames()
    _write_h5(tmp_path / "img_000000.h5", data)
    p = _params(tmp_path, ext_orig="h5", sum_frames=2)
    with open_source(p, 1) as src:
        assert src.n_frames == NFRAMES // 2
        block = src.read_block(0, 2)
    assert np.allclose(block[0], data[0] + data[1])
    assert np.allclose(block[1], data[2] + data[3])


def test_sum_frames_applies_to_row_reads_too(tmp_path):
    """read_rows feeds the median, so it has to sum the same way read_block does."""
    data = _frames()
    _write_h5(tmp_path / "img_000000.h5", data)
    p = _params(tmp_path, ext_orig="h5", sum_frames=2)
    with open_source(p, 1) as src:
        rows = src.read_rows([0, 1], 2, 5)
    assert np.allclose(rows[0], (data[0] + data[1])[2:5])
    assert np.allclose(rows[1], (data[2] + data[3])[2:5])


# ----------------------------------------------------------------------
# PixelScale -- the trap this parameter exists for
# ----------------------------------------------------------------------

def test_pixel_scale_divides_on_read(tmp_path):
    data = _frames(scale=64)
    _write_h5(tmp_path / "img_000000.h5", data)
    p = _params(tmp_path, ext_orig="h5", pixel_scale=64.0)
    with open_source(p, 1) as src:
        block = src.read_block(0, 1)
    assert np.allclose(block[0], data[0] / 64.0)


def test_pixel_scale_defaults_to_one_and_is_never_inferred(tmp_path):
    """x64-encoded data read with the default comes back UNCHANGED.

    The encoding is per SCAN -- the same 20-ID camera serial wrote x64 in one
    campaign and unscaled in the next -- so auto-detecting it would be
    guessing. The pipeline warns and leaves the value alone.
    """
    data = _frames(scale=64)
    _write_h5(tmp_path / "img_000000.h5", data)
    p = _params(tmp_path, ext_orig="h5")
    assert p.pixel_scale == 1.0
    with open_source(p, 1) as src:
        with pytest.warns(RuntimeWarning, match="multiple of 64"):
            block = src.read_block(0, 1)
    assert np.allclose(block[0], data[0])


def test_check_pixel_scale_warns_when_scale_64_meets_unscaled_data():
    unscaled = np.array([[0, 2, 4, 6, 4092]], dtype=np.float32)
    with pytest.warns(RuntimeWarning, match="thresholds the PEDESTAL"):
        check_pixel_scale(unscaled, 64.0)


def test_check_pixel_scale_is_quiet_when_the_setting_matches():
    import warnings

    scaled = (np.array([[0, 64, 128, 65472]], dtype=np.float32))
    unscaled = np.array([[0, 2, 4, 6, 4092]], dtype=np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        check_pixel_scale(scaled, 64.0)
        check_pixel_scale(unscaled, 1.0)


# ----------------------------------------------------------------------
# error paths -- each one names what to do next
# ----------------------------------------------------------------------

def test_missing_file_explains_the_per_distance_indexing(tmp_path):
    p = _params(tmp_path, ext_orig="h5", raw_start_nr=708)
    with pytest.raises(FileNotFoundError, match="RawStartNr"):
        open_source(p, 1)


def test_wrong_data_loc_lists_what_is_there(tmp_path):
    _write_h5(tmp_path / "img_000000.h5", _frames(), loc="entry/data")
    p = _params(tmp_path, ext_orig="h5")
    with pytest.raises(KeyError, match="entry"):
        open_source(p, 1)


def test_custom_data_loc_is_honoured(tmp_path):
    data = _frames()
    _write_h5(tmp_path / "img_000000.h5", data, loc="entry/data")
    p = _params(tmp_path, ext_orig="h5", data_loc="entry/data")
    with open_source(p, 1) as src:
        assert np.allclose(src.read_block(0, 1)[0], data[0])


def test_too_few_frames_points_at_the_omega_range(tmp_path):
    _write_h5(tmp_path / "img_000000.h5", _frames(n=4))
    p = _params(tmp_path, ext_orig="h5", nr_files_per_distance=NFRAMES)
    with pytest.raises(ValueError, match="omega range"):
        open_source(p, 1)


def test_frame_shape_mismatch_is_caught_at_open(tmp_path):
    _write_h5(tmp_path / "img_000000.h5", _frames()[:, :, :NY - 2])
    p = _params(tmp_path, ext_orig="h5")
    with pytest.raises(ValueError, match="shape"):
        open_source(p, 1)


def test_cli_run_reduces_an_hdf5_scan_end_to_end(tmp_path):
    """The route `midas-nf-pipeline run` actually takes.

    stages.run_process_images -> process_images.cli.run -> from_paramfile ->
    process_all. Nothing in the orchestrator needed changing for HDF5, and this
    is what says so.
    """
    from argparse import Namespace

    from midas_nf_preprocess.process_images.cli import run

    for d, start in enumerate((708, 709)):
        _write_h5(tmp_path / f"NF_{start:06d}.h5", _frames(seed=d))
    pf = tmp_path / "params.txt"
    pf.write_text(
        f"DataDirectory {tmp_path}\n"
        f"OutputDirectory {tmp_path}\n"
        "OrigFileName NF\n"
        "extOrig h5\n"
        "DataLoc exchange/data\n"
        "PixelScale 1\n"
        "RawStartNr 708\n"
        f"NrFilesPerDistance {NFRAMES}\n"
        "nDistances 2\n"
        f"NrPixelsY {NY}\n"
        f"NrPixelsZ {NZ}\n"
        "BlanketSubtraction 1\n"
        "DoLoGFilter 0\n"
    )
    rc = run(Namespace(parameter_file=str(pf), n_cpus=1, device="cpu",
                       dtype=None, all_layers=True, layer_nr=1, output=None))
    assert rc == 0
    out = tmp_path / "SpotsInfo.bin"
    assert out.exists() and out.stat().st_size > 0


def test_params_parse_the_new_keys(tmp_path):
    pf = tmp_path / "params.txt"
    pf.write_text(
        "DataDirectory /data\n"
        "OrigFileName NF_Au_cube_0802\n"
        "extOrig h5\n"
        "DataLoc exchange/data\n"
        "PixelScale 64\n"
        "StreamFrames 1\n"
        "MedianFrames 60\n"
        "MedianRowBlock 460\n"
        "RawStartNr 708\n"
        "NrFilesPerDistance 1440\n"
        "nDistances 3\n"
    )
    p = ProcessParams.from_paramfile(pf)
    assert p.ext_orig == "h5"
    assert p.data_loc == "exchange/data"
    assert p.pixel_scale == 64.0
    assert p.stream_frames == 1
    assert p.median_frames == 60
    assert p.median_row_block == 460
    assert layer_file(p, 3).endswith("NF_Au_cube_0802_000710.h5")
