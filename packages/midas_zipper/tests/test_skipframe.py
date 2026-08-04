"""SkipFrame is applied by the CONSUMER, not by the zipper. Do not "fix" it.

This test exists because the layered design is easy to misread and "correct"
into a double skip. The contract is:

  * the zarr holds ALL raw frames of the first file (and the full dark stack);
  * `SkipFrame` is recorded in the zarr's analysis parameters;
  * the consumer does the skipping —
      midas_peakfit/params.py:135      nFrames -= skipFrame
      midas_peakfit/orchestrator.py:181-183
                                       read_frame(data_file, frame_nr + skipFrame)
      midas_peakfit/zarr_io.py:301-302 dark_arr = dark_arr[skipFrame:]
  * `OmegaStart` in the parameter file is the omega of the first frame you want
    to USE (post-skip), so the zarr's `scan_parameters/start` — which is the
    omega of zarr frame 0, i.e. RAW frame 0 — is `OmegaStart - skipFrame*step`.
    The consumer then recovers omega(first used) = start + skipFrame*step
    = OmegaStart.

Making the zipper physically drop the frame as well skips it twice: a 1441-frame
sweep becomes 1439 processed frames instead of 1440. Verified on
bt_1id_jul26/Au3_cubes_ff_000008.

`skip_frames` IS applied at zip time to files 2+ (`i > 0`), because a multi-file
sweep repeats its leading frames at each file boundary — that is concatenation
de-duplication, a different thing from the throwaway-frame skip.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest

zarr = pytest.importorskip("zarr")

from midas_zipper import ff_zip


def test_start_omega_is_back_dated_to_raw_frame_zero(tmp_path):
    """`scan_parameters/start` must be OmegaStart - skipFrame*OmegaStep.

    OmegaStart is the first USED frame; `start` describes zarr frame 0, which
    is the raw (unskipped) frame. Flipping this sign shifts every frame's omega
    by one step for the whole scan.
    """
    ome_start, ome_step, skip = 180.0, -0.25, 1

    zRoot = zarr.open_group(str(tmp_path / "ome.zarr"), mode="w")
    z_groups = {
        "sp_pro_meas": zRoot.require_group(
            "measurement/process/scan_parameters"),
        "sp_pro_analysis": zRoot.require_group(
            "analysis/process/analysis_parameters"),
    }
    ff_zip.write_analysis_parameters(
        z_groups,
        {"OmegaStart": ome_start, "OmegaStep": ome_step, "SkipFrame": skip},
    )

    got = float(np.asarray(z_groups["sp_pro_meas"]["start"][...]).flat[0])
    assert got == pytest.approx(ome_start - skip * ome_step)     # 180.25
    # round-trip the consumer's arithmetic: omega of the first USED frame
    assert got + skip * ome_step == pytest.approx(ome_start)     # 180.00


def test_zipper_does_not_skip_frames_of_the_first_file():
    """The first file must contribute ALL its frames to the zarr.

    Guards against re-introducing the double skip.
    """
    src = inspect.getsource(ff_zip.process_hdf5_scan)
    assert "skip_frames if i > 0 else 0" in src, (
        "the zipper is skipping frames of the FIRST file — the consumer "
        "(midas_peakfit) already does that, so this double-skips")
    assert ("total_frames_to_write = frames_per_file + "
            "(frames_per_file - skip_frames) * (num_files - 1)") in src, (
        "first-file frame accounting changed; the zarr must contain every raw "
        "frame of file 1")
