"""midas_zipper — standalone zarr-zip generation for MIDAS workflows.

Pip-installable with no MIDAS source-tree or C-binary dependency. Wraps the
zarr/HDF5 ingest that the FF and PF pipelines need to turn raw detector data
into the ``*.MIDAS.zip`` files the rest of MIDAS consumes.

Public API
----------
- :func:`generate_ff_zip` — programmatic entry to FF zarr-zip generation
  (mirrors the ``ffGenerateZipRefactor`` CLI).
- ``python -m midas_zipper.ff_zip`` / ``midas-ff-zip`` — CLI.
- ``python -m midas_zipper.update_zarr`` / ``midas-update-zarr`` — CLI.
"""

from __future__ import annotations

__version__ = "0.1.2"

__all__ = ["generate_ff_zip", "__version__"]


def generate_ff_zip(
    *,
    result_folder: str,
    param_file: str,
    layer_nr: int = 1,
    data_fn: str = "",
    dark_fn: str = "",
    num_frame_chunks: int = -1,
    preproc_thresh: int = -1,
    num_files_per_scan: int = 1,
    extra_args: list[str] | None = None,
) -> int:
    """Generate a ``*.MIDAS.zip`` by invoking the ported ff_zip ``main()``.

    Runs in-process by populating ``sys.argv`` then calling
    :func:`midas_zipper.ff_zip.main`. Returns 0 on success; raises on error.
    Kept argv-driven so behavior is bit-identical to the legacy
    ``ffGenerateZipRefactor.py`` CLI the pipelines used to shell out to.
    """
    import sys
    from . import ff_zip

    argv = [
        "midas-ff-zip",
        "-resultFolder", str(result_folder),
        "-paramFN", str(param_file),
        "-LayerNr", str(layer_nr),
    ]
    if data_fn:
        argv += ["-dataFN", str(data_fn)]
    if dark_fn:
        argv += ["-darkFN", str(dark_fn)]
    if num_frame_chunks and num_frame_chunks > 0:
        argv += ["-numFrameChunks", str(num_frame_chunks)]
    if preproc_thresh and preproc_thresh > 0:
        argv += ["-preProcThresh", str(preproc_thresh)]
    if num_files_per_scan and num_files_per_scan > 1:
        argv += ["-numFilesPerScan", str(num_files_per_scan)]
    if extra_args:
        argv += [str(a) for a in extra_args]

    saved = sys.argv
    try:
        sys.argv = argv
        ff_zip.main()
    finally:
        sys.argv = saved
    return 0
