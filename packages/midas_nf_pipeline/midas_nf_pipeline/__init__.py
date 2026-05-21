"""midas-nf-pipeline: pure-Python NF-HEDM workflow driver.

Replaces the legacy ``NF_HEDM/workflows/nf_MIDAS.py`` and
``nf_MIDAS_Multiple_Resolutions.py`` shell-driver scripts with a
package-resident multi-resolution + multi-layer orchestrator built on:

  - :mod:`midas_nf_preprocess`    — hex grid, tomo filter, seed
                                     orientations, diffraction-spot
                                     simulation, image processing.
  - :mod:`midas_nf_fitorientation` — orientation fitting (NM-batched
                                     PyTorch / Triton on CUDA).
  - :mod:`midas_hkls`             — HKL list generation
                                     (``write_nf_hkls_csv``).
  - :mod:`midas_stress.orientation` — all crystal-symmetry / quaternion
                                     / misorientation primitives.

Single-resolution is a degenerate case of multi-resolution with
``NumLoops=0`` (one initial unseeded pass + post-processing).

Public API
----------

  - :func:`parse_mic.parse_mic`             — Python port of ``ParseMic``.
  - :func:`mic2grains.mic_to_grains`        — Python port of ``Mic2GrainsList``.
  - :func:`workflows.run_layer_pipeline`    — single-layer workflow driver.
  - :func:`workflows.run_multi_layer`       — outer multi-layer driver.
  - :class:`state.PipelineH5`               — incremental H5 state tracker.
  - :func:`consolidate.consolidate`         — bundle outputs into a
                                              consolidated H5.
"""

__version__ = "0.1.1"

# Submodules are imported lazily by the orchestrator so the package can be
# installed and tested without all optional deps present (e.g. denoise).
__all__ = ["__version__"]
