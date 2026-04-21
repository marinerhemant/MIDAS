"""midas-integrate-gpu: CUDA streaming backend for midas-integrate.

Installing this wheel alongside ``midas-integrate`` enables the GPU path
on ``Integrator(..., backend="gpu")`` and ``Server(..., backend="gpu")``:
the ``MIDASIntegratorGPU`` CUDA binary ships inside this wheel and is
discovered by ``midas_integrate._binaries.midas_bin`` via its
cross-package lookup chain.

Runtime requirements (user must satisfy):
    - NVIDIA GPU (compute capability 7.5+).
    - NVIDIA driver ≥ 525 (CUDA 12 forward-compat).
    - Linux x86_64. macOS and Windows aren't supported.

Installation:
    pip install 'midas-integrate[gpu]'

See ``midas_integrate_gpu.check_environment()`` to verify the runtime
prerequisites before invoking GPU integration.
"""

__version__ = "0.1.0"

from ._runtime import check_environment, is_gpu_available

__all__ = [
    "__version__",
    "check_environment",
    "is_gpu_available",
]
