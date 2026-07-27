"""Optional PyTorch dependency.

Every NumPy code path in midas-stress works without PyTorch. torch is needed
only when a caller passes a ``torch.Tensor`` — the differentiable code paths.
So that ``import midas_stress`` and the whole NumPy API keep working in
torch-free environments (e.g. beamline analysis envs where only the
orientation/misorientation math is used), modules import torch from **here**
rather than with a bare ``import torch``.

When torch is installed this is just ``import torch``. When it is not, a
stand-in module object is supplied whose only usable attribute is ``Tensor`` —
a real, never-matched type, so that the pervasive ``isinstance(x, torch.Tensor)``
dispatch checks stay valid and simply return ``False``. Any *actual* torch
operation on the stand-in raises a clear ``ModuleNotFoundError`` rather than an
obscure ``AttributeError`` on ``None``.

Modules that are torch-only (``torch_backend``, ``elastic_inverse_torch``) call
:func:`require_torch` so that using them without torch fails with a helpful
message.
"""

from __future__ import annotations

_MSG = (
    "PyTorch is not installed. The NumPy inputs and code paths of midas-stress "
    "need no torch; install it (pip install torch, or midas-stress[torch]) only "
    "for tensor / differentiable use."
)

try:
    import torch  # noqa: F401  (re-exported)
    HAS_TORCH = True
except ModuleNotFoundError:
    HAS_TORCH = False

    class _TorchMissing:
        """Stand-in for the ``torch`` module when it is not installed."""

        class Tensor:  # a real type that nothing outside torch is an instance of
            def __init__(self, *_a, **_k):
                raise ModuleNotFoundError(_MSG)

        def __getattr__(self, name):  # any real torch.* use -> clear error
            raise ModuleNotFoundError(f"torch.{name} was used, but {_MSG}")

    torch = _TorchMissing()


def require_torch(feature: str = "this feature"):
    """Return the torch module, or raise a clear error if it is not installed."""
    if not HAS_TORCH:
        raise ModuleNotFoundError(f"{feature} requires PyTorch. {_MSG}")
    return torch
