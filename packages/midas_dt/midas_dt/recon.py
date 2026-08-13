"""Reconstructing sinogram stacks, and propagating sigma through it.

Every reconstruction goes through ``midas_tomo`` -- never ``skimage.iradon``,
which is what the legacy scripts used for one of their two branches and is a
different algorithm from the gridrec the other branch used, so the two were
never comparable.

Sigma propagation
-----------------
Filtered back-projection is a **linear** operator: ``recon = A @ sino`` for a
fixed ``A`` that depends only on the geometry. The variance of a linear map is

    var(recon) = (A**2) @ var(sino)          # A**2 elementwise, NOT A @ A

and the trap is that pushing the variance sinogram through the engine computes
``A @ var``, which is a different quantity. It is not even non-negative: the
ramp filter has negative lobes, so ``A @ var`` can come out below zero, and
taking its absolute value to hide that would be manufacturing an error bar
rather than propagating one.

So :func:`reconstruct` estimates variance by **Monte Carlo**: resample the
sinogram from its own sigma, reconstruct, repeat, take the sample variance.
That is correct for any operator, linear or not, and costs ``n_samples``
extra reconstructions -- which is why it is opt-in rather than default.

The same reasoning is why SIRT/TV, when they land, cannot borrow a closed
form: they are iterative and nonlinear, so MC is the only honest route there
too.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from .conventions import RECON_SIGN, ScanKnownLimits, recon_size
from .sinogram import SinogramStack

__all__ = ["Reconstruction", "reconstruct"]

log = logging.getLogger(__name__)


@dataclass
class Reconstruction:
    """Reconstructed volume for one channel, with variance and provenance.

    ``intensity`` is ``(n_bins, X, X)`` -- one image per detector bin, in the
    same bin order as the :class:`~midas_dt.sinogram.SinogramStack` it came
    from, so :meth:`~midas_dt.sinogram.SinogramStack.bin_index` still applies.
    """

    intensity: np.ndarray
    variance: np.ndarray | None
    bin_shape: tuple[int, int]
    channel: object
    limits: ScanKnownLimits
    sign_applied: float = RECON_SIGN

    @property
    def n_bins(self) -> int:
        return int(self.intensity.shape[0])

    @property
    def size(self) -> int:
        return int(self.intensity.shape[-1])

    @property
    def sigma(self) -> np.ndarray | None:
        return None if self.variance is None else np.sqrt(self.variance)

    def voxel_pattern(self, iy: int, ix: int) -> np.ndarray:
        """The (n_eta, n_r) diffraction pattern at one voxel.

        This is what the reconstruct-then-fit branch fits. Reshaping to
        ``bin_shape`` here keeps the eta/r axes straight in one place.
        """
        n_eta, n_r = self.bin_shape
        return self.intensity[:, iy, ix].reshape(n_eta, n_r)

    def voxel_pattern_variance(self, iy: int, ix: int) -> np.ndarray | None:
        if self.variance is None:
            return None
        n_eta, n_r = self.bin_shape
        return self.variance[:, iy, ix].reshape(n_eta, n_r)


def reconstruct(
    stack: SinogramStack,
    *,
    shift: float = 0.0,
    filter_nr: int = 2,
    extra_pad: bool = True,
    variance_samples: int = 0,
    apply_sign: bool = True,
    n_cpus: int = 8,
    backend: str = "auto",
    deterministic: bool = False,
    rng: np.random.Generator | None = None,
) -> Reconstruction:
    """Reconstruct every bin of *stack*.

    Parameters
    ----------
    shift : float
        Rotation-axis offset in pixels.
    filter_nr : int
        0 none, 1 Shepp-Logan, 2 Hann (the 2023 runs' choice), 3 Hamming,
        4 ramp.
    extra_pad : bool
        Pad to 2x the next power of two, matching ``ExtraPadForTomo 1``.
    variance_samples : int
        Monte-Carlo samples for the per-voxel variance. ``0`` (default) skips
        it and leaves ``Reconstruction.variance`` as ``None``. Each sample is
        a full extra reconstruction of every bin, so this is the expensive
        knob; 16-32 is usually enough for a 10-20% estimate of sigma itself.

        There is deliberately no cheap "push the variance through FBP" option:
        that computes ``A @ var`` rather than ``A**2 @ var`` and can go
        negative. See the module docstring.
    apply_sign : bool
        Apply :data:`~midas_dt.conventions.RECON_SIGN`. With ``do_log=False``
        the engine back-projects intensity and returns a negative-going image;
        leaving this off inverts every peak, and a peak finder will still
        return plausible numbers from it.

    Notes
    -----
    ``do_log`` is forced ``False``: these are diffracted-intensity sinograms,
    not transmission, so there is no ``-log`` to take. Passing transmission
    data here would be a different pipeline.
    """
    try:
        from midas_tomo import run_tomo_from_sinos
    except ImportError as exc:
        raise ImportError(
            "reconstruction needs midas-tomo. Install with `pip install midas-dt`."
        ) from exc

    import tempfile
    from pathlib import Path

    n_bins = stack.n_bins
    expected = recon_size(stack.n_translations, extra_pad=extra_pad)
    log.info(
        "reconstructing %d bins of %s -> %dx%d",
        n_bins, stack.channel.label if stack.channel else "?", expected, expected,
    )

    with tempfile.TemporaryDirectory(prefix="midas_dt_recon_") as tmp:
        wd = Path(tmp)
        cube = run_tomo_from_sinos(
            stack.intensity, wd / "int", stack.omega_deg,
            shifts=float(shift), filter_nr=filter_nr, do_log=False,
            extra_pad=extra_pad, n_cpus=n_cpus, backend=backend,
            deterministic=deterministic, do_cleanup=True,
        )
        inten = np.asarray(cube[0])       # single shift

        var = None
        if variance_samples > 0:
            if variance_samples < 2:
                raise ValueError(
                    f"variance_samples must be 0 or >= 2, got {variance_samples}"
                )
            gen = rng if rng is not None else np.random.default_rng(0)
            sigma = np.sqrt(np.clip(stack.variance, 0.0, None))
            # Welford, so K samples cost K reconstructions and O(1) memory
            # rather than holding the whole stack of them.
            mean = np.zeros_like(inten, dtype=np.float64)
            m2 = np.zeros_like(inten, dtype=np.float64)
            for k in range(variance_samples):
                noisy = stack.intensity + sigma * gen.standard_normal(sigma.shape)
                sample = np.asarray(run_tomo_from_sinos(
                    noisy.astype(np.float32), wd / f"mc{k}", stack.omega_deg,
                    shifts=float(shift), filter_nr=filter_nr, do_log=False,
                    extra_pad=extra_pad, n_cpus=n_cpus, backend=backend,
                    deterministic=deterministic, do_cleanup=True,
                )[0], dtype=np.float64)
                delta = sample - mean
                mean += delta / (k + 1)
                m2 += delta * (sample - mean)
            var = m2 / (variance_samples - 1)
            log.info("variance from %d Monte-Carlo samples", variance_samples)

    if apply_sign:
        inten = inten * RECON_SIGN
        # Variance is sign-invariant: (-1)^2 = 1.

    return Reconstruction(
        intensity=inten, variance=var, bin_shape=stack.bin_shape,
        channel=stack.channel, limits=stack.limits,
        sign_applied=RECON_SIGN if apply_sign else 1.0,
    )
