"""End-to-end ProcessImagesCombined pipeline.

Orchestrates: TIFF load -> temporal median -> per-frame (median-subtract +
spatial median + multi-scale peak detection) -> SpotsInfo.bin.

The split between the differentiable path and the discrete spot mask:

    differentiable: filtered, log_response, spot_prob (autograd alive)
    detached:       labels, n_spots, SpotsBitMask    (graph cut)

Both ``process_layer(layer)`` and ``process_all(layers)`` are supported.
``process_layer`` is drop-in for the C executable's per-layer-invocation
pattern; ``process_all`` is the recommended Python API.
"""

from __future__ import annotations

import logging
import warnings

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence, Union

import numpy as np
import torch

from ..device import resolve_device, resolve_dtype, apply_cpu_threads
from .io import from_tensor, load_tiff_stack
from .log_filter import build_log_kernel
from .median import spatial_median, temporal_median
from .params import ProcessParams
from .peaks import PeakFindOutputs, find_peaks
from .spots_io import SpotsBitMask

LOGGER = logging.getLogger(__name__)


@dataclass
class FrameResult:
    """Per-frame output bundle.

    All tensors share device and (for floats) dtype.
    """

    frame_index: int
    layer_nr: int  # 1-indexed, matching the C convention
    filtered: torch.Tensor          # [Z, Y], autograd
    peaks: PeakFindOutputs          # log_response/spot_prob/labels/n_components

    @property
    def labels(self) -> torch.Tensor:
        return self.peaks.labels

    @property
    def n_spots(self) -> int:
        return self.peaks.n_components

    @property
    def spot_prob(self) -> torch.Tensor:
        return self.peaks.spot_prob

    @property
    def log_response(self) -> torch.Tensor:
        return self.peaks.log_response



def _nlm_denoise_residual(
    resid: "torch.Tensor",
    *,
    h_factor: float = 1.0,
    patch_size: int = 5,
    patch_distance: int = 6,
    h_absolute: float | None = None,
    backend: str = "auto",
) -> "torch.Tensor":
    """Non-local-means denoise of a median-corrected residual frame.

    The noise level is estimated with a robust MAD estimator rather than
    skimage's ``estimate_sigma``: the residual is mostly noise with SPARSE
    spots, so the median absolute deviation ignores the spots, and
    ``estimate_sigma`` additionally needs PyWavelets which is not a MIDAS
    dependency.

    Backends
    --------
    ``backend="auto"`` (default) runs the torch implementation
    (:mod:`~midas_nf_preprocess.process_images.denoise`) when the residual is
    already on a non-CPU device, and scikit-image otherwise.  ``"torch"`` and
    ``"skimage"`` force the choice.

    The torch path is what allows the stack to STAY on the device: scikit-image
    is CPU-only, so ``process_layer`` used to relocate the whole layer to the
    host whenever ``NLMDenoise 1`` was set.  Measured 42x per frame on an
    A6000 against one CPU thread (26.9 s -> 0.74 s at 4600x5320).

    The two backends give identical blob counts on real NF residuals but
    diverge inside saturated, densely-lit regions such as the direct beam --
    see :mod:`~midas_nf_preprocess.process_images.denoise` for the measurement.
    Spot finding excludes that region anyway.

    scikit-image's fast NLM releases the GIL (measured ~4x on 4 threads), which
    is why ``process_layer`` can thread the frame loop on the CPU path.
    """
    import numpy as np

    if backend not in ("auto", "torch", "skimage"):
        raise ValueError(f"backend must be auto|torch|skimage, got {backend!r}")
    dev, dt = resid.device, resid.dtype
    use_torch = (backend == "torch") or (backend == "auto" and dev.type != "cpu")

    # sigma_MAD on-device when we are staying on-device -- computing it via a
    # host round-trip would give back exactly the transfer the torch backend
    # exists to avoid.  Always in float32: a fp16 median is too coarse for a
    # scale estimate on near-counting data, where sigma_MAD is already
    # degenerate.
    if use_torch:
        r32 = resid.detach().to(torch.float32)
        sigma = float(1.4826 * torch.median(torch.abs(r32 - torch.median(r32))))
        a = None
    else:
        a = resid.detach().to("cpu", torch.float32).numpy()
        sigma = float(1.4826 * np.median(np.abs(a - np.median(a))))

    if h_absolute is not None and h_absolute > 0:
        # Absolute strength, in counts. Required on photon-starved data where
        # sigma_MAD is degenerate (see below).
        h = float(h_absolute)
        sigma = h
    elif np.isfinite(sigma) and sigma > 0:
        h = h_factor * sigma
    else:
        # sigma_MAD == 0 happens when the median-corrected residual is almost
        # entirely EXACTLY zero -- i.e. photon-starved, near-counting data.
        # Returning the frame undenoised here used to be SILENT, so
        # `NLMDenoise 1` became a no-op that nothing in the output revealed:
        # on 20-ID nfdev_jul26 the residual was 99.73% exact zeros and every
        # "median + NLM" reduction actually ran without the NLM.
        warnings.warn(
            "NLM skipped: sigma_MAD is 0, so h = NLMH * sigma_MAD is 0. The "
            "residual is almost entirely exact zeros (photon-starved data). "
            "Set NLMHAbsolute (in counts) to denoise anyway -- e.g. "
            "NLMHAbsolute 1.0 -- or the reduction silently runs without NLM.",
            RuntimeWarning, stacklevel=2,
        )
        return resid

    if use_torch:
        from .denoise import nl_means_torch
        out_t = nl_means_torch(
            resid.detach(), h=h, sigma=sigma,
            patch_size=patch_size, patch_distance=patch_distance,
            device=dev, dtype=torch.float32,
        )
        return out_t.to(dtype=dt)

    from skimage.restoration import denoise_nl_means
    out = denoise_nl_means(
        a, h=h, sigma=sigma, fast_mode=True,
        patch_size=patch_size, patch_distance=patch_distance,
        channel_axis=None,
    )
    return torch.from_numpy(np.ascontiguousarray(out)).to(device=dev, dtype=dt)

class ProcessImagesPipeline:
    """Orchestrator for the three-phase NF processing pipeline.

    Parameters
    ----------
    params : ProcessParams. Parsed parameter file.
    device : "cpu" | "cuda" | "mps" | torch.device | None. None auto-detects.
    dtype  : torch.dtype | str | None. None picks per-device default.
    n_cpus : optional int. Sets ``torch.set_num_threads`` on CPU only.

    Notes
    -----
    The C executable is invoked once per layer and accumulates into a shared
    ``SpotsInfo.bin`` via mmap. ``process_layer`` reproduces that pattern (also
    accepts an existing ``SpotsBitMask`` to write into). ``process_all``
    constructs the bitmask itself and processes every requested layer.
    """

    def __init__(
        self,
        params: ProcessParams,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[Union[str, torch.dtype]] = None,
        n_cpus: int = 0,
    ):
        self.params = params
        self.device = resolve_device(device)
        self.dtype = resolve_dtype(self.device, dtype)
        self.n_cpus = int(n_cpus or 0)
        apply_cpu_threads(n_cpus, self.device)

        # Pre-build LoG kernels once. Mirrors the C's two-scale pass:
        # primary (LoGMaskRadius, sigma) + fallback (4, 1.0).
        self._log_kernels = self._build_log_kernels()
        self._log_kernel_cache: dict[tuple, list[torch.Tensor]] = {}

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _build_log_kernels(self) -> list[torch.Tensor]:
        if not self.params.do_log_filter:
            return []
        primary = build_log_kernel(
            self.params.log_mask_radius,
            self.params.sigma,
            integer=False,
            device=self.device,
            dtype=self.dtype,
        )
        # C L999-L1003: hardcoded fallback (radius=4, sigma=1.0)
        fallback = build_log_kernel(
            4, 1.0, integer=False, device=self.device, dtype=self.dtype
        )
        return [primary, fallback]

    def _log_kernels_for(self, device: "torch.device") -> list[torch.Tensor]:
        """LoG kernels on ``device``, cached per device.

        The kernels are built once on ``self.device``, but the FRAME does not
        always stay there: ``process_layer`` relocates the stack to the host for
        the scikit-image NLM backend -- which is the DEFAULT backend. That left
        a CPU image meeting CUDA kernels in ``F.conv2d``, so `NLMDenoise 1` plus
        `DoLoGFilter 1` under `--device cuda` raised

            Input type (torch.FloatTensor) and weight type (torch.cuda.FloatTensor)
            should be the same

        for every frame. Following the image is the fix; caching keeps it off
        the per-frame path.
        """
        if device == self.device:
            return self._log_kernels
        key = (device.type, device.index)
        cached = self._log_kernel_cache.get(key)
        if cached is None:
            cached = [k.to(device) for k in self._log_kernels]
            self._log_kernel_cache[key] = cached
        return cached

    # ------------------------------------------------------------------
    # Phase 1: load
    # ------------------------------------------------------------------

    def load_layer(self, layer_nr: int) -> torch.Tensor:
        """Load all frames for one layer into a tensor [N, Z, Y]."""
        return load_tiff_stack(self.params, layer_nr, self.device, self.dtype)

    def from_stack(self, stack: torch.Tensor) -> torch.Tensor:
        """Validate-and-pass-through a user-supplied stack tensor."""
        return from_tensor(
            stack,
            nr_pixels_y=self.params.nr_pixels_y,
            nr_pixels_z=self.params.nr_pixels_z,
        ).to(device=self.device, dtype=self.dtype)

    # ------------------------------------------------------------------
    # Phase 2: temporal median
    # ------------------------------------------------------------------

    def temporal_median(self, stack: torch.Tensor) -> torch.Tensor:
        return temporal_median(stack)

    # ------------------------------------------------------------------
    # Phase 3: per-frame processing
    # ------------------------------------------------------------------

    def _matched_operating_point(self, resid: "torch.Tensor"):
        """``(sigma, threshold)`` for the matched detector, calibrated ONCE.

        Calibration sweeps sigma x threshold and counts false positives on the
        NEGATED residual, which costs ~126 connected-component passes -- far
        too much per frame.  It therefore runs on the FIRST frame that needs it
        and is reused for the whole run.  Set ``MatchedSigma`` and
        ``MatchedThreshold`` explicitly to skip it entirely.
        """
        cached = getattr(self, "_matched_cal", None)
        if cached is not None:
            return cached
        sigma = float(getattr(self.params, "matched_sigma", 0.0))
        thr = float(getattr(self.params, "matched_threshold", 0.0))
        if sigma > 0 and thr > 0:
            self._matched_cal = (sigma, thr)
            return self._matched_cal

        from .detect import calibrate_detector

        cal = calibrate_detector(
            resid.detach().to("cpu", torch.float32).numpy(),
            fp_budget=int(getattr(self.params, "matched_fp_budget", 5)),
            min_px=int(getattr(self.params, "matched_min_px", 4)),
        )
        if not cal.ok:
            warnings.warn(
                "matched-filter calibration found no operating point within the "
                f"false-positive budget; using the least-bad point "
                f"(sigma={cal.sigma:.2f}, threshold={cal.threshold:.3f}, "
                f"{cal.n_false} false positives). Raise MatchedFPBudget, or set "
                "MatchedSigma/MatchedThreshold explicitly.",
                RuntimeWarning, stacklevel=2,
            )
        self._matched_cal = (
            sigma if sigma > 0 else cal.sigma,
            thr if thr > 0 else cal.threshold,
        )
        self._matched_calibration = cal
        return self._matched_cal

    def process_frame(
        self,
        frame_idx: int,
        frame: torch.Tensor,
        median: torch.Tensor,
        layer_nr: int,
    ) -> FrameResult:
        """Run the full per-frame pipeline on a single frame.

        Steps:
          (a) median subtraction + blanket subtraction + clamp at 0
          (b) spatial median of the configured radius
          (c) multi-scale LoG peak finding (or CC over Image2 if do_log_filter=0)
        """
        if frame.shape != median.shape:
            raise ValueError(
                f"frame shape {tuple(frame.shape)} != median shape {tuple(median.shape)}"
            )

        # (a) median subtraction. Match C: clamp negative results at 0.
        # Note: the C does integer subtraction (uint16 - uint16 - int -> int -> 0-clamped uint16).
        # We do float subtraction; clamp keeps the math equivalent for non-negative inputs.
        #
        # NLM, when enabled, goes HERE: on the median-corrected residual, BEFORE
        # the blanket subtraction and the clamp. Denoising after thresholding
        # would be pointless (the noise is already baked into the mask), and
        # denoising the raw frame instead leaves the fixed-pattern background in.
        # Working on the residual is what lets BlanketSubtraction drop to ~0.7
        # sigma instead of ~3 sigma.
        resid = frame - median
        if int(getattr(self.params, "nlm_denoise", 0)) == 1:
            resid = _nlm_denoise_residual(
                resid,
                h_factor=float(getattr(self.params, "nlm_h", 1.0)),
                patch_size=int(getattr(self.params, "nlm_patch_size", 5)),
                patch_distance=int(getattr(self.params, "nlm_patch_distance", 6)),
                h_absolute=float(getattr(self.params, "nlm_h_absolute", 0.0)) or None,
                backend=str(getattr(self.params, "nlm_backend", "auto")),
            )
        # (a-bis) MATCHED-FILTER detection, if selected.
        #
        # This branches BEFORE the blanket subtraction and the clamp, and works
        # on the residual itself: the filter builds the MASK, and `filtered`
        # returned below is the UNTOUCHED residual, so downstream intensities
        # are the original ones.  That separation is the whole point -- NLM
        # gets a lower threshold by rewriting every pixel, this gets one
        # without touching any (process_images/detect.py).
        if str(getattr(self.params, "spot_detect", "log")) == "matched":
            from .detect import detect_labels_torch
            from .peaks import auto_temperature

            sigma, thr = self._matched_operating_point(resid)
            labels, n, score = detect_labels_torch(
                resid, sigma=sigma, threshold=thr,
                min_px=int(getattr(self.params, "matched_min_px", 4)),
            )
            t = self.params.soft_temperature
            T = (auto_temperature(score) if (isinstance(t, str) and t == "auto")
                 else float(t))
            T = float(T if not torch.is_tensor(T) else T.item()) or 1.0
            peaks = PeakFindOutputs(
                log_response=score,
                spot_prob=torch.sigmoid((score - thr) / T),
                labels=labels,
                n_components=n,
                temperature_img=T,
                temperature_log=T,
            )
            return FrameResult(
                frame_index=frame_idx, layer_nr=layer_nr,
                filtered=resid, peaks=peaks,
            )

        # _resolve_threshold sets this per layer; fall back to the absolute for
        # callers that invoke process_frame directly (tests, replay tooling).
        sub = resid - float(getattr(self, "_blanket", None)
                            if getattr(self, "_blanket", None) is not None
                            else self.params.blanket_subtraction)
        img = torch.clamp(sub, min=0)

        # (b) spatial median
        if self.params.mean_filt_radius > 0:
            img = spatial_median(img, radius=self.params.mean_filt_radius)

        # (c) peak finding
        if self.params.do_log_filter and self._log_kernels:
            peaks = find_peaks(
                img,
                self._log_kernels_for(img.device),
                soft_temperature=self.params.soft_temperature,
            )
        else:
            # No-LoG path: label connected components of (img > 0) directly.
            from .peaks import auto_temperature, label_components

            with torch.no_grad():
                mask = img.detach() > 0
                labels, n = label_components(mask, return_n=True)
            t = self.params.soft_temperature
            T_img = auto_temperature(img) if (isinstance(t, str) and t == "auto") else float(t)
            peaks = PeakFindOutputs(
                log_response=torch.zeros_like(img),
                spot_prob=torch.sigmoid(img / T_img),
                labels=labels,
                n_components=n,
                temperature_img=float(T_img if not torch.is_tensor(T_img) else T_img.item()),
                temperature_log=1.0,
            )
        return FrameResult(
            frame_index=frame_idx,
            layer_nr=layer_nr,
            filtered=img,
            peaks=peaks,
        )

    # ------------------------------------------------------------------
    # omega-persistence diagnostic (drives the SumFrames recommendation)
    # ------------------------------------------------------------------

    #: Excess-over-control below which a lag is not counted as persistent.
    #: Calibrated on ONE dataset -- sample A at 0.1 deg/frame, spots ~0.30 deg
    #: FWHM, where SumFrames 3 was measured best out of 14 configurations. It
    #: recovers 3 there (excess +0.30, +0.15, then +0.02) when the threshold is
    #: 7.5 sigma. It is NOT threshold-independent: at 3.5 sigma the same data
    #: gives 5, because the spot's omega tails clear a lower threshold and the
    #: excess then decays smoothly (+0.42, +0.25, +0.16, +0.11) with no cliff
    #: for the floor to find. Hence the log advises a direction, not a value.
    PERSIST_FLOOR = 0.10

    def _accumulate_persistence(self, masks: list) -> None:
        """Fold a run of CONSECUTIVE frame masks into the lag-overlap counters.

        Runs inside the normal frame loop on masks that already exist, so the
        diagnostic costs a few bitwise ANDs rather than a second reduction pass.

        The control pairs each frame with a NON-adjacent one from the same run.
        Without it the statistic is uninterpretable: NF frames share lit regions
        simply because the same rings are lit throughout, which put the control
        at 0.17-0.22 on real data -- far above the ~0.008 that random overlap of
        0.8%-lit frames would give.
        """
        n = len(masks)
        if n < 3:
            return
        for k in range(1, min(self._persist_max_lag, n - 1) + 1):
            inter = base = ctrl = 0
            for i in range(n - k):
                a = masks[i]
                inter += int((a & masks[i + k]).sum())
                base += int(a.sum())
                j = (i + k + n // 2) % n            # deterministic, non-adjacent
                ctrl += int((a & masks[j]).sum())
            self._persist_inter[k] = self._persist_inter.get(k, 0) + inter
            self._persist_base[k] = self._persist_base.get(k, 0) + base
            self._persist_ctrl[k] = self._persist_ctrl.get(k, 0) + ctrl

    def _log_sumframes_recommendation(self, layer_nr: int) -> None:
        """Log how many frames a spot spans, and what SumFrames that implies."""
        if not self._persist_base:
            return
        already = max(1, int(getattr(self.params, "sum_frames", 1) or 1))
        excess = {}
        for k in sorted(self._persist_base):
            b = self._persist_base[k]
            if b:
                excess[k] = (self._persist_inter[k] - self._persist_ctrl[k]) / b
        if not excess:
            return
        n_persist = 0
        for k in sorted(excess):
            if excess[k] > self.PERSIST_FLOOR and k == n_persist + 1:
                n_persist = k
        span = n_persist + 1
        detail = ", ".join(f"lag {k}: {v:+.3f}" for k, v in sorted(excess.items()))
        step = abs(float(getattr(self.params, "omega_step", 0.0)) or 0.0)
        deg = f" = {span * step:.2f} deg at {step:g} deg/frame" if step else ""
        LOGGER.info("layer %d: omega persistence (excess over control) %s",
                    layer_nr, detail)
        if span <= 1:
            LOGGER.info("layer %d: spots span 1 frame; SumFrames 1 is correct "
                        "(summing would add noise and no signal)", layer_nr)
        elif span > already:
            LOGGER.info(
                "layer %d: spots span ~%d frames%s but SumFrames is %d -- "
                "consider summing (2 is the safe step; frames beyond the spot "
                "width add only noise, and summing itself raises sigma ~1.5x "
                "for 3). NOTE the span is measured AT THE CURRENT THRESHOLD and "
                "grows as the threshold drops -- the same data gave 3 frames at "
                "7.5 sigma and 5 at 3.5 sigma -- so treat it as a direction, "
                "not a setting.",
                layer_nr, span, deg, already)
        else:
            LOGGER.info("layer %d: spots span ~%d frames%s; SumFrames %d is "
                        "matched", layer_nr, span, deg, already)

    # ------------------------------------------------------------------
    # Threshold calibration
    # ------------------------------------------------------------------

    def measure_threshold_sigma(self, stack, median, n_probe: int = 3) -> float:
        """sigma_MAD of the residual AS THRESHOLDED, i.e. after any denoising.

        Measured on ``n_probe`` frames spread across the layer and reduced by the
        median, because a single frame is not representative: the same scan gave
        0.268 and 0.282 on two frames.

        This is deliberately the POST-denoise number. NLM changes the noise by an
        order of magnitude (2.97 -> 0.27 counts here), so a sigma measured before
        it would describe a distribution that never meets the threshold.
        """
        idx = np.linspace(0, stack.shape[0] - 1, max(1, n_probe)).astype(int)
        sigmas = []
        for j in idx:
            resid = stack[int(j)] - median
            if int(getattr(self.params, "nlm_denoise", 0)) == 1:
                resid = _nlm_denoise_residual(
                    resid,
                    h_factor=float(getattr(self.params, "nlm_h", 1.0)),
                    patch_size=int(getattr(self.params, "nlm_patch_size", 5)),
                    patch_distance=int(getattr(self.params, "nlm_patch_distance", 6)),
                    h_absolute=float(getattr(self.params, "nlm_h_absolute", 0.0)) or None,
                    backend=str(getattr(self.params, "nlm_backend", "auto")),
                )
            r = resid.detach().to(torch.float32)
            sigmas.append(float(1.4826 * torch.median(torch.abs(r - torch.median(r)))))
        return float(np.median(sigmas))

    def _resolve_threshold(self, stack, median, layer_nr: int) -> float:
        """Effective blanket threshold for this layer, in counts.

        Absolute thresholds are not transferable between reductions: the same
        ``BlanketSubtraction 2`` is 7.5 sigma on an unsummed denoised residual and
        3.6 sigma on a 3-frame sum, and those two recovered 75 and 412 distinct
        orientations from identical data. Nothing in a parameter file reveals
        that, so the sigma is always measured and always logged.
        """
        want_sigma = float(getattr(self.params, "blanket_sigma", 0.0) or 0.0)
        absolute = float(self.params.blanket_subtraction)
        try:
            sigma = self.measure_threshold_sigma(stack, median)
        except Exception as exc:                     # never fail a run over a log line
            warnings.warn(f"threshold sigma not measured ({exc}); using "
                          f"BlanketSubtraction {absolute}", RuntimeWarning, stacklevel=2)
            return absolute

        # 1e-6, not 0: a residual of exact zeros can still produce a denormal
        # sigma, and dividing by it turns a disabled threshold into "0.0 sigma".
        if not np.isfinite(sigma) or sigma < 1e-6:
            # Photon-starved data: the residual is almost all exact zeros, so a
            # relative threshold is meaningless. _nlm_denoise_residual warns
            # about the same condition for h.
            if want_sigma > 0:
                warnings.warn(
                    f"BlanketSigma {want_sigma} ignored on layer {layer_nr}: "
                    f"sigma_MAD is {sigma}, i.e. the residual is essentially all "
                    f"zeros. Falling back to BlanketSubtraction {absolute}.",
                    RuntimeWarning, stacklevel=2)
            return absolute

        if want_sigma > 0:
            thr = want_sigma * sigma
            LOGGER.info("layer %d: BlanketSigma %.2f x sigma_MAD %.4f -> "
                        "BlanketSubtraction %.4f", layer_nr, want_sigma, sigma, thr)
            return thr

        if absolute <= 0:
            # Thresholding deliberately disabled; there is nothing to calibrate.
            LOGGER.info("layer %d: BlanketSubtraction 0 (no threshold), "
                        "sigma_MAD %.4f", layer_nr, sigma)
            return absolute

        in_sigma = absolute / sigma
        LOGGER.info("layer %d: BlanketSubtraction %.4g = %.2f sigma "
                    "(sigma_MAD %.4f, post-denoise)", layer_nr, absolute, in_sigma, sigma)
        if not (2.0 <= in_sigma <= 5.0):
            direction = ("far above the noise -- most real spots will be "
                         "discarded" if in_sigma > 5.0 else
                         "close to the noise -- expect many false detections")
            warnings.warn(
                f"BlanketSubtraction {absolute:g} is {in_sigma:.1f} sigma of this "
                f"layer's post-denoise residual (sigma_MAD {sigma:.4f}), {direction}. "
                f"Reductions on comparable NF data work best near 3.5 sigma; set "
                f"`BlanketSigma 3.5` to get {3.5 * sigma:.3f} automatically.",
                RuntimeWarning, stacklevel=2)
        return absolute

    # ------------------------------------------------------------------
    # Phase 3 + accumulate: per-layer
    # ------------------------------------------------------------------

    def process_layer(
        self,
        layer_nr: int,
        *,
        stack: Optional[torch.Tensor] = None,
        bitmask: Optional[SpotsBitMask] = None,
    ) -> SpotsBitMask:
        """Process all frames in one layer and return the populated SpotsBitMask.

        Parameters
        ----------
        layer_nr : 1-indexed layer number, matching the C ``argv[2]``.
        stack    : optional pre-loaded ``[N, Z, Y]`` tensor. If absent, loaded from disk.
        bitmask  : optional existing ``SpotsBitMask`` to write into. If absent, a
            fresh single-layer mask is allocated.
        """
        if stack is None:
            stack = self.load_layer(layer_nr)
        else:
            stack = self.from_stack(stack)
        median = self.temporal_median(stack)
        self._blanket = self._resolve_threshold(stack, median, layer_nr)
        self._persist_inter, self._persist_base, self._persist_ctrl = {}, {}, {}
        self._persist_max_lag = 6

        if bitmask is None:
            bitmask = SpotsBitMask(
                n_layers=self.params.n_distances,
                nr_files_per_layer=self.params.nr_files_per_distance,
                nr_pixels_y=self.params.nr_pixels_y,
                nr_pixels_z=self.params.nr_pixels_z,
            )

        # 0-indexed layer for the bitmask (matches C ``layer = nLayers - 1`` at L927).
        layer_idx = layer_nr - 1
        n_files = stack.shape[0]

        # Threaded frame loop.
        #
        # The per-frame work (NLM, spatial median, connected components) is the
        # dominant cost of the whole reduction and every frame is independent.
        # It used to be a plain serial loop, and `n_cpus` only ever reached
        # torch.set_num_threads on CPU -- so on device=cuda the cores sat idle.
        #
        # Threads, not processes: skimage's fast NLM and scipy's labeller both
        # release the GIL, so threads scale (measured ~4x on 4 threads) without
        # pickling 16 MB frames between workers.
        #
        # Frames are processed in batches and the bitmask writes are done in the
        # PARENT, in frame order: set_frame_from_labels mutates shared state, and
        # keeping the writes serial avoids needing a lock and keeps the output
        # bit-identical to the serial path regardless of worker scheduling.
        n_workers = max(1, int(getattr(self, "n_cpus", 1) or 1))

        # Threading the frame loop multiplies the PER-FRAME GPU temporaries by
        # the worker count: spatial_median's im2col alone OOM'd a 47 GB card at
        # 64 workers.
        #
        # This USED TO relocate the whole layer to the host whenever NLM was on,
        # because NLM was scikit-image only and therefore forced a host
        # round-trip per frame -- so the CPU was where the work had to happen
        # anyway. `denoise.nl_means_torch` removes that constraint (42x per
        # frame on an A6000), so the stack now STAYS on the device and the
        # frame loop is capped for memory instead, exactly as in the NLM-off
        # case. Set NLMBackend skimage to restore the old behaviour.
        nlm_on = int(getattr(self.params, "nlm_denoise", 0)) == 1
        nlm_backend = str(getattr(self.params, "nlm_backend", "auto"))
        if nlm_on and nlm_backend == "skimage" and stack.is_cuda:
            stack = stack.cpu()
            median = median.cpu()
        elif stack.is_cuda:
            # NLM on the device roughly doubles the per-frame temporaries
            # (~1.1 GB per 4600x5320 frame), so allow fewer concurrent frames.
            n_workers = min(n_workers, 4 if nlm_on else 8)

        if n_workers <= 1:
            buf = []
            for j in range(n_files):
                result = self.process_frame(j, stack[j], median, layer_nr)
                bitmask.set_frame_from_labels(layer_idx, j, result.labels)
                buf.append(result.labels > 0)
                if len(buf) >= 32:
                    self._accumulate_persistence(buf)
                    buf = []
            self._accumulate_persistence(buf)
            self._log_sumframes_recommendation(layer_nr)
            return bitmask

        from concurrent.futures import ThreadPoolExecutor

        batch = max(n_workers, 32)
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            for lo in range(0, n_files, batch):
                hi = min(lo + batch, n_files)
                results = list(ex.map(
                    lambda j: self.process_frame(j, stack[j], median, layer_nr),
                    range(lo, hi),
                ))
                for j, result in zip(range(lo, hi), results):
                    bitmask.set_frame_from_labels(layer_idx, j, result.labels)
                # ex.map preserves order, so a batch IS a consecutive run.
                self._accumulate_persistence([r.labels > 0 for r in results])
        self._log_sumframes_recommendation(layer_nr)
        return bitmask

    # ------------------------------------------------------------------
    # All layers
    # ------------------------------------------------------------------

    def process_all(
        self,
        layers: Optional[Iterable[int]] = None,
        *,
        bitmask: Optional[SpotsBitMask] = None,
    ) -> SpotsBitMask:
        """Process every requested layer into a single ``SpotsBitMask``.

        ``layers`` defaults to ``range(1, n_distances + 1)``.
        """
        if layers is None:
            layers = range(1, self.params.n_distances + 1)
        layers = list(layers)

        if bitmask is None:
            bitmask = SpotsBitMask(
                n_layers=self.params.n_distances,
                nr_files_per_layer=self.params.nr_files_per_distance,
                nr_pixels_y=self.params.nr_pixels_y,
                nr_pixels_z=self.params.nr_pixels_z,
            )
        for layer_nr in layers:
            self.process_layer(layer_nr, bitmask=bitmask)
        return bitmask
