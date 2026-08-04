"""Parameter file parser for ProcessImagesCombined.

Mirrors the field set parsed in NF_HEDM/src/ProcessImagesCombined.c L652-L778
and the backward-compat defaults from L782-L793.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Optional, Union


def _parse_temperature(token: str) -> Union[float, str]:
    """Accept 'auto' or a numeric string for SoftTemperature in the param file."""
    if token.lower() == "auto":
        return "auto"
    return float(token)


@dataclass
class ProcessParams:
    """All parameters consumed by the ProcessImagesCombined pipeline.

    Field names are snake_case versions of the C parameter file keys.
    """

    # I/O
    data_directory: str = "."
    output_directory: str = ""  # if empty, falls back to data_directory in __post_init__
    orig_filename: str = ""
    reduced_filename: str = ""
    ext_orig: str = "tif"
    ext_reduced: str = "bin"

    # Layout
    raw_start_nr: int = 0
    nr_pixels: int = 2048
    nr_pixels_y: int = 0  # 0 means "use nr_pixels"
    nr_pixels_z: int = 0  # 0 means "use nr_pixels_y after pixel resolution"
    wf_images: int = 0
    nr_files_per_distance: int = 0
    n_distances: int = 1
    # --- omega binning: sum SumFrames consecutive RAW frames into one ---------
    # A spot spans a finite omega range. Sampling finer than that range splits
    # one spot's photons across several frames, each carrying the FULL
    # background, so each is harder to detect than the spot really is. Summing N
    # frames that fall inside the spot recovers all of its signal while the
    # background noise grows only as sqrt(N) -- a genuine sqrt(N) SNR gain, as
    # opposed to lowering the threshold, which only admits more noise.
    #
    # Measured on nf_sampleC_htA_s2 (recon/omega_width.py): spot FWHM 0.30 deg at a
    # 0.1 deg step, so SumFrames 3 spans the spot and should give ~1.73x.
    #
    # ONLY summing frames that fall INSIDE the spot helps. Beyond the spot width
    # this adds background to a fixed signal and SNR DROPS as sqrt(N). Measure
    # the width before choosing N.
    #
    # NrFilesPerDistance and OmegaStep in the paramfile describe the POST-SUM
    # scan: for 1800 raw frames at 0.1 deg with SumFrames 3, set
    # NrFilesPerDistance 600 and OmegaStep -0.3. Everything downstream
    # (SpotsInfo sizing, frame indices, omega) then refers to summed frames.
    sum_frames: int = 1

    # Processing
    # float, not int: the arithmetic in process_frame was always float, so the
    # int was purely a parsing restriction -- and a consequential one. It put a
    # FLOOR on sensitivity, because on unsummed NLM-denoised data sigma_MAD is
    # ~0.27 counts, so the smallest legal threshold (1) is already 3.7 sigma and
    # nothing below that could be expressed at all.
    blanket_subtraction: float = 0.0
    # When > 0, the threshold is BlanketSigma * sigma_MAD of the POST-denoise
    # residual, measured per layer, and blanket_subtraction is ignored. This is
    # the transferable knob: the reduction catalog on sample A found every good
    # configuration sitting at ~3.5 sigma regardless of how it got there, while
    # the same absolute "BlanketSubtraction 2" meant 7.5 sigma unsummed (75
    # orientations recovered) and 3.6 sigma summed (412).
    blanket_sigma: float = 0.0
    # --- NLM denoise of the MEDIAN-CORRECTED residual (before thresholding) ---
    # Distinct from the pipeline's `Denoise` stage, which denoises RAW frames
    # before median subtraction. Denoising the residual instead lets the
    # threshold drop to well under 1 sigma: on nf_sampleB_htB_s2, NLM + threshold 2
    # recovered 5.3x the area in >=30 px blobs that raw + threshold 10 did,
    # with FEWER single-pixel specks.
    nlm_denoise: int = 0
    nlm_h: float = 1.0            # h = nlm_h * sigma_MAD
    nlm_patch_size: int = 5
    nlm_patch_distance: int = 6
    # Absolute NLM filter strength, in COUNTS.  Overrides ``nlm_h * sigma_MAD``
    # when > 0.  Needed on photon-starved detectors where the median-corrected
    # residual is almost entirely zero, so sigma_MAD is EXACTLY 0 and the
    # sigma-scaled h collapses -- in which case NLM would otherwise be skipped
    # silently.  0 = derive h from sigma_MAD (the historical behaviour).
    nlm_h_absolute: float = 0.0
    # Which NLM implementation to use: "skimage" (default), "torch", or "auto"
    # (torch when the data is already on a non-CPU device).
    #
    # The torch path is 42x faster per frame on an A6000 (26.9 s -> 0.74 s at
    # 4600x5320) and lets the frame loop STAY on the device -- scikit-image
    # being CPU-only is why the layer used to be relocated to the host whenever
    # NLMDenoise was set.
    #
    # It DEFAULTS TO skimage anyway, because the two are close but NOT
    # bit-equivalent: correlation 0.988-0.9999 depending on regime, blob counts
    # within +-2 of each other on dense synthetic data (exact on real NF
    # frames), diverging most in saturated regions.  Switching the default
    # would silently change existing reductions.  Opt in with NLMBackend torch.
    nlm_backend: str = "skimage"
    # --- spot DETECTION backend -------------------------------------------
    # "log"     : the historical multi-scale LoG path, mirroring the C.
    # "matched" : Gaussian matched filter on the median-corrected residual,
    #             used ONLY to build the mask -- intensities are then read from
    #             the UNTOUCHED residual, which is the point.  Measured 5.8-8.6x
    #             more detected spots than a raw threshold at an equal, measured
    #             false-positive budget, across 1-ID and 20-ID data; see
    #             process_images/detect.py.
    spot_detect: str = "log"
    # 0 = calibrate from the data (recommended, and what makes it generic).
    # The scan converged on sigma 0.70 on five frames spanning two beamlines,
    # two detectors and a 2.7x difference in pixel size.
    matched_sigma: float = 0.0
    matched_threshold: float = 0.0     # 0 = calibrate
    matched_fp_budget: int = 5         # false positives allowed, per frame
    matched_min_px: int = 4            # blobs smaller than this are dropped
    matched_calib_frames: int = 3      # frames used for the one-off calibration
    mean_filt_radius: int = 1  # spatial median radius (0=identity, 1=3x3, 2=5x5)
    do_log_filter: int = 1
    log_mask_radius: int = 4
    sigma: float = 1.0
    write_fin_image: int = 0
    do_deblur: int = 0
    write_legacy_bin: int = 0

    # Soft surrogate (extension over C). ``"auto"`` (default) lets find_peaks
    # pick per-image robust scales for the img and log_response sigmoids.
    # A positive float overrides for both sigmoids.
    soft_temperature: Union[float, str] = "auto"

    def __post_init__(self) -> None:
        # Mirror C L782-L789: NrPixelsY/Z fallback chain.
        if self.nr_pixels_y == 0 and self.nr_pixels_z == 0:
            self.nr_pixels_y = self.nr_pixels
            self.nr_pixels_z = self.nr_pixels
        elif self.nr_pixels_y != 0 and self.nr_pixels_z == 0:
            self.nr_pixels_z = self.nr_pixels_y
        elif self.nr_pixels_y == 0 and self.nr_pixels_z != 0:
            self.nr_pixels_y = self.nr_pixels_z
        # Mirror C L790-L791: deblur forces write_fin_image.
        if self.do_deblur != 0:
            self.write_fin_image = 1
        # Mirror C L792-L793: empty output dir falls back to data dir.
        if not self.output_directory:
            self.output_directory = self.data_directory

    @classmethod
    def from_paramfile(cls, path: Union[str, Path]) -> "ProcessParams":
        """Parse a MIDAS parameter file in the C ProcessImagesCombined format.

        Unknown keys are ignored (same behavior as the C parser).
        """
        # Map: (param-file key, dataclass field, type)
        keys: list[tuple[str, str, type]] = [
            ("RawStartNr", "raw_start_nr", int),
            ("DataDirectory", "data_directory", str),
            ("OutputDirectory", "output_directory", str),
            ("NrPixels", "nr_pixels", int),
            ("NrPixelsY", "nr_pixels_y", int),
            ("NrPixelsZ", "nr_pixels_z", int),
            ("WFImages", "wf_images", int),
            ("NrFilesPerDistance", "nr_files_per_distance", int),
            ("SumFrames", "sum_frames", int),
            ("OrigFileName", "orig_filename", str),
            ("ReducedFileName", "reduced_filename", str),
            ("extOrig", "ext_orig", str),
            ("extReduced", "ext_reduced", str),
            ("BlanketSubtraction", "blanket_subtraction", float),
            ("BlanketSigma", "blanket_sigma", float),
            ("NLMDenoise", "nlm_denoise", int),
            ("NLMH", "nlm_h", float),
            ("NLMHAbsolute", "nlm_h_absolute", float),
            ("NLMBackend", "nlm_backend", str),
            ("SpotDetect", "spot_detect", str),
            ("MatchedSigma", "matched_sigma", float),
            ("MatchedThreshold", "matched_threshold", float),
            ("MatchedFPBudget", "matched_fp_budget", int),
            ("MatchedMinPx", "matched_min_px", int),
            ("MatchedCalibFrames", "matched_calib_frames", int),
            ("NLMPatchSize", "nlm_patch_size", int),
            ("NLMPatchDistance", "nlm_patch_distance", int),
            ("MedFiltRadius", "mean_filt_radius", int),
            ("DoLoGFilter", "do_log_filter", int),
            ("LoGMaskRadius", "log_mask_radius", int),
            ("GaussFiltRadius", "sigma", float),
            ("WriteFinImage", "write_fin_image", int),
            ("Deblur", "do_deblur", int),
            ("nDistances", "n_distances", int),
            ("WriteLegacyBin", "write_legacy_bin", int),
            ("SoftTemperature", "soft_temperature", _parse_temperature),  # extension
        ]

        # Use a placeholder; only populate fields explicitly present in the file.
        # We construct the dataclass at the end so __post_init__ runs once.
        kwargs: dict = {}
        with open(path, "r") as f:
            for line in f:
                line = line.rstrip("\n")
                if not line or line.startswith("#"):
                    continue
                # The C uses prefix matching with a trailing space; we tokenize.
                parts = line.split(None, 1)
                if len(parts) < 2:
                    continue
                key, rest = parts
                value_token = rest.split(None, 1)[0]
                for c_key, py_field, cast in keys:
                    if key == c_key:
                        try:
                            kwargs[py_field] = cast(value_token)
                        except ValueError:
                            pass
                        break
        return cls(**kwargs)

    def with_overrides(self, **kwargs) -> "ProcessParams":
        """Return a copy with the given fields replaced (re-runs __post_init__)."""
        return replace(self, **kwargs)
