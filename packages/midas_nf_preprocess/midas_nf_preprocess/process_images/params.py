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

    # Processing
    blanket_subtraction: int = 0
    # --- NLM denoise of the MEDIAN-CORRECTED residual (before thresholding) ---
    # Distinct from the pipeline's `Denoise` stage, which denoises RAW frames
    # before median subtraction. Denoising the residual instead lets the
    # threshold drop to well under 1 sigma: on nf_Ce_ht525_s2, NLM + threshold 2
    # recovered 5.3x the area in >=30 px blobs that raw + threshold 10 did,
    # with FEWER single-pixel specks.
    nlm_denoise: int = 0
    nlm_h: float = 1.0            # h = nlm_h * sigma_MAD
    nlm_patch_size: int = 5
    nlm_patch_distance: int = 6
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
            ("OrigFileName", "orig_filename", str),
            ("ReducedFileName", "reduced_filename", str),
            ("extOrig", "ext_orig", str),
            ("extReduced", "ext_reduced", str),
            ("BlanketSubtraction", "blanket_subtraction", int),
            ("NLMDenoise", "nlm_denoise", int),
            ("NLMH", "nlm_h", float),
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
