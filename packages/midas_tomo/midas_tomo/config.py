"""``TomoConfig`` — the MIDAS_TOMO parameter file as a dataclass.

One field per keyword parsed by ``setGlobalOpts`` in ``c_src/tomo_utils.c``,
with the defaults taken from that function's initialisers rather than from
documentation, so the Python and C views cannot drift apart silently.

The legacy driver scripts each wrote this file with their own pile of
``f.write(f'...')`` lines, which is how ``run_tomo`` and ``run_tomo_from_sinos``
ended up with subtly different parameter sets for the same engine. Everything
that writes a parameter file now goes through :meth:`TomoConfig.to_param_file`.

Note on paths: the binary does **not** resolve relative paths against its own
working directory in every code path, so :meth:`to_param_file` writes absolute
paths. Passing relative ones is a silent-wrong-answer trap, not an error.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from math import ceil, log2
from pathlib import Path
from typing import Iterable, Sequence

__all__ = ["TomoConfig", "FILTERS", "next_power_of_2", "parse_shift_arg"]

#: ``filter`` keyword values, as documented in ``tomo_init.c``'s usage text.
FILTERS = {
    0: "none",
    1: "shepp-logan",
    2: "hann",
    3: "hamming",
    4: "ramp",
}


def next_power_of_2(n: int) -> int:
    """Smallest power of two >= *n* (and >= 1)."""
    if n <= 1:
        return 1
    return 1 << int(ceil(log2(n)))


def parse_shift_arg(shifts) -> tuple[float, float, float, int]:
    """Normalise a shift specification to ``(start, end, step, n_shifts)``.

    Accepts a scalar (single shift) or a ``[start, end, step]`` sequence. The
    count matches the C engine's own arithmetic in ``tomo_init.c``:
    ``round(|end - start| / step) + 1``.
    """
    if isinstance(shifts, (int, float)):
        return float(shifts), float(shifts), 1.0, 1
    seq = list(shifts)
    if len(seq) != 3:
        raise ValueError(
            f"shifts must be a scalar or [start, end, step]; got {shifts!r}"
        )
    start, end, step = (float(v) for v in seq)
    if step == 0:
        raise ValueError("shift step must be non-zero")
    n = round(abs(end - start) / abs(step)) + 1
    return start, end, step, int(n)


@dataclass
class TomoConfig:
    """Every keyword ``setGlobalOpts`` understands, plus output-shape helpers.

    Defaults mirror the C initialisers at the top of ``setGlobalOpts``
    (``tomo_utils.c``): ``doLog=1``, ``auto_centering=1``, ``stripeSnr=3.0``,
    ``stripeLaSize=61``, ``stripeSmSize=21``, ``saveReconSeparate=1``.
    Deviations from those are noted per field.
    """

    # ---- required I/O
    data_file: str | os.PathLike = ""
    recon_file: str | os.PathLike = ""
    are_sinos: bool = False
    det_xdim: int = 0
    det_ydim: int = 0

    # ---- angles: exactly one of theta_file / theta_range
    theta_file: str | os.PathLike | None = None
    theta_range: tuple[float, float, float] | None = None  # start, end, step

    # ---- reconstruction
    filter_nr: int = 2                 # Hann; C default is 0, but every caller passes 2
    shift_values: tuple[float, float, float] = (0.0, 0.0, 1.0)
    do_log: bool = True                # C: doLogProj = 1
    extra_pad: bool = False            # C: powerIncrement = 0
    auto_centering: bool = True        # C: auto_centering = 1
    slices_to_process: str | os.PathLike | int = -1   # -1 = all, or a file path
    save_recon_separate: bool = False  # C default is 1; the Python API wants one cube

    # ---- artefact removal
    ring_removal_coeff: float | None = None   # None => omit => C leaves it disabled
    do_stripe_removal: bool = False
    stripe_snr: float = 3.0
    stripe_la_size: int = 61
    stripe_sm_size: int = 21
    stripe_config_file: str | os.PathLike | None = None  # sweep mode

    # ---- HDF5 input (optional path through readRawHDF5)
    hdf5_file: str | os.PathLike | None = None
    image_dataset: str | None = None
    dark_dataset: str | None = None

    # ---- misc
    debug: bool = False

    #: Plan FFTs with FFTW_ESTIMATE instead of FFTW_MEASURE. Deterministic
    #: across runs and machines, at some speed cost, and with no wisdom file
    #: written to the cwd. NOT part of the parameter file -- passed on the
    #: command line, and only honoured by binaries that advertise the
    #: capability (see backend_c.supports_deterministic).
    deterministic: bool = False

    # ------------------------------------------------------------------ checks
    def validate(self) -> list[str]:
        """Return a list of problems; empty means the config is usable.

        Kept separate from :meth:`to_param_file` so callers can inspect
        problems without side effects.
        """
        problems: list[str] = []

        if not str(self.data_file):
            problems.append("data_file is required")
        if not str(self.recon_file):
            problems.append("recon_file is required")
        if self.det_xdim <= 0:
            problems.append(f"det_xdim must be positive, got {self.det_xdim}")
        if self.det_ydim <= 0:
            problems.append(f"det_ydim must be positive, got {self.det_ydim}")

        if (self.theta_file is None) == (self.theta_range is None):
            problems.append(
                "exactly one of theta_file / theta_range must be set "
                f"(got theta_file={self.theta_file!r}, theta_range={self.theta_range!r})"
            )

        if self.filter_nr not in FILTERS:
            problems.append(
                f"filter_nr must be one of {sorted(FILTERS)} "
                f"({', '.join(f'{k}={v}' for k, v in FILTERS.items())}), "
                f"got {self.filter_nr}"
            )

        start, end, step = self.shift_values
        if start != end and step == 0:
            problems.append("shift step must be non-zero when start != end")
        else:
            # The engine's usage text says "ENSURE TO GIVE A RANGE WITH EVEN
            # NUMBER OF SHIFTS" and its inner loop reconstructs shift pairs;
            # an odd count makes it exit non-zero with no useful message.
            # One shift is the exception -- that path is special-cased in the C.
            n = self.n_shifts
            if n > 1 and n % 2:
                problems.append(
                    f"shift_values {self.shift_values} gives {n} shifts; the "
                    f"engine reconstructs shifts in pairs and requires an even "
                    f"count (or exactly 1). Adjust the range or step - e.g. "
                    f"end={end + step:g} would give {n + 1}."
                )

        # The Vo median filters index a window of the stated size; even sizes
        # give an off-centre window, which the C does not guard against.
        if self.do_stripe_removal or self.stripe_config_file:
            for name, val in (
                ("stripe_la_size", self.stripe_la_size),
                ("stripe_sm_size", self.stripe_sm_size),
            ):
                if val % 2 == 0:
                    problems.append(f"{name} must be odd, got {val}")
                if val <= 0:
                    problems.append(f"{name} must be positive, got {val}")

        if self.hdf5_file is not None and self.image_dataset is None:
            problems.append("hdf5_file requires image_dataset")

        return problems

    def check(self) -> None:
        """Raise ``ValueError`` if :meth:`validate` found anything."""
        problems = self.validate()
        if problems:
            raise ValueError(
                "invalid TomoConfig:\n  - " + "\n  - ".join(problems)
            )

    # ------------------------------------------------------------- derived
    @property
    def n_shifts(self) -> int:
        """Number of rotation-axis shifts the engine will reconstruct."""
        return parse_shift_arg(list(self.shift_values))[3]

    @property
    def recon_xdim(self) -> int:
        """Width of each reconstructed slice.

        The engine pads to the next power of two, doubled again when
        ``extra_pad`` is set.
        """
        x = next_power_of_2(self.det_xdim)
        return x * 2 if self.extra_pad else x

    def output_path(self, n_slices: int, *, n_cleanup: int | None = None) -> Path:
        """Full path of the reconstruction cube the engine will write.

        The filename encodes the cube shape; ``n_cleanup`` is present only in
        the ``stripe_config_file`` sweep mode.
        """
        x = self.recon_xdim
        stem = str(self.recon_file)
        prefix = "" if n_cleanup is None else f"_NrCleanup_{n_cleanup:03d}"
        return Path(
            f"{stem}{prefix}"
            f"_NrShifts_{self.n_shifts:03d}"
            f"_NrSlices_{n_slices:05d}"
            f"_XDim_{x:06d}"
            f"_YDim_{x:06d}_float32.bin"
        )

    def wisdom_paths(self, workdir: str | os.PathLike) -> list[Path]:
        """Wisdom files the engine may drop in *workdir*.

        ``FFTW_MEASURE`` plans are cached to ``fftwf_wisdom_{1,2}d_<N>.txt`` in
        the process's working directory. The sizes are twice ``recon_xdim``.
        """
        n = 2 * self.recon_xdim
        return [
            Path(workdir) / f"fftwf_wisdom_1d_{n}.txt",
            Path(workdir) / f"fftwf_wisdom_2d_{n}.txt",
        ]

    # ------------------------------------------------------------- emission
    def to_lines(self) -> list[str]:
        """The parameter file as a list of lines (no trailing newlines).

        Only keywords the engine needs are emitted: optional ones are omitted
        rather than written with a sentinel, because ``setGlobalOpts`` treats
        the *presence* of some keys (notably ``ringRemovalCoeff``) as the
        enable switch.
        """
        def abspath(p) -> str:
            return str(Path(p).expanduser().resolve())

        lines = [
            f"saveReconSeparate {int(self.save_recon_separate)}",
            f"dataFileName {abspath(self.data_file)}",
            # Not resolved as a whole: recon_file is a *stem*, and its parent
            # must exist. resolve() on a non-existent leaf is fine.
            f"reconFileName {abspath(self.recon_file)}",
            f"areSinos {int(self.are_sinos)}",
            f"detXdim {self.det_xdim}",
            f"detYdim {self.det_ydim}",
            f"filter {self.filter_nr}",
            "shiftValues {:f} {:f} {:f}".format(*self.shift_values),
            f"doLog {int(self.do_log)}",
            f"ExtraPad {int(self.extra_pad)}",
            f"AutoCentering {int(self.auto_centering)}",
        ]

        if self.theta_file is not None:
            lines.append(f"thetaFileName {abspath(self.theta_file)}")
        else:
            lines.append("thetaRange {:f} {:f} {:f}".format(*self.theta_range))

        if isinstance(self.slices_to_process, (str, os.PathLike)):
            lines.append(f"slicesToProcess {abspath(self.slices_to_process)}")
        else:
            lines.append(f"slicesToProcess {self.slices_to_process}")

        # Presence-is-enable: writing `ringRemovalCoeff 0` would still turn
        # ring removal ON in the C (use_ring_removal = 1 on any match).
        if self.ring_removal_coeff:
            lines.append(f"ringRemovalCoeff {self.ring_removal_coeff}")

        if self.stripe_config_file is not None:
            lines.append("doStripeRemoval 1")
            lines.append(f"stripeConfigFile {abspath(self.stripe_config_file)}")
        elif self.do_stripe_removal:
            lines.append("doStripeRemoval 1")
            lines.append(f"stripeSnr {self.stripe_snr}")
            lines.append(f"stripeLaSize {self.stripe_la_size}")
            lines.append(f"stripeSmSize {self.stripe_sm_size}")

        if self.hdf5_file is not None:
            lines.append(f"HDF5FileName {abspath(self.hdf5_file)}")
            lines.append(f"ImageDatasetName {self.image_dataset}")
            if self.dark_dataset is not None:
                lines.append(f"DarkDatasetName {self.dark_dataset}")

        if self.debug:
            lines.append("debug 1")

        return lines

    def to_param_file(self, path: str | os.PathLike) -> Path:
        """Validate, then write the parameter file to *path*."""
        self.check()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(self.to_lines()) + "\n")
        return path

    @classmethod
    def from_param_file(cls, path: str | os.PathLike) -> "TomoConfig":
        """Parse a parameter file back into a config.

        Round-trips :meth:`to_param_file`. Handy for reading the 2023 DT
        ``tomo_config.txt`` files. Unknown keywords are ignored, matching the
        C parser, which also skips what it does not recognise.
        """
        cfg = cls()
        seen_range = False
        for raw in Path(path).read_text().splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            key, vals = parts[0], parts[1:]
            if not vals:
                continue
            if key == "dataFileName":
                cfg.data_file = vals[0]
            elif key == "reconFileName":
                cfg.recon_file = vals[0]
            elif key == "areSinos":
                cfg.are_sinos = bool(int(vals[0]))
            elif key == "detXdim":
                cfg.det_xdim = int(vals[0])
            elif key == "detYdim":
                cfg.det_ydim = int(vals[0])
            elif key == "filter":
                cfg.filter_nr = int(vals[0])
            elif key == "thetaFileName":
                cfg.theta_file = vals[0]
            elif key == "thetaRange":
                cfg.theta_range = (float(vals[0]), float(vals[1]), float(vals[2]))
                seen_range = True
            elif key == "shiftValues":
                cfg.shift_values = (float(vals[0]), float(vals[1]), float(vals[2]))
            elif key == "doLog":
                cfg.do_log = bool(int(vals[0]))
            elif key == "ExtraPad":
                cfg.extra_pad = bool(int(vals[0]))
            elif key == "AutoCentering":
                cfg.auto_centering = bool(int(vals[0]))
            elif key == "saveReconSeparate":
                cfg.save_recon_separate = bool(int(vals[0]))
            elif key == "slicesToProcess":
                try:
                    cfg.slices_to_process = int(vals[0])
                except ValueError:
                    cfg.slices_to_process = vals[0]
            # The C matches this with strncmp("ringRemovalCoeff"), so the
            # longer legacy spelling ringRemovalCoefficient matches too.
            elif key.startswith("ringRemovalCoeff"):
                cfg.ring_removal_coeff = float(vals[0])
            elif key == "doStripeRemoval":
                cfg.do_stripe_removal = bool(int(vals[0]))
            elif key == "stripeSnr":
                cfg.stripe_snr = float(vals[0])
            elif key == "stripeLaSize":
                cfg.stripe_la_size = int(vals[0])
            elif key == "stripeSmSize":
                cfg.stripe_sm_size = int(vals[0])
            elif key == "stripeConfigFile":
                cfg.stripe_config_file = vals[0]
            elif key == "HDF5FileName":
                cfg.hdf5_file = vals[0]
            elif key == "ImageDatasetName":
                cfg.image_dataset = vals[0]
            elif key == "DarkDatasetName":
                cfg.dark_dataset = vals[0]
            elif key == "debug":
                cfg.debug = bool(int(vals[0]))
        if seen_range:
            cfg.theta_file = None
        return cfg
