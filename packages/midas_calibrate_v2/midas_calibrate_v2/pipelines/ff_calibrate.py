"""Powder standard → FF-HEDM parameter file, in one call.

``calibrate_ff_from_files`` takes a calibrant exposure and the parameter file a
previous reconstruction used, and writes the parameter file the next one should
use: refined geometry and distortion from the powder, everything else
(thresholds, ring numbers, omega scan, lattice) carried over verbatim.

It exists because the individual pieces are each easy to get subtly wrong, and
none of the mistakes raise:

* **Frames, not an image.** A calibrant exposure is a stack. The generic
  ``--image`` loader in this CLI takes an HDF5 file's first top-level key,
  which on a ``.vrx.h5`` is the ``WM`` metadata *group*. Point
  ``data_group``/``dark_group`` at the real datasets instead.
* **The dark is not always in the obvious place.** On the 20-ID Varex the dark
  sits in ``/exchange/bright`` while ``/exchange/dark`` exists and is all
  zeros. Subtracting the zeros leaves a ~1500-count pedestal.
* **Never invent a beam centre.** :func:`~midas_calibrate_v2.calibrate`
  auto-seeds it; passing a guess overrides the seeder and the fit then cannot
  travel far enough to recover.
* **RhoD must be rewritten.** It is the distortion normalisation *and*,
  aliased to ``MaxRingRad``, the cap on hkl generation — and
  ``ff_paramstest_from_auto_result`` does not replace it, so a wrong value in
  the template survives calibration. Too large, the hkl generator emits
  hundreds of rings and overruns a fixed 500-ring array in the c-omp indexer,
  which then indexes zero seeds and reports success.
* **A strain number is not a calibration.** A converged fit can sit on the
  wrong ring assignment, so an overlay is written every time and the result
  carries a pass/fail against the 100 µε gate.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Sequence, Union

import numpy as np

#: No calibrant geometry above this belongs downstream.
STRAIN_GATE_UE = 100.0

DEFAULT_DATA_GROUP = "exchange/data"


@dataclass
class FFCalibrationResult:
    """What :func:`calibrate_ff_from_files` produced."""
    paramstest: Path
    Lsd_um: float
    BC_y_px: float
    BC_z_px: float
    tx_deg: float
    ty_deg: float
    tz_deg: float
    RhoD_um: float
    wavelength_A: float
    strain_uE: float                     # post-residual when available
    in_loop_strain_uE: float
    passes_gate: bool
    overlay_png: Optional[Path] = None
    geometry_json: Optional[Path] = None
    distortion: Dict[str, float] = field(default_factory=dict)

    def summary(self) -> str:
        v = "PASS" if self.passes_gate else "FAIL"
        return (f"Lsd {self.Lsd_um:.4f} um  BC ({self.BC_y_px:.4f}, "
                f"{self.BC_z_px:.4f}) px  tilts ({self.tx_deg:.6f}, "
                f"{self.ty_deg:.6f}, {self.tz_deg:.6f}) deg  "
                f"RhoD {self.RhoD_um:.1f} um  strain {self.strain_uE:.1f} ue [{v}]")


def load_calibrant_frame(
    path: Union[str, Path],
    *,
    data_group: str = DEFAULT_DATA_GROUP,
    dark_group: Optional[str] = None,
    reduce: str = "median",
) -> np.ndarray:
    """Reduce a calibrant exposure to one dark-subtracted 2-D frame.

    ``median`` (default) rejects zingers without smearing the rings. Raises if
    ``data_group`` is missing rather than guessing a dataset, and warns loudly
    if the named dark reduces to all zeros — the signature of pointing at the
    wrong group.
    """
    import h5py

    with h5py.File(str(path), "r") as f:
        if data_group not in f:
            raise KeyError(
                f"{path} has no dataset {data_group!r}. Top-level keys: "
                f"{list(f)}. Pass data_group explicitly — the generic image "
                "loader's 'first key' guess is wrong for these files.")
        data = np.asarray(f[data_group][...], dtype=np.float64)
        dark = None
        if dark_group:
            if dark_group not in f:
                raise KeyError(
                    f"{path} has no dataset {dark_group!r}. Top-level keys: "
                    f"{list(f)}.")
            dark = np.asarray(f[dark_group][...], dtype=np.float64)

    if data.ndim == 2:
        data = data[None]
    fn = np.median if reduce == "median" else np.mean
    img = fn(data, axis=0)
    if dark is not None:
        if dark.ndim == 2:
            dark = dark[None]
        if float(np.abs(dark).max()) == 0.0:
            raise ValueError(
                f"{dark_group!r} in {path} is all zeros, so subtracting it "
                "does nothing and the detector pedestal stays in the image. "
                "On 20-ID Varex the dark is in '/exchange/bright'.")
        img = img - fn(dark, axis=0)
    return np.clip(img, 0.0, None)


def rho_d_for(BC_y: float, BC_z: float, n_y: int, n_z: int, px: float) -> float:
    """RhoD in **micrometres**: beam-centre-to-farthest-corner, times the pitch.

    The same definition the calibration's own forward model uses. Expressing it
    in pixels (a natural-looking mistake — it is a pixel distance) makes the
    distortion polynomial explode; leaving it far too large makes it powerless
    *and* overruns the indexer's ring array.
    """
    return px * math.hypot(max(BC_y, n_y - 1 - BC_y), max(BC_z, n_z - 1 - BC_z))


def write_ring_overlay(image, result, wavelength_A, calibrant, out_png, *,
                       max_frac_of_frame: float = 0.75) -> int:
    """Predicted rings at the refined geometry, over the measured frame."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from ..seed.calibrant import CALIBRANTS

    a = float(CALIBRANTS[calibrant]["a"])
    seen, radii = set(), []
    for h in range(9):
        for k in range(9):
            for l in range(9):
                if h == k == l == 0 or not (h % 2 == k % 2 == l % 2):
                    continue
                s = h * h + k * k + l * l
                if s in seen:
                    continue
                seen.add(s)
                arg = wavelength_A / (2.0 * a / math.sqrt(s))
                if abs(arg) >= 1.0:
                    continue
                r = result.Lsd * math.tan(2.0 * math.asin(arg)) / result.pxY
                if 50 < r < max_frac_of_frame * image.shape[0]:
                    radii.append(r)

    vmax = np.percentile(image[image > 0], 99.5) if (image > 0).any() else 1.0
    fig, ax = plt.subplots(figsize=(11, 11))
    ax.imshow(image, cmap="gray", vmin=0, vmax=vmax, origin="lower")
    th = np.linspace(0, 2 * np.pi, 720)
    for r in radii:
        ax.plot(result.BC_y + r * np.cos(th), result.BC_z + r * np.sin(th),
                lw=0.7, color="red", alpha=0.85)
    ax.plot([result.BC_y], [result.BC_z], "+", color="cyan", ms=14, mew=2)
    ax.set_title(f"{calibrant}: predicted rings at the refined geometry "
                 f"(Lsd={result.Lsd:.0f} um)", fontsize=10)
    ax.set_xlabel("Y (px)")
    ax.set_ylabel("Z (px)")
    fig.tight_layout()
    fig.savefig(str(out_png), dpi=130)
    plt.close(fig)
    return len(radii)


#: Keys a template must supply that nothing can guess — they describe the
#: experiment, not the detector. Each maps to a ``synthesize_template``
#: argument of the same name.
REQUIRED_FROM_SCRATCH = (
    "wavelength_A", "px_um", "n_pixels", "im_trans",
    "lattice", "space_group",
    "raw_folder", "file_stem", "start_file_nr", "ext",
    "omega_start", "omega_step", "ring_thresh",
)

#: **None of these are calibration parameters.** The fit does not read one of
#: them. They are here only because ``--mode ff`` emits a parameter file the
#: *reconstruction* then consumes, and a file missing them will not run — so
#: when there is no template to inherit from, something has to write them.
#:
#: They are therefore assumptions, not results. They are printed when used,
#: annotated in the emitted file, and overridable one by one. If you have a
#: previous reconstruction, pass it as the template instead and none of this
#: applies: its own values carry through untouched.
#:
#: ``Rsample``/``Hbeam`` deserve their own warning. They are a **search bound,
#: not the sample size**. Setting them to the true dimensions plops grains onto
#: the bounding box and manufactures a pile-up of positions at the edges, so
#: the generous values below are deliberate and should not be "corrected"
#: downward to match the specimen.
FF_DEFAULTS: Dict[str, str] = {
    "Rsample": "1000", "Hbeam": "1000", "BeamThickness": "1000",
    "Vsample": "50000000", "GlobalPosition": "0",
    "NumPhases": "1", "PhaseNr": "1",
    "Completeness": "0.4", "MinNrSpots": "2",
    "MarginEta": "500", "MarginOme": "0.2",
    "MarginRadial": "1000", "MarginRadius": "1000",
    "MargABC": "2.5", "MargABG": "2.5",
    "OmeBinSize": "0.1", "EtaBinSize": "0.1",
    "StepSizeOrient": "0.1", "StepSizePos": "100",
    "MinEta": "6.0", "Width": "1500",
    "BoxSize": "-1000000 1000000 -1000000 1000000",
    "UseFriedelPairs": "1", "DiscModel": "0",
    "UpperBoundThreshold": "64000",
    "MinOmeSpotIDsToIndex": "-90", "MaxOmeSpotIDsToIndex": "90",
    "NrFilesPerSweep": "1", "SkipFrame": "0",
    "Padding": "6",
}


def synthesize_template(
    out_path: Union[str, Path],
    *,
    wavelength_A: float,
    px_um: float,
    n_pixels: int,
    im_trans: Sequence[int],
    lattice: Sequence[float],
    space_group: int,
    raw_folder: str,
    file_stem: str,
    start_file_nr: int,
    ext: str,
    omega_start: float,
    omega_step: float,
    ring_thresh: Sequence[Sequence[float]],
    dark_file: Optional[str] = None,
    dark_loc: Optional[str] = None,
    omega_range: Optional[Sequence[float]] = None,
    ring_to_index: Optional[int] = None,
    overrides: Optional[Dict[str, str]] = None,
) -> Path:
    """Write a minimal but complete FF parameter file from first principles.

    For the case where there is no previous reconstruction to borrow from. The
    caller supplies what describes *this experiment*; :data:`FF_DEFAULTS`
    supplies the rest. Geometry is left at placeholders — the calibration
    overwrites all of it.
    """
    if len(lattice) != 6:
        raise ValueError(f"lattice needs 6 values (a b c alpha beta gamma), "
                         f"got {len(lattice)}")
    if not ring_thresh:
        raise ValueError("ring_thresh is required: at least one (ring, "
                         "threshold) pair, e.g. [(1, 75), (2, 75)]")
    rings = [int(r) for r, _ in ring_thresh]
    vals = dict(FF_DEFAULTS)
    vals.update(overrides or {})

    if omega_range is None:
        span = abs(omega_step) * round(360.0 / abs(omega_step))
        omega_range = (-span / 2.0, span / 2.0)

    lines = [
        "# FF-HEDM parameter file written by midas-calibrate-v2 --mode ff.",
        "# Geometry below is a placeholder; the calibration replaces Lsd, BC,",
        "# tilts, distortion and RhoD. Everything else describes the experiment.",
        f"Wavelength {wavelength_A!r}".replace("'", ""),
        f"px {px_um}",
        f"NrPixels {int(n_pixels)}",
        f"NrPixelsY {int(n_pixels)}",
        f"NrPixelsZ {int(n_pixels)}",
        "ImTransOpt " + " ".join(str(int(t)) for t in im_trans) if im_trans
        else "ImTransOpt 0",
        f"SpaceGroup {int(space_group)}",
        "LatticeConstant " + " ".join(str(float(x)) for x in lattice),
        f"RawFolder {raw_folder if raw_folder.endswith('/') else raw_folder + '/'}",
        f"FileStem {file_stem}",
        f"StartFileNrFirstLayer {int(start_file_nr)}",
        f"Ext {ext if ext.startswith('.') else '.' + ext}",
        f"OmegaStart {omega_start}",
        f"OmegaStep {omega_step}",
        f"OmegaRange {omega_range[0]} {omega_range[1]}",
        f"OverAllRingToIndex {int(ring_to_index if ring_to_index else min(rings))}",
        "# ^ the ring indexing seeds from. It sets how many seeds you get, so",
        "# it changes the grain count: on 20-ID ti7al, ring 3 gave 4512 seeds",
        "# and 208 grains where ring 1 gave 1630 and 173. Choose a strong ring.",
        "# placeholder geometry — replaced by the calibration",
        "Lsd 1000000", "BC 0 0", "tx 0", "ty 0", "tz 0", "RhoD 0",
    ]
    if dark_file:
        lines.append(f"Dark {dark_file}")
    if dark_loc:
        lines.append(f"darkLoc {dark_loc}")
    for r, t in ring_thresh:
        lines.append(f"RingThresh {int(r)} {float(t):g}")
    lines += [
        "",
        "# ---------------------------------------------------------------",
        "# INDEXING / REFINEMENT SETTINGS -- NOT calibration results.",
        "# The calibration does not read any of these; they are written here",
        "# only so this file is runnable. Review them against your experiment.",
        "#",
        "# Rsample and Hbeam are a SEARCH BOUND, not the sample size. Setting",
        "# them to the real specimen dimensions pushes grains onto the box",
        "# edges and manufactures a pile-up there. Leave them generous.",
        "# ---------------------------------------------------------------",
    ]
    for k in sorted(vals):
        lines.append(f"{k} {vals[k]}")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")
    print(f"[ff-calib] no template: assumed indexing settings (NOT fitted) — "
          + ", ".join(f"{k}={vals[k]}" for k in
                      ("Rsample", "Hbeam", "Completeness", "MinNrSpots",
                       "StepSizeOrient", "StepSizePos") if k in vals))
    print("[ff-calib] Rsample/Hbeam are a SEARCH BOUND, not the sample size; "
          "review before relying on grain positions.")
    chosen = int(ring_to_index if ring_to_index else min(rings))
    if ring_to_index is None:
        print(f"[ff-calib] OverAllRingToIndex defaulted to {chosen} (the lowest "
              f"ring given, of {rings}). This sets the seed count and hence "
              "the grain count -- pass --ring-to-index to pick a stronger ring.")
    return out_path


def _im_trans_from_template(v1, template_paramstest: Path) -> tuple:
    """``ImTransOpt`` from the template, by whatever route survives.

    ``CalibrationParams`` has no ``ImTransOpt`` attribute — the key lands in
    ``.extra`` if anywhere — so reading it with ``getattr`` returns nothing and
    the calibration silently runs with **no image transform**. On a file whose
    reconstruction uses ``ImTransOpt 2`` that mirrors Z, and the fit converges
    happily onto the mirrored beam centre with a good strain number. Measured:
    BC_z 1411.59 instead of 1467.46 (= 2879 - 1467.46) at 55.6 ue, PASS.

    So: extra, then the raw text, and only an explicit ``ImTransOpt 0`` or a
    genuinely silent file is allowed to mean "no transform".
    """
    extra = getattr(v1, "extra", {}) or {}
    if "ImTransOpt" in extra:
        raw = extra["ImTransOpt"]
        vals = raw if isinstance(raw, (list, tuple)) else str(raw).split()
        return tuple(int(float(x)) for x in vals if int(float(x)) != 0)
    for line in Path(template_paramstest).read_text().splitlines():
        line = line.split("#", 1)[0].strip().rstrip(";").strip()
        t = line.split()
        if t[:1] == ["ImTransOpt"]:
            return tuple(int(float(x)) for x in t[1:] if int(float(x)) != 0)
    return ()


def _check_not_mirrored(res, v1, n_y: int, n_z: int, *, tol_px: float = 5.0):
    """Warn if the refined beam centre landed on the template's mirror.

    A wrong ``ImTransOpt`` does not raise and does not spoil the strain — it
    just flips an axis, and the fit follows. The signature is unmistakable
    though: the refined centre sits near ``N-1 - BC`` instead of near ``BC``.
    """
    msgs = []
    for axis, got, ref, n in (("y", res.BC_y, v1.BC_y, n_y),
                              ("z", res.BC_z, v1.BC_z, n_z)):
        if not ref:
            continue
        mirrored = (n - 1) - float(ref)
        if abs(got - mirrored) < tol_px and abs(got - float(ref)) > tol_px:
            msgs.append(
                f"BC_{axis} refined to {got:.3f}, which is the template's "
                f"{float(ref):.3f} mirrored about the detector centre "
                f"({mirrored:.3f}). That is what a wrong ImTransOpt looks "
                "like, and it does NOT show up in the strain.")
    return msgs


def calibrate_ff_from_files(
    calibrant_file: Union[str, Path],
    template_paramstest: Union[str, Path],
    out_paramstest: Union[str, Path],
    *,
    raw_folder: Optional[str] = None,
    calibrant: str = "CeO2",
    wavelength_A: Optional[float] = None,
    px_um: Optional[float] = None,
    n_pixels_y: Optional[int] = None,
    n_pixels_z: Optional[int] = None,
    data_group: str = DEFAULT_DATA_GROUP,
    dark_group: Optional[str] = None,
    reduce: str = "median",
    im_trans: Sequence[int] = (),
    initial_Lsd_um: float = 1_000_000.0,
    n_iter: int = 4,
    lm_max_iter: int = 200,
    strain_gate_uE: float = STRAIN_GATE_UE,
    overlay: bool = True,
    device: str = "cpu",
    verbose: bool = True,
) -> FFCalibrationResult:
    """Calibrate on a powder standard and write the FF parameter file.

    ``wavelength_A``, ``px_um``, ``n_pixels_*`` and ``im_trans`` default to the
    template's values, so the usual call names only the calibrant exposure, the
    template and the sample folder.

    ``im_trans`` **must** match what the reconstruction uses; a mismatch
    mirrors the geometry relative to the fit and nothing downstream detects it.
    """
    from midas_calibrate.params import CalibrationParams as V1Params
    from ..pipelines.auto import calibrate
    from ..compat.to_v1 import ff_paramstest_from_auto_result

    template_paramstest = Path(template_paramstest)
    out_paramstest = Path(out_paramstest)
    out_paramstest.parent.mkdir(parents=True, exist_ok=True)
    v1 = V1Params.from_file(template_paramstest)

    wavelength_A = float(wavelength_A if wavelength_A is not None else v1.Wavelength)
    px_um = float(px_um if px_um is not None else v1.pxY)
    n_y = int(n_pixels_y if n_pixels_y is not None else (v1.NrPixelsY or 0))
    n_z = int(n_pixels_z if n_pixels_z is not None else (v1.NrPixelsZ or n_y))
    if not (wavelength_A > 0 and px_um > 0 and n_y > 0 and n_z > 0):
        raise ValueError(
            f"need wavelength/px/NrPixels; got lambda={wavelength_A}, "
            f"px={px_um}, NrPixels={n_y}x{n_z}. Supply them explicitly if the "
            "template does not carry them.")
    if not im_trans:
        im_trans = _im_trans_from_template(v1, template_paramstest)

    image = load_calibrant_frame(calibrant_file, data_group=data_group,
                                 dark_group=dark_group, reduce=reduce)
    if verbose:
        print(f"[ff-calib] {calibrant_file}")
        print(f"[ff-calib] image {image.shape}, mean {image.mean():.1f}; "
              f"lambda {wavelength_A} A "
              f"({12.398419843320026 / wavelength_A:.4f} keV); "
              f"im_trans={tuple(im_trans)}; beam centre auto-seeded")

    res = calibrate(
        image, wavelength=wavelength_A, pxY=px_um, calibrant=calibrant,
        im_trans=tuple(im_trans), initial_Lsd=initial_Lsd_um,
        output_dir=str(out_paramstest.parent),
        n_iter=n_iter, lm_max_iter=lm_max_iter,
        refine_tilts=True, refine_distortion=True,
        device=device, verbose=verbose,
    )

    ff_paramstest_from_auto_result(
        res, template_paramstest, out_paramstest,
        raw_folder=raw_folder, n_pixels_y=n_y, n_pixels_z=n_z)

    # RhoD is outside the exporter's replaced-set, so rewrite it here or the
    # template's value survives — the single most damaging thing that can be
    # wrong in this file.
    rho_d = rho_d_for(res.BC_y, res.BC_z, n_y, n_z, px_um)
    kept = [ln for ln in out_paramstest.read_text().splitlines()
            if ln.strip().split()[:1] != ["RhoD"]]
    kept += [f"# RhoD = corner-to-beam-centre in um. Also caps hkl generation",
             f"# (aliased to MaxRingRad): too large overruns the indexer's",
             f"# {500}-ring array and it then indexes nothing, silently.",
             f"RhoD {rho_d:.4f}"]
    out_paramstest.write_text("\n".join(kept) + "\n")

    mirror_warnings = _check_not_mirrored(res, v1, n_y, n_z)
    for m in mirror_warnings:
        print(f"[ff-calib] *** {m} ***")

    strain = float(res.post_residual_strain_uE
                   if res.post_residual_strain_uE is not None
                   else res.in_loop_strain_uE)
    png = None
    if overlay:
        png = out_paramstest.parent / f"ring_overlay_{calibrant}.png"
        n = write_ring_overlay(image, res, wavelength_A, calibrant, png)
        if verbose:
            print(f"[ff-calib] overlay -> {png} ({n} rings). Look at it: a "
                  "converged fit can still sit on the wrong ring assignment.")

    out = FFCalibrationResult(
        paramstest=out_paramstest,
        Lsd_um=float(res.Lsd), BC_y_px=float(res.BC_y), BC_z_px=float(res.BC_z),
        tx_deg=float(res.tx), ty_deg=float(res.ty), tz_deg=float(res.tz),
        RhoD_um=rho_d, wavelength_A=wavelength_A,
        strain_uE=strain, in_loop_strain_uE=float(res.in_loop_strain_uE),
        passes_gate=(strain < strain_gate_uE) and not mirror_warnings,
        overlay_png=png, distortion=dict(res.distortion or {}),
    )
    gj = out_paramstest.parent / "geometry.json"
    gj.write_text(json.dumps({
        "source": str(calibrant_file), "template": str(template_paramstest),
        "paramstest": str(out_paramstest), "calibrant": calibrant,
        "Lsd_um": out.Lsd_um, "BC_y_px": out.BC_y_px, "BC_z_px": out.BC_z_px,
        "tx_deg": out.tx_deg, "ty_deg": out.ty_deg, "tz_deg": out.tz_deg,
        "RhoD_um": out.RhoD_um, "wavelength_A": out.wavelength_A,
        "in_loop_strain_uE": out.in_loop_strain_uE,
        "strain_uE": out.strain_uE, "passes_gate": out.passes_gate,
        "distortion": out.distortion,
    }, indent=2))
    out.geometry_json = gj

    if verbose:
        print(f"[ff-calib] {out.summary()}")
        print(f"[ff-calib] wrote {out_paramstest}")
        if not out.passes_gate:
            print(f"[ff-calib] *** {strain:.1f} ue exceeds the "
                  f"{strain_gate_uE:.0f} ue gate — do NOT use this geometry. "
                  "Check px, wavelength, ImTransOpt and the lattice constant, "
                  "and look at the overlay. ***")
    return out
