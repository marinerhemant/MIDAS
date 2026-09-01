"""Global powder-intensity correction (Phase C2/C3).

Per-detector ``calc_radius`` (Phase C1) derives each spot's ``GrainVolume``
from that ring's **observed** powder sum::

    powder_int(ring) = Σ IntegratedIntensity(observed spots on ring) / n_frames
    GrainVolume      ∝ I_spot / powder_int(ring)

When a ring is not fully sampled in η, that denominator is short by the
missing fraction while the numerator — one spot — is not, so every volume
on that ring is inflated by ``360° / coverage``. Two geometries do this:

* **multi-panel** layouts (pinwheel: each panel sees ~75° of η), and
* **a single panel** whose outer rings run off the detector edge. On a
  2048² GE at 1-ID the ninth ring is still complete at 926 px while the
  twentieth retains 5.2° of azimuth — a 69× volume error, 4.1× in radius.

This stage removes that bias. It runs **after** ``cross_det_merge`` and
**before** ``binning``.

Method
------
For each ring it builds a per-η-bin intensity model ``Î(ring, η)`` from the
observed spots, with multi-distance averaging where panels overlap, then
fills the η bins nobody covered by circular interpolation between the
nearest covered neighbours. The correction is the ratio of what was seen to
what the full ring would have given::

    scale(ring) = Σ_covered Î  /  Σ_all-360°  Î_filled        (≤ 1)
    GrainRadius *= scale ** (1/3)          (** 1/2 when DiscModel == 1)

Why interpolate ``Î`` rather than scale by ``coverage/360``
-----------------------------------------------------------
η is not a coordinate the ω sweep explores freely. ``G_z`` is invariant
under rotation about the ω axis, and at the diffraction condition
``G_z/|G| = cos θ · cos η``, so a reflection's η is fixed by the grain's
orientation relative to the rotation axis. For randomly oriented grains
``u = cos θ · cos η`` is uniform, which makes the **spot count** density
along a ring go as ``|sin η|`` — sparse near the rotation axis, dense at the
sides. Taken alone that would make ``coverage/360`` wrong, because the arcs a
truncated ring keeps are not a fair sample of the ring.

It is rescued by the rotation-method Lorentz factor: a reflection near η = 0
crosses the Ewald sphere slowly and is therefore integrated over many more ω
frames, so its ``IntegratedIntensity`` is enhanced by ``~1/|sin η|``. The two
cancel, and the *summed intensity* per unit η comes out flat.

Measured, not assumed — 1-ID GE5, LSHR, 258 165 spots over five
full-coverage rings (2026-09-01)::

    ring   count/bin CV   corr(|sin η|)    ΣI/bin CV   corr(|sin η|)
      1        0.440          +0.980          0.178       +0.623
      2        0.303          +0.897          0.214       -0.659
      3        0.385          +0.995          0.099       +0.412
      4        0.380          +0.995          0.087       -0.219
      5        0.415          +0.968          0.100       +0.184

The count density tracks ``|sin η|`` almost exactly; ``ΣI`` does not, and the
sign of its residual correlation is inconsistent between rings, i.e. noise
and texture rather than a geometric trend. Over the same bins ``NImgs`` runs
1.4 → 9.8 and ``DeltaOmega`` 0.35 → 2.45, which is the Lorentz enhancement
doing the cancelling.

So ``Î(η)`` is smooth and near-flat, interpolation across a gap is
well-conditioned, and on an untextured ring the correction reduces to exactly
``coverage/360``. Interpolating rather than hard-coding that factor is what
keeps the estimator usable when the sample *is* textured. Note the
cancellation is a property of a full ω sweep with per-spot ω integration; the
flatness of the observed profile is therefore re-measured at run time on this
layer's full-coverage rings and reported (``ProfileCV`` below), so a sample
that violates it is flagged rather than silently mis-corrected.

**Scaling, not recomputing.** An earlier version rebuilt ``GrainVolume``
from ``Vsample · I_spot / ΣI``, which silently dropped the ΔΘ, cos θ and
hkl-multiplicity factors that ``calc_radius`` applies per spot — so
enabling it would have rewritten every radius with a cruder formula rather
than merely correcting for coverage. Applying a per-ring scale on top of
``calc_radius``'s value keeps that physics intact and makes the stage an
**exact no-op at full coverage** (``scale`` is then 1.0 to the bit, and the
GrainRadius token is left untouched), which is what keeps it safe to enable
for every existing single-panel user.

Coverage source
---------------
``EtaCoverage_DetN`` rows in ``paramstest.txt`` when present. They are
emitted by ``midas_ff_pipeline``'s transforms stage but **not** by this
package's, so in practice they are usually absent and the coverage is
derived here from the panel geometry instead (``LsdFit``, ``YBCFit``,
``ZBCFit``, ``txFit``/``tyFit``/``tzFit``, ``NrPixelsY/Z``, ``px``, and the
``RingNumbers``/``RingRadii`` tables) via the same pixel-enumeration used to
write those rows. This stage deliberately does not write the rows back:
``midas-index`` and ``midas-fit-grain`` also consume them, and acquiring
them would change those stages' behaviour as a side effect of a radius fix.

Outputs
-------
``layer_dir/InputAll.csv``    ``GrainRadius`` column rescaled in place.
``layer_dir/PowderModel.csv`` one row per (ring, η-bin) — the model and its
                              interpolated fill, for diagnostics.
``layer_dir/PowderCoverage.csv`` one row per ring — coverage and the applied
                              scale, so a downstream consumer can filter out
                              rings whose correction rests on little data.
"""
from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

from ._base import StageContext
from .._logging import LOG, stage_timer
from ..eta_coverage import (
    CoverageArc,
    compute_panel_eta_coverage,
    parse_coverage_blocks,
    total_coverage_per_ring,
)
from ..results import StageResult


# -----------------------------------------------------------------------------
#  Bin grid
# -----------------------------------------------------------------------------

ETA_BIN_SIZE_DEG = 1.0   # 1° bins span [-180, 180); 360 bins per ring
N_ETA_BINS = int(round(360.0 / ETA_BIN_SIZE_DEG))

#: Below this covered fraction the extrapolation rests on so few η bins that
#: the corrected radius should be treated as indicative only. Rings under it
#: are still corrected (the arithmetic is right; it is the statistics that are
#: thin) and are reported in ``PowderCoverage.csv`` so they can be filtered.
LOW_COVERAGE_WARN_FRAC = 0.5

#: Ring-band half-width (μm) when ``paramstest.txt`` carries no ``Width``.
#: Matches the MIDAS ``Width`` default; coverage is only weakly sensitive to it.
DEFAULT_WIDTH_UM = 1500.0

#: Coarse bins used to judge how flat ``Î(η)`` is. 1° bins are too sparse to
#: measure a profile against; 10° bins held ~7000 spots per bin in the
#: reference measurement.
PROFILE_N_BINS = 36

#: Coefficient of variation of ``Î(η)`` on a *full-coverage* ring above which
#: the profile is too structured for gap interpolation to be trusted. The
#: reference measurement gave 0.087–0.214 (see module docstring), so this sits
#: comfortably above an untextured sample and below a strongly textured one.
PROFILE_CV_WARN = 0.35


def _eta_bin(eta_deg: float) -> int:
    """Map eta in [-180, 180] to bin index in [0, 360)."""
    b = int((eta_deg + 180.0) / ETA_BIN_SIZE_DEG)
    if b < 0:
        b = 0
    elif b >= N_ETA_BINS:
        b = N_ETA_BINS - 1
    return b


def _coverage_per_bin(arcs_by_det: Dict[int, list]) -> Dict[int, np.ndarray]:
    """Return ``cov[ring]`` = (N_ETA_BINS,) int — number of panels covering bin.

    A bin gets credit for one panel iff that panel has an arc on this
    ring containing the bin's centre. Multi-distance overlapping panels
    give bins a count > 1; gaps (no panel covers) give count 0 and the
    bin must be filled by interpolation before the ring total is taken.
    """
    out: Dict[int, np.ndarray] = {}
    for det_id, arcs in arcs_by_det.items():
        for arc in arcs:
            cov = out.setdefault(arc.ring_nr, np.zeros(N_ETA_BINS, dtype=np.int64))
            lo_b = _eta_bin(arc.eta_lo_deg)
            hi_b = _eta_bin(arc.eta_hi_deg)
            if lo_b <= hi_b:
                cov[lo_b:hi_b + 1] += 1
            else:                      # arc wraps ±180
                cov[lo_b:] += 1
                cov[:hi_b + 1] += 1
    return out


def _fill_uncovered(hat: np.ndarray, cov: np.ndarray) -> Tuple[np.ndarray, bool]:
    """Interpolate ``hat`` across η bins that no panel covered.

    η is periodic, so the covered samples are tiled at ``±N_ETA_BINS`` before
    interpolating; a gap that straddles ±180° is then filled from the covered
    bins on either side of the wrap rather than clamped to an endpoint.

    Returns ``(filled, is_full)``. When every bin is covered the array is
    returned unchanged (a copy), so the caller's ratio is exactly 1.0 and the
    stage is a bit-level no-op.
    """
    idx = np.nonzero(cov > 0)[0]
    n = int(cov.size)
    if idx.size == 0:
        return hat.copy(), False
    if idx.size == n:
        return hat.copy(), True
    xs = np.concatenate([idx - n, idx, idx + n]).astype(np.float64)
    ys = np.tile(hat[idx], 3)
    filled = np.interp(np.arange(n, dtype=np.float64), xs, ys)
    filled[idx] = hat[idx]          # keep observed bins exact
    return filled, False


def _profile_cv(hat: np.ndarray, cov: np.ndarray) -> float:
    """Coefficient of variation of ``Î(η)`` over coarse, fully-covered bins.

    A flat profile (the untextured expectation, see the module docstring)
    gives a small value; a textured or otherwise structured ring gives a
    large one, and gap interpolation on partial rings should not be trusted.

    Only coarse bins whose constituent fine bins are *all* covered are used,
    so a partial ring contributes only its intact portion. Returns ``nan``
    when too little of the ring is intact to judge.
    """
    n = int(hat.size)
    per = n // PROFILE_N_BINS
    if per < 1:
        return float("nan")
    usable, vals = 0, []
    for k in range(PROFILE_N_BINS):
        sl = slice(k * per, (k + 1) * per)
        if np.all(cov[sl] > 0):
            vals.append(float(hat[sl].sum()))
            usable += 1
    if usable < PROFILE_N_BINS // 2:
        return float("nan")
    arr = np.asarray(vals, dtype=np.float64)
    mean = arr.mean()
    if mean <= 0:
        return float("nan")
    return float(arr.std() / mean)


# -----------------------------------------------------------------------------
#  paramstest geometry
# -----------------------------------------------------------------------------

def _paramstest_kv(text: str) -> Dict[str, List[str]]:
    """Loose paramstest reader returning ``{key: [tokens-after-key, ...]}``."""
    out: Dict[str, List[str]] = {}
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip().rstrip(";").rstrip()
        if not line:
            continue
        toks = [t.rstrip(";") for t in line.split()]
        if not toks:
            continue
        out.setdefault(toks[0], []).append(" ".join(toks[1:]))
    return out


def _kv_float(kv: Dict[str, List[str]], key: str, default: float) -> float:
    try:
        return float(kv[key][0].split()[0])
    except (KeyError, IndexError, ValueError):
        return default


def _coverage_from_geometry(text: str) -> Dict[int, List[CoverageArc]]:
    """Derive single-panel η coverage by enumerating the panel's pixels.

    Returns ``{1: [CoverageArc, ...]}`` — shaped like ``parse_coverage_blocks``
    so the rest of the stage is indifferent to where coverage came from.
    Empty dict when the geometry or the ring table is missing.
    """
    kv = _paramstest_kv(text)

    rings = kv.get("RingNumbers", [])
    radii = kv.get("RingRadii", [])
    if not rings or len(rings) != len(radii):
        LOG.warning("  paramstest has %d RingNumbers and %d RingRadii — "
                    "cannot derive coverage from geometry",
                    len(rings), len(radii))
        return {}
    ring_radii_um: List[Tuple[int, float]] = []
    for rn_s, rad_s in zip(rings, radii):
        try:
            ring_radii_um.append((int(float(rn_s.split()[0])),
                                  float(rad_s.split()[0])))
        except (IndexError, ValueError):
            continue
    if not ring_radii_um:
        return {}

    lsd = _kv_float(kv, "LsdFit", 0.0) or _kv_float(kv, "Lsd", 0.0)
    if lsd <= 0:
        LOG.warning("  paramstest has no usable LsdFit/Lsd — "
                    "cannot derive coverage from geometry")
        return {}

    n_y = int(_kv_float(kv, "NrPixelsY", 0.0))
    n_z = int(_kv_float(kv, "NrPixelsZ", 0.0))
    n_pixels = n_y or n_z or int(_kv_float(kv, "NrPixels", 2048.0))
    if n_y and n_z and n_y != n_z:
        LOG.warning("  non-square panel (%d x %d): coverage enumeration "
                    "assumes square and will use %d", n_y, n_z, n_pixels)

    width_um = _kv_float(kv, "Width", DEFAULT_WIDTH_UM)

    arcs = compute_panel_eta_coverage(
        n_pixels=n_pixels,
        px_um=_kv_float(kv, "px", 200.0),
        lsd_um=lsd,
        y_bc_px=_kv_float(kv, "YBCFit", _kv_float(kv, "YBC", 0.0)),
        z_bc_px=_kv_float(kv, "ZBCFit", _kv_float(kv, "ZBC", 0.0)),
        tx_deg=_kv_float(kv, "txFit", _kv_float(kv, "tx", 0.0)),
        ty_deg=_kv_float(kv, "tyFit", _kv_float(kv, "ty", 0.0)),
        tz_deg=_kv_float(kv, "tzFit", _kv_float(kv, "tz", 0.0)),
        ring_radii_um=ring_radii_um,
        width_um=width_um,
    )
    if not arcs:
        return {}
    LOG.info("  derived η coverage from panel geometry "
             "(Lsd=%.1f µm, %d px, band ±%.0f µm)", lsd, n_pixels, width_um)
    return {1: arcs}


# -----------------------------------------------------------------------------
#  Spot list aggregation
# -----------------------------------------------------------------------------

def _read_radius_csvs(per_det_paths: Sequence[Path]) -> List[Tuple[int, float, float]]:
    """Collect ``(ring, eta_deg, integrated_intensity)`` from each panel.

    Only the η-resolved intensity is needed to build the powder model — the
    correction is applied per *ring*, so no spot-level identity is required
    and the brittle global→local SpotID remapping the recompute path needed
    is gone.
    """
    out: List[Tuple[int, float, float]] = []
    for path in per_det_paths:
        if not path.exists():
            continue
        with path.open() as fp:
            head = fp.readline().split()
            try:
                col_int = head.index("IntegratedIntensity")
                col_eta = head.index("Eta")
                col_ring = head.index("RingNr")
            except ValueError:
                continue
            need = max(col_int, col_eta, col_ring)
            for line in fp:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                toks = line.split()
                if len(toks) <= need:
                    continue
                try:
                    out.append((int(float(toks[col_ring])),
                                float(toks[col_eta]),
                                float(toks[col_int])))
                except ValueError:
                    continue
    return out


# -----------------------------------------------------------------------------
#  Run stage
# -----------------------------------------------------------------------------

def run(ctx: StageContext) -> StageResult:
    started = time.time()
    layer_dir = ctx.layer_dir
    input_all = layer_dir / "InputAll.csv"
    paramstest = layer_dir / "paramstest.txt"
    diag = layer_dir / "PowderModel.csv"
    cov_csv = layer_dir / "PowderCoverage.csv"

    # Tolerant skip when upstream FF stages have nothing to feed us —
    # mirrors how zip_convert / peakfit / transforms degrade in the
    # scaffold smoke-test path (no zarr present).
    if not input_all.exists() or not paramstest.exists():
        LOG.info("global_powder: missing InputAll.csv or paramstest.txt; skip.")
        return _empty_result(started, str(input_all))

    with stage_timer("global_powder"):
        ptext = paramstest.read_text()
        coverage_arcs = parse_coverage_blocks(ptext)
        source = "EtaCoverage_DetN rows"
        if not coverage_arcs:
            # Usual path for this package: transforms does not emit the rows.
            coverage_arcs = _coverage_from_geometry(ptext)
            source = "panel geometry"
        if not coverage_arcs:
            LOG.warning("  no η coverage available (no EtaCoverage rows and "
                        "geometry could not be read) — leaving GrainRadius "
                        "unchanged; partial rings will stay biased by "
                        "360°/coverage")
            return _empty_result(started, str(input_all))

        cov_per_bin = _coverage_per_bin(coverage_arcs)
        n_panels = len(coverage_arcs)
        cov_deg = total_coverage_per_ring(
            [a for arcs in coverage_arcs.values() for a in arcs]
        )
        LOG.info("  coverage from %s: %d panel(s), rings %s",
                 source, n_panels, sorted(cov_per_bin.keys()))

        # Per-spot intensity from per-detector Radius_*.csv.
        per_det_paths: List[Path] = []
        for det in ctx.detectors:
            det_dir = ctx.stage_dir(det)
            cands = sorted(det_dir.glob("Radius_StartNr_*_EndNr_*.csv"))
            if cands:
                per_det_paths.append(cands[0])
        if not per_det_paths:
            cands = sorted(layer_dir.glob("Radius_StartNr_*_EndNr_*.csv"))
            per_det_paths = cands[:1]
        spots = _read_radius_csvs(per_det_paths)
        LOG.info("  loaded %d per-detector spot intensity rows", len(spots))
        if not spots:
            LOG.warning("  no Radius_*.csv rows — leaving GrainRadius unchanged")
            return _empty_result(started, str(input_all))

        # Build the per-(ring, eta-bin) observed intensity sum.
        ring_intensity_sum: Dict[int, np.ndarray] = {}
        for ring, eta, inten in spots:
            ring_intensity_sum.setdefault(
                ring, np.zeros(N_ETA_BINS, dtype=np.float64)
            )[_eta_bin(eta)] += inten

        # Î(ring, η) = Σ_obs(I) / panels covering the bin, then interpolate
        # across bins nobody covered and integrate over the full 360°.
        i_hat: Dict[int, np.ndarray] = {}
        i_filled: Dict[int, np.ndarray] = {}
        scale: Dict[int, float] = {}
        cov_frac: Dict[int, float] = {}
        prof_cv: Dict[int, float] = {}
        low_rings: List[int] = []
        for r, ising in sorted(ring_intensity_sum.items()):
            cov = cov_per_bin.get(r, np.zeros(N_ETA_BINS, dtype=np.int64))
            with np.errstate(invalid="ignore"):
                hat = np.where(cov > 0, ising / np.maximum(cov, 1), 0.0)
            filled, is_full = _fill_uncovered(hat, cov)
            i_hat[r] = hat
            i_filled[r] = filled
            n_cov = int((cov > 0).sum())
            cov_frac[r] = n_cov / float(N_ETA_BINS)
            prof_cv[r] = _profile_cv(hat, cov)
            tot_obs = float(hat.sum())
            tot_360 = float(filled.sum())
            s = (tot_obs / tot_360) if tot_360 > 0 else 1.0
            # A ring can only lose intensity to the gaps, never gain.
            scale[r] = min(max(s, 0.0), 1.0)
            if not is_full and cov_frac[r] < LOW_COVERAGE_WARN_FRAC:
                low_rings.append(r)
            LOG.info("  ring %2d: coverage %6.1f° (%5.1f%%, %3d/%d bins)  "
                     "∫Î dη obs %.3e → 360° %.3e  scale %.4f",
                     r, cov_deg.get(r, 0.0), 100.0 * cov_frac[r], n_cov,
                     N_ETA_BINS, tot_obs * ETA_BIN_SIZE_DEG,
                     tot_360 * ETA_BIN_SIZE_DEG, scale[r])
        if low_rings:
            LOG.warning("  rings %s are below %.0f%% η coverage — their "
                        "correction is extrapolated from few bins and the "
                        "resulting GrainRadius is indicative only "
                        "(see PowderCoverage.csv)",
                        low_rings, 100.0 * LOW_COVERAGE_WARN_FRAC)

        # Does this layer actually satisfy the flat-Î(η) premise the gap
        # interpolation rests on? Judge it on the complete rings, which carry
        # no extrapolation of their own, and say so when it fails.
        intact_cv = [prof_cv[r] for r in sorted(prof_cv)
                     if cov_frac[r] >= 0.999 and not math.isnan(prof_cv[r])]
        profile_cv = float(np.median(intact_cv)) if intact_cv else float("nan")
        if intact_cv:
            LOG.info("  Î(η) profile CV on %d full-coverage ring(s): "
                     "median %.3f (untextured reference 0.09–0.21)",
                     len(intact_cv), profile_cv)
            if profile_cv > PROFILE_CV_WARN and any(
                    cov_frac[r] < 0.999 for r in scale):
                LOG.warning("  Î(η) is strongly structured (CV %.3f > %.2f) — "
                            "the sample looks textured, so interpolating "
                            "across the gaps of partial rings is unreliable. "
                            "Prefer restricting the analysis to full-coverage "
                            "rings.", profile_cv, PROFILE_CV_WARN)
        elif any(cov_frac[r] < 0.999 for r in scale):
            LOG.warning("  no full-coverage ring in this layer — the "
                        "flat-Î(η) premise behind the gap interpolation "
                        "could not be checked")

        # ── rescale GrainRadius in InputAll.csv ────────────────────────────
        # GrainVolume *= scale, so GrainRadius *= scale**(1/3); the disc model
        # is 2-D (R = sqrt(V/π)), so it takes the square root instead.
        kv = _paramstest_kv(ptext)
        disc_model = int(_kv_float(kv, "DiscModel", 0.0))
        expo = 0.5 if disc_model == 1 else (1.0 / 3.0)

        rows_in = input_all.read_text().splitlines()
        if not rows_in:
            return _empty_result(started, str(input_all))
        header = rows_in[0]
        col_names = header.split()
        try:
            col_ring = col_names.index("RingNumber")
            col_grad = col_names.index("GrainRadius")
        except ValueError as e:
            LOG.warning("  InputAll.csv column missing — leaving unchanged (%s)", e)
            return _empty_result(started, str(input_all))

        n_rewritten = 0
        out_lines = [header]
        for raw in rows_in[1:]:
            toks = raw.split()
            if len(toks) <= max(col_grad, col_ring):
                out_lines.append(raw)
                continue
            try:
                ring = int(float(toks[col_ring]))
            except ValueError:
                out_lines.append(raw)
                continue
            s = scale.get(ring, 1.0)
            if s == 1.0:
                # Full coverage: leave the row byte-identical, so this stage
                # is provably a no-op on complete rings.
                out_lines.append(raw)
                continue
            try:
                gr = float(toks[col_grad])
            except ValueError:
                out_lines.append(raw)
                continue
            toks[col_grad] = f"{gr * (s ** expo):.6f}"
            out_lines.append(" ".join(toks))
            n_rewritten += 1

        input_all.write_text("\n".join(out_lines) + "\n")
        LOG.info("  rescaled GrainRadius for %d / %d spots "
                 "(%d rings at full coverage were left untouched)",
                 n_rewritten, max(0, len(rows_in) - 1),
                 sum(1 for v in scale.values() if v == 1.0))

        # ── diagnostics ───────────────────────────────────────────────────
        with cov_csv.open("w") as fp:
            fp.write("RingNr CoverageDeg CoverageFrac NEtaBinsCovered "
                     "ScaleVolume ScaleRadius ProfileCV LowCoverage\n")
            for r in sorted(scale):
                cov = cov_per_bin.get(r, np.zeros(N_ETA_BINS, dtype=np.int64))
                fp.write(f"{r} {cov_deg.get(r, 0.0):.3f} {cov_frac[r]:.6f} "
                         f"{int((cov > 0).sum())} {scale[r]:.6e} "
                         f"{scale[r] ** expo:.6e} "
                         f"{prof_cv.get(r, float('nan')):.6f} "
                         f"{int(r in low_rings)}\n")

        with diag.open("w") as fp:
            fp.write("RingNr EtaBinLo EtaBinHi Intensity_hat "
                     "Intensity_filled NPanels\n")
            for r in sorted(i_hat):
                cov = cov_per_bin.get(r, np.zeros(N_ETA_BINS, dtype=np.int64))
                hat = i_hat[r]
                filled = i_filled[r]
                for b in range(N_ETA_BINS):
                    if hat[b] == 0 and cov[b] == 0 and filled[b] == 0:
                        continue
                    lo = -180.0 + b * ETA_BIN_SIZE_DEG
                    hi = lo + ETA_BIN_SIZE_DEG
                    fp.write(f"{r} {lo:.3f} {hi:.3f} {hat[b]:.6e} "
                             f"{filled[b]:.6e} {int(cov[b])}\n")

    finished = time.time()
    return StageResult(
        stage_name="global_powder",
        started_at=started,
        finished_at=finished,
        duration_s=finished - started,
        outputs={str(input_all): "", str(diag): "", str(cov_csv): ""},
        metrics={
            "n_panels": n_panels,
            "n_rings": len(scale),
            "n_spots_rescaled": n_rewritten,
            "coverage_source": source,
            "min_coverage_frac": (min(cov_frac.values()) if cov_frac else 1.0),
            "n_rings_low_coverage": len(low_rings),
            "profile_cv": profile_cv,
        },
    )


def expected_outputs(ctx: StageContext) -> list[Path]:
    return [ctx.layer_dir / "InputAll.csv"]


# -----------------------------------------------------------------------------
#  Helpers
# -----------------------------------------------------------------------------

def _empty_result(started: float, path: str) -> StageResult:
    finished = time.time()
    return StageResult(
        stage_name="global_powder",
        started_at=started,
        finished_at=finished,
        duration_s=finished - started,
        outputs={path: ""},
        metrics={"skipped": True},
    )
