"""Per-grain position uncertainty via :mod:`midas_propagate`.

For each final grain, we compute the data-driven (σ_X, σ_Y, σ_Z) by
inverting the Hessian of the spot-residual NLL on the 12-parameter
``(euler[3], latc[6], pos[3])`` block. This is the FROZEN-calibration
variant (assumes ``Σ_cc = 0``). The Schur-marginalised variant — which
propagates calibration uncertainty too — requires a Bayesian calibration
fit and lives downstream.

Cost is ~0.5 s/grain (jacfwd autograd for J_g, FD for J_c on the 5-param
calibration). For full datasets we recommend sampling 5-10k grains
unless you really want every grain (multi-day single-thread for 150k+).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import multiprocessing as _mp

import numpy as np
import pandas as pd
import torch


_FORK_ONE = None            # set just before a fork pool is created


def _fork_chunk(rng):
    """Run ``_FORK_ONE`` over a range in a forked child.

    Processes rather than threads: ``torch``'s forward-mode AD keeps its
    dual-level stack in GLOBAL interpreter state, so concurrent ``jacfwd``
    calls in one process corrupt each other (measured: 22/120 grains survived
    a 96-thread pool). Each forked child has its own interpreter and its own
    stack, so the same work parallelises cleanly.

    Caveat: Python warns that forking a multi-threaded process can deadlock.
    We set ``torch.set_num_threads(1)`` before creating the pool, which is the
    standard mitigation and also stops K children oversubscribing the box.
    Verified clean on 400 real grains (25.0 s serial -> 8.1 s at 8 jobs, with
    identical sigma), but that is empirical, not a guarantee: if you see a hang,
    set MIDAS_PG_SIGMA_JOBS=1.
    """
    return [_FORK_ONE(i) for i in rng]


__all__ = ["PerGrainSigmaResult", "PerGrainParameterSigmaResult",
           "compute_per_grain_position_sigma",
           "compute_per_grain_parameter_sigma"]


@dataclass
class PerGrainSigmaResult:
    """Per-grain position covariance + scalar σ.

    All arrays are length ``n_grains`` (matches the v4 leaf row order).
    ``ok`` is False if the grain failed the Hessian computation (too few
    matched spots, ill-conditioned, etc.).
    """

    sigma_X_um: np.ndarray
    sigma_Y_um: np.ndarray
    sigma_Z_um: np.ndarray
    n_spots_matched: np.ndarray
    residual_rms_px: np.ndarray
    ok: np.ndarray


@dataclass
class PerGrainParameterSigmaResult:
    """Per-grain 1-sigma on **every** refined parameter, not just position.

    The Hessian block is 12x12 over ``[euler(3), latc(6), pos(3)]`` and was
    always being computed in full — :class:`PerGrainSigmaResult` simply
    discarded nine of the twelve. These are the marginal standard deviations,
    ``sqrt(diag(inv(H_gg)))``, so each already accounts for correlation with
    the other eleven parameters.

    Units follow the MIDAS convention: **radians** for Euler angles, Angstrom
    for the lattice lengths, degrees for the lattice angles, micrometers for
    position.

    ``sigma_hydrostatic_strain`` is the derived quantity most people actually
    want, propagated from the (a, b, c) block *including* its covariance --
    for a cubic cell eps_hydro = (da/a + db/b + dc/c)/3, so treating the three
    as independent would misstate it in either direction depending on the sign
    of their correlation.
    """

    sigma_euler_rad: np.ndarray        # (N, 3)
    sigma_latc: np.ndarray             # (N, 6) — A, A, A, deg, deg, deg
    sigma_pos_um: np.ndarray           # (N, 3)
    sigma_hydrostatic_strain: np.ndarray   # (N,) dimensionless
    n_spots_matched: np.ndarray
    residual_rms_px: np.ndarray
    ok: np.ndarray
    cov: Optional[np.ndarray] = None   # (N, 12, 12) when return_cov=True
    #: {exception -> count} for grains that failed. NEVER leave this out of a
    #: report: a stage that produced nothing must say why, or "no result" reads
    #: as "the data could not support one".
    failures: Dict[str, int] = field(default_factory=dict)
    #: The ASSUMED spot noise these sigmas were computed at. Every sigma scales
    #: linearly with it, so it must travel with the numbers or they cannot be
    #: interpreted, let alone rescaled later.
    sigma_obs_px: float = 1.0
    sigma_obs_frames: float = 1.0
    sigma_obs_is_default: bool = True
    #: (N,) robust scale of THIS grain's own residual, in units of the assumed
    #: sigma_obs. 1.0 means the assumption matched the data. See
    #: :attr:`sigma_pos_emp_um`.
    resid_scale: Optional[np.ndarray] = None
    #: (N,) spots the sigma was actually computed on, and where they came from.
    spot_source: str = "processkey"

    @property
    def sigma_pos_emp_um(self) -> Optional[np.ndarray]:
        """(N, 3) EMPIRICAL 1-sigma: Fisher sigma rescaled by each grain's own
        residual. Two things this fixes, both measured against EBSD truth on
        `shade_LSHR` (2026-09-03):

        1. **It needs no ``sigma_obs_px``.** The Fisher sigma is proportional to
           the assumed noise and ``resid_scale`` is inversely proportional to it,
           so the assumption CANCELS. A placeholder value can no longer set the
           scale of the answer.
        2. **It responds to how well the grain actually fitted.** The Fisher
           sigma is expected information -- it detaches the observations, so a
           grain with a terrible residual and many spots still gets a small
           sigma. Measured: Fisher sigma had rank correlation ~0 with the true
           per-grain position error (rho -0.08 to -0.16), i.e. no power to say
           WHICH grains are badly placed.

        Still a within-model statement: it cannot see a systematic both the
        model and the data share.
        """
        if self.resid_scale is None:
            return None
        return self.sigma_pos_um * self.resid_scale[:, None]

    @property
    def sigma_pos_3d_um(self) -> np.ndarray:
        """(N,) Euclidean 1-sigma on grain position: sqrt(sx^2+sy^2+sz^2).

        This is the honest single number for "how well is this grain placed".
        It is EXACT rather than an approximation: for a 3-D error with
        covariance Sigma, E[|dr|^2] = trace(Sigma), and the trace is unchanged
        by the off-diagonal correlations between X, Y and Z. So no information
        is lost by collapsing the three components this way, which is not true
        of quoting them separately or averaging them.

        Reporting the three components pooled into one HISTOGRAM, by contrast,
        is misleading — Z is better constrained than X and Y, so the pooled
        distribution is bimodal for a purely geometric reason. Use this scalar,
        or plot the components separately; never pool them.
        """
        return np.sqrt(np.nansum(self.sigma_pos_um ** 2, axis=1))

    #: parameter order of the 12-block, for labelling output tables
    PARAM_NAMES = ("euler0", "euler1", "euler2",
                   "a", "b", "c", "alpha", "beta", "gamma",
                   "X", "Y", "Z")

    def as_columns(self) -> dict:
        """Flat ``{name: (N,) array}`` for writing a CSV."""
        out = {}
        for k in range(3):
            out[f"sigma_{self.PARAM_NAMES[k]}_rad"] = self.sigma_euler_rad[:, k]
        for k in range(6):
            out[f"sigma_{self.PARAM_NAMES[3+k]}"] = self.sigma_latc[:, k]
        for k in range(3):
            out[f"sigma_{self.PARAM_NAMES[9+k]}_um"] = self.sigma_pos_um[:, k]
        out["sigma_pos_3d_um"] = self.sigma_pos_3d_um
        out["sigma_eps_hydro"] = self.sigma_hydrostatic_strain
        out["residual_rms_px"] = self.residual_rms_px
        if self.resid_scale is not None:
            out["resid_scale"] = self.resid_scale
            emp = self.sigma_pos_emp_um
            for k in range(3):
                out[f"sigma_{self.PARAM_NAMES[9+k]}_emp_um"] = emp[:, k]
            out["sigma_pos_3d_emp_um"] = np.sqrt(np.nansum(emp ** 2, axis=1))
        out["n_spots_matched"] = self.n_spots_matched
        out["ok"] = self.ok
        return out


def compute_per_grain_parameter_sigma(
    *,
    grain_OM: np.ndarray,                  # (N, 3, 3) — consensus OMs (FZ-canonical)
    grain_pos_um: np.ndarray,              # (N, 3)
    rep_cand_idx: np.ndarray,              # (N,) — OPF row index of the grain's rep
    pk_path: Path,                          # ProcessKey.bin
    spot_ids_per_grain: Optional[Sequence[np.ndarray]] = None,
    inputall_df: pd.DataFrame,              # InputAll (SpotID-indexed) with YLab,ZLab,Omega
    hkls,                                   # HklTable
    geometry,                               # midas_diffract.HEDMGeometry
    latc: np.ndarray,                       # (6,) lattice (a,b,c,α,β,γ)
    calibration_names: Sequence[str] = ("Lsd", "BC_y", "BC_z", "ty", "tz"),
    calibration_map: Optional[np.ndarray] = None,    # (n_c,) MAP values
    sigma_obs_px: float = 1.0,
    sigma_obs_frames: Optional[float] = None,
    max_match_dist_px: float = 10.0,
    omega_start_deg: float = -180.0,
    omega_step_deg: float = 0.25,
    method: str = "fisher",
    device: Optional[str] = None,
    return_cov: bool = False,
    log=None,
) -> PerGrainParameterSigmaResult:
    """Per-grain 1-sigma on all 12 refined parameters, by Hessian inversion.

    ``latc`` may be a single ``(6,)`` cell shared by every grain, or a
    ``(N, 6)`` array of the **per-grain fitted** cells. Prefer the latter
    whenever you care about the lattice or strain uncertainties: evaluating
    every grain's Hessian at one global cell puts the linearisation at the
    wrong point for every grain except the average one, and the lattice block
    is exactly where that bites.

    Parameters
    ----------
    grain_OM : (N, 3, 3) float64
        Consensus FZ-canonical orientation matrix per grain.
    grain_pos_um : (N, 3) float64
        Grain position in sample frame (µm).
    rep_cand_idx : (N,) int64
        ProcessKey row index for each grain's representative candidate.
    spot_ids_per_grain : sequence of (M_i,) int64, optional
        The spots each grain was **refined** on, normally from ``SpotMatrix.csv``.
        **Pass this whenever you have it.** ``ProcessKey`` holds the spots the
        INDEXER matched to the representative *seed*, which is a strict subset:
        measured on `shade_LSHR`, median 58 against the refined grain's 268.
        Since sigma ~ 1/sqrt(n) (measured rho +0.985 against 1/sqrt(n)), taking
        the seed set inflates every sigma by sqrt(268/58) = 2.15x -- it describes
        the precision of the seed, not of the grain that was reported.
        (== OPF row index = alive_idx[local_rep] from the v4 pipeline).
    pk_path : Path
        ``Results/ProcessKey.bin``.
    inputall_df : DataFrame
        InputAll table indexed by SpotID with columns YLab, ZLab, Omega.
    hkls : HklTable
        Output of :func:`compute.hkl_ingest.read_hkls_csv`.
    geometry : HEDMGeometry
        Detector + scan geometry (used both to build the forward model
        and to convert observed YLab,ZLab to detector pixel coordinates).
    latc : (6,) float64
        Lattice parameters (a, b, c, α, β, γ) — Å and degrees.
    calibration_names, calibration_map : optional
        Calibration parameter names + MAP values forwarded to
        ``midas_propagate.per_grain_hessian_blocks``. Defaults to the
        five calibration parameters the Schur path expects. Position
        covariance does not require Σ_cc; this argument is only used to
        construct the (g, c) Hessian blocks. Pass calibration_map = MAP
        values from paramstest.
    sigma_obs_px : float
        Spot measurement noise on the DETECTOR coordinates (y, z), in pixels.
        **Every sigma this function returns scales LINEARLY with it, and it is
        an assumption you supply — not something measured from your data.**
        The default 1.0 is a placeholder, not a typical value: on 1-ID GE5 the
        measured figure was 0.35 px, so the default inflated every error bar by
        ~2.9x (verified 2026-09-02, four lenses). Measure it for your setup —
        ``midas-grain-qa --stage calibration`` does this from grains re-found in
        an adjacent layer — and pass it. A warning is emitted if you do not.

        Do NOT estimate it as chi-squared/dof over the refiner's residuals:
        MIDAS minimises a sum of ABSOLUTE internal angles (an L1/LAD estimator,
        ``FitPosOrStrainsOMP.c``), the residual is heavy-tailed, and chi2/dof
        answers a question about a least-squares fit that is never performed.
        On 1-ID it gave 0.84 px against a true 0.35.
    sigma_obs_frames : float, optional
        Noise on the OMEGA coordinate, in frames. Defaults to ``sigma_obs_px``,
        which is a unit confusion worth avoiding: a frame is not a pixel. On
        1-ID one frame is 0.25 deg while the measured omega core was 0.098 deg,
        so carrying the pixel value across over-assumed the omega channel ~2.5x
        on its own.
    max_match_dist_px : float
        Association radius at MAP, in the mixed (pixel, pixel, frame) metric.
        **Was 2.0 until 2026-09-03, which silently discarded ~78 % of each
        grain's real spots** -- measured on `shade_LSHR`, 58 of 267 matched, and
        since sigma ~ 1/sqrt(n) every sigma came out 2.20x too large. The value
        is safe because the result SATURATES: 2.0 -> 58 spots, 5.0 -> 261,
        10.0 -> 267, and 25.0 and 60.0 also give 267 with sigma unchanged to
        3 significant figures. A threshold past the saturation point admits no
        further spots, so it cannot be pulling in false associations. If you
        lower it, check n_spots_matched against the grain's row count in
        SpotMatrix.csv -- a large gap means sigma describes a smaller
        measurement than the one that produced the grain.
        Nearest-neighbor association threshold at MAP for spot matching.
        Default 2.0 (= 2 px).
    method : 'fisher' | 'hessian'
        See ``midas_propagate.joint_nll.per_grain_hessian_blocks``.
    log : callable or None
        Verbose progress logger.

    Returns
    -------
    PerGrainSigmaResult
    """
    from midas_diffract.forward import HEDMForwardModel
    from midas_propagate.joint_nll import GrainObs, per_grain_hessian_blocks
    from midas_stress.orientation import orient_mat_to_euler

    if log is None:
        log = lambda *a, **k: None
    N = int(grain_OM.shape[0])

    # Device routing: MPS or CUDA if available, CPU otherwise
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    dev = torch.device(device)
    if device != "cpu":
        log(f"  per-grain σ: using device={device}")

    # Load ProcessKey for matched-spot lookup -- only when no explicit refined
    # spot list was supplied (see `spot_ids_per_grain`).
    if spot_ids_per_grain is None:
        pk_rows = os.path.getsize(pk_path) // (5000 * 4)
        PK = np.memmap(pk_path, dtype=np.int32, mode="r", shape=(pk_rows, 5000))
    else:
        pk_rows, PK = 0, None
        if len(spot_ids_per_grain) != N:
            raise ValueError(
                f"spot_ids_per_grain has {len(spot_ids_per_grain)} entries for "
                f"{N} grains; it must be aligned with grain_OM")

    hkls_cart = torch.from_numpy(hkls.g_crystal.astype(np.float64))
    thetas = torch.from_numpy(np.deg2rad(hkls.theta_deg.astype(np.float64)))
    hkls_int = torch.from_numpy(np.stack(
        [hkls.h, hkls.k, hkls.l], axis=1,
    ).astype(np.int64))

    # Detector geometry from the HEDMGeometry instance
    y_BC = float(geometry.y_BC); z_BC = float(geometry.z_BC)
    px = float(geometry.px)
    if calibration_map is None:
        calibration_map = np.array([
            float(geometry.Lsd), y_BC, z_BC,
            float(geometry.ty), float(geometry.tz),
        ], dtype=np.float64)
    calibration_map = torch.from_numpy(np.asarray(calibration_map, dtype=np.float64))
    # (y, z) in pixels, omega in FRAMES — different units, so different scales.
    s_ome = float(sigma_obs_px if sigma_obs_frames is None else sigma_obs_frames)
    sigma_obs = torch.tensor([float(sigma_obs_px), float(sigma_obs_px), s_ome],
                             dtype=torch.float64)
    if float(sigma_obs_px) == 1.0:
        log("  per-grain sigma: ** sigma_obs_px is at its PLACEHOLDER default of "
            "1.0 px. Every sigma below scales linearly with it and none of them "
            "carry information from your data. Measure it "
            "(midas-grain-qa --stage calibration) and pass it. On 1-ID the true "
            "value was 0.35 px, so this default inflated the bars ~2.9x. **")
    if sigma_obs_frames is None:
        log("  per-grain sigma: sigma_obs_frames not given; using the pixel "
            "value for the omega channel. A frame is not a pixel — check it.")
    latc_arr = np.asarray(latc, dtype=np.float64)
    if latc_arr.ndim == 1:
        latc_arr = np.broadcast_to(latc_arr.reshape(1, 6), (N, 6))
    elif latc_arr.shape != (N, 6):
        raise ValueError(f"latc must be (6,) or ({N}, 6); got {latc_arr.shape}")
    latc_all = torch.from_numpy(np.ascontiguousarray(latc_arr))

    sig12 = np.full((N, 12), np.nan, dtype=np.float64)
    sig_eps = np.full(N, np.nan, dtype=np.float64)
    cov_all = np.full((N, 12, 12), np.nan, dtype=np.float64) if return_cov else None
    n_match = np.zeros(N, dtype=np.int32)
    resid_rms = np.full(N, np.nan, dtype=np.float64)
    ok_arr = np.zeros(N, dtype=bool)

    # ── Per-grain σ is embarrassingly parallel over i. The dominant cost
    # (per_grain_hessian_blocks: jacfwd autograd + Hessian invert) is torch
    # C++ that releases the GIL, so a thread pool gives near-core-count
    # scaling with NO data duplication (PK memmap / inputall_df / hkls
    # tensors are all read-only shared). Numerics are identical to the
    # former serial loop; only the iteration is concurrent. Worker count
    # via MIDAS_PG_SIGMA_JOBS (default min(96, N)).
    import os as _os
    from collections import Counter
    _fail: "Counter[str]" = Counter()
    # Default to SERIAL. The 96-way pool was measured to fail for ~90% of
    # grains on 1-ID LSHR (40/400 vs 400/400 serial) because the autograd
    # inside per_grain_hessian_blocks is not safe to run concurrently this
    # way. Opt in with MIDAS_PG_SIGMA_JOBS if you have verified it on your
    # data by comparing the pass rate against a serial run.
    _njobs = int(_os.environ.get("MIDAS_PG_SIGMA_JOBS", "0")) or 0
    if _njobs == 0:                       # not set: choose a safe default
        _njobs = 1 if str(dev) != "cpu" else min(_os.cpu_count() or 1, 16)
    # CUDA does not survive fork, and on GPU the serial path is already ~0.05
    # s/grain, so parallelism buys nothing there.
    _use_fork = (_njobs > 1 and str(dev) == "cpu"
                 and "fork" in _mp.get_all_start_methods())
    if _njobs > 1 and not _use_fork:
        _njobs = 1
    if _use_fork:
        # each child gets one intra-op thread so K children do not oversubscribe
        try:
            torch.set_num_threads(1)
        except Exception:
            pass

    def _one(i):
        if spot_ids_per_grain is not None:
            sids = np.asarray(spot_ids_per_grain[i], dtype=np.int64).ravel()
            # SpotMatrix carries UNMATCHED PREDICTION rows with SpotID <= 0
            # (a theoretical reflection the grain should have shown and did
            # not). Those have no InputAll entry, so keeping them makes the
            # lookup raise and the whole grain report as failed -- which is
            # how this first went in: 900/900 grains -> 276/900.
            sids = sids[sids > 0]
        else:
            rep = int(rep_cand_idx[i])
            if not (0 <= rep < pk_rows):
                return None
            sids = PK[rep]; sids = sids[sids != 0].astype(np.int64)
        sids = np.unique(sids)
        if len(sids) < 4:
            return None
        try:
            ia_g = inputall_df.loc[sids].dropna()
        except KeyError:
            return None
        if len(ia_g) < 4:
            return None
        y_pix = y_BC - ia_g["YLab"].to_numpy(np.float64) / px
        z_pix = z_BC + ia_g["ZLab"].to_numpy(np.float64) / px
        frame = (ia_g["Omega"].to_numpy(np.float64) - omega_start_deg) / omega_step_deg
        obs_det = torch.from_numpy(np.column_stack([y_pix, z_pix, frame]))
        OM = np.asarray(grain_OM[i], dtype=np.float64).reshape(3, 3)
        euler_rad = torch.from_numpy(orient_mat_to_euler(OM).astype(np.float64))
        pos_um = torch.from_numpy(np.asarray(grain_pos_um[i], dtype=np.float64))
        grain_obs = GrainObs(
            spot_id=i, euler_rad=euler_rad, latc=latc_all[i],
            pos_um=pos_um, observed_detector=obs_det,
        )
        try:
            res = per_grain_hessian_blocks(
                grain_obs,
                hkls_cart=hkls_cart, hkls_int=hkls_int, thetas=thetas,
                base_geometry=geometry, scan_config=None,
                calibration_names=list(calibration_names),
                calibration_map=calibration_map,
                sigma_obs_detector=sigma_obs,
                method=method,
                max_match_dist=max_match_dist_px,
            )
        except Exception as e:                          # noqa: BLE001
            # RECORD, never swallow. A bare `return None` here hid a
            # CONCURRENCY failure for a long time: across the 96-thread pool
            # below this raised for ~90% of grains, which the result then
            # reported merely as ok=False — indistinguishable from "the data
            # cannot support a fit". Run single-threaded the SAME grains all
            # succeed and give the same sigma to 3 significant figures. A
            # silent except turns a bug into a physics conclusion.
            _fail[f"{type(e).__name__}: {str(e)[:90]}"] += 1
            return None
        # CONDITIONING GATE. The ridge below exists to keep the inverse from
        # blowing up, but if the Hessian is rank-deficient in some direction
        # the "uncertainty" it returns there is just sqrt(1/ridge) — a
        # property of the regularisation, not of the data. Measured on 1-ID
        # LSHR layer 6: without this gate ~half the grains reported
        # sigma_X = 3.162e4 um, which is exactly sqrt(1/1e-9). Report those as
        # FAILED rather than as a very large uncertainty, because a caller
        # filtering on "sigma < threshold" would silently keep them.
        RIDGE = 1e-9
        H_raw = res.H_gg
        try:
            evals = torch.linalg.eigvalsh(
                0.5 * (H_raw + H_raw.transpose(-1, -2)))
            lo = float(evals.min()); hi = float(evals.max())
        except Exception:
            return None
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= 0:
            return None
        # The test must be RELATIVE. An absolute floor rejects a perfectly
        # well-conditioned Hessian that merely has a small overall scale --
        # on this data an absolute cut at RIDGE*1e3 threw away 77% of grains
        # that had ~190 spots for 12 parameters. What actually matters is
        # (a) the condition number and (b) whether the ridge is large enough
        # relative to the smallest eigenvalue to be doing the work.
        if lo <= 0 or (hi / lo) > 1e12 or lo <= RIDGE * 10.0:
            return None
        H_gg = H_raw + RIDGE * torch.eye(H_raw.shape[0], dtype=H_raw.dtype)
        try:
            Sigma_g = torch.linalg.inv(H_gg)
        except Exception:
            Sigma_g = torch.linalg.pinv(H_gg)
        S = Sigma_g.detach().cpu().numpy()
        sig = np.sqrt(np.maximum(np.diag(S), 0.0))          # all 12
        # eps_hydro = (da/a + db/b + dc/c)/3 for a cubic cell. Propagate with
        # the FULL (a,b,c) covariance: J Sigma_abc J^T. Ignoring the
        # off-diagonals would misstate this either way, since a, b and c are
        # strongly correlated through the common radial scale.
        abc = latc_arr[i, :3]
        J = (1.0 / (3.0 * abc))
        var_eps = float(J @ S[3:6, 3:6] @ J)
        # ROBUST scale of this grain's own residual, in units of the assumed
        # sigma_obs (the residual is returned already divided by it). 1.0 means
        # the assumption matched this grain. MAD, not rms: a couple of
        # misassigned spots are common and would drag an rms badly -- measured
        # on 1-ID, 15-17% of spots carried 82-86% of the sum of squares.
        _r = res.residual_at_map.detach().cpu().numpy().ravel()
        _r = _r[np.isfinite(_r)]
        _scale = (1.4826 * float(np.median(np.abs(_r - np.median(_r))))
                  if _r.size >= 4 else float("nan"))
        return (i, sig, float(np.sqrt(max(var_eps, 0.0))),
                int(res.n_spots_matched),
                float(torch.sqrt((res.residual_at_map ** 2).mean()).item()),
                S if return_cov else None, _scale)

    log(f"  per-grain σ: {N:,} grains, "
        + (f"{_njobs} forked processes" if _use_fork else "serial"))
    n_ok = 0; n_fail = 0; _done = 0
    resid_scale = np.full(N, np.nan)

    def _results():
        if not _use_fork:
            for i in range(N):
                yield _one(i)
            return
        global _FORK_ONE
        _FORK_ONE = _one
        step = max(1, (N + _njobs - 1) // _njobs)
        chunks = [range(a, min(a + step, N)) for a in range(0, N, step)]
        ctx = _mp.get_context("fork")
        with ctx.Pool(_njobs) as pool:
            for batch in pool.imap_unordered(_fork_chunk, chunks):
                for r in batch:
                    yield r

    if True:
        for r in _results():
            _done += 1
            if r is None:
                n_fail += 1
            else:
                i, sig, se, nm, rr, S, rs = r
                sig12[i] = sig; sig_eps[i] = se
                if return_cov and S is not None:
                    cov_all[i] = S
                n_match[i] = nm; resid_rms[i] = rr; ok_arr[i] = True
                resid_scale[i] = rs
                n_ok += 1
            if _done % 1000 == 0:
                log(f"    σ progress: {_done:,}/{N:,}  ({n_ok} ok, {n_fail} fail)")

    if _fail:
        log(f"  per-grain σ: {sum(_fail.values())} grains failed; reasons:")
        for _m, _c in _fail.most_common(5):
            log(f"      {_c:6d}  {_m}")
        if _njobs > 1:
            log(f"  NOTE {_njobs} worker threads were used. Re-run with "
                f"MIDAS_PG_SIGMA_JOBS=1 before believing a high failure rate — "
                f"this path has a measured concurrency failure.")

    return PerGrainParameterSigmaResult(
        sigma_euler_rad=sig12[:, 0:3], sigma_latc=sig12[:, 3:9],
        sigma_pos_um=sig12[:, 9:12], sigma_hydrostatic_strain=sig_eps,
        n_spots_matched=n_match, residual_rms_px=resid_rms, ok=ok_arr,
        cov=cov_all, failures=dict(_fail),
        sigma_obs_px=float(sigma_obs_px), sigma_obs_frames=s_ome,
        sigma_obs_is_default=bool(float(sigma_obs_px) == 1.0),
        resid_scale=resid_scale,
        spot_source=("spotmatrix(refined)" if spot_ids_per_grain is not None
                     else "processkey(seed)"),
    )


def compute_per_grain_position_sigma(**kw) -> PerGrainSigmaResult:
    """Position-only view of :func:`compute_per_grain_parameter_sigma`.

    Kept because ``v4_pipeline`` calls it and only wants (σ_X, σ_Y, σ_Z).
    It is a slice of the same computation, not a second implementation —
    the 12x12 Hessian was always being inverted in full.
    """
    kw.pop("return_cov", None)
    full = compute_per_grain_parameter_sigma(**kw)
    return PerGrainSigmaResult(
        sigma_X_um=full.sigma_pos_um[:, 0],
        sigma_Y_um=full.sigma_pos_um[:, 1],
        sigma_Z_um=full.sigma_pos_um[:, 2],
        n_spots_matched=full.n_spots_matched,
        residual_rms_px=full.residual_rms_px, ok=full.ok,
    )
