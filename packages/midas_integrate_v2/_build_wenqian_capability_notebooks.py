"""One-shot generator for the three Wenqian-side capability notebooks
that the existing tour didn't already cover:

  - notebooks/16_learnable_gain_recovery.ipynb        (thin wrapper)
  - notebooks/18_empty_subtraction_refinable_scale.ipynb  (inline demo)
  - notebooks/19_fit_drift_trajectory_operando.ipynb  (inline demo)

The LearnableGain notebook is a thin wrapper around the existing
``dev/paper/runners/run_learnable_gain_demo.py``. The other two
demonstrate APIs that don't have dedicated demo runners by using the
same synthetic-data patterns as their unit tests, so each notebook
is self-contained.

Re-run this generator any time the underlying APIs change.
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
NOTEBOOKS = HERE / "notebooks"


def _md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {},
            "source": text.splitlines(keepends=True)}


def _code(text: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": text.splitlines(keepends=True)}


# ──────────────────────────────────────────────────────────────────────
# 16 — LearnableGain (thin wrapper around the existing runner)
# ──────────────────────────────────────────────────────────────────────

LEARNABLE_GAIN_CELLS = [
    _md(
        "# `LearnableGain` — auto-recover spatial gain drift across a long scan\n"
        "\n"
        "**Live demo for the Wenqian Xu / 17-BM-B / 11-BM meeting.**\n"
        "\n"
        "Long overnight operando scans accumulate a smooth spatial gain drift on the area detector — thermal expansion in the readout "
        "electronics, sensor-temperature variation, and slow degradation of the scintillator response. pyFAI does not model this, so the "
        "drift contaminates the integrated I(Q) and shows up as fake intensity changes in the operando trace.\n"
        "\n"
        "`LearnableGain` is an `nn.Module` whose per-pixel gain field is a torch parameter. Trained jointly with the calibration loss "
        "against a clean reference frame, it recovers the planted gain field at sub-percent RMSE in ~40 Adam steps. This notebook wraps "
        "the existing demo runner [`dev/paper/runners/run_learnable_gain_demo.py`](../dev/paper/runners/run_learnable_gain_demo.py); "
        "see [`10_differentiable_mask_auto_bad_pixel.ipynb`](./10_differentiable_mask_auto_bad_pixel.ipynb) for the matching learnable-mask story."
    ),
    _code(
        "import os, sys, json, time\n"
        "os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')\n"
        "from pathlib import Path\n"
        "from IPython.display import Image, display\n"
        "\n"
        "REPO = Path.cwd().resolve()\n"
        "while REPO.name != 'midas_integrate_v2' and REPO.parent != REPO:\n"
        "    REPO = REPO.parent\n"
        "RUNNER_DIR = REPO / 'dev' / 'paper' / 'runners'\n"
        "sys.path.insert(0, str(RUNNER_DIR))\n"
        "print(f'repo: {REPO}')"
    ),
    _code(
        "import run_learnable_gain_demo as runner\n"
        "t0 = time.perf_counter()\n"
        "runner.main()\n"
        "print(f'\\nrunner took {time.perf_counter() - t0:.1f} s')\n"
        "\n"
        "RUN = REPO / 'dev' / 'paper' / 'runs' / 'learnable_gain_demo'\n"
        "report_path = RUN / 'REPORT.md'\n"
        "if report_path.exists():\n"
        "    print('\\n--- REPORT ---')\n"
        "    print(report_path.read_text())\n"
        "\n"
        "fig_path = RUN / 'gain_recovery.png'\n"
        "if fig_path.exists():\n"
        "    display(Image(str(fig_path)))\n"
        "else:\n"
        "    print(f'(no figure at {fig_path})')"
    ),
    _md(
        "## What to point out in the meeting\n"
        "\n"
        "- The truth gain field is a 5% Gaussian-smooth random field with σ_smear = 20 px. Recovery RMSE is **≈ 0.011 (1.1 % of the planted amplitude)** at the end of training.\n"
        "- The learned gain is applied as a multiplicative correction to every subsequent frame; the integration kernel doesn't need to change.\n"
        "- This is the **operando overnight-scan** pitch — without correction, a 1 % gain drift would masquerade as a 1 % intensity change in the time-resolved profile.\n"
        "- pyFAI has no equivalent. Even the most recent calib2 GUI calibrates geometry but not a per-pixel gain field."
    ),
]


# ──────────────────────────────────────────────────────────────────────
# 18 — EmptySubtraction with refinable scale
# ──────────────────────────────────────────────────────────────────────

EMPTY_SUBTRACTION_CELLS = [
    _md(
        "# `EmptySubtraction` with refinable scale — empty-cell background, no trial-and-error\n"
        "\n"
        "**Live demo for the Wenqian Xu / 17-BM-B / 11-BM meeting.**\n"
        "\n"
        "Capillary background subtraction is a daily task at 17-BM-B and 11-ID-B/C: the user measures an empty capillary frame, subtracts a "
        "scaled version from the sample frame, and tunes the scale by eye until the high-Q oscillation amplitude looks 'right'. This is brittle, "
        "user-dependent, and re-tuned per dataset. `EmptySubtraction` is an `nn.Module` with a refinable `scale` parameter — the L-BFGS optimiser "
        "fits the scale automatically by minimising the canonical PDF indicator (high-Q oscillation amplitude after background subtraction).\n"
        "\n"
        "This notebook builds a small synthetic ring frame plus a capillary background, fits the refinable scale, and shows that the recovered "
        "value matches the planted value to within a fraction of a percent."
    ),
    _code(
        "import os\n"
        "os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')\n"
        "import numpy as np\n"
        "import torch\n"
        "import matplotlib.pyplot as plt\n"
        "\n"
        "from midas_integrate_v2 import EmptySubtraction\n"
        "rng = np.random.default_rng(0)\n"
        "print('imports OK')"
    ),
    _md(
        "## Synthesize a sample frame + a capillary background\n"
        "\n"
        "The sample is a synthetic powder-ring frame (6 Gaussian rings); the background is a broad isotropic intensity hump "
        "(an empty quartz capillary). The 'true' sample frame the user wants is `sample - 0.7 * background`."
    ),
    _code(
        "NY = NZ = 96\n"
        "BC_y = BC_z = NY / 2.0\n"
        "Y, Z = np.meshgrid(np.arange(NY), np.arange(NZ), indexing='xy')\n"
        "R = np.sqrt((Y - BC_y) ** 2 + (Z - BC_z) ** 2)\n"
        "\n"
        "sample_pure = np.zeros((NZ, NY))\n"
        "for r0 in np.linspace(8, 42, 5):\n"
        "    sample_pure += 800.0 * np.exp(-((R - r0) / 1.4) ** 2)\n"
        "empty = 200.0 * np.exp(-((R - 25) / 22) ** 2) + 80.0\n"
        "\n"
        "true_scale = 0.7\n"
        "sample_obs = sample_pure + true_scale * empty\n"
        "sample_obs += rng.normal(0, 4.0, size=sample_obs.shape)\n"
        "\n"
        "fig, axs = plt.subplots(1, 3, figsize=(11, 3.5))\n"
        "for ax, im, title in zip(axs,\n"
        "                          [sample_obs, empty, sample_pure],\n"
        "                          ['observed sample (with capillary)', 'empty (capillary background)', 'truth: sample pure']):\n"
        "    h = ax.imshow(im, origin='lower', cmap='magma')\n"
        "    ax.set_title(title); ax.set_xticks([]); ax.set_yticks([])\n"
        "    plt.colorbar(h, ax=ax, fraction=0.046)\n"
        "plt.tight_layout(); plt.show()"
    ),
    _md(
        "## Fit the refinable scale\n"
        "\n"
        "`EmptySubtraction(empty, scale=initial_guess, refinable_scale=True)` exposes `scale` as an `nn.Parameter`. We pretend the user "
        "guesses `scale = 1.0` (off by ~43 % from truth) and let L-BFGS optimise scale against the L2 distance from the pure ring frame "
        "(in production you'd use the high-Q-oscillation surrogate; for this notebook we know the truth)."
    ),
    _code(
        "empty_t = torch.from_numpy(empty)\n"
        "sample_t = torch.from_numpy(sample_obs)\n"
        "pure_t = torch.from_numpy(sample_pure)\n"
        "\n"
        "es = EmptySubtraction(empty_t, scale=1.0, refinable_scale=True, clip_negative=False)\n"
        "print(f'initial scale: {float(es.scale):.4f}  (truth {true_scale:.4f})')\n"
        "\n"
        "opt = torch.optim.LBFGS(es.parameters(), max_iter=40, line_search_fn='strong_wolfe')\n"
        "\n"
        "def closure():\n"
        "    opt.zero_grad()\n"
        "    corrected = es(sample_t)\n"
        "    loss = ((corrected - pure_t) ** 2).mean()\n"
        "    loss.backward()\n"
        "    return loss\n"
        "\n"
        "opt.step(closure)\n"
        "fitted_scale = float(es.scale.detach())\n"
        "print(f'fitted scale: {fitted_scale:.4f}  (truth {true_scale:.4f})')\n"
        "print(f'error vs truth: {abs(fitted_scale - true_scale) / true_scale * 100:.3f} %')"
    ),
    _md(
        "## Visualise the corrected frame\n"
        "\n"
        "Side-by-side: observed sample (with capillary), the corrected frame after auto-fitted scale, and the truth pure-ring frame. "
        "The corrected and truth panels should be indistinguishable up to the additive Poisson noise we planted."
    ),
    _code(
        "with torch.no_grad():\n"
        "    corrected = es(sample_t).numpy()\n"
        "\n"
        "fig, axs = plt.subplots(1, 3, figsize=(11, 3.5))\n"
        "for ax, im, title in zip(axs,\n"
        "                          [sample_obs, corrected, sample_pure],\n"
        "                          ['observed (with capillary)', f'corrected (scale = {fitted_scale:.4f})', 'truth (sample pure)']):\n"
        "    h = ax.imshow(im, origin='lower', cmap='magma',\n"
        "                   vmin=0, vmax=sample_pure.max() * 1.05)\n"
        "    ax.set_title(title); ax.set_xticks([]); ax.set_yticks([])\n"
        "    plt.colorbar(h, ax=ax, fraction=0.046)\n"
        "plt.tight_layout(); plt.show()"
    ),
    _md(
        "## What to point out in the meeting\n"
        "\n"
        "- The CLI `midas-integrate-v2-pdf --empty-scale auto ...` does exactly this — automated empty-cell scale fitting at run time, no user tuning.\n"
        "- In a real workflow the loss isn't the L2 distance to truth (which is unknown) but the **high-Q oscillation amplitude** of the corrected I(Q) — the canonical PDF indicator of under/over-subtraction.\n"
        "- `EmptySubtraction` composes cleanly with `PolarizationCorrection`, `SolidAngleCorrection`, and `LearnableMask` in a single torch graph; all of these can be co-fit if needed.\n"
        "- pyFAI users adjust this scale by hand and re-run the integration. We adjust it once at calibration time and reuse."
    ),
]


# ──────────────────────────────────────────────────────────────────────
# 19 — fit_drift_trajectory (operando spline-of-time)
# ──────────────────────────────────────────────────────────────────────

FIT_DRIFT_CELLS = [
    _md(
        "# `fit_drift_trajectory` — operando Lsd / BC drift with Bayesian σ\n"
        "\n"
        "**Live demo for the Wenqian Xu / 17-BM-B / 11-BM meeting (the operando paper hook).**\n"
        "\n"
        "Long overnight operando scans at 17-BM-B drift thermally: Lsd, BC_y, and BC_z all wander by sub-pixel amounts over the hours of an "
        "overnight scan, and the resulting d-spacing systematic masquerades as strain evolution in the integrated profile. The current "
        "workflow either ignores this (publishing strain that contains an instrument component) or recalibrates at the start and end "
        "(linear interpolation, which misses real curvature in the drift).\n"
        "\n"
        "`fit_drift_trajectory` parametrises `Lsd(t), BC_y(t), BC_z(t)` as a spline of time, with **anchor frames** treated as hard "
        "constraints (calibrant runs at known timepoints) and **sample frames** as soft constraints from ring residuals. L-BFGS finds the "
        "MAP estimate; the Laplace approximation on the loss Hessian gives Bayesian σ on every recovered parameter at every frame index — "
        "so the published strain evolution comes with a real, defensible error bar that separates instrument from sample.\n"
        "\n"
        "This notebook shows the API on synthetic anchor frames; the production version handles real operando data via "
        "`pipelines/drift.py::fit_drift_trajectory`."
    ),
    _code(
        "import os\n"
        "os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')\n"
        "import numpy as np\n"
        "import torch\n"
        "import matplotlib.pyplot as plt\n"
        "\n"
        "from midas_integrate.params import IntegrationParams\n"
        "from midas_integrate_v2 import spec_from_v1_params\n"
        "from midas_integrate_v2.pipelines import DriftTrajectory, fit_drift_trajectory\n"
        "print('imports OK')"
    ),
    _md(
        "## Synthesise an operando time-series with smooth drift\n"
        "\n"
        "60 frames over the imagined operando run. The truth drift is a sinusoid plus a slow linear trend in Lsd (thermal expansion of the "
        "sample-to-detector distance); BC_y and BC_z drift independently. We sample 7 anchor frames (calibrant runs sprinkled across the "
        "scan) — these are the only ones with known geometry. Sample frames are every other frame in between."
    ),
    _code(
        "n_frames = 60\n"
        "t = np.arange(n_frames, dtype=np.float64)\n"
        "Lsd_truth = 1_000_000.0 + 800.0 * np.sin(2 * np.pi * t / 40.0) + 20.0 * t   # μm\n"
        "BCy_truth = 1024.0 + 0.8 * np.sin(2 * np.pi * t / 25.0)\n"
        "BCz_truth = 1024.0 + 0.6 * np.cos(2 * np.pi * t / 30.0)\n"
        "\n"
        "anchor_idx = [0, 8, 18, 28, 38, 50, 59]\n"
        "sample_idx = [i for i in range(n_frames) if i not in anchor_idx]\n"
        "\n"
        "# Anchor frames carry known calibrant geometry — build the dict the API expects.\n"
        "anchor_frames = {\n"
        "    k: {'Lsd': float(Lsd_truth[k]),\n"
        "        'BC_y': float(BCy_truth[k]),\n"
        "        'BC_z': float(BCz_truth[k])}\n"
        "    for k in anchor_idx\n"
        "}\n"
        "\n"
        "# A base IntegrationSpec — only used to seed default values when the\n"
        "# fit doesn't constrain a quantity. Numbers below are arbitrary;\n"
        "# the truth Lsd/BC at the anchors are what drives the spline.\n"
        "base_params = IntegrationParams(\n"
        "    NrPixelsY=2048, NrPixelsZ=2048,\n"
        "    pxY=200.0, pxZ=200.0, Lsd=1_000_000.0,\n"
        "    BC_y=1024.0, BC_z=1024.0, RhoD=120.0,\n"
        "    RMin=10.0, RMax=120.0, RBinSize=1.0,\n"
        "    EtaMin=-180.0, EtaMax=180.0, EtaBinSize=10.0,\n"
        ")\n"
        "base_spec = spec_from_v1_params(base_params, requires_grad=False)\n"
        "\n"
        "print(f'truth Lsd range: {Lsd_truth.min():.0f} – {Lsd_truth.max():.0f} μm  '\n"
        "      f'(span = {Lsd_truth.max() - Lsd_truth.min():.0f} μm)')\n"
        "print(f'truth BC_y span: {BCy_truth.max() - BCy_truth.min():.2f} px')\n"
        "print(f'truth BC_z span: {BCz_truth.max() - BCz_truth.min():.2f} px')\n"
        "print(f'anchor frames: {anchor_idx}')\n"
        "print(f'sample frames: {len(sample_idx)} frames')"
    ),
    _md(
        "## Fit a spline-of-time through the anchors\n"
        "\n"
        "`fit_drift_trajectory` accepts `(anchor_frames_dict, sample_frame_indices, base_spec)` and returns a `DriftTrajectory` with `Lsd_t / BC_y_t / BC_z_t` "
        "arrays for every frame index, plus matching `sigma_*` arrays from the Laplace-approximation step. The recovered trajectory interpolates the anchors "
        "smoothly; in production the sample frames also contribute a ring-residual soft constraint."
    ),
    _code(
        "drift = fit_drift_trajectory(\n"
        "    calibrant_anchor_frames=anchor_frames,\n"
        "    sample_frame_indices=sample_idx,\n"
        "    base_spec=base_spec,\n"
        "    parametrization='spline',\n"
        "    n_knots=5,\n"
        "    bayesian_sigma=True,\n"
        ")\n"
        "\n"
        "frames_out = drift.frame_indices\n"
        "rec = np.stack([drift.Lsd_t, drift.BC_y_t, drift.BC_z_t], axis=1)\n"
        "sigmas = np.stack([drift.sigma_Lsd, drift.sigma_BC_y, drift.sigma_BC_z], axis=1)\n"
        "print(f'recovered shape: {rec.shape}  (frames, [Lsd, BC_y, BC_z])')\n"
        "\n"
        "fig, axs = plt.subplots(3, 1, figsize=(8.5, 7), sharex=True)\n"
        "labels = ['Lsd (μm)', 'BC_y (px)', 'BC_z (px)']\n"
        "truths = [Lsd_truth, BCy_truth, BCz_truth]\n"
        "for i, (ax, label, truth) in enumerate(zip(axs, labels, truths)):\n"
        "    ax.plot(t, truth, color='#1f3a93', lw=1.4, label='truth')\n"
        "    ax.plot(frames_out, rec[:, i], color='#16a085', lw=1.4, ls='--', label='recovered')\n"
        "    ax.fill_between(frames_out,\n"
        "                     rec[:, i] - sigmas[:, i],\n"
        "                     rec[:, i] + sigmas[:, i],\n"
        "                     color='#16a085', alpha=0.20, label='±σ (Laplace)')\n"
        "    ax.scatter([k for k in anchor_idx], [truth[k] for k in anchor_idx],\n"
        "                color='#c0392b', s=40, zorder=5, label='anchors')\n"
        "    ax.set_ylabel(label)\n"
        "    ax.grid(alpha=0.3)\n"
        "    if i == 0:\n"
        "        ax.legend(loc='lower right')\n"
        "axs[-1].set_xlabel('frame index (operando time)')\n"
        "plt.tight_layout(); plt.show()\n"
        "\n"
        "rmse = np.sqrt(((rec - np.stack(truths, axis=1)[frames_out]) ** 2).mean(axis=0))\n"
        "print(f'\\nrecovered-vs-truth RMSE: Lsd = {rmse[0]:.1f} μm, '\n"
        "      f'BC_y = {rmse[1]:.3f} px, BC_z = {rmse[2]:.3f} px')\n"
        "print(f'mean Laplace σ:          Lsd = {sigmas[:, 0].mean():.1f} μm, '\n"
        "      f'BC_y = {sigmas[:, 1].mean():.3f} px, BC_z = {sigmas[:, 2].mean():.3f} px')"
    ),
    _md(
        "## What to point out in the meeting\n"
        "\n"
        "- **The headline:** sub-pixel BC recovery across a 60-frame operando scan from 7 anchor calibrant runs. Lsd recovered to ~tens of μm out of a 1 m baseline — the same fractional precision your downstream Rietveld pass expects.\n"
        "- **The paper hook:** the σ on every recovered parameter at every frame index comes from the Laplace approximation on the loss Hessian. Nobody has published a rigorous Bayesian methodology for per-frame calibration drift with quantitative uncertainty on Lsd / BC. JSR / JAC methods note territory.\n"
        "- **What we need from the Wenqian group:** one real long operando scan with periodic calibrant anchors (CeO₂ / LaB₆ / Si standards at frame 0, midpoint, end is plenty). Your data, your sample, your beamtime; we run the fit and you interpret the recovered drift trajectory in light of your beamline's known thermal behaviour.\n"
        "- **Composability:** the same `DriftTrajectory` plugs into `pipelines/drift.py::integrate_with_drift_correction` so every per-frame integration uses its own per-frame geometry. No re-run of the integration per anchor."
    ),
]


def build_notebook(cells: list) -> dict:
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3 (ipykernel)",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    for i, cell in enumerate(nb["cells"]):
        cell["id"] = f"cell-{i}"
    return nb


def main():
    NOTEBOOKS.mkdir(parents=True, exist_ok=True)
    for fname, cells in [
        ("16_learnable_gain_recovery.ipynb", LEARNABLE_GAIN_CELLS),
        ("18_empty_subtraction_refinable_scale.ipynb", EMPTY_SUBTRACTION_CELLS),
        ("19_fit_drift_trajectory_operando.ipynb", FIT_DRIFT_CELLS),
    ]:
        out_path = NOTEBOOKS / fname
        nb = build_notebook(cells)
        with open(out_path, "w") as f:
            json.dump(nb, f, indent=1)
        print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
