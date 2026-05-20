"""Builds notebook 16: LearnableGain spatial-gain-drift recovery.

Self-contained version of the dev/paper/runners/run_learnable_gain_demo.py
demo: plants a smooth 5% Gaussian spatial gain field on a synthetic ring
frame and trains a ``LearnableGain`` module to reverse it, all inline using
the public ``midas_integrate_v2`` API. No runner import, no C binaries, CPU
only.
"""
from pathlib import Path
from _nb_helper import write_notebook


CELLS = [
    ("md", """\
# 16 — `LearnableGain`: auto-recover spatial gain drift across a long scan

Long overnight operando scans accumulate a smooth **spatial gain drift** on
the area detector — thermal expansion in the readout electronics,
sensor-temperature variation, and slow scintillator degradation. pyFAI does
not model this, so the drift leaks into the integrated I(Q) and shows up as
fake intensity changes in the operando trace.

`LearnableGain` is an `nn.Module` whose per-pixel gain field is a torch
parameter. Trained jointly with the integration kernel, it absorbs the drift
into a multiplicative correction so the integrated profile stays clean.

This notebook is **self-contained** — it plants a known gain field on a
synthetic frame and recovers it inline with the public API, on CPU. No C
binaries, no external data.
"""),
    ("code", """\
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch
from scipy.ndimage import gaussian_filter

from midas_integrate.params import IntegrationParams
from midas_integrate_v2 import (
    LearnableGain, gain_smoothness_prior, gain_unity_prior,
    integrate_with_corrections, spec_from_v1_params,
)

torch.set_default_dtype(torch.float64)
"""),
    ("md", """\
## 1. Synthesize a ring frame and a planted gain field

The clean frame is a stack of six concentric Debye–Scherrer rings with a
little Poisson-ish read noise. The planted gain `gain_truth` is a 5%
Gaussian-smooth random field (σ = 20 px) — exactly the kind of slow spatial
drift we want to recover.
"""),
    ("code", """\
NY = NZ = 256
rng = np.random.default_rng(0)

BC_y, BC_z = NY / 2.0, NZ / 2.0
Y, Z = np.meshgrid(np.arange(NY), np.arange(NZ), indexing="xy")
R = np.sqrt((Y - BC_y) ** 2 + (Z - BC_z) ** 2)

img = np.zeros((NZ, NY))
for r0 in np.linspace(20, 110, 6):
    img += 1000.0 * np.exp(-((R - r0) / 1.4) ** 2)
img += rng.normal(0.0, 5.0, size=img.shape)
img = np.clip(img, 0, None)

raw = rng.normal(0.0, 1.0, size=(NZ, NY))
smoothed = gaussian_filter(raw, sigma=20.0)
smoothed /= np.max(np.abs(smoothed)) + 1e-30
gain_truth = 1.0 + 0.05 * smoothed          # ~+/-5% smooth drift

img_drifted = img * gain_truth
print(f"frame {img.shape}, planted gain range "
      f"[{gain_truth.min():.3f}, {gain_truth.max():.3f}]")
"""),
    ("md", """\
## 2. Integration spec

`spec_from_v1_params` turns a v1 `IntegrationParams` into the differentiable
v2 spec consumed by `integrate_with_corrections`. The **clean-frame**
integration is our training target; the gain module has to make the drifted
frame integrate to the same profile.
"""),
    ("code", """\
p = IntegrationParams(
    NrPixelsY=NY, NrPixelsZ=NZ,
    pxY=200.0, pxZ=200.0, Lsd=1_000_000.0,
    BC_y=BC_y, BC_z=BC_z, RhoD=120.0,
    RMin=10.0, RMax=120.0, RBinSize=1.0,
    EtaMin=-180.0, EtaMax=180.0, EtaBinSize=10.0,
)
spec = spec_from_v1_params(p, requires_grad=False)

img_t = torch.as_tensor(img, dtype=torch.float64)
img_drift_t = torch.as_tensor(img_drifted, dtype=torch.float64)
target = integrate_with_corrections(img_t, spec).detach()
print("target profile shape:", tuple(target.shape))
"""),
    ("md", """\
## 3. Train `LearnableGain` to reverse the drift

We divide the drifted frame by the current gain estimate, integrate, and
minimise the mismatch against the clean target. Two priors keep the solution
well-posed:

- `gain_unity_prior` — pulls the mean gain toward 1 (fixes the global scale
  gauge that integration alone can't pin down),
- `gain_smoothness_prior` — penalises high spatial frequency, matching the
  smooth physical drift.
"""),
    ("code", """\
gain = LearnableGain(NrPixelsZ=NZ, NrPixelsY=NY, scale=0.1)
optim = torch.optim.Adam(gain.parameters(), lr=0.02)

for step in range(40):
    optim.zero_grad()
    adjusted = img_drift_t / gain().clamp(min=1e-6)
    out = integrate_with_corrections(adjusted, spec)
    loss = (out - target).pow(2).mean()
    loss = (loss
            + 1e-4 * gain_unity_prior(gain)
            + 1e-3 * gain_smoothness_prior(gain))
    loss.backward()
    optim.step()
    if step % 10 == 0:
        print(f"  step {step:3d}  loss={float(loss):.6e}")

gain_recovered = gain().detach().numpy()
"""),
    ("md", "## 4. How close did we get?"),
    ("code", """\
rmse = float(np.sqrt(np.mean((gain_recovered - gain_truth) ** 2)))
amp = float(gain_truth.max() - gain_truth.min())
print(f"recovered-vs-truth RMSE : {rmse:.4e}")
print(f"planted drift amplitude : {amp:.4e}")
print(f"RMSE as %% of amplitude : {100 * rmse / amp:.2f} %%")
assert rmse < 0.05, "gain recovery did not converge"
"""),
    ("md", """\
## 5. Why this matters

- The truth gain is a 5% Gaussian-smooth random field (σ = 20 px); the
  learned field tracks it to a small fraction of the planted amplitude.
- The learned gain is applied as a **multiplicative per-pixel correction** to
  every subsequent frame — the integration kernel itself never changes.
- This is the operando overnight-scan story: without correction, a 1% gain
  drift would masquerade as a 1% intensity change in the time-resolved
  profile. pyFAI / calib2 calibrate geometry but not a per-pixel gain field.

To turn the recovered field into figures or a report, the
`dev/paper/runners/run_learnable_gain_demo.py` script wraps this exact logic
and writes `gain_recovered.npy`, `gain_truth.npy`, and a `REPORT.md` under
`dev/paper/runs/learnable_gain_demo/`.
"""),
]


def main():
    out = Path(__file__).parent / "16_learnable_gain_recovery.ipynb"
    write_notebook(out, CELLS)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
