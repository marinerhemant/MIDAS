# Track C: EM Spot-Ownership Model

## What It Does

Replaces hard (binary) spot-to-grain assignment with soft probabilistic ownership using Expectation-Maximization, analogous to RELION's approach in cryo-EM.

**Current MIDAS behavior:** Each spot is assigned to at most one grain (max-intensity rule in `findSingleSolutionPFRefactored.c`). At grain boundaries, this creates jagged, noisy boundaries. Shared spots (e.g., twin-related reflections) are removed from BOTH sinograms, losing signal.

**EM approach:** Each spot gets an ownership probability for every nearby grain. Shared spots are SPLIT proportionally between grains (preserving total intensity). Noise spots are heavily downweighted by distance-based ownership. Grain orientations are optionally refined via gradient descent in the M-step.

## Pipeline Integration

When `-useEM 1` is passed to `pf_MIDAS.py` (requires `-doTomo 1`):

1. Indexing runs as normal
2. `findSingleSolutionPFRefactored` finds unique grain orientations
3. **Pre-tomo refinement** (FitOrStrainsScanningOMP) sharpens per-voxel orientations
4. Refined orientations are propagated to unique grain set
5. **EM spot-ownership** runs: E-step (soft assignment) + M-step (orientation refinement)
6. **Weighted sinograms** are generated from EM ownership probabilities
7. Tomo reconstruction runs on the improved sinograms
8. Standard mic-seeded re-indexing and final refinement follow

## Files

| File | Purpose |
|------|---------|
| `fwd_sim/em_spot_ownership.py` | `EMSpotOwnership` class (E-step, M-step, angular wrapping, ring filtering) |
| `FF_HEDM/workflows/em_pf_integration.py` | Pipeline integration: data loading, weighted sinogram generation |
| `FF_HEDM/workflows/pf_MIDAS.py` | Main driver (modified: `-useEM` flag and EM insertion point) |
| `fwd_sim/hedm_forward.py` | Forward model (used internally by EM) |
| `fwd_sim/tests/test_em_spot_ownership.py` | Unit tests for EM model |

## Quick Start (CLI)

```bash
# Standard pf-HEDM with EM-weighted sinograms and MLEM tomo:
pf_MIDAS.py -paramFile params.txt -nCPUs 32 \
  -doTomo 1 -useEM 1 -reconMethod mlem -mlemIter 50

# EM parameters (all optional, showing defaults):
#   -emIter 10           # EM iterations
#   -emSigmaInit 0.02    # Initial Gaussian width (radians)
#   -emSigmaMin 0.005    # Annealing floor
#   -emSigmaDecay 0.9    # Sigma decay per iteration
#   -emRefineOrientations 1  # 1=full EM with M-step, 0=E-step only
#   -emOptSteps 5        # Gradient steps per M-step
#   -emLR 0.005          # Adam learning rate for M-step
```

## Prerequisites

```bash
source /path/to/miniconda3/bin/activate midas_env
# Needs: torch, numpy
# Needs: MIDAS build for HKL generation
# Input: observed spots from pf-HEDM or FF-HEDM analysis
```

---

## Conceptual Overview

```
E-step:  For each (spot, voxel) pair:
           P(spot s ← voxel v) ∝ exp(-||predicted_v - observed_s||² / 2σ²)
         Normalize so each spot's probabilities sum to 1.

M-step:  For each voxel:
           Optimize orientation to minimize:
             Σ_s  P(s←v) · ||predicted_v(s) - observed_s||²
         (weighted least squares, a few gradient steps per voxel)

Anneal:  σ decreases each iteration (sharp assignments emerge)

Repeat for n_iter iterations.
```

---

## Step 1: Prepare Observed Spots

### From MIDAS FF-HEDM output (consolidated peaks)

```python
import numpy as np
import torch
import sys
sys.path.insert(0, "/Users/hsharma/opt/MIDAS/fwd_sim")

DEG2RAD = 3.14159265358979323846 / 180.0

# Load spots from SpotMatrixGen.csv (if ForwardSimulationCompressed ran with WriteSpots=1)
spot_data = np.loadtxt("SpotMatrixGen.csv", skiprows=1, delimiter="\t")
# Columns: 0=GrainID, 2=Omega, 6=Eta, 10=Theta

obs_2theta = spot_data[:, 10] * DEG2RAD * 2  # Theta -> 2*Theta
obs_eta    = spot_data[:, 6] * DEG2RAD
obs_omega  = spot_data[:, 2] * DEG2RAD
obs_spots  = torch.tensor(
    np.column_stack([obs_2theta, obs_eta, obs_omega]),
    dtype=torch.float64,
)
print(f"Loaded {obs_spots.shape[0]} observed spots")
```

### From pf-HEDM peak search output

```python
# pf-HEDM stores spots in InputAllExtraInfoFittingAll.csv or similar
# Columns typically include: SpotID, Omega, Eta, Ttheta, YCen, ZCen, ...
# Adapt the column indices to your specific output format:

peak_data = np.loadtxt("InputAllExtraInfoFittingAll.csv", delimiter=",", skiprows=1)
obs_spots = torch.tensor(
    np.column_stack([
        peak_data[:, 3] * DEG2RAD,  # 2theta column (check your format)
        peak_data[:, 4] * DEG2RAD,  # eta column
        peak_data[:, 2] * DEG2RAD,  # omega column
    ]),
    dtype=torch.float64,
)
```

---

## Step 2: Prepare Initial Orientations and Positions

### From FF-HEDM indexing output

```python
import h5py

# Read grain orientations from consolidated output
with h5py.File("consolidated_Output.h5", "r") as f:
    grp = f["grains"]
    grain_ids = sorted(grp.keys())
    euler_list = []
    pos_list = []
    for gid in grain_ids:
        g = grp[gid]
        euler_list.append(g["euler_angles"][()])  # radians
        pos_list.append(g["position"][()])         # micrometers

initial_euler = torch.tensor(np.array(euler_list), dtype=torch.float64)  # (N, 3)
positions     = torch.tensor(np.array(pos_list), dtype=torch.float64)    # (N, 3)
print(f"Loaded {initial_euler.shape[0]} grain orientations")
```

### From pf-HEDM voxel grid

```python
# If you have per-voxel orientations from a previous pf-HEDM reconstruction:
# (e.g., from the .mic file or HDF5 output)
# Format: each row = [x, y, euler1, euler2, euler3]
voxel_data = np.loadtxt("reconstruction.mic", skiprows=4)
positions = torch.tensor(voxel_data[:, 3:5], dtype=torch.float64)  # (N, 2) x,y
positions = torch.nn.functional.pad(positions, (0, 1))  # pad z=0 -> (N, 3)
initial_euler = torch.tensor(voxel_data[:, 7:10], dtype=torch.float64)  # (N, 3) radians
```

---

## Step 3: Create the Forward Model

```python
from hedm_forward import HEDMForwardModel, HEDMGeometry
# (same as Track A -- see track_a_forward_model.md for details)

geometry = HEDMGeometry(
    Lsd=1_000_000.0, y_BC=1024.0, z_BC=1024.0, px=200.0,
    omega_start=0.0, omega_step=0.25, n_frames=1440,
    n_pixels_y=2048, n_pixels_z=2048, min_eta=6.0,
    wavelength=0.172979,
)

model = HEDMForwardModel(hkls=hkls_cart, thetas=thetas, geometry=geometry)
```

---

## Step 4: Run EM

```python
from em_spot_ownership import EMSpotOwnership

em = EMSpotOwnership(
    forward_model=model,
    sigma_init=0.02,     # initial Gaussian width (radians, ~1 degree)
    sigma_min=0.005,     # floor (~0.3 degrees)
    sigma_decay=0.9,     # anneal by 10% each iteration
    max_distance=0.1,    # only consider spots within 0.1 rad (~6 degrees)
)

final_euler, ownership = em.fit(
    obs_spots=obs_spots,            # (S, 3) observed spot coordinates
    initial_euler=initial_euler,    # (N, 3) starting orientations
    positions=positions,            # (N, 3) voxel/grain positions
    n_iter=10,                      # EM iterations
    n_opt_steps=10,                 # gradient steps per M-step per voxel
    lr=0.005,                       # Adam learning rate for M-step
    verbose=True,
)

# final_euler: (N, 3) refined Euler angles
# ownership:   (S, N) final spot-to-voxel probabilities
```

Output looks like:
```
  EM iter   0: sigma=0.0200, assigned=162/168, avg_confidence=0.9523
  EM iter   1: sigma=0.0180, assigned=165/168, avg_confidence=0.9701
  EM iter   2: sigma=0.0162, assigned=168/168, avg_confidence=0.9940
  ...
```

---

## Step 5: Analyze Results

### Check orientation improvement

```python
from hedm_forward import HEDMForwardModel
RAD2DEG = 180.0 / 3.14159265358979323846

for i in range(initial_euler.shape[0]):
    R_init = HEDMForwardModel.euler2mat(initial_euler[i])
    R_final = HEDMForwardModel.euler2mat(final_euler[i])
    trace = torch.trace(R_init.T @ R_final)
    change_deg = torch.acos(torch.clamp((trace - 1) / 2, -1, 1)) * RAD2DEG
    print(f"  Grain {i}: orientation changed by {change_deg.item():.4f} deg")
```

### Inspect spot ownership at boundaries

```python
# Find spots with split ownership (shared between grains)
max_ownership = ownership.max(dim=1).values
ambiguous = (max_ownership < 0.8) & (max_ownership > 0.1)
print(f"\nAmbiguous spots (ownership < 80%): {ambiguous.sum().item()}")

# For each ambiguous spot, show which grains share it:
for s in torch.where(ambiguous)[0][:10]:
    probs = ownership[s]
    top2 = torch.topk(probs, 2)
    print(f"  Spot {s.item()}: grain {top2.indices[0].item()} "
          f"({top2.values[0]:.2f}), grain {top2.indices[1].item()} "
          f"({top2.values[1]:.2f})")
```

### Save results

```python
import numpy as np

np.savetxt("em_refined_euler.csv", final_euler.numpy(),
           header="phi1_rad Phi_rad phi2_rad", delimiter="\t")
np.save("em_ownership.npy", ownership.numpy())
```

---

## Tuning Parameters

| Parameter | Default | When to change |
|-----------|---------|----------------|
| `sigma_init` | 0.02 | Increase for heavily deformed materials (broader peaks) |
| `sigma_min` | 0.005 | Decrease for high-precision final assignment |
| `sigma_decay` | 0.9 | Slower (0.95) for complex microstructures |
| `max_distance` | 0.1 | Increase if grains are poorly indexed initially |
| `n_iter` | 10 | More for many-grain problems |
| `n_opt_steps` | 5-10 | More for larger perturbations |
| `lr` | 0.005 | Decrease if M-step oscillates |

## Limitations

- Currently processes voxels sequentially (not batched). For >1000 voxels, this will be slow. GPU batching is a Phase 2 improvement.
- Requires all observed spots in memory. For very large datasets (>1M spots), consider spatial partitioning.
- The M-step uses Adam with nearest-neighbor matching (not fully soft). A fully soft M-step is possible but more expensive.
