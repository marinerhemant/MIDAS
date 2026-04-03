# Track A: Differentiable Forward Model & Optimization

## What It Does

A fully differentiable PyTorch forward model that predicts diffraction spot positions from crystallographic orientations, positions, and lattice parameters. Supports all three HEDM modalities (NF, FF, pf-HEDM). Validated pixel-exact against the C executables `ForwardSimulationCompressed` and `simulateNF`.

## Files

| File | Purpose |
|------|---------|
| `fwd_sim/hedm_forward.py` | Core forward model (`HEDMForwardModel`) |
| `fwd_sim/hedm_losses.py` | Loss functions (NCC, L2, spot matching) |
| `fwd_sim/single_grain_optimization_ff.py` | Complete FF optimization demo |

## Prerequisites

```bash
source /path/to/miniconda3/bin/activate midas_env
# Needs: torch, numpy
# For HKL generation: MIDAS build (GetHKLList executable)
```

---

## Step 1: Prepare HKL List

The forward model needs the HKL reflections for your material. Generate them from your parameter file:

```python
import subprocess
import numpy as np
import torch
from pathlib import Path

MIDAS_BIN = Path("/Users/hsharma/opt/MIDAS/build/bin")
DEG2RAD = 3.14159265358979323846 / 180.0

# Option A: From an existing MIDAS parameter file
work_dir = Path("/path/to/your/analysis")
subprocess.run([str(MIDAS_BIN / "GetHKLList"), str(work_dir / "params.txt")],
               cwd=str(work_dir), check=True)

# Option B: Create a minimal parameter file
with open(work_dir / "params_hkl.txt", "w") as f:
    f.write("LatticeParameter 4.08 4.08 4.08 90 90 90\n")
    f.write("Wavelength 0.172979\n")
    f.write("SpaceGroup 225\n")
    f.write("Lsd 1000000\n")       # doesn't matter for HKL generation
    f.write("MaxRingRad 500000\n")  # include all rings
subprocess.run([str(MIDAS_BIN / "GetHKLList"), str(work_dir / "params_hkl.txt")],
               cwd=str(work_dir), check=True)

# Parse hkls.csv
data = np.loadtxt(work_dir / "hkls.csv", skiprows=1)
hkls_cart = torch.tensor(data[:, 5:8], dtype=torch.float64)  # g1, g2, g3
thetas    = torch.tensor(data[:, 8] * DEG2RAD, dtype=torch.float64)  # Theta in rad
hkls_int  = torch.tensor(data[:, 0:3], dtype=torch.float64)  # h, k, l integers
print(f"Loaded {hkls_cart.shape[0]} HKL reflections")
```

---

## Step 2: Create the Forward Model

```python
from hedm_forward import HEDMForwardModel, HEDMGeometry

# --- FF-HEDM geometry (from your parameter file) ---
geometry = HEDMGeometry(
    Lsd=1_000_000.0,       # sample-detector distance in um (from params.txt)
    y_BC=1024.0,           # beam center y in pixels
    z_BC=1024.0,           # beam center z in pixels
    px=200.0,              # pixel size in um
    omega_start=0.0,       # degrees
    omega_step=0.25,       # degrees
    n_frames=1440,         # = 360 / omega_step
    n_pixels_y=2048,
    n_pixels_z=2048,
    min_eta=6.0,           # exclude polar regions (degrees)
    wavelength=0.172979,   # Angstroms
    flip_y=True,           # True for FF/PF, False for NF
)

# --- NF-HEDM geometry (multi-distance) ---
geometry_nf = HEDMGeometry(
    Lsd=[8289.15, 10290.72],        # list of distances
    y_BC=[985.42, 985.16],          # list of beam centers
    z_BC=[17.51, 24.51],
    px=1.48,
    omega_start=180.0,
    omega_step=-0.25,
    n_frames=1440,
    n_pixels_y=2048,
    n_pixels_z=2048,
    min_eta=6.0,
    wavelength=0.172979,
    flip_y=False,                    # NF convention
)

model = HEDMForwardModel(
    hkls=hkls_cart,
    thetas=thetas,
    geometry=geometry,           # or geometry_nf
    hkls_int=hkls_int,          # needed for strain (CorrectHKLsLatC)
)
```

---

## Step 3: Forward Simulate Spots

### From Euler angles (most common)

```python
# Single grain
euler_rad = torch.tensor([[0.5, 0.3, 0.7]], dtype=torch.float64)  # (1, 3) radians
position  = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float64)  # (1, 3) micrometers

spots = model(euler_rad, position)  # returns SpotDescriptors

# Extract spot coordinates
from hedm_forward import HEDMForwardModel
coords, valid = HEDMForwardModel.predict_spot_coords(spots, space="angular")
# coords: (1, 2, M, 3) = [2theta, eta, omega] in radians
# valid:  (1, 2, M)     = 1 for valid spots, 0 otherwise

# Get valid spots as flat array
mask = valid.squeeze() > 0.5
valid_spots = coords.squeeze()[mask]  # (N_valid, 3)
print(f"{valid_spots.shape[0]} valid spots predicted")
```

### With strain (lattice parameters)

```python
# Lattice params: [a, b, c, alpha_deg, beta_deg, gamma_deg]
latc = torch.tensor([4.082, 4.079, 4.081, 90.01, 89.99, 90.02],
                     dtype=torch.float64)

spots = model(euler_rad, position, lattice_params=latc)
```

### From existing MIDAS output (consolidated_Output.h5)

```python
import h5py

with h5py.File("consolidated_Output.h5", "r") as f:
    grp = f["grains"]
    for gid in grp:
        g = grp[gid]
        orient = g["orientation"][()]         # (3, 3) rotation matrix
        euler  = g["euler_angles"][()]        # (3,) radians
        pos    = g["position"][()]            # (3,) micrometers
        latc   = g["lattice_params_fit"][()]  # (6,) [a,b,c,alpha,beta,gamma]
        radius = float(g["radius"][()])

        euler_t = torch.tensor(euler, dtype=torch.float64).unsqueeze(0)
        pos_t   = torch.tensor(pos, dtype=torch.float64).unsqueeze(0)
        latc_t  = torch.tensor(latc, dtype=torch.float64)

        spots = model(euler_t, pos_t, lattice_params=latc_t)
        coords, valid = HEDMForwardModel.predict_spot_coords(spots, "angular")
        n_valid = (valid > 0.5).sum().item()
        print(f"Grain {gid}: {n_valid} predicted spots")
```

---

## Step 4: Optimize Orientation and Strain

### Setup: observed spots from MIDAS SpotMatrixGen.csv

```python
import numpy as np

# After running ForwardSimulationCompressed with WriteSpots=1:
spot_data = np.loadtxt("SpotMatrixGen.csv", skiprows=1, delimiter="\t")
# Columns: GrainID, SpotID, Omega, DetHor, DetVert, OmeRaw, Eta, RingNr, ...
# Column 10: Theta, Column 2: Omega

# For a specific grain:
grain_mask = spot_data[:, 0] == 1  # grain ID = 1
grain_spots = spot_data[grain_mask]
obs_2theta = grain_spots[:, 10] * DEG2RAD  # Theta -> 2*Theta
obs_eta    = grain_spots[:, 6] * DEG2RAD
obs_omega  = grain_spots[:, 2] * DEG2RAD
obs_coords = torch.tensor(
    np.column_stack([2 * obs_2theta, obs_eta, obs_omega]),
    dtype=torch.float64
)
```

### Three-phase optimization

```python
from hedm_losses import SpotMatchingLoss

# Resolution-based weights
sigma_2th = 0.5 * px / Lsd
sigma_eta = 0.5 / 500.0  # typical ring radius ~500 pixels
sigma_ome = 0.25 * abs(omega_step) * DEG2RAD
weights = torch.tensor([sigma_2th, sigma_eta, sigma_ome], dtype=torch.float64)
weights = weights / weights.mean()
loss_fn = SpotMatchingLoss(metric="l2", weights=weights)

# Starting point (from MIDAS indexing output)
opt_euler = torch.tensor(initial_euler_rad, dtype=torch.float64, requires_grad=True)
opt_latc  = torch.tensor(initial_latc, dtype=torch.float64)

def closure():
    if opt_euler.grad is not None: opt_euler.grad.zero_()
    if opt_latc.requires_grad and opt_latc.grad is not None: opt_latc.grad.zero_()
    spots = model(opt_euler.unsqueeze(0), pos.unsqueeze(0), lattice_params=opt_latc)
    coords, valid = HEDMForwardModel.predict_spot_coords(spots, "angular")
    pred = coords.squeeze().reshape(-1, 3)
    v = valid.squeeze().reshape(-1)
    pred_valid = pred[v > 0.5]
    dists = torch.cdist(obs_coords, pred_valid)
    min_d, nn = dists.min(dim=1)
    keep = min_d < 0.5
    loss = loss_fn(pred_valid[nn[keep]], obs_coords[keep])
    loss.backward()
    return loss

# Phase 1: Orientation only
opt_latc.requires_grad_(False)
optimizer = torch.optim.LBFGS([opt_euler], lr=1.0, max_iter=20,
                               line_search_fn="strong_wolfe")
for step in range(15):
    optimizer.step(closure)

# Phase 2: Strain only
opt_euler.requires_grad_(False)
opt_latc.requires_grad_(True)
optimizer = torch.optim.LBFGS([opt_latc], lr=1.0, max_iter=20,
                               line_search_fn="strong_wolfe")
for step in range(15):
    optimizer.step(closure)

# Phase 3: Joint
opt_euler.requires_grad_(True)
optimizer = torch.optim.LBFGS([opt_euler, opt_latc], lr=0.5, max_iter=20,
                               line_search_fn="strong_wolfe")
for step in range(10):
    optimizer.step(closure)

print(f"Optimized Euler: {opt_euler.detach() / DEG2RAD}")
print(f"Optimized Lattice: {opt_latc.detach()}")
```

---

## Step 5: Run the Complete Demo

```bash
cd /Users/hsharma/opt/MIDAS/fwd_sim
python single_grain_optimization_ff.py           # both clean and noisy
python single_grain_optimization_ff.py --clean   # noise-free only
python single_grain_optimization_ff.py --noisy   # noisy only
```

---

## API Reference

### HEDMGeometry

| Parameter | Type | Description |
|-----------|------|-------------|
| `Lsd` | float or list[float] | Sample-detector distance(s) in um |
| `y_BC` | float or list[float] | Beam center Y in pixels |
| `z_BC` | float or list[float] | Beam center Z in pixels |
| `px` | float | Pixel size in um |
| `omega_start` | float | Starting omega in degrees |
| `omega_step` | float | Omega step in degrees |
| `n_frames` | int | Number of omega frames |
| `n_pixels_y` | int | Detector Y size |
| `n_pixels_z` | int | Detector Z size |
| `min_eta` | float | Minimum eta in degrees |
| `wavelength` | float | X-ray wavelength in Angstroms |
| `flip_y` | bool | True for FF/PF, False for NF |

### HEDMForwardModel

| Method | Inputs | Returns |
|--------|--------|---------|
| `forward(euler, pos, latc)` | `(N,3)`, `(N,3)`, `(6,)` | `SpotDescriptors` |
| `predict_spot_coords(spots, space)` | SpotDescriptors, "angular"/"detector" | `(coords, valid)` |
| `predict_images(spots, nF, nY, nZ)` | SpotDescriptors, grid dims | `(nF, nY, nZ)` tensor |
| `euler2mat(angles)` | `(..., 3)` radians | `(..., 3, 3)` orthogonal matrix |
| `correct_hkls_latc(latc)` | `(..., 6)` | `(hkls_cart, thetas)` |
| `orthogonalize(R)` | `(..., 3, 3)` | `(..., 3, 3)` SO(3)-projected |

### SpotDescriptors fields

| Field | Shape | Description |
|-------|-------|-------------|
| `omega` | `(..., K, M)` | Omega angle in radians |
| `eta` | `(..., K, M)` | Eta angle in radians |
| `two_theta` | `(..., K, M)` | 2*theta in radians |
| `y_pixel` | `(..., K, M)` or `(D,...,K,M)` | Detector Y pixel |
| `z_pixel` | `(..., K, M)` or `(D,...,K,M)` | Detector Z pixel |
| `frame_nr` | `(..., K, M)` | Omega frame number |
| `valid` | `(..., K, M)` | 1=valid, 0=filtered |

K = 2*N (two omega solutions per position), M = number of HKL reflections.
