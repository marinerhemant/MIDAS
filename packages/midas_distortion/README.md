# midas-distortion

Canonical MIDAS radial-distortion model — the single source of truth for the
detector distortion layout and the v1↔v2 coefficient mapping, shared by:

- **midas-calibrate-v2** — fits the distortion (v2 canonical names).
- **midas-peakfit** — applies it to spot geometry (numpy).
- **midas-transforms** — applies it to spot geometry (torch).

## Model

Multiplicative factor on the projected radius `R` (with `ρ = R / RhoD`,
`η' = 90° − η`):

```
D(ρ, η) = 1
    + iso_R2·ρ² + iso_R4·ρ⁴ + iso_R6·ρ⁶              (isotropic)
    + a1·ρ⁴·cos( η' + phi1)                           (1-fold; ρ⁴ is a v1 quirk)
    + a2·ρ²·cos(2η' + phi2)
    + a3·ρ³·cos(3η' + phi3)
    + a4·ρ⁴·cos(4η' + phi4)
    + a5·ρ⁵·cos(5η' + phi5)
    + a6·ρ⁶·cos(6η' + phi6)
```

The legacy v1 ordering (`p0..p14`, phases scattered) is `v1_term_layout()`;
v1↔v2 reindexing is exact (`v1_to_v2_coeffs` / `v2_to_v1_coeffs`).

## Backend-agnostic kernel

`distortion_factor(R_norm, eta_deg, p_coeffs, terms=...)` dispatches `cos` /
`ones_like` on the input's own array library, so numpy and torch consumers
evaluate bit-for-bit the same model (up to floating-point reassociation).

```python
import numpy as np
from midas_distortion import distortion_factor, v1_term_layout, v1_to_v2_coeffs

p_v1 = np.zeros(15)          # legacy paramstest p0..p14
D = distortion_factor(R/RhoD, eta_deg, p_v1, terms=v1_term_layout())
# …or convert once and use the v2 layout (default):
D = distortion_factor(R/RhoD, eta_deg, v1_to_v2_coeffs(p_v1))
```
