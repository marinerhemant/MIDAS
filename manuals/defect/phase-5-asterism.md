# Phase 5 — asterism, dislocation density, sub-grains

Needs the Bragg/diffuse split from `phase-3` to know which voxels are "near-Bragg".

## Asterism

```python
from midas_defect.asterism_fit import (predict_hkl_positions, fit_asterism_patches,
                                       fit_single_patch, strain_tensor_from_centroids)
```

Per-hkl patches of extended intensity hugging each Bragg core, encoding the dislocation
strain field and the orientation gradient. Fitted, not thresholded.

## Dislocation density — relative here, not absolute

```python
from midas_defect.williamson_hall import (williamson_hall, modified_williamson_hall,
                                          contrast_factors_for_fits,
                                          dislocation_density_per_grain)
from midas_defect.contrast_factor import ...        # cubic
from midas_defect.contrast_factor_hex import ...    # hexagonal
```

Modified Williamson–Hall on radial breadth, with a contrast factor.

**Read `ENVELOPE.md` §8 before quoting a number.** The package uses a cubic `H²` anisotropy
correction in `q_U`; the reference re-analysis used a per-grain radial-breadth fit, and the
two do **not** agree in absolute density. The real-data regression therefore compares
**relative** quantities — matrix/twin ratio, `q_U` ordering — and so should you, until the
contrast factor is pinned against an independent method.

Ratios and orderings between populations measured the same way: supported. Absolute m⁻²: not.

## Sub-grains

```python
from midas_defect.subgrain import decompose_asterism_patches
```

DBSCAN per hkl patch; each sub-cluster is a coherent sub-grain whose orientation deviates
slightly from the average `U`. **Cross-hkl matching is not implemented** — this is a
diagnostic, not a sub-grain list, and a "sub-grain" here is a per-hkl cluster that has not
been shown to be the same object in another reflection.

Two cautions carried from real data:

* At permissive thresholds a splitting detector fires on nearly every reflection, which
  includes noise. Ask how often a **non-reflection** position "splits" before quoting a
  population rate.
* A split is separated **azimuthally at constant |G|**, not radially. A radial spread is a
  d-spacing spread — strain or dispersion — not a misorientation.

## Burgers population

```python
from midas_defect.burgers_population import (burgers_type_parameters,
                                             solve_burgers_population)
```

Decomposition into Burgers-vector types. Check the conditioning before promising a
population: a poorly separated basis returns a confident split of an unidentifiable mixture.

## Peak shape and size

```python
from midas_defect.peakshape import (fit_size_strain_mosaic, crystallite_size_from_width,
                                    recover_size_distribution)
```

Same rule as the rods: any size from a width is a **lower bound** while the mosaic is in the
measurement. `ENVELOPE.md` §2.

## Detector artifacts

```python
from midas_defect.detector_qc import flag_fixed_pixel_artifacts
```

Run it. A fixed-pixel artifact is coherent across frames and therefore survives every
statistical test built to reject noise.
