# midas-grain-odf

Differentiable per-grain orientation distribution function (ODF) inversion
from far-field HEDM data. Closes the gap left open by Nygren et al. 2019
(*IOP Conf. Ser. MSE* **580** 012018) — recovers the intensity-weighted
ODF, not just the GOE envelope.

See [dev/implementation_plan.md](dev/implementation_plan.md) for the
design rationale and roadmap.

## Status

Alpha. Synthetic plant-and-recover validation only. Real-data path under
construction.

## Install

```bash
cd packages/midas_grain_odf
pip install -e .[dev]
```

The package depends on the differentiable HEDM forward model in
`fwd_sim/hedm_forward.py`. That module is loaded by relative import; ensure
`MIDAS/fwd_sim` is on `PYTHONPATH` when using the package outside the
MIDAS repo.

## Notebooks

Worked-example Jupyter notebooks live in `notebooks/`. They are **not shipped with `pip install`** — get them by cloning the [MIDAS repository](https://github.com/marinerhemant/MIDAS/tree/master/packages/midas_grain_odf/notebooks).

## Quickstart (synthetic)

```bash
cd packages/midas_grain_odf
python -m pytest tests/test_synth_particle.py -xvs
```

## Public API

```python
from midas_grain_odf import (
    ParticleODF, BinghamMixtureODF, VoxelGridODF,
    fit_grain_odf,
)
```
