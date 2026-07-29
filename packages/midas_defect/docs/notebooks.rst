Notebooks
=========

Four runnable tutorials in ``notebooks/`` walk through the package end-to-end.
Each uses the synthetic Cu-Al-like fixture from ``tests/conftest.py``, so they
run in a few seconds with no external data dependency.

01 - Quickstart
---------------

``notebooks/01_quickstart.ipynb`` -- 30-line minimum-viable pipeline:
load orientations + voxels, run K-medoids variant assignment, compute
Schmid factors, fit modified-WH dislocation density.

02 - Full pipeline
------------------

``notebooks/02_full_pipeline.ipynb`` -- every published module in
dependency order, ending with a master inventory CSV and the four
publication-figure scaffolds.

03 - Bootstrap UQ
-----------------

``notebooks/03_bootstrap_uq.ipynb`` -- three composition patterns:
custom ``stat_fn``, cross-analysis paired ratios, ``n_boot`` tuning.

04 - Advanced
-------------

``notebooks/04_advanced.ipynb`` -- Mecking-Kocks variant ``k_2`` ratio,
energy-balance closure, Schmid tercile stratification, Taylor visible
fraction.
