midas_defect
============

Phase-agnostic diffuse-scattering defect metrology on top of standard FF-HEDM.
Drives a 16-module pipeline from indexed grains and a sparse voxel cloud to
per-grain dislocation density, twin-shear projection, energy partition, and a
master inventory CSV with full bootstrap UQ retained.

The package targets FCC, BCC, and HCP, and is :doc:`differentiable-friendly
<api/index>` so each step can be re-used in CPFEM coupling (:mod:`midas_defect.cpfem`).

Quick start
-----------

.. code-block:: python

   from midas_defect.variants import assign_variants_kmeans
   from midas_defect.types import CrystalPhase

   out = assign_variants_kmeans(OM, n_variants=2, phase=CrystalPhase.FCC)
   labels = out["labels"]

See :doc:`notebooks` for end-to-end walk-throughs:

* ``01_quickstart``     -- 30-line minimum-viable pipeline
* ``02_full_pipeline``  -- every module in dependency order
* ``03_bootstrap_uq``   -- composing CIs across analyses
* ``04_advanced``       -- MK closure, energy balance, Schmid stratification

Module map
----------

.. toctree::
   :maxdepth: 2
   :caption: Foundation

   api/types
   api/bootstrap
   api/phases

.. toctree::
   :maxdepth: 2
   :caption: Per-grain analyses

   api/variants
   api/stress
   api/strain
   api/schmid
   api/energy
   api/spatial
   api/gnd
   api/asterism
   api/polytype
   api/line_profile
   api/debye_waller
   api/distributions
   api/thermodynamics
   api/reports

.. toctree::
   :maxdepth: 2
   :caption: Coupling + tutorials

   api/cpfem
   notebooks

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
