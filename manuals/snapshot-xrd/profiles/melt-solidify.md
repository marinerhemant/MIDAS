# Profile -- laser melting / solidification series

> Part of the snapshot-xrd doc set. Spine: [`../README.md`](../README.md).

Typical acquisition: a fixed beam on a sample while a laser passes; hundreds to thousands of frames
at a few hundred Hz; a single matrix phase dominates; minor phases are sparse.

- **Traces:** the halo index rises when the illuminated volume melts and falls on solidification;
  several excursions mean several melt events. The matrix scale jumps with the hot solid next to the
  melt and relaxes during cooling; it does not return to the pre-event value within seconds.
- **Windows:** before = frames prior to the first excursion (minus a guard); after = the end of the
  series. Exclude everything between.
- **Matrix:** use every allowed matrix line to the detector edge for classification; the per-window
  fit uses the strongest low-angle lines.
- **Minor phases:** populations are sparse and local (they can concentrate at a few sample
  positions); test per series and pooled, and leave-one-series-out. A population that is present
  before and absent after can mean dissolution in the melt pool or removal from the beam; the
  diffraction alone does not tell which.
- **Candidates:** from the alloy's element set, one source and one rule; include phases that share a
  structure type with each other only as one cell family (they are not separable by pattern).
