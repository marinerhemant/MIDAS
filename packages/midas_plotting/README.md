# midas-plotting

Standard plots for MIDAS reconstructions.

```python
from midas_plotting import read_mic, orientation_map, compare_maps

m = read_mic("Ce5Y_mr.2.mic")
print(m.summary())
orientation_map(m, space_group=225, cmin=0.3)
```

```bash
midas-plot Ce5Y.0.mic Ce5Y_sum3thr2.0.mic --kind orientation --cmin 0.3 \
    --titles "baseline|sum3+thr2" -o compare.png
```

## Why

IPF colouring, `.mic` parsing and map plotting had been re-implemented in
several one-off analysis scripts, each with its own conventions. Two things that
kept going wrong and are now handled in one place:

- **Euler→RGB is not an orientation map.** Two orientations a fraction of a
  degree apart can produce very different Euler triplets near gimbal lock, so a
  single grain renders as several colours. `ipf_rgb` colours by the crystal
  direction along a sample axis instead.
- **A permissive confidence cut fills the whole grid.** The fit returns *an*
  orientation for every voxel it evaluates, so plotting at C ≥ 0.1 shows
  plausible microstructure whether or not material is there. `orientation_map`
  annotates the figure when asked to plot below `TRUST_FLOOR` (0.3).

Symmetry operators come from `midas_stress`; nothing is hand-listed here.

Implemented Laue families: cubic (SG 195–230) and hexagonal (168–194).
Anything else raises rather than silently falling back to cubic.

## Far-field (`Grains.csv`)

```python
from midas_plotting import ff, read_grains

g = read_grains("Grains.csv")
print(len(g), g.space_group)        # symmetry is read from the file's header

ff.summary(g)                        # one-page overview
ff.grain_map(g, color="ipf")         # IPF-coloured grain centres
ff.ipf_legend(g.space_group)         # the colour key
ff.pole_figure(g, hkl=(1, 1, 1))
ff.strain_map(g, kind="vonmises")
```

```bash
midas-plot Grains.csv --kind summary -o overview.png
midas-plot Grains.csv --kind pole --hkl 1,1,1
midas-plot Grains.csv --kind strain --strain-kind hydrostatic
```

FF output is a **grain list**, not a voxel grid, so these are scatter and
distribution plots. They are namespaced under `ff` rather than exported flat
because both modalities have a `grain_map` and they mean different things:
`maps.grain_map` labels a near-field voxel grid, `ff.grain_map` scatters
far-field grain centres.

Things the module will not let you get wrong:

* **Symmetry comes from the file.** `Grains.csv` states its space group in the
  preamble; the plots use it. Defaulting to cubic would colour a hexagonal
  sample with the wrong IPF triangle and produce a plausible, wrong figure.
* **Columns are read by name.** `Grains.csv` has 47 columns and
  `midas-fit-grain` 0.5.6 shipped a cyclic rotation of three of them; a
  positional reader inherits that silently.
* **Euler angles are cross-checked against `O11..O33`.** They describe the same
  orientation, so disagreement means the row is being sliced wrong — you get a
  warning instead of a wrong colour.
* **Strain is already microstrain.** The `eFab`/`eKen` columns are not
  dimensionless; they are not rescaled.

Two caveats the plots cannot fix: FF grain positions are good to ~100 µm (not
the six decimals the file prints), and `GrainRadius` is only correct with
`midas-process-grains >= 0.6.1`.
