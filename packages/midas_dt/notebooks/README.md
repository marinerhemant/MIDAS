# midas-dt notebooks

| # | File | Topic |
|---|---|---|
| 0 | `00_calibration.ipynb` | detector geometry from a calibrant — **do this first** |
| 1 | `01_dt_recon_walkthrough.ipynb` | XRD-CT scan → per-voxel maps, start to finish |

Start at `00`. Notebook `01` asks you to type in `Lsd`, the beam centre, the
pixel size and the wavelength; `00` is where those come from, and it writes a
`calibration.json` that `01` reads so you never retype a number.

Both notebooks run on macOS and Linux. Calibration used to segfault on macOS —
numba's OpenMP threading layer colliding with torch's — and that is **fixed in
`midas-integrate`**, which now picks numba's `workqueue` threadpool there. On an
older install, run with `NUMBA_THREADING_LAYER=workqueue`.

The notebook **generates its own synthetic scan on first run**, so it works on
any machine with no beamline data. Step through it once to see what each stage
should look like, then set `USE_DEMO_DATA = False` in §1 and point it at your
own files. Everything below §1 runs unchanged.

It ships with its outputs saved, so you can read it without running anything.

## Running it

```bash
pip install "midas-dt[direct]"      # [direct] adds torch: branch C, SIRT/TV, absorption
pip install jupyterlab matplotlib
jupyter lab                         # open 01_dt_recon_walkthrough.ipynb
```

From a source checkout instead of a pip install, the setup cell puts the
sibling packages on `sys.path` for you — launch Jupyter from this directory.

On macOS, if importing dies with an OpenMP error, `export
KMP_DUPLICATE_LIB_OK=TRUE` before launching. Conda ships more than one
`libomp` and the reconstruction engine links its own.

## Two things it will not decide for you

`midas-dt`'s defaults come from APS 1-ID, and two of them are **wrong for other
instruments and fail silently** — a clean reconstruction of the wrong thing,
with no error:

- **`NEGATE_OMEGA`** — the sign of the rotation angle. Wrong ⇒ you reconstruct
  a mirror image, which looks entirely reasonable.
- **`DROP_FIRST_FRAME`** — whether the detector writes a throwaway frame at the
  start of each acquisition. Wrong ⇒ every projection is one angular step off.

§2 prints the frames-per-file count so you can settle the second one from the
data. For the first, ask your beamline scientist which way the stage turns
looking along the beam. The notebook says this too, at the point where it
matters.

## Is XRD-CT the right technique at all?

Only if the diffraction rings are **continuous**. Coarse grains make them a
string of spots, the azimuthal integral stops being meaningful, and the answer
is scanning-3DXRD (`midas_index` PF mode + `midas_pf_odf`) rather than this.
§4 measures it on your own frames instead of assuming, and masks detector
module gaps first — counting those as azimuthal structure makes *every* ring
look spotty.

## Files

- `00_calibration.ipynb` — detector calibration; recovers a planted geometry to
  **1.1 ppm in Lsd and 0.003 px in beam centre**, and quotes a real
  single-detector run that reproduces an archived fit to 33 ppm / 0.015 px
- `01_dt_recon_walkthrough.ipynb` — the walkthrough
- `_demo.py` — writes the synthetic scan and the synthetic calibrant. Deliberately emits **real raw files**
  (header + contiguous int32 frames) rather than handing back arrays, so the
  notebook exercises the same reader path your data takes. A demo that bypasses
  the reader proves nothing about the reader.

Running the notebook creates `dt_output/` here (demo data, maps, figures). It
is gitignored and safe to delete; re-running regenerates it.

## Beyond the notebook

- `../README.md` — the conventions table, the three branches, and the full
  known-limits ledger
- `../scripts/` — the diagnostic scripts used against real beamline data
  (`look_at_frame.py`, `inspect_u3o8_lineout.py`, `index_u3o8_rings.py`)
- `../examples/run_dt_recon.py` — the same workflow as a batch script
- `midas-dt --help` — the command-line equivalent
