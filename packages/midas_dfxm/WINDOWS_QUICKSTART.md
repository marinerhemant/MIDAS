# midas-dfxm on Windows — quickstart

`midas-dfxm` is public on PyPI (current version 0.3.2). The install route below was
validated 2026-07-31 against 0.1.0 by a clean-environment install + run from public
PyPI (pure Python; the only compiled dependency is PyTorch, which ships Windows CPU
wheels). No GPU, no CUDA, no C compiler, no admin rights needed.

## 1. Python
Install Python 3.9–3.12 (python.org) or Miniconda. Then make an isolated env:

```bat
py -m venv dfxm_env
dfxm_env\Scripts\activate
python -m pip install --upgrade pip
```
(conda: `conda create -n dfxm python=3.11 && conda activate dfxm`)

## 2. Install
One command, straight from PyPI:

```bat
pip install "midas-dfxm[viz]"
```

That pulls PyTorch (CPU), the six MIDAS sibling packages (`midas-stress`,
`midas-hkls`, `midas-2d`, `midas-defect`, `midas-invert`, `midas-distortion`),
and matplotlib (the `[viz]` extra), all from PyPI. Verified in a clean venv:
torch 2.x, numpy 2.x, matplotlib 3.x, all at their released versions,
`midas-dfxm` 0.1.0.

## 3. One environment variable (set once)
PyTorch and the MIDAS siblings can both load an OpenMP runtime; on some machines
that aborts at import. Set this to be safe:

```bat
set KMP_DUPLICATE_LIB_OK=TRUE        :: this shell only
setx KMP_DUPLICATE_LIB_OK TRUE       :: persist for future shells
```

## 4. Where the examples live
From 0.3.2 the tutorials ship inside the package (`midas_dfxm.examples`), so
there is nothing extra to download — run them as modules with `python -m`.

Figures are written to a `figures\` folder in whatever directory you run from.
Set `MIDAS_FIGDIR` to send them elsewhere.

## 5. Run the 60-second demo
```bat
python -m midas_dfxm.examples.demo_quickstart
```
Runs in ~2 s on CPU. It builds a curved+strained crystal, renders a realistic
DFXM image, and recovers the full nine-component deformation-gradient field,
printing the round-trip error (max |dF| ~3e-6) and saving `demo_quickstart.png`:
a DFXM image, the orientation channel (what a center-of-mass mosaicity scan
gives), and the planted-vs-recovered strain map (the extra information the
differentiable inverse adds).

For the NX-school, `tutorial_school_dfxm.py` opens as an interactive notebook in
VSCode (Run Cell on each `# %%`) with no Jupyter setup, and also runs as a plain
script — find it next to the others under `midas_dfxm\examples\` in site-packages,
or run it as a module. Two more go deeper:
```bat
python -m midas_dfxm.examples.tutorial_field_forward
python -m midas_dfxm.examples.tutorial_dislocation_typing
```

## 6. Confirm the install (optional, from the cloned repo)
```bat
pip install pytest
python -m pytest tests -q
```

## Smoke test (one-liner)
```bat
python -c "import torch, midas_dfxm; from midas_dfxm import stroh_dislocation; print('midas-dfxm', midas_dfxm.__version__, 'OK')"
```
