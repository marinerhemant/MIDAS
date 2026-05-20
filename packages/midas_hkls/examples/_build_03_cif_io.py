"""Builds example notebook 03: CIF read / write round-trip."""
from pathlib import Path
from _nb_helper import write_notebook


CELLS = [
    ("md", """\
# 03 — CIF input / output

`midas_hkls.read_cif` / `write_cif` move crystal structures between CIF files
and the in-memory `Crystal` object (lattice + space group + asymmetric-unit
atoms). The reader prefers `gemmi` and falls back to `pycifrw`; the writer
prefers `gemmi` and falls back to a pure-Python emitter.

This needs the **`[cif]`** extra (`gemmi`) or **`[cif-pure]`** (`pycifrw`). If
neither is installed, the notebook documents the API and stops.

The example is self-contained: we build a structure in code, write it to a
temporary CIF, read it back, and confirm the round-trip — no external files.
"""),
    ("code", """\
import importlib.util

HAVE_GEMMI = importlib.util.find_spec("gemmi") is not None
HAVE_PYCIFRW = importlib.util.find_spec("CifFile") is not None
HAVE_CIF = HAVE_GEMMI or HAVE_PYCIFRW
print(f"gemmi: {HAVE_GEMMI}   pycifrw: {HAVE_PYCIFRW}")
if not HAVE_CIF:
    print("Install with: pip install 'midas-hkls[cif]' — feature deferred.")
"""),
    ("md", """\
## Build a structure in code

CeO₂ — fluorite, space group 225 (Fm-3m), a = 5.4112 Å — Ce at (0,0,0) and
O at (¼,¼,¼). Lengths in Å, angles in degrees per the package convention.
"""),
    ("code", """\
from midas_hkls import Atom, Crystal, Lattice, SpaceGroup

sg = SpaceGroup.from_number(225)
lat = Lattice.for_system("cubic", a=5.4112)
ceo2 = Crystal(
    lattice=lat,
    space_group=sg,
    atoms=[
        Atom("Ce", (0.0, 0.0, 0.0), B_iso=0.30),
        Atom("O", (0.25, 0.25, 0.25), B_iso=0.50),
    ],
    name="ceo2",
)
print(f"{ceo2.name}: SG {ceo2.space_group.number}, a = {ceo2.lattice.a} Å, "
      f"{len(ceo2.atoms)} asymmetric-unit atoms")
print(f"full unit cell expands to {len(ceo2.unit_cell_atoms())} atoms")
"""),
    ("md", "## Write to a CIF file"),
    ("code", """\
import tempfile
from pathlib import Path
from midas_hkls import write_cif, read_cif

if HAVE_CIF:
    tmp = Path(tempfile.mkdtemp())
    cif_path = tmp / "ceo2.cif"
    write_cif(ceo2, cif_path)
    text = cif_path.read_text()
    print(f"wrote {cif_path}  ({len(text)} bytes)")
    # Show the cell + first atom lines so the format is visible
    for line in text.splitlines():
        if any(k in line for k in ("data_", "_cell_length_a", "_space_group",
                                    "_symmetry_Int", "Ce", "O ")):
            print("   ", line.strip())
else:
    print("(deferred — no CIF backend)")
"""),
    ("md", """\
## Read it back and verify the round-trip

The parsed structure must reproduce the space group, lattice, element list,
fractional coordinates, occupancy, and B-factors we started from.
"""),
    ("code", """\
if HAVE_CIF:
    rt = read_cif(cif_path)
    print(f"SG  {rt.space_group.number} (was {ceo2.space_group.number})")
    print(f"a   {rt.lattice.a:.4f} Å (was {ceo2.lattice.a})")
    print(f"elements {sorted(a.element for a in rt.atoms)}")

    assert rt.space_group.number == ceo2.space_group.number
    assert abs(rt.lattice.a - ceo2.lattice.a) < 1e-4
    assert len(rt.atoms) == len(ceo2.atoms)
    for a, b in zip(sorted(ceo2.atoms, key=lambda x: x.element),
                    sorted(rt.atoms, key=lambda x: x.element)):
        assert a.element == b.element
        for u, v in zip(a.fract, b.fract):
            assert abs(u - v) < 1e-4
        assert abs(a.B_iso - b.B_iso) < 1e-3
    print("round-trip OK")
else:
    print("(deferred — no CIF backend)")
"""),
    ("md", """\
## Straight into the HKL / structure-factor pipeline

A `Crystal` read from CIF feeds the rest of `midas_hkls` directly — here we
generate its reflection list.
"""),
    ("code", """\
if HAVE_CIF:
    from midas_hkls import generate_hkls
    refs = generate_hkls(rt.space_group, rt.lattice,
                         wavelength_A=0.173, two_theta_max_deg=12.0)
    print(f"{len(refs)} reflections to 2θ = 12°; first few:")
    for r in refs[:5]:
        print(f"   ({r.h}{r.k}{r.l})  d = {r.d_spacing:.4f} Å  "
              f"2θ = {r.two_theta_deg:.3f}°  mult = {r.multiplicity}")
else:
    print("(deferred — no CIF backend)")
"""),
    ("md", """\
That closes the loop: structures move losslessly through CIF and drop straight
into HKL generation and structure-factor calculation. The reader and writer
each have a pure-Python fallback, so CIF I/O still works without gemmi as long
as `pycifrw` is present.
"""),
]


def main():
    out = Path(__file__).parent / "03_cif_io.ipynb"
    write_notebook(out, CELLS)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
