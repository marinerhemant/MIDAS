#!/bin/bash
# Self-test of the kit without data: syntax of every script, imports of the helpers, the environment, and a tiny synthetic run of the line finder.
#   export EXP=<experiment>; source kit_env.sh; ./selftest.sh      (EXP may be the _template: only syntax and imports are checked then)
set -u
export PYTHONDONTWRITEBYTECODE=1
KIT=$(cd "$(dirname "$0")" && pwd); fail=0
while IFS= read -r -d '' f; do bash -n "$f" || { echo "SYNTAX FAIL $f"; fail=1; }; done < <(find "$KIT" -name '*.sh' -print0)
P=${MIDAS_ENV:-/home/beams12/S1IDUSER/opt/envs/midas}/bin; PY=$P/python; [ -x "$PY" ] || PY=$(command -v python3)
while IFS= read -r -d '' f; do $PY -B -I -c "import ast,sys; ast.parse(open(sys.argv[1]).read())" "$f" || { echo "SYNTAX FAIL $f"; fail=1; }; done < <(find "$KIT" -name '*.py' -print0)
$PY -B -I - <<PYE || fail=1
import sys; sys.path.insert(0, "$KIT/tools")
import numpy as np
for m in ("h5py", "scipy", "numpy", "midas_hkls", "matplotlib", "pandas"):
    __import__(m)
import phase_hunt as ph
rng = np.random.default_rng(0); rad = np.r_[rng.uniform(150, 1440, 20000), rng.normal(500, 0.4, 80)]   # planted line at 500 px on a flat background
lines, *_ = ph.hist_lines(rad)
assert any(abs(l[0] - 500) < 1.0 for l in lines), "the line finder missed a planted 80-spot line on 20000 background clusters"
n, a = ph.best_cubic(np.array([3.482, 2.970, 2.843, 5.687, 2.011]), "F")
assert n >= 4 and abs(a - 9.85) < 0.1, (n, a)      # Li6PS5Cl (220)(311)(222)(111)(422), F lattice
print("imports ok; line finder and cubic indexing behave on synthetic input")
PYE
[ $fail = 0 ] && echo "SELFTEST PASS" || { echo "SELFTEST FAIL"; exit 1; }
