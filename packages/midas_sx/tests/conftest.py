"""Put tests/ and tests/support/ on sys.path: the tests import the independent simulator (`import simulate as sim`) and, in one case, another test module."""
import sys
from pathlib import Path

_here = Path(__file__).resolve().parent
for _p in (_here / "support", _here):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
