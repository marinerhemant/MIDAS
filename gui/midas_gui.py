#!/usr/bin/env python3
"""MIDAS unified GUI launcher.

Thin entry-point so users can run::

    python gui/midas_gui.py [data_directory]

The application code lives in ``gui/midas_app/``.
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from midas_app.main import main  # noqa: E402

if __name__ == '__main__':
    sys.exit(main())
