"""Make `scripts/` importable-agnostic: ensure the repo root is on sys.path.

Import this at the top of each script (``import _bootstrap``) so scripts run
from anywhere without an editable install.
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
