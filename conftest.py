"""Make ``src`` importable from tests without a full package install.

pytest picks this file up before collecting tests, so putting the repo
root on ``sys.path`` here lets ``from src.core.evaluation import ...``
work on CI (where the runnable scripts' ``sys.path.insert(0, ".")`` does
not run before pytest imports the test modules).
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
