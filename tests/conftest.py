from __future__ import annotations

import importlib
from pathlib import Path
import sys


_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_ROOT = _REPO_ROOT / "src"

if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

# The shared environment can point editable installs at sibling worktrees.
# Force pytest collection in this checkout to resolve `moe_surgeon` from this
# repo's own `src/` tree instead of whichever editable install was last active.
for module_name in tuple(sys.modules):
    if module_name == "moe_surgeon" or module_name.startswith("moe_surgeon."):
        del sys.modules[module_name]

importlib.invalidate_caches()
