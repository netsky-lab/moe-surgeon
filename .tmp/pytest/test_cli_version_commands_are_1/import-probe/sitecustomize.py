from __future__ import annotations

import atexit
import json
import os
import sys


def _dump_imports() -> None:
    output_path = os.environ["MOE_SURGEON_IMPORT_PROBE_OUTPUT"]
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(sorted(sys.modules), handle)


atexit.register(_dump_imports)
