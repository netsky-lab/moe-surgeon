from __future__ import annotations

import os
from pathlib import Path
import tempfile

from sample import answer


def test_answer() -> None:
    assert answer() == 42


def test_bootstrap_tempdir() -> None:
    expected = Path(os.environ["TMPDIR"])
    assert Path(tempfile.gettempdir()) == expected
    assert expected.name in {"system", "pytest"}
    assert expected.is_dir()
