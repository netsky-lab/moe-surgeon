# Known Gotchas

- 2026-04-17: [Define canonical data contracts and JSON serialization schema] `python -m moe_surgeon` is not executable and exits `No module named moe_surgeon.__main__`, so the requested `python -m moe_surgeon`-style import/run path is currently broken (`src/moe_surgeon/__init__.py` exists but no `__main__.py`).