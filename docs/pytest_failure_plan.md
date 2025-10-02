# Pytest Failure Remediation Plan (2024-XX-XX)

All 18 collection errors stem from Python not being able to import `src.*` modules when pytest runs. The failures hit every module under `src/utils`, `src/sim`, `src/rl`, etc., so nothing past collection executes.

## Root Cause Hypothesis

Pytest is running without the project root on `sys.path`, so namespace imports like `src.utils.config_loader` fail. This also causes pytest to discover stray tests under `bin/` because the usual `testpaths = ["tests"]` configuration is being bypassed once import errors trigger.

## Remediation Steps

1. **Stabilise the import path**
   - Ensure the repo root is appended to `sys.path` early in test collection. The simplest fix is to add a `tests/conftest.py` that does:
     ```python
     import sys
     from pathlib import Path
     ROOT = Path(__file__).resolve().parents[1]
     if str(ROOT) not in sys.path:
         sys.path.insert(0, str(ROOT))
     ```
   - Alternative/complimentary: add a lightweight `src/__init__.py` that sets `__all__` and confirms the directory is treated as a regular package (namespace packages can be brittle across tooling).

2. **Tighten pytest discovery**
   - Explicitly exclude `bin/` (and other non-test dirs) via `norecursedirs` in `pytest.ini` to prevent accidental collection once imports succeed.
   - Verify `pytest.ini` is actually being loaded (run `pytest --show-config` after the path fix).

3. **Re-run pytest**
   - After the import path guard is in place, rerun `pytest -m "not slow"` to surface the next layer of failures (if any).
   - Update the plan once new errors appear, focusing on functional regressions introduced by recent feature work.

## Open Questions

- Why did pytest bypass the existing `testpaths` setting? Confirm once imports succeed.
- Do we need to update developer docs (`README.md` / `PROJECT_STATUS.md`) to mention sourcing the virtualenv or installing `-e .` before running tests?

## Next Actions

- [ ] Implement the path guard (`tests/conftest.py`) and optional `src/__init__.py` sanity import.
- [ ] Update `pytest.ini` to set `norecursedirs`.
- [ ] Rerun pytest and iterate on the resulting failures.

