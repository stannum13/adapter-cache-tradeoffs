# Agent Instructions

- Use `uv` for dependency management and command execution.
- Run `uv run pytest tests -q` before finalizing code changes.
- Run `uv run ruff check .` and `uv run ruff format . --check` before finalizing.
- Keep GPU, vLLM, and external model-server tests optional.
- Do not require internet or GPU for unit tests.
- Write typed, testable code with focused unit tests for every new policy, cache model, runner, or parser.
- Every benchmark runner must produce JSONL request logs, `summary.json`, `config_resolved.yaml`, and `manifest.json`.
- Keep raw run artifacts, generated reports, cloud keys, caches, and virtualenvs out of git.
- Prefer reproducible configs under `configs/benchmark/` over ad hoc command lines.
- Use causal-transformer terminology in public docs except when historical context requires otherwise.
