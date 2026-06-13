# Agent Notes

- Use `uv` for dependency management and commands.
- Run `uv run pytest tests -q` before final handoff.
- Keep GPU and real serving tests optional.
- Write typed, testable code.
- Every benchmark must produce JSONL logs and summary JSON.
- Every policy/cache model must have unit tests.
- Do not require internet or GPU for unit tests.
- Real vLLM integration stays behind optional flags; skip integration tests unless `RUN_VLLM_TESTS=1`.

