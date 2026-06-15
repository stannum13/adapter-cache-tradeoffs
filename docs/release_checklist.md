# Release checklist

Use this before pushing a public repository.

## Public surface

- README describes the project as a cache/routing benchmark harness, not a paper
  with model-quality conclusions.
- Generated reports, figures, tables, and run artifacts are ignored by git.
- Public docs use relative links only.
- `AGENTS.md`, local planning notes, credentials, and machine-specific paths are
  not tracked.

## Eval path

- Validate the source-backed eval bundle:

  ```bash
  make validate-source-eval
  ```

- Run the CPU mock path for systems sanity:

  ```bash
  make source-eval
  ```

- For real model quality, run the vLLM-backed source eval:

  ```bash
  make vllm-source-eval
  ```

- If you do not have vLLM running, use the local Hugging Face causal LM smoke
  path:

  ```bash
  make transformers-source-eval
  ```

- Do not commit generated local results as claims unless the exact serving stack,
  model, adapters, prompts, seeds, and hardware are documented.

## Verification

```bash
uv sync --extra dev
uv run pytest tests -q
uv run ruff check .
uv run ruff format . --check
```

Optional serving tests stay opt-in:

```bash
RUN_VLLM_TESTS=1 uv run pytest tests/test_optional_integrations.py -q
```

## Clean public history

This repo currently has no remote configured. To publish with clean history,
push a squashed public branch rather than the local build-out history:

```bash
git switch public/main
git remote add origin git@github.com:<owner>/adapter-cache-tradeoffs.git
git push -u origin public/main:main
```

If a remote already exists, inspect it first with `git remote -v` and avoid
force-pushing over shared work without coordination.
