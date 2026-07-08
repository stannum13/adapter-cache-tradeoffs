# Release checklist

Use this before pushing a public repository.

## Public surface

- README describes the project as a cache/routing benchmark harness, not a paper
  with model-quality conclusions.
- Generated reports, figures, tables, and run artifacts are ignored by git.
- Public docs use relative links only.
- `AGENTS.md` is tracked intentionally as public contributor guidance when
  present.
- Local private planning notes, credentials, cloud keys, caches, virtualenvs,
  raw artifacts, and machine-specific paths are not tracked.
- Legacy runs missing current lifecycle files are either rerun, explicitly
  grandfathered with scoped wording, or excluded from claim-supporting bundles.

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

Publish with a squashed public branch rather than the local build-out history:

```bash
public_commit=$(printf 'Initial public release\n' | git commit-tree HEAD^{tree})
git branch -f public/main "$public_commit"
git push --force-with-lease origin public/main:main
```

Inspect `git remote -v` first and avoid force-pushing over shared work without
coordination.
