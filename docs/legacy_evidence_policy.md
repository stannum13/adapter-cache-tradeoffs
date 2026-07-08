# Legacy Evidence Policy

This policy covers benchmark runs created before the current runner lifecycle
contract required every run to write:

- `requests.jsonl`
- `summary.json`
- `config_resolved.yaml`
- `manifest.json`
- `status.json`

Runs that are missing `status.json` or other lifecycle files are legacy runs.
They can remain useful as historical context, but they are not complete evidence
under the current runner contract. Evidence-bundle validation uses the
non-raw evidence files: `summary.json`, `config_resolved.yaml`, `manifest.json`,
and `status.json`.

## Ground Rules

- Do not backfill or fabricate missing lifecycle files for historical runs.
- Do not treat a legacy run as a fully validated evidence-bundle entry unless
  the required non-raw evidence-bundle files are present.
- Keep raw artifacts under `artifacts/runs/`; do not commit request logs, metric
  scrapes, model outputs, adapters, caches, or local generated reports.
- Scope public claims to the files that actually exist: summaries, resolved
  configs, manifests, run docs, server notes, and the bundle validation report.
- Use causal-transformer wording in public docs unless a historical file name or
  serving tool requires more specific terminology.

## Decision Path

For every legacy run considered for a public claim, choose exactly one path.

### Rerun

Use when the result is claim-critical, feasible to repeat, or needed for a new
release frontier. Replace the historical result with a current-contract run and
require strict bundle validation for the selected evidence.

### Grandfather

Use when the result is historical context, rerun cost is not justified yet, and
enough provenance exists to explain the stack and limits. Cite it only as
scoped historical evidence and state the missing lifecycle-file limitation.
Strict bundle validation still reports the run incomplete.

### Exclude

Use when provenance is insufficient, the result cannot be interpreted cleanly,
or the claim would depend on missing raw context. Leave it out of public
evidence bundles and claim tables.

Grandfathering is a documentation decision, not a validator bypass. A
grandfathered run can be described with limitations, but `--strict` evidence
bundle validation should still fail if required files are missing.

## Strict-Mode Semantics

Evidence bundle strict mode validates the required non-raw bundle files plus
the selected generated reports, figures, and tables. It records raw artifacts
as excluded rather than copying or requiring them. Strict mode is suitable for
release candidates, new evidence, and bridge runs that should be treated as
claim-supporting.

```bash
make public-evidence-bundle STRICT=1
uv run acb bundle --bundle-name public-review --run-glob 'source-eval-*' --strict
```

If selected legacy runs are missing `status.json`, strict mode writes the
manifest and then exits nonzero. That output is a completeness report, not a
fully validated evidence bundle. Use it to decide whether to rerun, grandfather,
or exclude each legacy result.

For new claim-supporting evidence, strict validation should pass before the
result is promoted into public claims:

```bash
make evidence-bundle \
  BUNDLE=vllm-bridge-reset-g8 \
  RUN_GLOBS="vllm-bridge-reset-g8*" \
  STRICT=1
```

## Public Wording

Use wording like:

> Historical runs are scoped to available summaries, manifests, run docs, and
> server notes because they predate the `status.json` lifecycle contract.

Avoid wording like:

> The `public-review` bundle fully validates all historical benchmark evidence.

The correct public boundary is: legacy results can motivate hypotheses and
document prior observations, but new release claims should rely on reruns or
strictly validated evidence bundles.
