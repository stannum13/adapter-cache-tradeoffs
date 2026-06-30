# Evidence Bundles

Evidence bundles are machine-readable manifests for a selected set of benchmark run
directories under `artifacts/runs`. They are intended to make a claim auditable without
requiring raw run artifacts to be committed.

Build a bundle with:

```bash
make evidence-bundle BUNDLE=release-candidate RUNS="run-a run-b" \
  REPORTS="docs/release_report.md" FIGURES="docs/figures/quality_vs_p95_ttft.png"
```

By default the output is `evidence/<bundle-name>/bundle_manifest.json`. Use
`OUTPUT=/path/to/bundle` to write somewhere else, or `RUN_GLOBS="large-model-*"` to
select runs by directory-name pattern. If no `RUNS` or `RUN_GLOBS` are provided, all
immediate child directories under `artifacts/runs` are recorded.

Each run entry records:

- The run ID and source run directory.
- Presence of `summary.json`, `config_resolved.yaml`, and `manifest.json`.
- SHA256 and size for those included evidence files when present.
- Run-level git metadata from `manifest.json` when available.
- Raw artifacts such as `requests.jsonl` and backend metric scrapes as explicitly
  excluded. The bundle generator does not copy those files.

Generated reports and figures can be listed with `REPORTS=...` and `FIGURES=...`; their
paths, hashes, and sizes are recorded when they exist.
