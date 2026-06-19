# External eval path

The included external JSONL fixture is source-backed public-domain data derived
from online public-domain texts. It is enough to validate the benchmark harness
and provide a license-clear external path, but stronger claims should still
prefer a separately curated eval set.

Use `configs/benchmark/external_eval_vllm_template.yaml` as the starting point.
The default points at `data/eval/external_public_domain_eval.jsonl`, a 500-row
fixture with source provenance, repeated shared-prefix groups, balanced task
types, and both prompt layouts:

```bash
make validate-external-eval
make vllm-external-eval
```

To replace the fixture:

1. Create a JSONL file with the schema in [eval_datasets.md](eval_datasets.md).
2. Include repeated shared documents so cache locality is measurable.
3. Balance `qa`, `json`, `summary`, and `code` tasks.
4. Keep `document_id` / `shared_prefix_id`, `tenant_id`, and `trust_group_id`
   stable and intentional.
5. Record provenance and license for every source.
6. Point `workload.dataset_path` at the new file and run validation before
   serving.

`make validate-external-eval` is intentionally stricter than the smoke tests. It
requires at least 500 rows, all four task types, both prompt layouts, balanced
task counts, repeated shared-prefix groups, tenant/trust-group fields, source
provenance, and public-domain licenses for the included fixture.

For multi-model comparisons, use
`configs/benchmark/model_family_vllm_template.yaml`. Each model family needs
compatible specialist and multitask adapters served as distinct vLLM model
names. Do not compare adapter quality across model families unless each family
has adapters trained on the same SFT protocol and evaluated on the same held-out
records.
