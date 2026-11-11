Run outputs land here by default (`bench run spec.yaml --output-dir results`).

Each run writes:

- `results/<spec-name>/manifest.json` — run identity, hardware, config, completed rung ids (used to resume)
- `results/<spec-name>/results.jsonl` — one row per measured rung

These files are gitignored. Whether long-term results should be committed here, stored in another repo, or kept only on the bench host is a TODO for the operator.
