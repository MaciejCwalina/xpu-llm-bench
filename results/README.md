Run outputs land here by default (`bench run spec.yaml --output-dir results`).

Each run writes, under `results/<spec-name>/`:

- `manifest.json` — run identity, hardware, config, site config used, launch check, hold state, and `completed_rung_ids` (used by `--resume`). `status` is `in_progress`, `completed`, `aborted`, or `refused` (preflight / bench-hold refused; nothing ran).
- `results.jsonl` — one row per measured rung.
- `sessions.json` — the state `--resume` needs to rebuild every live conversation. It is compact: each user turn is stored as its corpus segment index, length, opener flag and a SHA-256, and is regenerated from the seeded prompt corpus on load; only the model's own replies are stored verbatim. A resume refuses to continue if a regenerated turn does not match its hash (the corpus, seed or code changed). Never edit it.
- `greedy_outputs.jsonl` / `logprob_scores.jsonl` — quality probe outputs (when enabled).

A `--dry-run` writes only to `results/<spec-name>/dry-run/`: a manifest with `status: dry_run` and `planned_rung_ids` (never completed ids, so `--resume` ignores it) and `DRY_RUN` placeholder rows. It never writes `sessions.json` and never touches a real run's files.

`bench report results/<spec-name>` prints the markdown report to stdout (`--output FILE` writes it to a file instead; `--csv FILE` writes every column).

These files are gitignored. Whether long-term results should be committed here, stored in another repo, or kept only on the bench host is a TODO for the operator.
