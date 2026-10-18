# Saved experiment outputs

Copies of the outputs in `work/` that the report's numbers come from, kept here so they can be
checked without re-running the pipeline. `work/` itself (including the 3.6 GB DuckDB database)
stays git-ignored. Re-running a command writes to `work/`, not here; copy a file over deliberately
if you want to update it.

| File | Produced by | Used in report |
|---|---|---|
| `detector_metrics.json` | `python cli.py detect` | Detector table (single run that produced the alerts) |
| `detector_seeds.json` | `python cli.py detect-seeds` | Detector table, mean ± sd over seeds |
| `reports_heldout/` | `investigate --skip 10 --top 50`, then `verify` | Held-out agent evaluation (alerts 11–60). `<id>.json` = agent report, `<id>.verified.json` = verifier output |
| `verify_eval.txt`, `eval_summary.csv` | `verify`, `summarize` (verifier v2) | Groundedness results |
| `verify_eval_v1.txt`, `eval_summary_v1.csv` | `verify --v1`, `summarize` | Same reports under the lenient v1 rules, for comparison |
| `verifier_perturbation.json` | `eval-verifier` (v2) | Error-injection results |
| `verifier_perturbation_v1.json` | `eval-verifier --v1` | Error injection against v1, showing the gaps that motivated v2 |
| `judge_baseline.json` | `judge-baseline` | Deterministic verifier vs LLM-as-judge comparison |
| `reports_dev/v1` … `v5` | `investigate` with each prompt version (v1–v2 on the first 3 dev alerts, v3–v5 on alerts 1–10) | Prompt development on alerts 1–10 (not used for the held-out results) |

Not included: the CERT data, the database, claim-labelling sheets and pilot responses (these
contain participants' answers and stay local).
