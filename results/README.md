# Saved experiment outputs

Copies of the outputs in `work/` that the report's numbers come from, kept here so they can be
checked without re-running the pipeline. `work/` itself (including the 3.6 GB DuckDB database)
stays git-ignored. Re-running a command writes to `work/`, not here; copy a file over deliberately
if you want to update it.

| File | Produced by | Used in report |
|---|---|---|
| `detector_metrics.json` | `python cli.py detect` | Detector table (single run that produced the alerts). Default parameters, loose `insiders_caught@k` (see note below) |
| `detector_seeds.json` | `python cli.py detect-seeds` | Original sklearn detectors, default parameters, mean ± sd over 5 seeds. Loose `insiders_caught@k` (see note below); superseded for insider counts by `detector_seeds_default_and_deep.json` |
| `detector_seeds_default_and_deep.json` | `detect-seeds --seeds 0 1 2 3 4` (sklearn, default parameters), `detect-seeds --models torch_ae --seeds 0 1 2 3 4`, `detect-seeds --models transformer --seeds 0 1 2` | Detector table: default-parameter baselines plus the PyTorch autoencoder (5 seeds) and event-sequence Transformer (3 seeds). Strict `insiders_caught@k` |
| `detector_seeds_tuned.json` | `detect-seeds --params best_params.json --seeds 0 1 2 3 4` | Detector table: tuned Isolation Forest, One-Class SVM and autoencoder, mean ± sd over 5 seeds. Strict `insiders_caught@k` |
| `tuning.json` | `tune --seeds 0 1 2` | Hyperparameter tuning: every grid configuration with validation metrics, plus tuned-vs-default test results (single seed 42). Split: train / validation slice / test, all by time |
| `best_params.json` | `tune --seeds 0 1 2` | Selected hyperparameters per model; loaded by `detect-seeds --params` |
| `reports_heldout/` | `investigate --skip 10 --top 50`, then `verify` | Held-out agent evaluation (alerts 11–60). `<id>.json` = agent report, `<id>.verified.json` = verifier output |
| `verify_eval.txt`, `eval_summary.csv` | `verify`, `summarize` (verifier v2) | Groundedness results |
| `verify_eval_v1.txt`, `eval_summary_v1.csv` | `verify --v1`, `summarize` | Same reports under the lenient v1 rules, for comparison |
| `verifier_perturbation.json` | `eval-verifier` (v2) | Error-injection results |
| `verifier_perturbation_v1.json` | `eval-verifier --v1` | Error injection against v1, showing the gaps that motivated v2 |
| `judge_baseline.json` | `judge-baseline` | Deterministic verifier vs LLM-as-judge comparison |
| `claims_to_label.csv` | `export-claims --sample 120`, then labelled by hand | Manual labelling: 120 random held-out claims. `human_label` = Wayne Lee, `human_label_2` = Oh Jia Rong, labelled independently and blind to the verifier (s / u / i) |
| `claims_to_label_KEY.csv` | `export-claims` | The verifier's verdict for each claim, kept separate so labelling stayed blind |
| `claims_score.json` | `score-claims results/claims_to_label.csv` | Verifier vs labeller 1 (agreement, precision/recall on unsupported claims) and inter-rater agreement / Cohen's kappa (Table VI) |
| `reports_dev/v1` … `v5` | `investigate` with each prompt version (v1–v2 on the first 3 dev alerts, v3–v5 on alerts 1–10) | Prompt development on alerts 1–10 (not used for the held-out results) |

**Detector metric note.** In `detector_metrics.json` and the original `detector_seeds.json`,
`insiders_caught@k` counted insiders with *any* session in the top k, including benign sessions, so it
could be above zero while `precision@k` was zero. In the newer files it counts only insiders with at
least one *malicious* session in the top k, matching `precision@k`. The old definition is kept as
`insider_users_in_top@k`. Insider counts are therefore not comparable between the two sets of files.

**Tuning protocol.** Hyperparameters were selected on a validation slice immediately before the
test period (validation PR-AUC, averaged over 3 seeds); the test period was scored once per
configuration after refitting on train + validation. Selection used validation labels, so tuned
detectors are weakly supervised in their hyperparameters. The PyTorch autoencoder and Transformer
have fixed architectures and were not tuned.

Not included: the CERT data and the database. The user-study app was tested but no study with participants was run,
so there are no pilot results.
