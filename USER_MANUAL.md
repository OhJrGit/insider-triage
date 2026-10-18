# Evidence-Grounded Agentic Insider-Threat Triage: User Manual

ICT3214 Security Analytics, Coursework 1, Group 16.

This manual explains how to install the system, run every stage of the pipeline, reproduce the results in
the report, run the demo and the human pilot study, and remove everything afterwards.

---

## Contents
1. [What the system does](#1-what-the-system-does)
2. [Requirements](#2-requirements)
3. [Installation](#3-installation)
4. [Getting the CERT dataset](#4-getting-the-cert-dataset)
5. [Running the pipeline](#5-running-the-pipeline)
6. [Reproducing the report's results](#6-reproducing-the-reports-results)
7. [Demo application](#7-demo-application)
8. [Human pilot study](#8-human-pilot-study)
9. [Command reference](#9-command-reference)
10. [Output files](#10-output-files)
11. [The claim format (agent ↔ verifier contract)](#11-the-claim-format)
12. [Configuration](#12-configuration)
13. [Troubleshooting](#13-troubleshooting)
14. [Project structure](#14-project-structure)
15. [Uninstalling / freeing disk space](#15-uninstalling--freeing-disk-space)

---

## 1. What the system does

```
CERT r4.2 logs ──▶ DuckDB ──▶ sessions ──▶ features ──▶ anomaly detectors ──▶ ranked alerts
                                                                                    │
  analyst UI ◀── deterministic verifier ◀── LLM investigation agent (cited JSON claims)
```

1. **Ingest:** the CERT CSV files are loaded into a single DuckDB database, with one table of
   32.8 million events.
2. **Sessions and features:** every logon→logoff becomes a session, described by about 24 behavioural
   features, including deviations from that user's own past behaviour.
3. **Detection:** unsupervised detectors score each session: Isolation Forest, One-Class SVM,
   an autoencoder and, optionally, a PyTorch autoencoder and a Transformer over event sequences.
   The highest-scoring sessions become **alerts**.
4. **Investigation:** a local LLM (qwen2.5:7b through Ollama) investigates each alert with read-only
   tools and writes a report made of **structured claims**, each citing log event IDs.
5. **Verification:** a deterministic program, with no LLM involved, checks every claim against the
   raw logs and marks it **verified**, **unsupported** or **interpretive**.
6. **Evaluation:** detector metrics, error-injection testing of the verifier, a comparison with an
   LLM-as-judge baseline, manual claim labelling and a human pilot study.

## 2. Requirements

| | Minimum | Used for the report |
|---|---|---|
| OS | Windows 10/11, Linux or macOS | Windows 11 |
| Python | 3.10+ | 3.12.6 |
| RAM | 16 GB (8 GB will be slow) | 32 GB |
| Disk | ~25 GB free | |
| GPU | Optional; strongly recommended for the LLM | NVIDIA RTX 4070 Ti (12 GB) |
| CPU | Any modern x86-64 | AMD Ryzen 7 5800X3D |

Disk breakdown: extracted CERT r4.2 is about 15 GB, the database about 3.6 GB, the Python packages
about 1 GB, the LLM about 4.7 GB, and PyTorch with CUDA (optional) about 3 GB.

Expected LLM speed: about 20–70 s per report on an RTX 4070 Ti class GPU, several minutes on a
6 GB GPU, and 5–20 minutes on CPU only.

## 3. Installation

```bash
# 1. unzip the submission, then open a terminal in the insider-triage folder
cd insider-triage

# 2. create a virtual environment (Windows: use the py launcher)
py -3.12 -m venv .venv            # Linux/macOS: python3 -m venv .venv

# 3. install the packages
.\.venv\Scripts\python.exe -m pip install -r requirements.txt        # Linux/macOS: .venv/bin/python -m pip ...

# 4. check the installation (about 5 seconds; uses a built-in synthetic dataset)
.\.venv\Scripts\python.exe -m pytest -q
```

Every command in this manual is written as `python cli.py ...`. On Windows, either activate the
environment first with `.\.venv\Scripts\Activate.ps1`, or replace `python` with
`.\.venv\Scripts\python.exe`.

**Optional deep models** (PyTorch autoencoder and Transformer):
```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cu124   # NVIDIA GPU
python -m pip install torch                                                       # CPU only
```

**LLM (Ollama):**
1. Install Ollama from https://ollama.com.
2. Run `ollama pull qwen2.5:7b`. This is a 4.7 GB one-off download.
3. Ollama runs in the background as a local server on port 11434. Nothing else needs configuring.

## 4. Getting the CERT dataset

1. Open the CERT Insider Threat Test Dataset on CMU KiltHub: https://doi.org/10.1184/R1/12841247.v1
2. Download **r4.2.tar.bz2**, **answers.tar.bz2** and **CHECKSUMS-sha256.txt**.
3. Check that the downloads aren't corrupted:
   ```powershell
   Get-FileHash r4.2.tar.bz2, answers.tar.bz2 -Algorithm SHA256     # Linux: sha256sum r4.2.tar.bz2 answers.tar.bz2
   ```
   The hashes must match the r4.2 and answers lines in `CHECKSUMS-sha256.txt`.
4. Extract both archives (`tar -xjf r4.2.tar.bz2` and `tar -xjf answers.tar.bz2`). You should get:
   ```
   r4.2/      logon.csv device.csv http.csv email.csv file.csv psychometric.csv LDAP/
   answers/   insiders.csv  r4.2-1/  r4.2-2/  r4.2-3/  (plus other releases, which are ignored)
   ```

**No dataset yet?** Run `python cli.py sample-data`. It writes a small synthetic dataset in the same
format to `datasets/sample/`, so you can try the whole pipeline in seconds. Never report results from it.

## 5. Running the pipeline

Replace `<CERT>` and `<ANSWERS>` with the paths of the extracted folders.

| Step | Command | Time* | What you should see |
|---|---|---|---|
| 1 Ingest | `python cli.py ingest --cert-dir <CERT> --answers-dir <ANSWERS> --release r4.2 --no-content` | ~1 min | 32,770,222 events; 7,323 malicious events, 70 insiders, and no "missing ids" warning |
| 2 Sessions | `python cli.py sessions` | ~1 min | 470,591 sessions |
| 3 Features | `python cli.py features` | <1 min | 1,134 sessions labelled malicious |
| 4 Detect | `python cli.py detect --primary ocsvm` | 5–10 min | ROC-AUC, PR-AUC, P@200, insiders@200 for each model; the `alerts` table is created |
| 5 Investigate | `python cli.py investigate --backend ollama --model qwen2.5:7b --top 10` | ~30 s/alert | one line per report: claims, verdict, time, tokens |
| 6 Verify | `python cli.py verify --show` | seconds | each claim with [OK]/[BAD]/[INT] and the reasons |

\* Times measured on the hardware in Section 2.

`--no-content` drops the web page and email body text, which saves about 10 GB. The agent and
verifier never use it.

## 6. Reproducing the report's results

Run these in order after steps 1–4 above. They match the experiments in the Evaluation section.

```bash
# (a) prompt development: alerts ranked 1-10 (dev set); outputs kept per prompt version
python cli.py investigate --backend ollama --model qwen2.5:7b --top 10

# (b) main evaluation on 50 unseen alerts (ranks 11-60), frozen prompt
python cli.py investigate --backend ollama --model qwen2.5:7b --skip 10 --top 50
python cli.py verify > work/verify_eval.txt          # verifier v2 (default)
python cli.py verify --v1 > work/verify_eval_v1.txt  # verifier v1, for comparison
python cli.py summarize                              # groundedness, verdict vs label, cost
python cli.py eval-verifier                          # error injection, 9 error types
python cli.py eval-verifier --v1

# (c) baseline comparison: deterministic verifier vs LLM-as-judge
python cli.py judge-baseline --backend ollama --model qwen2.5:7b --max-claims 150 --repeats 2

# (d) detector robustness over 5 seeds (read-only; the alerts are not changed)
python cli.py detect-seeds --seeds 0 1 2 3 4
python cli.py detect-seeds --seeds 0 1 2 --models torch_ae transformer      # needs PyTorch

# (e) manual claim labelling (blind: the sheet does not show the verifier's verdict)
python cli.py export-claims --sample 120 --seed 0
#    open work/claims_to_label.csv in Excel; for each row read text + evidence and put
#    s (supported), u (unsupported) or i (interpretive) in human_label; a second person fills
#    human_label_2 on ~30-40 rows. The verifier's answers are in claims_to_label_KEY.csv.
python cli.py score-claims work/claims_to_label.csv   # agreement, precision/recall, Cohen's kappa

# (f) human pilot (Section 8)
python cli.py pilot-prepare --cases 8
streamlit run app/pilot.py
python cli.py pilot-analyze
```

LLM outputs can vary slightly between runs and machines, even at temperature 0. The verifier and the
detectors are deterministic for a fixed seed.

**Don't rerun `detect` after generating reports.** It would rebuild the alerts so they no longer match
the saved reports, so the command refuses unless you pass `--overwrite-alerts`. Use `detect-seeds` to
evaluate models without touching the alerts.

## 7. Demo application

```bash
streamlit run app/demo.py
```
A browser tab opens:
- **Sidebar:** the ranked alerts. Tick "Show ground truth" to see which alerts are real insiders
  (for evaluation only).
- **Run mock agent / Mock agent + 2 hallucinations:** creates a report instantly without an LLM.
  The second button plants two wrong claims so you can watch the verifier catch them.
- **Run LLM agent:** investigates the selected alert with Ollama, taking about 30 s.
- **Verified report tab:** each claim as a green (verified), red (unsupported) or blue (interpretive)
  box. Expand a claim to see the reasons and the exact log rows it cites.
- **All session events tab:** the raw log for the session.

The LLM button uses the environment variables in Section 12; the default is Ollama with qwen2.5:7b.

## 8. Human pilot study

The pilot tests whether the verified report helps a person triage an alert, compared with the alert
alone.

**Design.** Within-subjects and counterbalanced. Each participant sees all cases (default 8: 4 real
insider sessions and 4 benign ones, drawn from the evaluated alerts) in their own random order, half
with the report (condition B) and half without (condition A). Across participants, every case appears
in both conditions. Both conditions show the alert summary and the full raw log; the only difference
is the verified report. The agent's own verdict is never shown.

**Procedure:**
1. `python cli.py pilot-prepare --cases 8` freezes the case set in `work/pilot/cases.json`. Do this once.
2. `streamlit run app/pilot.py` starts the study app.
   - Participants can use your PC directly.
   - Alternatively, run `streamlit run app/pilot.py --server.address 0.0.0.0` and give them
     `http://<your-ip>:8501` on the same network.
3. Give each participant a different number (1, 2, 3, ...). The number decides their case order and
   which cases get the report. If a session is interrupted, entering the same number resumes it.
4. The participant reads the instructions, agrees, and decides on each alert (insider / not insider,
   plus confidence 1–5). The time per alert is recorded automatically. A short questionnaire follows.
5. `python cli.py pilot-analyze` reports accuracy, time and confidence per condition, the paired
   difference for each participant, and Wilcoxon signed-rank p-values once there are 5 or more
   participants. It also splits decisions into hits and false alarms and gives the signal-detection
   measures **d′** (how well insiders are told apart from benign sessions) and **c** (bias: below 0
   means quicker to say "insider"), so it can tell "the report made people better" apart from "the
   report made people flag more". `evidence_checking` shows how often participants opened the evidence
   behind verified, unsupported and interpretive claims.

Each participant gets half the insider cases and half the benign cases with the report, so the two
conditions have the same case mix. Don't change the code mid-study: a participant's case order and
conditions come from the code, so resuming under a different version would mix two schedules.

Responses are stored in `work/pilot/responses.csv`, `work/pilot/interactions.csv` (what was opened)
and `work/pilot/survey.csv`. No names are
collected. Check with the course team whether a consent form is required.

## 9. Command reference

All commands accept `--db PATH` (default `work/cert.duckdb`). Run `python cli.py <command> -h` for details.

| Command | Purpose | Key options |
|---|---|---|
| `sample-data` | Write a synthetic CERT-format dataset | `--out --users --days --seed` |
| `ingest` | Load the CSVs and answers into DuckDB | `--cert-dir --answers-dir --release --no-content --lenient` |
| `sessions` | Build sessions and assign events to them | `--cap-hours` (default 14) |
| `features` | Session features and labels | |
| `detect` | Train/score detectors and write `alerts` | `--models --primary --top-k --train-frac --overwrite-alerts` |
| `detect-seeds` | Mean/std of detector metrics over several seeds (read-only) | `--seeds --models --top-k` |
| `investigate` | Run the agent on alerts | `--backend mock/ollama/openai --model --top --skip --session --num-ctx --hallucinate` |
| `verify` | Verify the saved reports | `--show --strict-text --v1 --session` |
| `summarize` | Per-report table plus headline numbers | `--reports --out --v1` |
| `eval-verifier` | Error-injection test of the verifier | `--seed --v1` |
| `judge-baseline` | Deterministic verifier vs LLM-as-judge | `--max-claims --repeats --model` |
| `export-claims` / `score-claims` | Blind manual-labelling sheet with evidence; agreement + Cohen's kappa | `--out --sample --seed` / CSV path |
| `pilot-prepare` / `pilot-analyze` | Human pilot study | `--cases --seed` |

## 10. Output files

Everything is written under `work/`:

| File | Contents |
|---|---|
| `cert.duckdb` | Database: raw tables, `events`, `sessions`, `features`, `scores`, `alerts`, `malicious_events` |
| `detector_metrics.json` | Metrics from the last `detect` |
| `detector_seeds.json` | Mean/std over seeds |
| `reports/<session>.json` | Agent report, the full tool-call trace, tokens, seconds and normalisation log |
| `reports/<session>.verified.json` | Verifier result for each claim, with reasons and warnings |
| `reports_v1 ... reports_v5/` | Prompt-development runs on the dev alerts |
| `verify_eval.txt`, `eval_summary.csv` | Main evaluation (v2); `*_v1` files are verifier v1 |
| `verifier_perturbation.json` | Error-injection results, including missed examples |
| `judge_baseline.json` | LLM-as-judge vs deterministic verifier, every judged item |
| `claims_to_label.csv`, `claims_to_label_KEY.csv` | Blind labelling sheet (with evidence) and the verifier's answers |
| `pilot/` | Pilot cases, responses and survey |

## 11. The claim format

The agent must output claims like this; the verifier checks them:
```json
{"text": "CRD6991 visited wikileaks.org 3 times.", "type": "factual",
 "citations": ["{A1..}", "{B2..}", "{C3..}"],
 "asserted": {"source": "http", "url_contains": "wikileaks.org"}, "count": 3}
{"text": "The session lasted 107 minutes with 8 USB connections.", "type": "factual",
 "session": {"duration_min": 107.0, "n_usb_connect": 8}}
{"text": "In the last 30 days the user averaged 0 USB connections.", "type": "factual",
 "baseline": {"avg_usb_connects": 0.0}}
{"text": "This is outside the user's usual hours.", "type": "interpretive"}
```

**Verifier rules (v2).** A factual claim is *verified* only if all of these hold:
- every cited ID exists and belongs to the session's user;
- every `asserted` field matches **every** cited event (`_contains` gives a substring match; dates are
  checked at the precision given, and `"A to B"` means a range);
- cited events are described by at least one asserted field, and asserted fields have cited events;
- `count` equals the number of distinct citations;
- `baseline` and `session` values match values the verifier recomputes itself, at the precision stated.

Anything else is *unsupported*. Interpretive claims are not fact-checked, but any IDs they cite must
exist, and numbers inside them trigger a warning (`--strict-text` turns the warning into a failure).
The complete field lists are in `triage/agent/schema.py`.

## 12. Configuration

| Setting | Where | Default |
|---|---|---|
| Database path | `TRIAGE_DB` environment variable or `--db` | `work/cert.duckdb` |
| Company email domain | `TRIAGE_ORG_DOMAIN` | `dtaa.com` |
| Office hours, session cap, URL keyword lists | `triage/config.py` | 07–19 h, 14 h |
| LLM backend / model / URL / key / context size | `TRIAGE_LLM_BACKEND`, `TRIAGE_LLM_MODEL`, `TRIAGE_LLM_BASE_URL`, `TRIAGE_LLM_API_KEY`, `TRIAGE_LLM_NUM_CTX` | ollama, qwen2.5:7b, localhost:11434, -, 32768 |

To use a cloud API instead of Ollama:
```
python cli.py investigate --backend openai --base-url https://api.openai.com/v1 --model gpt-4o-mini
```
with your key in `TRIAGE_LLM_API_KEY`. This works with any OpenAI-compatible endpoint.

Ollama speed-ups for NVIDIA GPUs (set once, then restart Ollama):
`setx OLLAMA_FLASH_ATTENTION 1` and `setx OLLAMA_KV_CACHE_TYPE q8_0`.

## 13. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `.venv\Scripts\activate` not found | Use `.\.venv\Scripts\Activate.ps1`, or call `.\.venv\Scripts\python.exe` directly |
| `.venv\Scripts` has only `python.exe` | The venv was made by MSYS2/another Python. Delete `.venv` and recreate it with `py -3.12 -m venv .venv` |
| "running scripts is disabled" | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` |
| "The system cannot find the path specified" | The terminal isn't in the `insider-triage` folder, so `cd` into it first |
| "Could not find platform independent libraries" | A harmless warning from some Windows Python installs |
| `OverflowError` in ingest | Old version; update `triage/ingest.py` |
| CSV parse error during ingest | Rerun with `--lenient` and check how many rows were skipped |
| `cannot reach http://localhost:11434` | Ollama isn't running; start the Ollama app |
| "[warn] prompt filled the context window" | Increase `--num-ctx` (e.g. 65536) |
| An LLM request times out | It is retried once automatically. Check `ollama ps` shows "100% GPU", and close GPU-heavy apps |
| `detect` refuses to run | It protects alerts that already have reports. Use `detect-seeds`, or `--overwrite-alerts` |
| DuckDB "database is locked" | Another process has the database open for writing. Read-only commands can run together |
| PyTorch models fail | Install PyTorch (Section 3) |

## 14. Project structure

```
cli.py                    command-line entry point (Section 9)
triage/config.py          paths, office hours, keyword lists
triage/db.py              DuckDB helpers
triage/ingest.py          CSV + answers -> DuckDB
triage/sessions.py        logon/logoff sessions
triage/features.py        session features + labels
triage/detect.py          IForest, OCSVM, autoencoder; multi-seed evaluation
triage/deep.py            PyTorch autoencoder, event-sequence Transformer
triage/agent/schema.py    claim/report schema and prompt rules
triage/agent/tools.py     read-only tools for the agent
triage/agent/llm.py       Ollama / OpenAI-compatible clients
triage/agent/investigate.py  agent loop, normalisation, mock agent
triage/verifier/verify.py    deterministic verifier
triage/verifier/perturb.py   error injection
triage/evaluation.py      verifier evaluation, claim labelling
triage/judge.py           LLM-as-judge baseline
triage/summary.py         per-run results table
triage/pilot.py           pilot study design and analysis
app/demo.py               analyst demo UI
app/pilot.py              pilot study UI
scripts/make_sample_data.py  synthetic CERT-format data
tests/test_pipeline.py    automated tests
```

## 15. Uninstalling / freeing disk space

| Item | Remove with |
|---|---|
| Python packages | Delete `insider-triage\.venv` (run `.\.venv\Scripts\python.exe -m pip cache purge` first) |
| Database and results | Delete `insider-triage\work` |
| CERT data | Delete the extracted `r4.2` and `answers` folders and the `.tar.bz2` files |
| LLM | `ollama rm qwen2.5:7b`, then uninstall Ollama in Windows Settings › Apps |
| PyTorch (if installed) | Included in `.venv`; deleting `.venv` removes it |
