"""Human pilot study: does a verified report help an analyst triage an alert?

Design (within-subjects, counterbalanced):
  * `prepare` picks N cases from the evaluated alerts -- half real insider sessions, half benign --
    and freezes them in work/pilot/cases.json.
  * Every participant sees ALL cases, in their own random order. For participant p, case i is shown
    in condition B (alert + verified report) if (i + p) is odd, else condition A (alert only), so each
    participant gets half A / half B and across participants every case appears in both conditions.
  * Both conditions show the alert (user, PC, time, anomaly rank, session totals) AND the raw event
    log, so the only difference is the verified report.
  * Measured per case: decision (insider / not insider), confidence 1-5, seconds to decide, and which
    claims' evidence and whether the raw log were opened (work/pilot/interactions.csv).
  * `analyze` reports accuracy / time / confidence per condition, the outcome split (hit rate, false-alarm
    rate, d' and criterion c, so "better at telling insiders apart" can be separated from "more willing
    to say insider"), paired per-participant differences, a Wilcoxon signed-rank test (small-n: report
    effect sizes, not just p) and how often participants checked the evidence.
"""
from __future__ import annotations

import csv
import json
import random
import statistics as st
from datetime import datetime
from pathlib import Path
from statistics import NormalDist

from . import config

PILOT_DIR = config.WORK_DIR / "pilot"
CASES = PILOT_DIR / "cases.json"
RESPONSES = PILOT_DIR / "responses.csv"
SURVEY = PILOT_DIR / "survey.csv"
INTERACTIONS = PILOT_DIR / "interactions.csv"
RESPONSE_FIELDS = ["participant", "case_no", "session_id", "condition", "decision", "correct",
                   "confidence", "seconds", "timestamp"]
STATUSES = ("verified", "unsupported", "interpretive")
INTERACTION_FIELDS = ["participant", "case_no", "condition", "raw_log_opened",
                      *[f"n_{s}" for s in STATUSES], *[f"opened_{s}" for s in STATUSES], "timestamp"]


def prepare(con, n_cases: int = 8, seed: int = 0, reports_dir: Path = config.REPORTS_DIR) -> list[dict]:
    have = {p.stem for p in Path(reports_dir).glob("*.json") if not p.name.endswith(".verified.json")}
    rows = con.execute("SELECT session_id, label FROM features WHERE session_id IN (SELECT unnest(?::VARCHAR[]))",
                       [sorted(have)]).fetchall()
    ins = sorted(s for s, lab in rows if lab == 1)
    ben = sorted(s for s, lab in rows if lab == 0)
    rng = random.Random(seed)
    rng.shuffle(ins)
    rng.shuffle(ben)
    k = n_cases // 2
    if len(ins) < k or len(ben) < n_cases - k:
        raise ValueError(f"need {k} insider and {n_cases - k} benign reported sessions, have {len(ins)}/{len(ben)}")
    cases = [{"session_id": s, "label": 1} for s in ins[:k]] + [{"session_id": s, "label": 0} for s in ben[:n_cases - k]]
    rng.shuffle(cases)
    for i, c in enumerate(cases):
        c["case_no"] = i
    PILOT_DIR.mkdir(parents=True, exist_ok=True)
    CASES.write_text(json.dumps({"seed": seed, "created": datetime.now().isoformat(timespec="seconds"),
                                 "reports_dir": str(reports_dir), "cases": cases}, indent=2))
    return cases


def load_cases() -> list[dict]:
    return json.loads(CASES.read_text())["cases"]


def schedule(participant: int, cases: list[dict]) -> list[dict]:
    """Order + condition for one participant (deterministic, so a restart resumes the same plan).

    Conditions alternate *within* the insider cases and *within* the benign cases, so every participant
    judges (about) half the insiders and half the benign cases with the report. Without this, one
    participant could get, say, 3 insiders with the report and 1 without, and a difference between
    conditions would partly be a difference in case mix. Odd and even participant numbers get the
    opposite assignment, so across participants every case appears in both conditions.
    """
    cond = {}
    for label in (0, 1):
        same = sorted(i for i, c in enumerate(cases) if c["label"] == label)
        for j, i in enumerate(same):
            cond[i] = "B_report" if (j + participant) % 2 == 1 else "A_alert_only"
    order = list(range(len(cases)))
    random.Random(1000 + participant).shuffle(order)
    return [{**cases[i], "condition": cond[i]} for i in order]


def record(participant: int, case: dict, decision: str, confidence: int, seconds: float) -> None:
    PILOT_DIR.mkdir(parents=True, exist_ok=True)
    new = not RESPONSES.exists()
    with open(RESPONSES, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=RESPONSE_FIELDS)
        if new:
            w.writeheader()
        w.writerow({"participant": participant, "case_no": case["case_no"], "session_id": case["session_id"],
                    "condition": case["condition"], "decision": decision,
                    "correct": int((decision == "insider") == (case["label"] == 1)),
                    "confidence": confidence, "seconds": round(seconds, 1),
                    "timestamp": datetime.now().isoformat(timespec="seconds")})


def record_interactions(participant: int, case: dict, raw_log_opened: bool,
                        claims: dict[str, int], opened: dict[str, int]) -> None:
    """What the participant looked at before deciding: whether they opened the raw log, and, per claim
    status, how many claims had evidence to show (`claims`) and how many of those they opened (`opened`).
    A behavioural check on the survey's self-reported "I double-checked claims against the raw logs"."""
    PILOT_DIR.mkdir(parents=True, exist_ok=True)
    new = not INTERACTIONS.exists()
    with open(INTERACTIONS, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=INTERACTION_FIELDS)
        if new:
            w.writeheader()
        w.writerow({"participant": participant, "case_no": case["case_no"], "condition": case["condition"],
                    "raw_log_opened": int(raw_log_opened),
                    **{f"n_{s}": claims.get(s, 0) for s in STATUSES},
                    **{f"opened_{s}": opened.get(s, 0) for s in STATUSES},
                    "timestamp": datetime.now().isoformat(timespec="seconds")})


def record_survey(participant: int, answers: dict) -> None:
    PILOT_DIR.mkdir(parents=True, exist_ok=True)
    new = not SURVEY.exists()
    with open(SURVEY, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["participant", *answers.keys(), "timestamp"])
        if new:
            w.writeheader()
        w.writerow({"participant": participant, **answers, "timestamp": datetime.now().isoformat(timespec="seconds")})


def done_cases(participant: int) -> set[int]:
    if not RESPONSES.exists():
        return set()
    return {int(r["case_no"]) for r in csv.DictReader(open(RESPONSES, newline=""))
            if int(r["participant"]) == participant}


def analyze(responses_csv: Path | None = None) -> dict:
    responses_csv = Path(responses_csv or RESPONSES)
    if not responses_csv.exists():
        return {"participants": 0, "note": f"no responses yet ({responses_csv})"}
    rows = list(csv.DictReader(open(responses_csv, newline="")))
    for r in rows:
        r["correct"], r["confidence"], r["seconds"] = int(r["correct"]), int(r["confidence"]), float(r["seconds"])
    conds = ["A_alert_only", "B_report"]

    def summ(rs):
        if not rs:
            return {}
        ins = [r for r in rs if r["decision"] == "insider"]
        return {"decisions": len(rs), "accuracy": round(st.mean(r["correct"] for r in rs), 3),
                "mean_seconds": round(st.mean(r["seconds"] for r in rs), 1),
                "median_seconds": round(st.median(r["seconds"] for r in rs), 1),
                "mean_confidence": round(st.mean(r["confidence"] for r in rs), 2),
                "flagged_insider": len(ins), **outcomes(rs)}

    out = {"participants": len({r["participant"] for r in rows}),
           "by_condition": {c: summ([r for r in rows if r["condition"] == c]) for c in conds}}
    metrics = ("correct", "seconds", "confidence", "d_prime", "criterion_c")
    paired = {}
    for p in sorted({r["participant"] for r in rows}):
        a = [r for r in rows if r["participant"] == p and r["condition"] == conds[0]]
        b = [r for r in rows if r["participant"] == p and r["condition"] == conds[1]]
        if a and b:
            paired[p] = {m: (st.mean(x[m] for x in b) - st.mean(x[m] for x in a))
                         for m in ("correct", "seconds", "confidence")}
            oa, ob = outcomes(a), outcomes(b)
            for m in ("d_prime", "criterion_c"):
                if oa[m] is not None and ob[m] is not None:
                    paired[p][m] = ob[m] - oa[m]
    out["paired_differences_B_minus_A"] = {
        m: {"mean": round(st.mean(diffs), 3) if diffs else None, "per_participant": [round(x, 3) for x in diffs]}
        for m in metrics for diffs in [[d[m] for d in paired.values() if m in d]]}
    if len(paired) >= 5:
        from scipy.stats import wilcoxon
        tests = {}
        for m in metrics:
            diffs = [d[m] for d in paired.values() if m in d]
            if len(diffs) >= 5 and any(x != 0 for x in diffs):
                tests[m] = round(float(wilcoxon(diffs).pvalue), 4)
        out["wilcoxon_p"] = tests
    else:
        out["note"] = "fewer than 5 paired participants: report descriptive statistics only"
    interactions = responses_csv.parent / INTERACTIONS.name
    if interactions.exists():
        out["evidence_checking"] = _evidence_checking(list(csv.DictReader(open(interactions, newline=""))), rows)
    if SURVEY.exists():
        srows = list(csv.DictReader(open(SURVEY, newline="")))
        keys = [k for k in srows[0] if k not in ("participant", "timestamp")] if srows else []
        out["survey_means"] = {k: round(st.mean(float(r[k]) for r in srows if r[k] not in ("", None)), 2)
                               for k in keys if all(_isnum(r[k]) for r in srows)}
    return out


def outcomes(rs: list[dict]) -> dict:
    """Split decisions into hits / misses / false alarms / correct rejections, plus the signal-detection
    measures d' (how well insiders are told apart from benign sessions) and c (bias: below 0 = quick to
    say "insider", above 0 = reluctant). Accuracy alone mixes the two: a report that makes people flag
    everything raises hits and false alarms together and can leave accuracy unchanged.

    The true label is recovered from decision + correct, so older responses.csv files work too.
    Rates use the log-linear correction (add 0.5 to each count, 1 to each total; Hautus 1995) so that
    a rate of 0 or 1, which is common with a handful of cases, does not give an infinite z-score.
    """
    hits = misses = fas = crs = 0
    for r in rs:
        said_insider = r["decision"] == "insider"
        is_insider = said_insider == bool(r["correct"])
        if is_insider:
            hits, misses = hits + said_insider, misses + (not said_insider)
        else:
            fas, crs = fas + said_insider, crs + (not said_insider)
    out = {"hits": hits, "misses": misses, "false_alarms": fas, "correct_rejections": crs,
           "hit_rate": round(hits / (hits + misses), 3) if hits + misses else None,
           "false_alarm_rate": round(fas / (fas + crs), 3) if fas + crs else None,
           "d_prime": None, "criterion_c": None}
    if hits + misses and fas + crs:   # d' and c need at least one insider and one benign decision
        z = NormalDist().inv_cdf
        zh, zf = z((hits + 0.5) / (hits + misses + 1)), z((fas + 0.5) / (fas + crs + 1))
        out["d_prime"], out["criterion_c"] = round(zh - zf, 3), round(-(zh + zf) / 2, 3)
    return out


def _evidence_checking(irows: list[dict], responses: list[dict]) -> dict:
    """How often participants opened the evidence behind a claim, by claim status (report condition only),
    how often they opened the raw log in each condition, and accuracy on report cases split by whether
    they opened any evidence at all."""
    out = {}
    b = [r for r in irows if r["condition"] == "B_report"]
    for s in STATUSES:
        n = sum(int(r[f"n_{s}"]) for r in b)
        out[f"{s}_claims_opened"] = f"{sum(int(r[f'opened_{s}']) for r in b)}/{n}"
        out[f"{s}_open_rate"] = round(sum(int(r[f"opened_{s}"]) for r in b) / n, 3) if n else None
    out["raw_log_opened_rate"] = {
        c: round(st.mean(int(r["raw_log_opened"]) for r in rs), 3) if rs else None
        for c in ("A_alert_only", "B_report") for rs in [[r for r in irows if r["condition"] == c]]}
    checked = {(r["participant"], r["case_no"]) for r in b if any(int(r[f"opened_{s}"]) for s in STATUSES)}
    rb = [r for r in responses if r["condition"] == "B_report"]
    for name, grp in (("opened_any_evidence", [r for r in rb if (r["participant"], r["case_no"]) in checked]),
                      ("opened_no_evidence", [r for r in rb if (r["participant"], r["case_no"]) not in checked])):
        out[f"accuracy_when_{name}"] = round(st.mean(r["correct"] for r in grp), 3) if grp else None
        out[f"cases_{name}"] = len(grp)
    return out


def _isnum(v) -> bool:
    try:
        float(v)
        return True
    except (TypeError, ValueError):
        return False
