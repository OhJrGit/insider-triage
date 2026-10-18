"""Human pilot study: does a verified report help an analyst triage an alert?

Design (within-subjects, counterbalanced):
  * `prepare` picks N cases from the evaluated alerts -- half real insider sessions, half benign --
    and freezes them in work/pilot/cases.json.
  * Every participant sees ALL cases, in their own random order. For participant p, case i is shown
    in condition B (alert + verified report) if (i + p) is odd, else condition A (alert only), so each
    participant gets half A / half B and across participants every case appears in both conditions.
  * Both conditions show the alert (user, PC, time, anomaly rank, session totals) AND the raw event
    log, so the only difference is the verified report.
  * Measured per case: decision (insider / not insider), confidence 1-5, seconds to decide.
  * `analyze` reports accuracy / time / confidence per condition plus paired per-participant
    differences and a Wilcoxon signed-rank test (small-n: report effect sizes, not just p).
"""
from __future__ import annotations

import csv
import json
import random
import statistics as st
from datetime import datetime
from pathlib import Path

from . import config

PILOT_DIR = config.WORK_DIR / "pilot"
CASES = PILOT_DIR / "cases.json"
RESPONSES = PILOT_DIR / "responses.csv"
SURVEY = PILOT_DIR / "survey.csv"
RESPONSE_FIELDS = ["participant", "case_no", "session_id", "condition", "decision", "correct",
                   "confidence", "seconds", "timestamp"]


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
    """Order + condition for one participant (deterministic, so a restart resumes the same plan)."""
    order = list(range(len(cases)))
    random.Random(1000 + participant).shuffle(order)
    plan = []
    for i in order:
        cond = "B_report" if (i + participant) % 2 == 1 else "A_alert_only"
        plan.append({**cases[i], "condition": cond})
    return plan


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
                "flagged_insider": len(ins)}

    out = {"participants": len({r["participant"] for r in rows}),
           "by_condition": {c: summ([r for r in rows if r["condition"] == c]) for c in conds}}
    paired = {}
    for p in sorted({r["participant"] for r in rows}):
        a = [r for r in rows if r["participant"] == p and r["condition"] == conds[0]]
        b = [r for r in rows if r["participant"] == p and r["condition"] == conds[1]]
        if a and b:
            paired[p] = {m: (st.mean(x[m] for x in b) - st.mean(x[m] for x in a))
                         for m in ("correct", "seconds", "confidence")}
    out["paired_differences_B_minus_A"] = {
        m: {"mean": round(st.mean(d[m] for d in paired.values()), 3) if paired else None,
            "per_participant": [round(d[m], 3) for d in paired.values()]}
        for m in ("correct", "seconds", "confidence")}
    if len(paired) >= 5:
        from scipy.stats import wilcoxon
        tests = {}
        for m in ("correct", "seconds", "confidence"):
            diffs = [d[m] for d in paired.values()]
            if any(x != 0 for x in diffs):
                tests[m] = round(float(wilcoxon(diffs).pvalue), 4)
        out["wilcoxon_p"] = tests
    else:
        out["note"] = "fewer than 5 paired participants: report descriptive statistics only"
    if SURVEY.exists():
        srows = list(csv.DictReader(open(SURVEY, newline="")))
        keys = [k for k in srows[0] if k not in ("participant", "timestamp")] if srows else []
        out["survey_means"] = {k: round(st.mean(float(r[k]) for r in srows if r[k] not in ("", None)), 2)
                               for k in keys if all(_isnum(r[k]) for r in srows)}
    return out


def _isnum(v) -> bool:
    try:
        float(v)
        return True
    except (TypeError, ValueError):
        return False
