"""Baseline: LLM-as-a-judge vs the deterministic verifier.

The existing approach we compare against (e.g. Audit-LLM-style evidence debate, LLM-as-judge
evaluation) asks a model whether a claim is supported by the evidence. We give the judge exactly
what a careful reviewer would get -- the claim plus the raw log rows it cites (and the session
summary / user baseline for aggregate claims) -- and ask for supported / unsupported.

Test set with known answers, built from the evaluation reports:
  * GOOD claims: factual claims the deterministic verifier accepted (true facts from the logs)
  * BAD claims : the same claims with exactly one injected error (perturb.py)
Both systems are scored on the same set: accuracy, recall on BAD claims (errors caught),
false-alarm rate on GOOD claims, cost per judgement, and (optionally) self-consistency when the
judge is asked the same question twice.
"""
from __future__ import annotations

import json
import random
import statistics as st
import time
from pathlib import Path

from . import config
from .agent.investigate import _parse_json, load_report
from .agent.tools import Tools
from .verifier.perturb import MUTATIONS, mutate
from .verifier.verify import fetch_events, verify_report

JUDGE_PROMPT = """You are auditing an AI-written security investigation report. Decide whether ONE claim
is fully supported by the evidence shown. A claim is "supported" only if every fact in it (event
types, counts, times, URLs, files, PCs, numbers) matches the evidence exactly. If any part is wrong,
missing from the evidence, or the cited events do not show what the claim says, it is "unsupported".

Reply with ONE JSON object: {"verdict": "supported" | "unsupported", "reason": "<one sentence>"}"""


def _evidence(con, tools: Tools, session_id: str, claim, max_events: int = 40) -> str:
    parts = []
    if claim.citations:
        found = fetch_events(con, claim.citations[:max_events])
        lines = []
        for cid in claim.citations[:max_events]:
            e = found.get(cid)
            lines.append(json.dumps({k: v for k, v in e.items() if k != "session_id"}, default=str)
                         if e else json.dumps({"id": cid, "error": "no event with this id exists in the logs"}))
        if len(claim.citations) > max_events:
            lines.append(f"... and {len(claim.citations) - max_events} more cited events (not shown)")
        parts.append("Cited events:\n" + "\n".join(lines))
    if claim.session or claim.baseline or not claim.citations:
        s = tools.get_session(session_id)
        keep = ["user", "pc", "start_ts", "end_ts", "duration_min", "start_hour", "end_hour", "after_hours",
                "weekend", "foreign_pc", "n_usb_connect", "n_file", "n_file_exec_archive", "n_http",
                "n_http_leak", "n_http_job", "n_http_hack", "n_email", "n_email_external", "n_email_attach"]
        parts.append("Session summary:\n" + json.dumps({k: s.get(k) for k in keep}, default=str))
        parts.append("User baseline (30 days before):\n"
                     + json.dumps(tools.get_user_baseline(s["user"], s["start_ts"]), default=str))
    return "\n\n".join(parts)


def _claim_view(claim) -> str:
    d = {"text": claim.text, "citations": claim.citations, "asserted": claim.asserted}
    for k in ("count", "baseline", "session"):
        v = getattr(claim, k)
        if v not in (None, {}, []):
            d[k] = v
    return json.dumps(d, default=str)


def build_test_set(con, reports_dir: Path, max_good: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    goods = []
    for p in sorted(Path(reports_dir).glob("*.json")):
        if p.name.endswith(".verified.json"):
            continue
        rep = load_report(p)
        res = verify_report(con, rep)
        for cr, cl in zip(res["claims"], rep.claims):
            if cl.type == "factual" and cr["status"] == "verified":
                goods.append((rep.session_id, cl))
    rng.shuffle(goods)
    goods = goods[:max_good]
    items = []
    for sid, cl in goods:
        user = con.execute('SELECT "user" FROM sessions WHERE session_id = ?', [sid]).fetchone()[0]
        items.append({"session_id": sid, "claim": cl, "truth": "supported", "mutation": None})
        kinds = MUTATIONS[:]
        rng.shuffle(kinds)
        for k in kinds:                     # one injected error per good claim
            bad = mutate(con, cl, k, rng, user)
            if bad is not None:
                items.append({"session_id": sid, "claim": bad, "truth": "unsupported", "mutation": k})
                break
    return items


def run(con, llm, reports_dir: Path = config.REPORTS_DIR, max_good: int = 120, seed: int = 0,
        repeats: int = 1, out_dir: Path = config.WORK_DIR) -> dict:
    tools = Tools(con)
    items = build_test_set(con, reports_dir, max_good, seed)
    rows = []
    for i, it in enumerate(items):
        cl, sid = it["claim"], it["session_id"]
        det = verify_report(con, load_single(sid, cl))["claims"][0]["status"]
        det_verdict = "supported" if det == "verified" else "unsupported"
        msg = (f"CLAIM:\n{_claim_view(cl)}\n\nEVIDENCE:\n{_evidence(con, tools, sid, cl)}")
        verdicts, secs, toks = [], [], []
        for _ in range(repeats):
            llm.reset_usage()
            t = time.time()
            try:
                out = _parse_json(llm.chat([{"role": "system", "content": JUDGE_PROMPT},
                                            {"role": "user", "content": msg}]))
                v = str(out.get("verdict", "")).strip().lower()
                v = v if v in ("supported", "unsupported") else "invalid"
            except Exception:
                v = "invalid"
            verdicts.append(v)
            secs.append(time.time() - t)
            toks.append(llm.usage["prompt_tokens"] + llm.usage["completion_tokens"])
        rows.append({"session_id": sid, "truth": it["truth"], "mutation": it["mutation"],
                     "deterministic": det_verdict, "judge": verdicts[0], "judge_all": verdicts,
                     "judge_seconds": round(st.mean(secs), 2), "judge_tokens": round(st.mean(toks)),
                     "text": cl.text})
        if (i + 1) % 20 == 0:
            print(f"  judged {i + 1}/{len(items)}")

    def score(key):
        n = len(rows)
        correct = sum(r[key] == r["truth"] for r in rows)
        bad = [r for r in rows if r["truth"] == "unsupported"]
        good = [r for r in rows if r["truth"] == "supported"]
        caught = sum(r[key] == "unsupported" for r in bad)
        false_alarm = sum(r[key] != "supported" for r in good)
        by_mut = {}
        for r in bad:
            m = by_mut.setdefault(r["mutation"], [0, 0])
            m[0] += r[key] == "unsupported"
            m[1] += 1
        return {"accuracy": round(correct / n, 3) if n else None,
                "errors_caught": f"{caught}/{len(bad)}", "recall_on_errors": round(caught / len(bad), 3) if bad else None,
                "false_alarms_on_true_claims": f"{false_alarm}/{len(good)}",
                "false_alarm_rate": round(false_alarm / len(good), 3) if good else None,
                "caught_by_mutation": {k: f"{a}/{b}" for k, (a, b) in sorted(by_mut.items())}}

    res = {"items": len(rows), "good": sum(r["truth"] == "supported" for r in rows),
           "bad": sum(r["truth"] == "unsupported" for r in rows), "judge_model": llm.describe(),
           "deterministic_verifier": score("deterministic"), "llm_judge": score("judge"),
           "judge_invalid_outputs": sum(r["judge"] == "invalid" for r in rows),
           "judge_seconds_per_claim": round(st.mean(r["judge_seconds"] for r in rows), 2) if rows else None,
           "judge_tokens_per_claim": round(st.mean(r["judge_tokens"] for r in rows)) if rows else None}
    if repeats > 1:
        res["judge_self_consistency"] = round(
            sum(len(set(r["judge_all"])) == 1 for r in rows) / len(rows), 3)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "judge_baseline.json").write_text(json.dumps({"summary": res, "rows": rows}, indent=2, default=str))
    return res


def load_single(session_id, claim):
    from .agent.schema import Report
    return Report(session_id=session_id, claims=[claim])
