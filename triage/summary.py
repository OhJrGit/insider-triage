"""One table per evaluation run: every report joined with its verification result, the
ground-truth label of the session, and the cost of producing it.

    python cli.py summarize                       # work/reports -> work/eval_summary.csv
    python cli.py summarize --reports work/reports_v5 --out work/v5_summary.csv
"""
from __future__ import annotations

import csv
import json
import statistics as st
from collections import Counter
from pathlib import Path

from .agent.investigate import load_report
from .verifier.verify import verify_report


def summarize(con, reports_dir: Path, out_csv: Path, strict_citations: bool = True) -> dict:
    rows = []
    for p in sorted(Path(reports_dir).glob("*.json")):
        if p.name.endswith(".verified.json"):
            continue
        raw = json.loads(p.read_text())
        rep = load_report(p)
        res = verify_report(con, rep, strict_citations=strict_citations)
        s = res["summary"]
        trace = raw.get("trace", [])
        usage = next((t["usage"] for t in trace if "usage" in t), {})
        seconds = next((t["seconds"] for t in trace if "seconds" in t), None)
        normalized = sum(len(t.get("normalized", [])) for t in trace)
        lab = con.execute(
            "SELECT label, scenario, \"user\" FROM features WHERE session_id = ?", [rep.session_id]).fetchone()
        rank = con.execute("SELECT rank FROM alerts WHERE session_id = ?", [rep.session_id]).fetchone()
        factual = s["verified"] + sum(1 for c in res["claims"] if c["type"] == "factual" and c["status"] == "unsupported")
        rows.append({
            "session_id": rep.session_id, "alert_rank": rank[0] if rank else None,
            "user": lab[2] if lab else None,
            "true_label": None if lab is None or lab[0] is None else int(lab[0]),
            "scenario": lab[1] if lab else None,
            "agent_verdict": rep.verdict, "claims": s["claims"], "factual": factual,
            "verified": s["verified"], "unsupported": s["unsupported"], "interpretive": s["interpretive"],
            "groundedness": s["groundedness"], "claims_with_warnings": s.get("warnings", 0),
            "normalized_fixes": normalized, "llm_calls": usage.get("calls"),
            "prompt_tokens": usage.get("prompt_tokens"), "completion_tokens": usage.get("completion_tokens"),
            "seconds": seconds, "backend": raw.get("backend"),
        })
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        with open(out_csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)

    def agg(rs):
        f = sum(r["factual"] for r in rs)
        v = sum(r["verified"] for r in rs)
        return {"reports": len(rs), "factual": f, "verified": v,
                "groundedness": round(v / f, 3) if f else None,
                "fully_grounded_reports": sum(r["unsupported"] == 0 for r in rs)}

    out = {"all": agg(rows)}
    if rows and rows[0]["true_label"] is not None:
        out["insider_sessions"] = agg([r for r in rows if r["true_label"] == 1])
        out["benign_sessions"] = agg([r for r in rows if r["true_label"] == 0])
        out["verdict_vs_label"] = {
            f"label={lab}": dict(Counter(r["agent_verdict"] for r in rows if r["true_label"] == lab))
            for lab in (1, 0)}
    secs = [r["seconds"] for r in rows if r["seconds"] is not None]
    toks = [(r["prompt_tokens"] or 0) + (r["completion_tokens"] or 0) for r in rows]
    if secs:
        out["latency_seconds"] = {"median": st.median(secs), "mean": round(st.mean(secs), 1), "max": max(secs)}
    if toks:
        out["tokens_per_report"] = {"median": st.median(toks), "mean": round(st.mean(toks)), "max": max(toks)}
    out["normalized_fixes_total"] = sum(r["normalized_fixes"] for r in rows)
    return out
