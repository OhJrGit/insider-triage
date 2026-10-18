"""Command-line entry point. Run `python cli.py -h` or `python cli.py <command> -h`."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from triage import config, db


def cmd_sample_data(a):
    sys.path.insert(0, str(config.ROOT / "scripts"))
    from make_sample_data import build
    build(Path(a.out), a.users, a.days, a.seed)


def cmd_ingest(a):
    from triage import ingest
    con = db.connect(a.db)
    t = time.time()
    ingest.run(con, Path(a.cert_dir), Path(a.answers_dir) if a.answers_dir else None, a.release,
               keep_content=not a.no_content, lenient=a.lenient)
    print(f"done in {time.time() - t:.1f}s -> {a.db or config.DB_PATH}")


def cmd_sessions(a):
    from triage import sessions
    n, m = sessions.run(db.connect(a.db), a.cap_hours)
    print(f"  {n:,} sessions, {m:,} events assigned to a session")


def cmd_features(a):
    from triage import features
    con = db.connect(a.db)
    n = features.run(con)
    lab = con.execute("SELECT sum(label) FROM features").fetchone()[0]
    print(f"  {n:,} session feature rows" + (f", {int(lab)} labelled malicious" if lab is not None else ""))


def cmd_detect(a):
    from triage import detect
    cfg = detect.DetectConfig(train_frac=a.train_frac, models=tuple(a.models), primary=a.primary,
                              top_k=a.top_k, train_benign_only=a.train_benign_only,
                              overwrite_alerts=a.overwrite_alerts)
    detect.run(db.connect(a.db), cfg)
    print(f"  alerts table = top {a.top_k} by {a.primary}; metrics in {config.WORK_DIR / 'detector_metrics.json'}")


def cmd_detect_seeds(a):
    from triage import detect
    cfg = detect.DetectConfig(train_frac=a.train_frac, models=tuple(a.models), top_k=a.top_k)
    res = detect.run_seeds(db.connect(a.db, read_only=True), cfg, a.seeds)
    for name, m in res["models"].items():
        k = f"insiders_caught@{a.top_k}"
        print(f"  {name:<12} ROC-AUC {m['roc_auc']['mean']:.3f}±{m['roc_auc']['std']:.3f}  "
              f"PR-AUC {m['pr_auc']['mean']:.3f}±{m['pr_auc']['std']:.3f}  "
              f"P@{a.top_k} {m[f'precision@{a.top_k}']['mean']:.3f}±{m[f'precision@{a.top_k}']['std']:.3f}  "
              f"insiders {m[k]['mean']} ({m[k]['min']}-{m[k]['max']}) of {m[k]['of']}")
    print(f"  -> {config.WORK_DIR / 'detector_seeds.json'}  (alerts table left unchanged)")


def cmd_judge(a):
    from triage import judge
    from triage.agent.llm import make_llm
    llm = make_llm(a.backend, a.model, a.base_url, a.num_ctx)
    print(f"  judge: {llm.describe()}")
    res = judge.run(db.connect(a.db, read_only=True), llm, Path(a.reports), a.max_claims, a.seed, a.repeats)
    print(json.dumps(res, indent=2))
    print(f"  details -> {config.WORK_DIR / 'judge_baseline.json'}")


def cmd_pilot_prepare(a):
    from triage import pilot
    cases = pilot.prepare(db.connect(a.db, read_only=True), a.cases, a.seed, Path(a.reports))
    print(f"  {len(cases)} cases ({sum(c['label'] for c in cases)} insider) -> {pilot.CASES}")
    print("  start the study with:  streamlit run app/pilot.py")


def cmd_pilot_analyze(a):
    from triage import pilot
    print(json.dumps(pilot.analyze(), indent=2))


def _session_ids(con, a) -> list[str]:
    if a.session:
        return a.session
    return [r[0] for r in con.execute("SELECT session_id FROM alerts ORDER BY rank LIMIT ? OFFSET ?",
                                      [a.top, a.skip]).fetchall()]


def cmd_investigate(a):
    from triage.agent.investigate import MockInvestigator, investigate, save_report
    from triage.agent.llm import make_llm
    con = db.connect(a.db, read_only=True)
    ids = _session_ids(con, a)
    if a.backend == "mock":
        agent, label = MockInvestigator(hallucinate=a.hallucinate), f"mock(hallucinate={a.hallucinate})"
    else:
        llm = make_llm(a.backend, a.model, a.base_url, a.num_ctx)
        label = llm.describe()
        print(f"  using {label}")
    ok = 0
    for sid in ids:
        t = time.time()
        try:
            if a.backend == "mock":
                rep, trace = agent.investigate(con, sid)
            else:
                llm.reset_usage()
                rep, trace = investigate(con, sid, llm, a.max_steps)
                trace.append({"usage": dict(llm.usage), "seconds": round(time.time() - t, 1)})
        except Exception as e:
            print(f"  {sid}: FAILED {type(e).__name__}: {e}")
            continue
        ok += 1
        p = save_report(rep, trace, label)
        extra = "" if a.backend == "mock" else (
            f", {llm.usage['calls']} calls, {llm.usage['prompt_tokens'] + llm.usage['completion_tokens']:,} tokens")
        print(f"  {sid}: {len(rep.claims)} claims, verdict={rep.verdict} ({time.time() - t:.1f}s{extra}) -> {p.name}")
    print(f"  {ok}/{len(ids)} reports written")


def cmd_verify(a):
    from triage.agent.investigate import load_report
    from triage.verifier.verify import verify_report
    con = db.connect(a.db, read_only=True)
    paths = ([config.REPORTS_DIR / f"{s}.json" for s in a.session] if a.session
             else sorted(p for p in config.REPORTS_DIR.glob("*.json") if not p.name.endswith(".verified.json")))
    totals = {"verified": 0, "unsupported": 0, "interpretive": 0}
    for p in paths:
        res = verify_report(con, load_report(p), strict_text=a.strict_text, strict_citations=not a.v1)
        p.with_suffix(".verified.json").write_text(json.dumps(res, indent=2))
        s = res["summary"]
        for k in totals:
            totals[k] += s[k]
        print(f"  {res['session_id']}: {s['verified']} verified, {s['unsupported']} unsupported, "
              f"{s['interpretive']} interpretive (groundedness {s['groundedness']})")
        if a.show:
            for c in res["claims"]:
                mark = {"verified": "OK ", "unsupported": "BAD", "interpretive": "INT"}[c["status"]]
                print(f"     [{mark}] {c['text']}")
                for r in c["reasons"]:
                    print(f"           - {r}")
                for w in c["warnings"]:
                    print(f"           ! {w}")
    print(f"  total: {totals}")


def cmd_eval_verifier(a):
    from triage.evaluation import eval_perturbation
    res = eval_perturbation(db.connect(a.db, read_only=True), seed=a.seed, strict_citations=not a.v1)
    out = config.WORK_DIR / "verifier_perturbation.json"
    out.write_text(json.dumps(res, indent=2))
    print(f"  {res['reports']} reports")
    for k, v in res["by_mutation"].items():
        print(f"  {k:<14} caught {v['caught']:>4}/{v['tried']:<4} ({v['detection_rate']:.0%})")
    print(f"  overall detection rate: {res['overall_detection_rate']}  (details + missed examples in {out})")


def cmd_summarize(a):
    from triage.summary import summarize
    res = summarize(db.connect(a.db, read_only=True), Path(a.reports), Path(a.out), strict_citations=not a.v1)
    print(json.dumps(res, indent=2, default=str))
    print(f"  per-report table -> {a.out}")


def cmd_export_claims(a):
    from triage.evaluation import export_claims
    n = export_claims(db.connect(a.db, read_only=True), Path(a.out), strict_citations=not a.v1,
                      sample=a.sample, seed=a.seed)
    print(f"  wrote {n} claims to {a.out} (blind: no verifier output in it)")
    print("  fill human_label with s / u / i (supported / unsupported / interpretive); a second person fills")
    print("  human_label_2 on some rows for inter-rater agreement. Then: cli.py score-claims " + str(a.out))


def cmd_score_claims(a):
    from triage.evaluation import score_claims
    print(json.dumps(score_claims(Path(a.labelled)), indent=2))


def main():
    ap = argparse.ArgumentParser(description="Agentic insider-threat triage pipeline")
    ap.add_argument("--db", default=None, help=f"DuckDB path (default {config.DB_PATH})")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("sample-data", help="write a small synthetic CERT-format dataset")
    p.add_argument("--out", default="datasets/sample")
    p.add_argument("--users", type=int, default=30)
    p.add_argument("--days", type=int, default=120)
    p.add_argument("--seed", type=int, default=7)
    p.set_defaults(fn=cmd_sample_data)

    p = sub.add_parser("ingest", help="load CERT CSVs (+answers) into DuckDB")
    p.add_argument("--cert-dir", required=True)
    p.add_argument("--answers-dir")
    p.add_argument("--release", default="r4.2")
    p.add_argument("--no-content", action="store_true", help="drop content columns (saves lots of space)")
    p.add_argument("--lenient", action="store_true", help="skip malformed CSV rows instead of failing")
    p.set_defaults(fn=cmd_ingest)

    p = sub.add_parser("sessions", help="build sessions and assign events")
    p.add_argument("--cap-hours", type=int, default=config.SESSION_CAP_HOURS)
    p.set_defaults(fn=cmd_sessions)

    p = sub.add_parser("features", help="build per-session features + labels")
    p.set_defaults(fn=cmd_features)

    p = sub.add_parser("detect", help="train/score anomaly detectors, write alerts")
    p.add_argument("--train-frac", type=float, default=0.5)
    p.add_argument("--models", nargs="+", default=["iforest", "ocsvm", "autoencoder"])
    p.add_argument("--primary", default="iforest")
    p.add_argument("--top-k", type=int, default=200)
    p.add_argument("--train-benign-only", action="store_true")
    p.add_argument("--overwrite-alerts", action="store_true",
                   help="rebuild alerts even though work/reports holds reports for the current ones")
    p.set_defaults(fn=cmd_detect)

    p = sub.add_parser("detect-seeds", help="detector metrics over several random seeds (mean/std); read-only")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--train-frac", type=float, default=0.5)
    p.add_argument("--models", nargs="+", default=["iforest", "ocsvm", "autoencoder"])
    p.add_argument("--top-k", type=int, default=200)
    p.set_defaults(fn=cmd_detect_seeds)

    p = sub.add_parser("judge-baseline", help="compare the deterministic verifier with an LLM-as-judge")
    p.add_argument("--reports", default=str(config.REPORTS_DIR))
    p.add_argument("--max-claims", type=int, default=120, help="true claims sampled (plus one broken copy each)")
    p.add_argument("--repeats", type=int, default=1, help=">1 also measures judge self-consistency")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--backend", choices=["ollama", "openai"], default="ollama")
    p.add_argument("--model")
    p.add_argument("--base-url")
    p.add_argument("--num-ctx", type=int, default=None)
    p.set_defaults(fn=cmd_judge)

    p = sub.add_parser("pilot-prepare", help="freeze the case set for the human pilot study")
    p.add_argument("--cases", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--reports", default=str(config.REPORTS_DIR))
    p.set_defaults(fn=cmd_pilot_prepare)

    p = sub.add_parser("pilot-analyze", help="analyse work/pilot/responses.csv")
    p.set_defaults(fn=cmd_pilot_analyze)

    p = sub.add_parser("investigate", help="run the agent on alerts")
    p.add_argument("--session", nargs="+", help="session ids (default: top alerts)")
    p.add_argument("--top", type=int, default=10, help="how many alerts to investigate")
    p.add_argument("--skip", type=int, default=0,
                   help="skip the first N alerts (e.g. --skip 10 to evaluate on alerts not used for prompt tuning)")
    p.add_argument("--backend", choices=["mock", "ollama", "openai"], default="mock",
                   help="'ollama' = local Ollama; 'openai' = any OpenAI-compatible API (OpenAI, Groq, LM Studio...)")
    p.add_argument("--model", help="e.g. qwen2.5:7b for ollama")
    p.add_argument("--base-url")
    p.add_argument("--num-ctx", type=int, default=None, help="ollama context window (default 32768)")
    p.add_argument("--max-steps", type=int, default=8)
    p.add_argument("--hallucinate", type=int, default=0, help="mock only: inject N wrong claims per report")
    p.set_defaults(fn=cmd_investigate)

    p = sub.add_parser("verify", help="verify saved reports")
    p.add_argument("--session", nargs="+")
    p.add_argument("--show", action="store_true", help="print every claim")
    p.add_argument("--strict-text", action="store_true", help="text-anchoring failures count as unsupported")
    p.add_argument("--v1", action="store_true", help="use verifier v1 rules (lenient citations) for comparison")
    p.set_defaults(fn=cmd_verify)

    p = sub.add_parser("eval-verifier", help="error-injection evaluation of the verifier")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--v1", action="store_true", help="use verifier v1 rules (lenient citations) for comparison")
    p.set_defaults(fn=cmd_eval_verifier)

    p = sub.add_parser("summarize", help="per-report results table + headline numbers for a run")
    p.add_argument("--reports", default=str(config.REPORTS_DIR))
    p.add_argument("--out", default=str(config.WORK_DIR / "eval_summary.csv"))
    p.add_argument("--v1", action="store_true", help="use verifier v1 rules (lenient citations) for comparison")
    p.set_defaults(fn=cmd_summarize)

    p = sub.add_parser("export-claims", help="CSV of claims for manual labelling")
    p.add_argument("--out", default=str(config.WORK_DIR / "claims_to_label.csv"))
    p.add_argument("--sample", type=int, default=None, help="label a random sample of N claims")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--v1", action="store_true", help="use verifier v1 rules (lenient citations) for comparison")
    p.set_defaults(fn=cmd_export_claims)

    p = sub.add_parser("score-claims", help="compare verifier with manual labels")
    p.add_argument("labelled")
    p.set_defaults(fn=cmd_score_claims)

    a = ap.parse_args()
    config.ensure_dirs()
    a.fn(a)


if __name__ == "__main__":
    main()
