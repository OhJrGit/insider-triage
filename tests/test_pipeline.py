"""End-to-end tests on the synthetic sample. Run: pytest -q"""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from make_sample_data import build  # noqa: E402
from triage import config, db, detect, features, ingest, sessions  # noqa: E402
from triage.agent.investigate import MockInvestigator, investigate  # noqa: E402
from triage.agent.schema import Claim, Report  # noqa: E402
from triage.verifier.verify import check_field, verify_report  # noqa: E402


@pytest.fixture(scope="session")
def con(tmp_path_factory):
    d = tmp_path_factory.mktemp("cert")
    config.WORK_DIR, config.REPORTS_DIR = d, d / "reports"   # keep test output out of ./work
    build(d, n_users=15, n_days=110, seed=3)
    c = db.connect(d / "t.duckdb")
    ingest.run(c, d / "r4.2", d / "answers", "r4.2")
    sessions.run(c)
    features.run(c)
    detect.run(c, detect.DetectConfig(models=("iforest",), top_k=30))
    return c


def test_answers_join_events(con):
    missing = con.execute("SELECT count(*) FROM malicious_events ANTI JOIN events USING (id)").fetchone()[0]
    assert missing == 0
    assert con.execute("SELECT sum(label) FROM features").fetchone()[0] > 0


def test_every_event_in_at_most_one_session(con):
    dup = con.execute("SELECT count(*) - count(DISTINCT id) FROM events").fetchone()[0]
    assert dup == 0


def test_detector_ranks_insiders_high(con):
    top = con.execute("SELECT count(DISTINCT \"user\") FROM alerts WHERE label = 1").fetchone()[0]
    assert top >= 2


def _top_session(con):
    return con.execute("SELECT session_id FROM alerts ORDER BY rank LIMIT 1").fetchone()[0]


def test_mock_report_fully_verified(con):
    rep, _ = MockInvestigator().investigate(con, _top_session(con))
    res = verify_report(con, rep)
    assert res["summary"]["unsupported"] == 0
    assert res["summary"]["verified"] >= 1


def test_hallucinations_are_caught(con):
    rep, _ = MockInvestigator(hallucinate=3, seed=1).investigate(con, _top_session(con))
    res = verify_report(con, rep)
    assert res["summary"]["unsupported"] == 3


def test_check_field_rules():
    ev = {"id": "x", "source": "email", "date": "2010-03-01 22:14:05", "user": "U", "pc": "PC-1",
          "activity": "Send", "to": "a@dtaa.com;b@gmail.com", "cc": "", "bcc": "", "from": "U@dtaa.com",
          "size": 1234, "attachments": 2}
    assert check_field(ev, "date", "2010-03-01 22:14") is None          # minute precision
    assert check_field(ev, "date", "2010-03-01") is None                # day precision
    assert check_field(ev, "date", "2010-03-01 22:15") is not None
    assert check_field(ev, "time", "22:14") is None
    assert check_field(ev, "to", "b@gmail.com") is None                 # one of the recipients
    assert check_field(ev, "to_contains", "gmail") is None
    assert check_field(ev, "attachments", 2) is None
    assert check_field(ev, "attachments", "3") is not None
    assert check_field(ev, "url", "http://x") is not None               # wrong source for field
    assert check_field(ev, "colour", "red") is not None                 # unknown field


def test_factual_claim_needs_checkable_content(con):
    sid = _top_session(con)
    eid = con.execute("SELECT id FROM events WHERE session_id = ? LIMIT 1", [sid]).fetchone()[0]
    rep = Report(session_id=sid, claims=[
        Claim(text="The user uploaded secrets to wikileaks.", type="factual", citations=[eid])])
    assert verify_report(con, rep)["claims"][0]["status"] == "unsupported"


class ScriptedLLM:
    """Plays back fixed responses: one tool call, then a final report citing a real event."""

    def __init__(self, event):
        self.event, self.calls = event, 0

    def chat(self, messages):
        self.calls += 1
        if self.calls == 1:
            return json.dumps({"action": "tool", "tool": "get_user_baseline",
                               "args": {"user": self.event["user"], "before": self.event["date"]}})
        assert "TOOL RESULT" in messages[-1]["content"]
        e = self.event
        return "```json\n" + json.dumps({"action": "final", "report": {
            "verdict": "suspicious", "summary": "test",
            "claims": [
                {"text": f"Logged on to {e['pc']} at {e['date']}", "type": "factual", "citations": [e["id"]],
                 "asserted": {"source": "logon", "pc": e["pc"], "date": e["date"]}},
                {"text": "Unusual for this user", "type": "interpretive"},
            ]}}) + "\n```"


def test_llm_loop_with_scripted_model(con):
    from triage.agent.tools import Tools
    sid = _top_session(con)
    logon = [e for e in Tools(con).get_session_events(sid) if e["source"] == "logon"][0]
    rep, trace = investigate(con, sid, ScriptedLLM(logon))
    assert trace[0]["tool"] == "get_user_baseline"
    res = verify_report(con, rep)
    assert [c["status"] for c in res["claims"]] == ["verified", "interpretive"]


def test_ollama_backend_against_fake_server(con):
    """OllamaLLM speaks /api/chat, sends num_ctx, and counts tokens."""
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    from triage.agent.llm import OllamaLLM
    from triage.agent.tools import Tools

    sid = _top_session(con)
    logon = [e for e in Tools(con).get_session_events(sid) if e["source"] == "logon"][0]
    scripted = ScriptedLLM(logon)
    seen = []

    class H(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen.append(body)
            out = {"message": {"role": "assistant", "content": scripted.chat(body["messages"])},
                   "prompt_eval_count": 100, "eval_count": 20}
            data = json.dumps(out).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *a):
            pass

    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        llm = OllamaLLM(model="fake", base_url=f"http://127.0.0.1:{srv.server_port}/v1", num_ctx=8192)
        rep, _ = investigate(con, sid, llm)
    finally:
        srv.shutdown()
    assert seen[0]["options"]["num_ctx"] == 8192 and seen[0]["format"] == "json"
    assert llm.usage == {"calls": 2, "prompt_tokens": 200, "completion_tokens": 40}
    assert verify_report(con, rep)["summary"]["verified"] == 1


def test_normalize_moves_count_out_of_asserted():
    from triage.agent.investigate import normalize_payload
    p = {"claims": [{"text": "22 files", "type": "factual", "citations": "x",
                     "asserted": {"count": 22, "source": "file", "pc": None}}]}
    notes = normalize_payload(p)
    c = p["claims"][0]
    assert c["count"] == 22 and c["asserted"] == {"source": "file"} and c["citations"] == ["x"]
    assert len(notes) == 3


def test_unknown_field_reported_once(con):
    sid = _top_session(con)
    ids = [r[0] for r in con.execute("SELECT id FROM events WHERE session_id = ? LIMIT 5", [sid]).fetchall()]
    rep = Report(session_id=sid, claims=[Claim(text="x", type="factual", citations=ids, asserted={"colour": "red"})])
    reasons = verify_report(con, rep)["claims"][0]["reasons"]
    assert reasons == ["unknown field 'colour'"]


class EmptyThenGoodLLM(ScriptedLLM):
    """First answers with an empty report, then behaves like ScriptedLLM."""
    def chat(self, messages):
        if not getattr(self, "_nudged", False):
            self._nudged = True
            return json.dumps({"action": "final", "report": {"verdict": "unclear", "claims": []}})
        return super().chat(messages)


def test_empty_report_gets_a_retry(con):
    from triage.agent.tools import Tools
    sid = _top_session(con)
    logon = [e for e in Tools(con).get_session_events(sid) if e["source"] == "logon"][0]
    rep, trace = investigate(con, sid, EmptyThenGoodLLM(logon))
    assert any("no claims" in t.get("error", "") for t in trace)
    assert len(rep.claims) == 2


def test_baseline_claims_are_checked(con):
    from triage.agent.tools import Tools
    sid = _top_session(con)
    s = Tools(con).get_session(sid)
    bl = Tools(con).get_user_baseline(s["user"], s["start_ts"])
    good = Claim(text="x", type="factual", baseline={"avg_http": bl["avg_http"], "usual_pc": bl["usual_pc"]})
    rounded = Claim(text="x", type="factual", baseline={"avg_http": round(bl["avg_http"])})
    bad = Claim(text="x", type="factual", baseline={"avg_http": bl["avg_http"] + 50})
    unknown = Claim(text="x", type="factual", baseline={"avg_mood": 3})
    res = verify_report(con, Report(session_id=sid, claims=[good, rounded, bad, unknown]))
    assert [c["status"] for c in res["claims"]] == ["verified", "verified", "unsupported", "unsupported"]


def test_interpretive_numbers_warn(con):
    sid = _top_session(con)
    rep = Report(session_id=sid, claims=[
        Claim(text="Usual hours are 7:24 to 16:26.", type="interpretive"),
        Claim(text="This looks unusual.", type="interpretive")])
    res = verify_report(con, rep)
    assert res["claims"][0]["warnings"] and not res["claims"][1]["warnings"]
    strict = verify_report(con, rep, strict_text=True)
    assert strict["claims"][0]["status"] == "unsupported"


def test_session_claims_are_checked(con):
    from triage.agent.tools import Tools
    sid = _top_session(con)
    s = Tools(con).get_session(sid)
    good = Claim(text="x", type="factual", session={"duration_min": s["duration_min"], "pc": s["pc"],
                                                    "n_http": int(s["n_http"])})
    bad = Claim(text="x", type="factual", session={"n_http": s["n_http"] + 7})
    unknown = Claim(text="x", type="factual", session={"dev_n_http": 1})
    res = verify_report(con, Report(session_id=sid, claims=[good, bad, unknown]))
    assert [c["status"] for c in res["claims"]] == ["verified", "unsupported", "unsupported"]


def test_count_words_anchor():
    from triage.verifier.verify import _anchor_warnings
    assert _anchor_warnings("visited two websites", {}, 2) == []
    assert _anchor_warnings("visited it twice", {}, 2) == []
    assert _anchor_warnings("visited several websites", {}, 2)


def test_date_range_assertion():
    ev = {"id": "x", "source": "file", "date": "2010-03-01 22:14:05", "user": "U", "pc": "PC-1",
          "activity": "File", "filename": "a.doc"}
    assert check_field(ev, "date", "2010-03-01 22:00:00 to 2010-03-01 23:00:00") is None
    assert check_field(ev, "date", "2010-03-01 22:14 to 2010-03-01 22:30") is None
    assert check_field(ev, "date", "2010-03-01 22:20:00 to 2010-03-01 23:00:00") is not None


def test_normalize_moves_session_fields():
    from triage.agent.investigate import normalize_payload
    p = {"claims": [{"text": "x", "type": "factual", "citations": [],
                     "asserted": {"start_hour": 20.5, "pc": "PC-1", "n_file_exec_archive": 1}}]}
    notes = normalize_payload(p)
    c = p["claims"][0]
    assert c["asserted"] == {"pc": "PC-1"} and c["session"] == {"start_hour": 20.5, "n_file_exec_archive": 1}
    assert len(notes) == 2


def test_summarize(con, tmp_path):
    from triage.agent.investigate import save_report
    from triage.summary import summarize
    for sid in [r[0] for r in con.execute("SELECT session_id FROM alerts ORDER BY rank LIMIT 3").fetchall()]:
        rep, trace = MockInvestigator().investigate(con, sid)
        save_report(rep, trace + [{"usage": {"calls": 1, "prompt_tokens": 10, "completion_tokens": 5}, "seconds": 1.0}],
                    "mock", out_dir=tmp_path / "r")
    res = summarize(con, tmp_path / "r", tmp_path / "s.csv")
    assert res["all"]["reports"] == 3 and res["all"]["groundedness"] == 1.0
    assert "verdict_vs_label" in res and (tmp_path / "s.csv").exists()


def test_strict_citation_rules(con):
    sid = _top_session(con)
    eid = con.execute("SELECT id FROM events WHERE session_id = ? LIMIT 1", [sid]).fetchone()[0]
    count_only = Claim(text="1 event", type="factual", citations=[eid], count=1)
    no_cite = Claim(text="on PC-0", type="factual", asserted={"pc": "PC-0"}, baseline={"sessions": 0})
    rep = Report(session_id=sid, claims=[count_only, no_cite])
    v2 = [c["status"] for c in verify_report(con, rep)["claims"]]
    assert v2 == ["unsupported", "unsupported"]
    v1 = verify_report(con, rep, strict_citations=False)["claims"]
    assert v1[0]["status"] == "verified"


class FakeJudge:
    """Says 'supported' to everything -- lets us check the scoring logic."""
    def __init__(self):
        self.reset_usage()

    def reset_usage(self):
        self.usage = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0}

    def chat(self, messages):
        self.usage["calls"] += 1
        return '{"verdict": "supported", "reason": "looks fine"}'

    def describe(self):
        return "fake-judge"


def test_judge_baseline(con, tmp_path):
    from triage import judge
    from triage.agent.investigate import save_report
    for sid in [r[0] for r in con.execute("SELECT session_id FROM alerts ORDER BY rank LIMIT 3").fetchall()]:
        rep, trace = MockInvestigator().investigate(con, sid)
        save_report(rep, trace, "mock", out_dir=tmp_path / "r")
    res = judge.run(con, FakeJudge(), tmp_path / "r", max_good=10, out_dir=tmp_path, repeats=2)
    assert res["llm_judge"]["false_alarm_rate"] == 0.0 and res["llm_judge"]["recall_on_errors"] == 0.0
    assert res["deterministic_verifier"]["recall_on_errors"] >= 0.9
    assert res["judge_self_consistency"] == 1.0


def test_detect_seeds(con):
    from triage import detect
    res = detect.run_seeds(con, detect.DetectConfig(models=("iforest",), top_k=30), [0, 1])
    assert "roc_auc" in res["models"]["iforest"] and res["models"]["iforest"]["roc_auc"]["std"] >= 0


def test_pilot_flow(con, tmp_path, monkeypatch):
    from triage import pilot
    from triage.agent.investigate import save_report
    monkeypatch.setattr(pilot, "PILOT_DIR", tmp_path / "pilot")
    monkeypatch.setattr(pilot, "CASES", tmp_path / "pilot" / "cases.json")
    monkeypatch.setattr(pilot, "RESPONSES", tmp_path / "pilot" / "responses.csv")
    monkeypatch.setattr(pilot, "SURVEY", tmp_path / "pilot" / "survey.csv")
    monkeypatch.setattr(pilot, "INTERACTIONS", tmp_path / "pilot" / "interactions.csv")
    ins = [r[0] for r in con.execute("SELECT session_id FROM features WHERE label = 1 LIMIT 2").fetchall()]
    ben = [r[0] for r in con.execute("SELECT session_id FROM features WHERE label = 0 LIMIT 2").fetchall()]
    for sid in ins + ben:
        rep, trace = MockInvestigator().investigate(con, sid)
        save_report(rep, trace, "mock", out_dir=tmp_path / "r")
    cases = pilot.prepare(con, 4, 0, tmp_path / "r")
    assert sum(c["label"] for c in cases) == 2
    for p in range(1, 7):
        plan = pilot.schedule(p, cases)
        assert sorted(c["condition"] for c in plan).count("B_report") == 2   # half/half
        # ...and half of the insiders / half of the benign cases in each condition
        assert sorted((c["label"], c["condition"]) for c in plan) == [
            (0, "A_alert_only"), (0, "B_report"), (1, "A_alert_only"), (1, "B_report")]
        for c in plan:
            pilot.record(p, c, "insider" if c["condition"] == "B_report" else "not insider", 3, 10.0)
            pilot.record_interactions(p, c, c["condition"] == "A_alert_only",
                                      {"verified": 3, "unsupported": 1} if c["condition"] == "B_report" else {},
                                      {"verified": 1, "unsupported": 1} if c["condition"] == "B_report" else {})
    assert pilot.done_cases(1) == {0, 1, 2, 3}
    res = pilot.analyze(pilot.RESPONSES)
    assert res["participants"] == 6 and "wilcoxon_p" in res
    # B: everything flagged (all hits, all false alarms); A: nothing flagged. Same accuracy, opposite bias
    a, b = res["by_condition"]["A_alert_only"], res["by_condition"]["B_report"]
    assert a["accuracy"] == b["accuracy"] == 0.5
    assert (b["hit_rate"], b["false_alarm_rate"], a["hit_rate"], a["false_alarm_rate"]) == (1.0, 1.0, 0.0, 0.0)
    assert a["d_prime"] == b["d_prime"] == 0.0 and b["criterion_c"] < 0 < a["criterion_c"]
    ev = res["evidence_checking"]
    assert ev["verified_claims_opened"] == "12/36" and ev["unsupported_open_rate"] == 1.0
    assert ev["raw_log_opened_rate"] == {"A_alert_only": 1.0, "B_report": 0.0}
    # every case shown in both conditions across participants
    seen = {(c["case_no"], c["condition"]) for p in range(1, 7) for c in pilot.schedule(p, cases)}
    assert len(seen) == 8


def test_pilot_outcomes():
    from triage.pilot import outcomes

    def rows(hits, misses, fas, crs):
        return ([{"decision": "insider", "correct": 1}] * hits + [{"decision": "not insider", "correct": 0}] * misses
                + [{"decision": "insider", "correct": 0}] * fas + [{"decision": "not insider", "correct": 1}] * crs)
    a, b = outcomes(rows(12, 8, 8, 12)), outcomes(rows(16, 4, 13, 7))
    assert (a["hit_rate"], a["false_alarm_rate"], b["hit_rate"], b["false_alarm_rate"]) == (0.6, 0.4, 0.8, 0.65)
    assert a["criterion_c"] == 0.0 and b["criterion_c"] < -0.5        # B: much quicker to say "insider"
    assert abs(a["d_prime"] - b["d_prime"]) < 0.15                    # ...but not better at telling them apart
    perfect = outcomes(rows(5, 0, 0, 5))                              # rates of 1 and 0 stay finite
    assert 0 < perfect["d_prime"] < 4 and perfect["criterion_c"] == 0.0


def test_event_sequences(con):
    from triage.deep import MAX_LEN, load_sequences
    seqs = load_sequences(con)
    sid = _top_session(con)
    assert seqs[sid] and all("|" in t for t in seqs[sid]) and len(seqs[sid]) < MAX_LEN
    assert seqs[sid][0].startswith("logon|")


def test_deep_models(con):
    pytest.importorskip("torch")
    from triage import detect
    res = detect.run_seeds(con, detect.DetectConfig(models=("torch_ae", "transformer"), top_k=30), [0])
    for m in ("torch_ae", "transformer"):
        assert 0.0 <= res["models"][m]["roc_auc"]["mean"] <= 1.0


def test_blind_labelling_roundtrip(con, tmp_path):
    import csv as _csv
    from triage.agent.investigate import save_report
    from triage.evaluation import export_claims, score_claims
    for sid in [r[0] for r in con.execute("SELECT session_id FROM alerts ORDER BY rank LIMIT 2").fetchall()]:
        rep, trace = MockInvestigator(hallucinate=1).investigate(con, sid)
        save_report(rep, trace, "mock", out_dir=tmp_path / "r")
    out = tmp_path / "lab.csv"
    n = export_claims(con, out, tmp_path / "r", sample=6, seed=1)
    rows = list(_csv.DictReader(open(out, encoding="utf-8-sig")))
    assert n == 6 and "verifier_status" not in rows[0] and rows[0]["evidence"]
    key = {r["claim_id"]: r["verifier_status"] for r in _csv.DictReader(open(tmp_path / "lab_KEY.csv"))}
    for r in rows:   # a perfect labeller
        st = key[r["claim_id"]]
        r["human_label"] = r["human_label_2"] = {"verified": "s", "unsupported": "u", "interpretive": "i"}[st]
    with open(out, "w", newline="", encoding="utf-8-sig") as fh:
        w = _csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    res = score_claims(out)
    assert res["agreement_with_verifier"] == 1.0 and res["cohens_kappa"] in (1.0, None) or res["cohens_kappa"] == 1.0
