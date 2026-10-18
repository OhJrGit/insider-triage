"""Investigation agent: a small tool-calling loop that ends in a structured Report.

The protocol is plain JSON (not provider-specific function calling) so it works with any
OpenAI-compatible model, including local Ollama models:
    {"action": "tool", "tool": "<name>", "args": {...}}
    {"action": "final", "report": {...Report JSON...}}

`MockInvestigator` builds a correct report from rules, with optional injected
hallucinations. Use it to develop the verifier and demo without an LLM.
"""
from __future__ import annotations

import json
import random
import re
from datetime import datetime, timedelta
from pathlib import Path

from .. import config
from .schema import ASSERTABLE_FIELDS, SESSION_FIELDS, Claim, Report, schema_prompt
from .tools import Tools

PRELOAD_EVENTS = 120
MAX_EMPTY_NUDGES = 2

SYSTEM_PROMPT = """You are a SOC analyst assistant investigating a user session flagged by an anomaly detector
in an organisation's activity logs (logons, removable-device use, web browsing, email, file copies).
Your job: work out what happened in the session, compare it with the user's normal behaviour, and
write an evidence-grounded report. Every factual statement will be checked automatically against
the raw logs, so only state what the events show, and cite them.

Respond with ONE JSON object and nothing else, either:
  {{"action": "tool", "tool": "<tool name>", "args": {{...}}}}
or, when done:
  {{"action": "final", "report": {{...}}}}

The session's events and the user's 30-day baseline are already given to you. Only call a tool if
you need something that is not there (e.g. more http events, or earlier activity). Never repeat a call.

Tools:
{tools}

{schema}

EXAMPLE claims (ids are made up; copy the SHAPE, never these values):
  {{"text": "ABC1234 logged on to PC-1111 at 2010-05-04 22:10:05.", "type": "factual",
    "citations": ["{{AAAA-11111111-11111111}}"],
    "asserted": {{"source": "logon", "activity": "Logon", "pc": "PC-1111", "date": "2010-05-04 22:10:05"}}}}
  {{"text": "ABC1234 logged off at 2010-05-04 23:01:44.", "type": "factual",
    "citations": ["{{BBBB-22222222-22222222}}"],
    "asserted": {{"source": "logon", "activity": "Logoff", "date": "2010-05-04 23:01:44"}}}}
  {{"text": "A removable device was connected 2 times.", "type": "factual",
    "citations": ["{{CCCC-33333333-33333333}}", "{{DDDD-44444444-44444444}}"],
    "asserted": {{"source": "device", "activity": "Connect"}}, "count": 2}}
  {{"text": "The user visited wikileaks.org 3 times.", "type": "factual",
    "citations": ["{{EEEE-...}}", "{{FFFF-...}}", "{{GGGG-...}}"],
    "asserted": {{"source": "http", "url_contains": "wikileaks.org"}}, "count": 3}}
  {{"text": "Over the previous 30 days the user averaged 0 USB connections and started work around 08:12.",
    "type": "factual", "citations": [], "baseline": {{"avg_usb_connects": 0.0, "avg_start_hour": 8.2}}}}
  {{"text": "The session lasted 107 minutes and included 8 USB connections.", "type": "factual",
    "citations": [], "session": {{"duration_min": 107.0, "n_usb_connect": 8}}}}
  {{"text": "This activity is outside the user's usual working hours.", "type": "interpretive", "citations": []}}
Note how the logon and the logoff are two separate claims: asserted values must hold for EVERY cited
event, so events that differ in time, URL, filename etc. need separate claims (or a _contains match
that is true for all of them). "count" is a top-level field of the claim, never inside "asserted".
A report must contain claims; include the most important observed facts even if the session looks benign.
"""


def _parse_json(text: str) -> dict:
    text = text.strip()
    m = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.S)
    if m:
        text = m.group(1)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("no JSON object in response")
    return json.loads(text[start:end + 1])


def _preload(tools: Tools, session_id: str) -> tuple[list[dict], int]:
    """All non-http events plus as many http events as fit in PRELOAD_EVENTS."""
    events = tools.get_session_events(session_id, limit=100_000)
    if len(events) <= PRELOAD_EVENTS:
        return events, 0
    non_http = [e for e in events if e["source"] != "http"]
    http = [e for e in events if e["source"] == "http"]
    keep = non_http + http[: max(PRELOAD_EVENTS - len(non_http), 0)]
    keep.sort(key=lambda e: (e["date"], e["id"]))
    return keep, len(events) - len(keep)


def normalize_payload(payload: dict) -> list[str]:
    """Fix purely structural slips WITHOUT changing what a claim says. Every fix is logged in the
    trace so it can be reported (e.g. "n% of claims needed normalisation")."""
    notes = []
    for i, c in enumerate(payload.get("claims") or []):
        if not isinstance(c, dict):
            continue
        a = c.get("asserted")
        if isinstance(a, dict):
            if "count" in a and c.get("count") is None:
                try:
                    c["count"] = int(a.pop("count"))
                    notes.append(f"claim {i}: moved 'count' out of 'asserted'")
                except (TypeError, ValueError):
                    pass
            for k in [k for k in a if k in SESSION_FIELDS and k not in ASSERTABLE_FIELDS]:
                c.setdefault("session", {})
                if c["session"] is None:
                    c["session"] = {}
                c["session"].setdefault(k, a.pop(k))
                notes.append(f"claim {i}: moved session field '{k}' from 'asserted' to 'session'")
            for k in [k for k, v in a.items() if v is None or v == ""]:
                a.pop(k)
                notes.append(f"claim {i}: dropped empty asserted field '{k}'")
        elif a is None:
            c["asserted"] = {}
        for agg in ("baseline", "session"):
            if c.get(agg) is None:
                c[agg] = {}
        if c.get("citations") is None:
            c["citations"] = []
        elif isinstance(c.get("citations"), str):
            c["citations"] = [c["citations"]]
            notes.append(f"claim {i}: wrapped single citation in a list")
    return notes


def investigate(con, session_id: str, llm, max_steps: int = 8) -> tuple[Report, list[dict]]:
    tools = Tools(con)
    session = tools.get_session(session_id)
    events, hidden = _preload(tools, session_id)
    baseline = tools.get_user_baseline(session["user"], session["start_ts"])
    system = SYSTEM_PROMPT.format(tools=tools.spec_text(), schema=schema_prompt())
    user_msg = (
        f"Investigate session {session_id}.\n\nSession summary:\n{json.dumps(session, default=str)}\n\n"
        f"User baseline (30 days before this session):\n{json.dumps(baseline, default=str)}\n\n"
        f"Events in the session ({len(events)} shown"
        + (f", {hidden} more http events available via get_session_events with source='http'" if hidden else "")
        + "):\n" + "\n".join(json.dumps(e, default=str) for e in events)
    )
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user_msg}]
    trace: list[dict] = []
    seen_calls: set[str] = set()
    empty_nudges = 0

    for step in range(max_steps + 1):
        if step == max_steps:
            messages.append({"role": "user", "content": "Step limit reached. Return your final report now."})
        raw = llm.chat(messages)
        messages.append({"role": "assistant", "content": raw})
        try:
            obj = _parse_json(raw)
        except Exception as e:
            trace.append({"step": step, "error": f"bad JSON: {e}", "raw": raw[:500]})
            messages.append({"role": "user", "content": "That was not valid JSON. Reply with one JSON object."})
            continue
        if obj.get("action") == "tool" and step < max_steps:
            key = json.dumps([obj.get("tool"), obj.get("args")], sort_keys=True, default=str)
            if key in seen_calls:
                trace.append({"step": step, "repeated_tool": obj.get("tool"), "args": obj.get("args")})
                messages.append({"role": "user", "content":
                                 "You already made this exact call; its result is above. Do not repeat calls. "
                                 "Use what you have and return your final report."})
                continue
            seen_calls.add(key)
            result = tools.call(obj.get("tool", ""), obj.get("args", {}))
            trace.append({"step": step, "tool": obj.get("tool"), "args": obj.get("args")})
            messages.append({"role": "user", "content": "TOOL RESULT:\n" + json.dumps(result, default=str)[:30_000]})
            continue
        payload = obj.get("report", obj)
        if not isinstance(payload, dict):
            payload = {}
        payload["session_id"] = session_id
        if not payload.get("claims") and events and empty_nudges < MAX_EMPTY_NUDGES and step < max_steps:
            empty_nudges += 1
            trace.append({"step": step, "error": "final report had no claims; asked again"})
            messages.append({"role": "user", "content":
                             "Your report contains no claims. Write the report again with the key factual "
                             "claims (with citations and asserted fields) plus your interpretation."})
            continue
        notes = normalize_payload(payload)
        try:
            report = Report.model_validate(payload)
            trace.append({"step": step, "final": True, **({"normalized": notes} if notes else {})})
            return report, trace
        except Exception as e:
            trace.append({"step": step, "error": f"invalid report: {e}"})
            messages.append({"role": "user", "content": f"Report did not match the schema: {e}. Fix it."})
    raise RuntimeError(f"agent did not produce a valid report for {session_id}")


# ---------------------------------------------------------------------------------------
class MockInvestigator:
    """Rule-based stand-in for the LLM. Produces claims in exactly the agreed format.
    `hallucinate=k` appends k plausible but WRONG factual claims (fake id, shifted time,
    wrong PC...) so you can watch the verifier catch them."""

    def __init__(self, hallucinate: int = 0, seed: int = 0):
        self.hallucinate = hallucinate
        self.rng = random.Random(seed)

    def investigate(self, con, session_id: str) -> tuple[Report, list[dict]]:
        tools = Tools(con)
        s = tools.get_session(session_id)
        ev = tools.get_session_events(session_id, limit=100_000)
        claims: list[Claim] = []
        logon = next((e for e in ev if e["source"] == "logon" and e["activity"] == "Logon"), None)
        if logon:
            claims.append(Claim(
                text=f"{logon['user']} logged on to {logon['pc']} at {logon['date']}.",
                type="factual", citations=[logon["id"]],
                asserted={"source": "logon", "activity": "Logon", "pc": logon["pc"], "date": logon["date"]}))
        usb = [e for e in ev if e["source"] == "device" and e["activity"] == "Connect"]
        if usb:
            claims.append(Claim(
                text=f"A removable device was connected {len(usb)} time(s) during the session.",
                type="factual", citations=[e["id"] for e in usb], count=len(usb),
                asserted={"source": "device", "activity": "Connect"}))
        files = [e for e in ev if e["source"] == "file"]
        if files:
            claims.append(Claim(
                text=f"{len(files)} file event(s) were recorded, e.g. {files[0]['filename']}.",
                type="factual", citations=[e["id"] for e in files], count=len(files),
                asserted={"source": "file"}))
        for label, pat in [("leak", config.LEAK_URL_PATTERN), ("job-search", config.JOB_URL_PATTERN),
                           ("hacking-tool", config.HACK_URL_PATTERN)]:
            hits = [e for e in ev if e["source"] == "http" and re.search(pat, e["url"].lower())]
            by_dom: dict[str, list[dict]] = {}
            for e in hits:
                by_dom.setdefault(re.sub(r"^https?://", "", e["url"]).split("/")[0], []).append(e)
            for dom, es in by_dom.items():
                claims.append(Claim(
                    text=f"The user visited {dom} ({label} site) {len(es)} time(s).",
                    type="factual", citations=[e["id"] for e in es], count=len(es),
                    asserted={"source": "http", "url_contains": dom}))
        ext = [e for e in ev if e["source"] == "email"
               and "@" in (e["to"] + e["cc"] + e["bcc"]).lower().replace(f"@{config.ORG_DOMAIN}", "")]
        if ext:
            claims.append(Claim(
                text=f"{len(ext)} email(s) were sent to external addresses.",
                type="factual", citations=[e["id"] for e in ext], count=len(ext),
                asserted={"source": "email"}))
        claims.append(Claim(
            text=f"The session lasted {s['duration_min']:.0f} minutes with {s['n_http']:.0f} web visits.",
            type="factual", session={"duration_min": round(s["duration_min"]), "n_http": int(s["n_http"])}))
        bl = tools.get_user_baseline(s["user"], s["start_ts"])
        if bl.get("sessions"):
            claims.append(Claim(
                text=f"In the previous 30 days the user averaged {bl['avg_usb_connects']} USB connections per session.",
                type="factual", baseline={"avg_usb_connects": bl["avg_usb_connects"]}))
        if s.get("after_hours"):
            claims.append(Claim(text="The session took place outside normal working hours.",
                                type="interpretive", citations=[logon["id"]] if logon else []))
        if usb and (files or any("leak" in c.text for c in claims)):
            claims.append(Claim(text="Removable media use combined with file or upload activity is "
                                     "consistent with possible data exfiltration.", type="interpretive"))

        for _ in range(self.hallucinate):
            claims.append(self._fake(ev, s))

        verdict = "suspicious" if any(c.type == "interpretive" for c in claims) else "unclear"
        report = Report(session_id=session_id, verdict=verdict,
                        summary=f"Session of {s['user']} on {s['pc']} from {s['start_ts']} to {s['end_ts']}.",
                        claims=claims)
        return report, [{"mock": True, "hallucinated": self.hallucinate}]

    def _fake(self, ev: list[dict], s: dict) -> Claim:
        kind = self.rng.choice(["fake_id", "shift_time", "wrong_pc"])
        base = self.rng.choice(ev)
        if kind == "fake_id":
            fid = "{FAKE-" + "".join(self.rng.choice("ABCDEFGH0123456789") for _ in range(8)) + "-00000000}"
            return Claim(text="The user copied payroll_2010.xls to a removable drive.", type="factual",
                         citations=[fid], asserted={"source": "file", "filename": "payroll_2010.xls"})
        if kind == "shift_time":
            t = datetime.fromisoformat(base["date"]) + timedelta(hours=self.rng.choice([-3, -2, 2, 3]))
            return Claim(text=f"A {base['source']} event occurred at {t:%Y-%m-%d %H:%M:%S}.", type="factual",
                         citations=[base["id"]], asserted={"source": base["source"], "date": f"{t:%Y-%m-%d %H:%M:%S}"})
        return Claim(text=f"The {base['source']} activity happened on PC-0000.", type="factual",
                     citations=[base["id"]], asserted={"pc": "PC-0000"})


def save_report(report: Report, trace: list[dict], backend: str, out_dir: Path = config.REPORTS_DIR) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / f"{report.session_id}.json"
    p.write_text(json.dumps({"backend": backend, "report": report.model_dump(), "trace": trace}, indent=2,
                            default=str))
    return p


def load_report(path: Path) -> Report:
    data = json.loads(Path(path).read_text())
    return Report.model_validate(data.get("report", data))
