"""Deterministic evidence verifier. No LLM involved.

For each claim in a Report:
  interpretive -> 'interpretive' (any cited ids must still exist and belong to the user,
                  otherwise 'unsupported')
  factual      -> 'verified' only if ALL of these hold, else 'unsupported':
      1. it cites at least one event id
      2. every cited id exists in the events table
      3. every cited event belongs to the session's user
      4. it asserts at least one checkable field (or a count)
      5. every asserted field matches EVERY cited event
      6. if `count` is set, it equals the number of distinct cited ids
Warnings (do not change the status, but are reported):
  - a cited event lies outside the investigated session
  - an asserted value does not appear in the claim text ("text anchoring").
    Pass strict_text=True to make this an error. This guards against a claim whose
    structured part is correct but whose prose says something else -- a known gap of
    structured verification that is worth discussing in the report.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from datetime import datetime

from ..agent.schema import ASSERTABLE_FIELDS, BASELINE_FIELDS, SESSION_FIELDS, Report
from ..agent.tools import EVENT_COLS, Tools, _row_to_event

DATE_FORMATS = [("%Y-%m-%d %H:%M:%S", "second"), ("%Y-%m-%d %H:%M", "minute"), ("%Y-%m-%d", "day"),
                ("%Y-%m-%dT%H:%M:%S", "second"), ("%m/%d/%Y %H:%M:%S", "second")]
TIME_FORMATS = [("%H:%M:%S", "second"), ("%H:%M", "minute")]
SOURCE_FIELDS = {"url": "http", "filename": "file", "to": "email", "cc": "email", "bcc": "email",
                 "from": "email", "size": "email", "attachments": "email"}
LIST_FIELDS = {"to", "cc", "bcc"}
INT_FIELDS = {"size", "attachments"}


@dataclass
class ClaimResult:
    index: int
    text: str
    type: str
    status: str                      # verified | unsupported | interpretive
    reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    citations: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------------------------
def _parse_dt(value: str, formats) -> tuple[datetime, str] | None:
    for fmt, precision in formats:
        try:
            return datetime.strptime(str(value).strip(), fmt), precision
        except ValueError:
            continue
    return None


def _truncate(t: datetime, precision: str) -> tuple:
    full = (t.year, t.month, t.day, t.hour, t.minute, t.second)
    return full[: {"day": 3, "minute": 5, "second": 6}[precision]]


def check_field(event: dict, key: str, value) -> str | None:
    """Return None if the asserted key=value holds for the event, else a reason string."""
    base, contains = (key[:-9], True) if key.endswith("_contains") else (key, False)
    if base not in ASSERTABLE_FIELDS:
        return f"unknown field '{key}'"
    if base in SOURCE_FIELDS and event["source"] != SOURCE_FIELDS[base]:
        return f"field '{base}' does not apply to a {event['source']} event ({event['id']})"
    if base == "date" and isinstance(value, str) and re.search(r"\s(to|-|–)\s", value.strip()):
        # A range "A to B": every cited event must fall inside it (at the precision given).
        parts = re.split(r"\s(?:to|-|–)\s", value.strip(), maxsplit=1)
        lo, hi = _parse_dt(parts[0], DATE_FORMATS), _parse_dt(parts[1], DATE_FORMATS)
        if lo is None or hi is None:
            return f"unparseable date range '{value}'"
        actual = datetime.fromisoformat(event["date"])
        if not (_truncate(lo[0], lo[1]) <= _truncate(actual, lo[1]) and
                _truncate(actual, hi[1]) <= _truncate(hi[0], hi[1])):
            return f"date range mismatch on {event['id']}: claimed {value}, log says {event['date']}"
        return None
    if base == "date":
        parsed = _parse_dt(value, DATE_FORMATS)
        if parsed is None:
            return f"unparseable date '{value}'"
        claimed, prec = parsed
        actual = datetime.fromisoformat(event["date"])
        if _truncate(claimed, prec) != _truncate(actual, prec):
            return f"date mismatch on {event['id']}: claimed {value}, log says {event['date']}"
        return None
    if base == "time":
        parsed = _parse_dt(value, TIME_FORMATS)
        if parsed is None:
            return f"unparseable time '{value}'"
        claimed, prec = parsed
        actual = datetime.fromisoformat(event["date"])
        n = 3 if prec == "second" else 2
        if (claimed.hour, claimed.minute, claimed.second)[:n] != (actual.hour, actual.minute, actual.second)[:n]:
            return f"time mismatch on {event['id']}: claimed {value}, log says {event['date'][11:]}"
        return None
    actual = event.get(base)
    if base in INT_FIELDS and not contains:
        try:
            ok = int(float(value)) == int(actual)
        except (TypeError, ValueError):
            ok = False
        return None if ok else f"{base} mismatch on {event['id']}: claimed {value}, log says {actual}"
    a = ("" if actual is None else str(actual)).strip().lower()
    v = str(value).strip().lower()
    if contains:
        ok = v in a
    elif base in LIST_FIELDS:
        logged = [y.strip() for y in a.split(";") if y.strip()]
        v_list = [x.strip() for x in v.split(";") if x.strip()]
        ok = len(v_list) > 0 and all(x in logged for x in v_list)
    else:
        ok = v == a
    return None if ok else f"{key} mismatch on {event['id']}: claimed '{value}', log says '{actual}'"


NUMBER_WORDS = {1: ("one", "once", "a single"), 2: ("two", "twice"), 3: ("three",), 4: ("four",),
                5: ("five",), 6: ("six",), 7: ("seven",), 8: ("eight",), 9: ("nine",), 10: ("ten",),
                11: ("eleven",), 12: ("twelve",)}


def _anchor_warnings(text: str, asserted: dict, count: int | None) -> list[str]:
    t = text.lower()
    out = []
    for k, v in asserted.items():
        base = k[:-9] if k.endswith("_contains") else k
        if base in ("source", "activity", "user", "size", "attachments"):
            continue  # usually paraphrased ("connected a USB drive"), don't demand literal text
        sv = str(v).strip().lower()
        if base in ("date", "time"):
            m = re.search(r"\d{1,2}:\d{2}", sv)   # look for HH:MM if a time was given
            sv = m.group(0) if m else sv
        if sv and sv not in t:
            out.append(f"asserted {k}='{v}' is not mentioned in the claim text")
    if count is not None and not re.search(rf"\b{count}\b", t) and not any(
            re.search(rf"\b{w}\b", t) for w in NUMBER_WORDS.get(count, ())):
        out.append(f"count {count} is not mentioned in the claim text")
    return out


def _check_aggregate(actual: dict, key: str, value, allowed: dict, label: str,
                     text_keys: tuple[str, ...]) -> str | None:
    """Compare a claimed aggregate with the recomputed one, at the precision the claim used
    (claiming 143 for 142.95 is fine; claiming 150 is not)."""
    if key not in allowed:
        return f"unknown {label} field '{key}'"
    a = actual.get(key)
    if key in text_keys:
        ok = str(value).strip().lower() == str(a or "").strip().lower()
        return None if ok else f"{label} {key} mismatch: claimed '{value}', logs say '{a}'"
    try:
        v = float(value)
        sv = str(value).strip()
        decimals = 0 if v.is_integer() else (len(sv.split(".")[1]) if "." in sv else 0)
        ok = a is not None and round(float(a), decimals) == round(v, decimals)
    except (TypeError, ValueError):
        ok = False
    return None if ok else f"{label} {key} mismatch: claimed {value}, logs say {a}"


def check_baseline(actual: dict, key: str, value) -> str | None:
    return _check_aggregate(actual, key, value, BASELINE_FIELDS, "baseline", ("usual_pc",))


def check_session(actual: dict, key: str, value) -> str | None:
    return _check_aggregate(actual, key, value, SESSION_FIELDS, "session", ("pc",))


NUMBERISH = re.compile(r"\d")
ID_TOKENS = re.compile(r"\b[A-Z]{3}\d{4}\b|\bPC-\d+\b") #this is only suitable for CERT dataset, need to change if using other dataset.


# --------------------------------------------------------------------------------------------
def fetch_events(con, ids: list[str]) -> dict[str, dict]:
    if not ids:
        return {}
    cur = con.execute(f"SELECT {EVENT_COLS}, session_id FROM events WHERE id IN (SELECT unnest(?::VARCHAR[]))",
                      [list(set(ids))])
    cols = [c[0] for c in cur.description]
    out = {}
    for r in cur.fetchall():
        e = _row_to_event(cols, r)
        e["session_id"] = r[-1]
        out[e["id"]] = e
    return out


def verify_report(con, report: Report, strict_text: bool = False, strict_citations: bool = True) -> dict:
    """strict_citations (verifier v2, default): every cited event must be described by at least one
    asserted field, and asserted fields need cited events to be checked against. v1 (False) only
    checked that cited events exist, which let a citation be swapped for any event of the same user
    when a claim asserted nothing about its events (found by the error-injection test)."""
    sess = con.execute('SELECT "user", start_ts FROM sessions WHERE session_id = ?', [report.session_id]).fetchone()
    session_user = sess[0] if sess else None
    baseline_actual = None   # computed lazily, only if some claim uses `baseline`
    session_actual = None    # likewise for `session`
    events = fetch_events(con, [c for cl in report.claims for c in cl.citations])
    results: list[ClaimResult] = []

    for i, cl in enumerate(report.claims):
        r = ClaimResult(index=i, text=cl.text, type=cl.type, status="", citations=list(cl.citations))
        for cid in cl.citations:
            e = events.get(cid)
            if e is None:
                r.reasons.append(f"cited event {cid} does not exist")
            elif session_user and e["user"] != session_user:
                r.reasons.append(f"cited event {cid} belongs to {e['user']}, not {session_user}")
            elif e["session_id"] != report.session_id:
                r.warnings.append(f"cited event {cid} is outside the investigated session")

        if cl.baseline:
            if sess is None:
                r.reasons.append("baseline claim but the session is unknown")
            else:
                if baseline_actual is None:
                    baseline_actual = Tools(con).get_user_baseline(session_user, str(sess[1]))
                for key, val in cl.baseline.items():
                    reason = check_baseline(baseline_actual, key, val)
                    if reason:
                        r.reasons.append(reason)

        if cl.session:
            if sess is None:
                r.reasons.append("session claim but the session is unknown")
            else:
                if session_actual is None:
                    session_actual = Tools(con).get_session(report.session_id)
                for key, val in cl.session.items():
                    reason = check_session(session_actual, key, val)
                    if reason:
                        r.reasons.append(reason)

        aggregate = bool(cl.baseline or cl.session)
        if cl.type == "factual":
            if not cl.citations and not aggregate:
                r.reasons.append("factual claim has no citation")
            if not cl.asserted and cl.count is None and not aggregate:
                r.reasons.append("factual claim asserts no checkable field, count, baseline or session value")
            if strict_citations and cl.citations and not cl.asserted:
                r.reasons.append("cited events are not described by any asserted field "
                                 "(only their existence could be checked)")
            if strict_citations and cl.asserted and not cl.citations:
                r.reasons.append("asserted fields but no cited events to check them against")
            for key, val in cl.asserted.items():
                base = key[:-9] if key.endswith("_contains") else key
                if base not in ASSERTABLE_FIELDS:
                    r.reasons.append(f"unknown field '{key}'")   # report once, not once per citation
                    continue
                for cid in cl.citations:
                    if cid in events:
                        reason = check_field(events[cid], key, val)
                        if reason:
                            r.reasons.append(reason)
            if cl.count is not None and cl.count != len(set(cl.citations)):
                r.reasons.append(f"count says {cl.count} but {len(set(cl.citations))} distinct events are cited")
            anchors = _anchor_warnings(cl.text, cl.asserted, cl.count)
            (r.reasons if strict_text else r.warnings).extend(anchors)
            r.reasons = list(dict.fromkeys(r.reasons))           # de-duplicate, keep order
            r.status = "unsupported" if r.reasons else "verified"
        else:
            if NUMBERISH.search(ID_TOKENS.sub("", cl.text)):
                msg = "interpretive claim states specific numbers/times that nobody checks; make it factual"
                (r.reasons if strict_text else r.warnings).append(msg)
            r.status = "unsupported" if r.reasons else "interpretive"
        results.append(r)

    factual = [r for r in results if r.type == "factual"]
    n_ver = sum(r.status == "verified" for r in results)
    summary = {
        "claims": len(results),
        "verified": n_ver,
        "unsupported": sum(r.status == "unsupported" for r in results),
        "interpretive": sum(r.status == "interpretive" for r in results),
        "groundedness": round(n_ver / len(factual), 3) if factual else None,
        "warnings": sum(len(r.warnings) > 0 for r in results),
    }
    return {"session_id": report.session_id, "summary": summary, "claims": [asdict(r) for r in results]}
