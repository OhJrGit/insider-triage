"""The contract between the investigation agent and the verifier.

Every report is a list of atomic claims. A FACTUAL claim must:
  * cite one or more event ids (`citations`)
  * state, in `asserted`, the field values it relies on -- these are checked against EVERY
    cited event by the verifier (see ASSERTABLE_FIELDS)
  * set `count` if the text states a number of events; it must equal len(set(citations))
An INTERPRETIVE claim (judgement, motive, "unusual for this user") needs no asserted
fields, but any ids it cites must still exist and belong to the user.
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

# Keys the agent may use in `asserted`. Append `_contains` to any string field for a
# case-insensitive substring match, e.g. {"url_contains": "wikileaks.org"}.
ASSERTABLE_FIELDS = {
    "source": "logon | device | http | email | file",
    "activity": "Logon, Logoff, Connect, Disconnect, Visit, Send, File (exact value from the event)",
    "user": "user id",
    "pc": "machine id",
    "date": "timestamp 'YYYY-MM-DD HH:MM[:SS]', a date 'YYYY-MM-DD', or a range 'A to B' (all cited events inside it)",
    "time": "time of day 'HH:MM[:SS]'",
    "url": "http events: full URL",
    "filename": "file events: file name",
    "to": "email events: one recipient address",
    "cc": "email events: one cc address",
    "bcc": "email events: one bcc address",
    "from": "email events: sender",
    "size": "email events: size in bytes (integer)",
    "attachments": "email events: number of attachments (integer)",
}


# Keys a factual claim may use in `baseline` -- statements about the user's USUAL behaviour.
# The verifier recomputes these with the same get_user_baseline() the agent was shown.
BASELINE_FIELDS = {
    "sessions": "number of sessions in the 30-day window",
    "avg_start_hour": "average session start hour (e.g. 7.4 = 07:24)",
    "avg_end_hour": "average session end hour",
    "frac_after_hours": "fraction of sessions outside office hours (0-1)",
    "avg_usb_connects": "average USB connects per session",
    "avg_files": "average file events per session",
    "avg_http": "average web visits per session",
    "avg_job_site_visits": "average job-site visits per session",
    "avg_external_emails": "average external emails per session",
    "usual_pc": "the PC the user normally uses",
}


# Keys a factual claim may use in `session` -- totals for the investigated session, as shown in the
# session summary. The verifier recomputes them from the features table.
SESSION_FIELDS = {
    "pc": "machine the session ran on",
    "duration_min": "session length in minutes",
    "start_hour": "start time as a decimal hour (22.5 = 22:30)",
    "end_hour": "end time as a decimal hour",
    "after_hours": "1 if outside office hours, else 0",
    "weekend": "1 if on a weekend, else 0",
    "foreign_pc": "1 if not the user's usual PC, else 0",
    "n_usb_connect": "number of USB connects",
    "n_file": "number of file events",
    "n_file_exec_archive": "number of .exe/.zip/.rar/... file events",
    "n_http": "number of web visits",
    "n_http_leak": "visits to leak/file-sharing sites",
    "n_http_job": "visits to job sites",
    "n_http_hack": "visits to hacking/keylogger sites",
    "n_email": "emails sent",
    "n_email_external": "emails to external addresses",
    "n_email_attach": "emails with attachments",
}


class Claim(BaseModel):
    text: str
    type: Literal["factual", "interpretive"]
    citations: list[str] = Field(default_factory=list)
    asserted: dict[str, Any] = Field(default_factory=dict)
    count: int | None = None
    baseline: dict[str, Any] = Field(default_factory=dict)
    session: dict[str, Any] = Field(default_factory=dict)


class Report(BaseModel):
    session_id: str
    verdict: Literal["malicious", "suspicious", "benign", "unclear"] = "unclear"
    summary: str = ""
    claims: list[Claim] = Field(default_factory=list)


def schema_prompt() -> str:
    fields = "\n".join(f"    - {k}: {v}" for k, v in ASSERTABLE_FIELDS.items())
    bfields = "\n".join(f"    - {k}: {v}" for k, v in BASELINE_FIELDS.items())
    sfields = "\n".join(f"    - {k}: {v}" for k, v in SESSION_FIELDS.items())
    return f"""Report JSON format:
{{
  "verdict": "malicious" | "suspicious" | "benign" | "unclear",
  "summary": "2-4 sentence plain-English summary for an analyst",
  "claims": [
    {{"text": "...", "type": "factual" | "interpretive",
      "citations": ["<event id>", ...],
      "asserted": {{"<field>": "<value copied exactly from the cited events>"}},
      "count": <integer, only if the text states a number of events>,
      "baseline": {{"<baseline field>": <value copied from the user baseline>}},
      "session": {{"<session field>": <value copied from the session summary>}}}}
  ]
}}
Rules for claims:
  - One fact per claim. Factual claims MUST cite event ids you actually saw in tool output.
  - Every value in "asserted" must be true for EVERY cited event. Copy values exactly.
  - Do NOT count events yourself. For any total ("22 files", "8 USB connections") use a "session"
    claim with the number copied from the session summary. Use "count" only for a few events you list
    individually, and then cite exactly that many.
  - Statements about the user's USUAL behaviour (averages, usual hours, usual PC) are FACTUAL claims
    with a "baseline" object holding the exact numbers from the user baseline you were given; they
    need no citations. Example: "baseline": {{"avg_http": 142.95}}.
  - Totals for THIS session (duration, number of USB connects / files / web visits / emails) are
    FACTUAL claims with a "session" object holding the exact values from the session summary, e.g.
    "session": {{"duration_min": 107.0, "n_usb_connect": 8}}. They need no citations. When you DO cite
    events for a count, cite exactly the events of that kind (e.g. only Connect events, not Disconnect).
  - Opinions, motives and risk judgements are "interpretive". Interpretive claims must NOT contain
    numbers, times or dates -- state those in a factual claim instead.
  - _contains values are plain text substrings: no regex, no "|", no wildcards. Use separate claims.
  - Allowed "asserted" fields (add the suffix _contains to any text field for a substring match):
{fields}
  - Allowed "baseline" fields:
{bfields}
  - Allowed "session" fields:
{sfields}"""
