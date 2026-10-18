"""Human pilot study UI:  streamlit run app/pilot.py

Prepare the cases once first:  python cli.py pilot-prepare --cases 8
Responses are appended to work/pilot/responses.csv; analyse with:  python cli.py pilot-analyze
"""
import sys
import time
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json  # noqa: E402

from triage import db, pilot  # noqa: E402
from triage.agent.investigate import load_report  # noqa: E402
from triage.agent.tools import Tools  # noqa: E402
from triage.verifier.verify import fetch_events, verify_report  # noqa: E402

st.set_page_config(page_title="Alert triage study", layout="wide")


@st.cache_resource
def get_con():
    return db.connect(read_only=True)


con = get_con()
if not pilot.CASES.exists():
    st.error("No study cases yet. Run `python cli.py pilot-prepare` first.")
    st.stop()
cases = pilot.load_cases()
ss = st.session_state

# ---------------------------------------------------------------- start screen
if "participant" not in ss:
    st.title("Insider-threat alert triage study")
    st.markdown(f"""
You will review **{len(cases)} security alerts** from a (synthetic) company's activity logs. Each alert is
a user session that an anomaly detector flagged. For each one, decide whether it is a **real insider
threat** or **not**, and say how confident you are.

* For some alerts you only see the alert and the raw logs. For others you also see an **AI-written
  investigation report** in which every factual statement has been automatically checked against the
  logs: green = verified, red = not supported by the logs, blue = interpretation.
* Work at your normal pace; time per alert is recorded. There is no penalty for being wrong.
* Your answers are recorded under a participant number only, and no personal data is collected.
  You can stop at any time.
""")
    pid = st.number_input("Participant number (given by the researcher)", min_value=1, step=1, value=None)
    agree = st.checkbox("I have read the above and agree to take part.")
    if st.button("Start", disabled=not (pid and agree)):
        ss.participant = int(pid)
        ss.plan = pilot.schedule(ss.participant, cases)
        st.rerun()
    st.stop()

plan, pid = ss.plan, ss.participant
done = pilot.done_cases(pid)
remaining = [c for c in plan if c["case_no"] not in done]

# ---------------------------------------------------------------- survey / end
if not remaining:
    if ss.get("survey_done"):
        st.title("Thank you!")
        st.write("All done. You can close this window.")
        st.stop()
    st.title("Last step: a few quick questions")
    st.caption("1 = strongly disagree, 5 = strongly agree. About the alerts that came with an AI report:")
    q1 = st.slider("The green/red verification marks helped me decide.", 1, 5, 3)
    q2 = st.slider("I trusted claims marked VERIFIED.", 1, 5, 3)
    q3 = st.slider("I double-checked claims against the raw logs.", 1, 5, 3)
    q4 = st.slider("The report saved me time compared with reading the logs.", 1, 5, 3)
    q5 = st.slider("I would want reports like this in a real SOC.", 1, 5, 3)
    exp = st.selectbox("Your security-analysis experience", ["student, no SOC experience",
                                                              "student with SOC/CTF experience",
                                                              "industry experience"])
    comment = st.text_area("Anything else? (optional)")
    if st.button("Submit"):
        pilot.record_survey(pid, {"marks_helped": q1, "trusted_verified": q2, "double_checked": q3,
                                  "saved_time": q4, "would_use": q5, "experience": exp,
                                  "comment": comment.replace("\n", " ")})
        ss.survey_done = True
        st.rerun()
    st.stop()

# ---------------------------------------------------------------- one case
case = remaining[0]
sid = case["session_id"]
key = f"start_{pid}_{case['case_no']}"
if key not in ss:
    ss[key] = time.time()

st.progress((len(plan) - len(remaining)) / len(plan),
            text=f"Alert {len(plan) - len(remaining) + 1} of {len(plan)}")
s = Tools(con).get_session(sid)
rank = con.execute("SELECT rank FROM alerts WHERE session_id = ?", [sid]).fetchone()

st.subheader("Alert")
c1, c2, c3, c4 = st.columns(4)
c1.metric("User", s["user"])
c2.metric("PC", s["pc"])
c3.metric("Session", f"{s['start_ts'][:16]} → {s['end_ts'][11:16]}")
c4.metric("Anomaly rank", f"#{rank[0]}" if rank else "-")
totals = {"duration (min)": s["duration_min"], "USB connects": s["n_usb_connect"], "file copies": s["n_file"],
          "web visits": s["n_http"], "emails": s["n_email"], "external emails": s["n_email_external"],
          "after hours": bool(s["after_hours"]), "usual PC?": not bool(s["foreign_pc"])}
st.dataframe(pd.DataFrame([totals]), hide_index=True, width="stretch")

if case["condition"] == "B_report":
    st.subheader("AI investigation report (automatically verified)")
    report = load_report(Path(json.loads(pilot.CASES.read_text())["reports_dir"]) / f"{sid}.json")
    res = verify_report(con, report)
    if report.summary:
        st.markdown(f"_{report.summary}_")
    cited = fetch_events(con, [c for cl in report.claims for c in cl.citations])
    for c in res["claims"]:
        box = {"verified": st.success, "unsupported": st.error, "interpretive": st.info}[c["status"]]
        label = {"verified": "VERIFIED", "unsupported": "NOT SUPPORTED BY LOGS", "interpretive": "INTERPRETATION"}
        box(f"**{label[c['status']]}** · {c['text']}")
        rows = [cited[x] for x in c["citations"] if x in cited]
        if rows or c["reasons"]:
            with st.expander("evidence"):
                for r in c["reasons"]:
                    st.markdown(f"- {r}")
                if rows:
                    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

with st.expander("Raw event log for this session", expanded=case["condition"] == "A_alert_only"):
    ev = con.execute("SELECT ts, source, activity, detail FROM events WHERE session_id = ? ORDER BY ts",
                     [sid]).df()
    st.dataframe(ev, hide_index=True, width="stretch", height=350)

st.subheader("Your decision")
decision = st.radio("Is this a real insider threat?", ["insider", "not insider"], index=None, horizontal=True)
confidence = st.slider("How confident are you?", 1, 5, 3, help="1 = guessing, 5 = certain")
if st.button("Submit and continue", disabled=decision is None, type="primary"):
    pilot.record(pid, case, decision, confidence, time.time() - ss[key])
    st.rerun()
