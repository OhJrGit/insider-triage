"""Verifier evaluation.

1) eval_perturbation: inject known errors into verified claims, measure detection per error type.
2) export_claims / score_claims: the manual-labelling loop.
   export_claims writes every claim + verifier status to a CSV with an empty `human_label`
   column. Team members fill in supported / unsupported / interpretive by reading the raw
   logs, then score_claims reports agreement, precision and recall of the verifier.
"""
from __future__ import annotations

import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from . import config
from .agent.investigate import load_report
from .verifier.perturb import MUTATIONS, mutate
from .verifier.verify import verify_report


def _reports(reports_dir: Path):
    for p in sorted(Path(reports_dir).glob("*.json")):
        if p.name.endswith(".verified.json"):
            continue
        yield p, load_report(p)


def eval_perturbation(con, reports_dir: Path = config.REPORTS_DIR, seed: int = 0,
                      strict_citations: bool = True) -> dict:
    rng = random.Random(seed)
    tried, caught = Counter(), Counter()
    misses: dict[str, list] = defaultdict(list)
    n_reports = 0
    for _, rep in _reports(reports_dir):
        n_reports += 1
        res = verify_report(con, rep, strict_citations=strict_citations)
        user = con.execute('SELECT "user" FROM sessions WHERE session_id = ?', [rep.session_id]).fetchone()[0]
        for cr, claim in zip(res["claims"], rep.claims):
            if cr["status"] != "verified":
                continue
            for kind in MUTATIONS:
                broken = mutate(con, claim, kind, rng, user)
                if broken is None:
                    continue
                single = rep.model_copy(update={"claims": [broken]})
                status = verify_report(con, single, strict_citations=strict_citations)["claims"][0]["status"]
                tried[kind] += 1
                if status == "unsupported":
                    caught[kind] += 1
                elif len(misses[kind]) < 5:
                    misses[kind].append({"session": rep.session_id, "claim": broken.model_dump()})
    out = {"reports": n_reports, "by_mutation": {}}
    for k in MUTATIONS:
        if tried[k]:
            out["by_mutation"][k] = {"tried": tried[k], "caught": caught[k],
                                     "detection_rate": round(caught[k] / tried[k], 3)}
    total_t, total_c = sum(tried.values()), sum(caught.values())
    out["overall_detection_rate"] = round(total_c / total_t, 3) if total_t else None
    out["example_misses"] = misses
    return out


def _evidence_text(con, claim, session_id) -> str:
    """Everything a labeller needs to judge the claim, without opening the database."""
    from .agent.tools import Tools
    from .verifier.verify import fetch_events
    parts = []
    if claim.citations:
        ev = fetch_events(con, claim.citations)
        for cid in claim.citations[:25]:
            e = ev.get(cid)
            if e is None:
                parts.append(f"{cid}: NOT FOUND")
                continue
            detail = e.get("url") or e.get("filename") or e.get("to") or ""
            parts.append(f"{e['date']} {e['source']}/{e['activity']} {e['pc']} {detail}".strip())
        if len(claim.citations) > 25:
            parts.append(f"... +{len(claim.citations) - 25} more")
    if claim.session or claim.baseline:
        t = Tools(con)
        s = t.get_session(session_id)
        if claim.session:
            parts.append("SESSION " + json.dumps({k: s.get(k) for k in claim.session}, default=str))
        if claim.baseline:
            b = t.get_user_baseline(s["user"], s["start_ts"])
            parts.append("BASELINE " + json.dumps({k: b.get(k) for k in claim.baseline}, default=str))
    return " || ".join(parts)


def export_claims(con, out_csv: Path, reports_dir: Path = config.REPORTS_DIR,
                  strict_citations: bool = True, sample: int | None = None, seed: int = 0) -> int:
    """Writes a BLIND labelling sheet (no verifier output, so labels aren't biased) and a separate
    key file with the verifier's verdicts that score_claims joins back on claim_id."""
    rows, key = [], []
    for _, rep in _reports(reports_dir):
        res = verify_report(con, rep, strict_citations=strict_citations)
        for cr, claim in zip(res["claims"], rep.claims):
            cid = f"{rep.session_id}#{cr['index']}"
            rows.append({
                "claim_id": cid, "type": claim.type, "text": claim.text,
                "evidence": _evidence_text(con, claim, rep.session_id),
                "asserted": json.dumps(claim.asserted), "count": claim.count,
                "human_label": "", "human_label_2": "", "notes": "",
            })
            key.append({"claim_id": cid, "verifier_status": cr["status"],
                        "verifier_reasons": " | ".join(cr["reasons"])})
    if sample and sample < len(rows):
        idx = sorted(random.Random(seed).sample(range(len(rows)), sample))
        rows = [rows[i] for i in idx]
    random.Random(seed + 1).shuffle(rows)          # labeller sees claims in random order
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]) if rows else ["claim_id"])
        w.writeheader()
        w.writerows(rows)
    with open(_key_path(out_csv), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["claim_id", "verifier_status", "verifier_reasons"])
        w.writeheader()
        w.writerows(key)
    return len(rows)


def _key_path(labels_csv: Path) -> Path:
    return Path(labels_csv).with_name(Path(labels_csv).stem + "_KEY.csv")


def _norm_label(x: str) -> str:
    x = (x or "").strip().lower()
    return {"supported": "verified", "s": "verified", "v": "verified", "u": "unsupported",
            "i": "interpretive"}.get(x, x)


def _kappa(a: list[str], b: list[str]) -> float | None:
    if not a:
        return None
    cats = sorted(set(a) | set(b))
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    pe = sum((a.count(c) / n) * (b.count(c) / n) for c in cats)
    return round((po - pe) / (1 - pe), 3) if pe < 1 else 1.0


def score_claims(labelled_csv: Path, key_csv: Path | None = None) -> dict:
    """Labels: verified/supported (s), unsupported (u), interpretive (i). 'unsupported' is the
    positive class: the verifier's job is to catch bad claims."""
    key = {r["claim_id"]: r["verifier_status"]
           for r in csv.DictReader(open(key_csv or _key_path(labelled_csv), newline="", encoding="utf-8"))}
    tp = fp = fn = tn = agree = n = 0
    h1, h2 = [], []
    for row in csv.DictReader(open(labelled_csv, newline="", encoding="utf-8-sig")):
        human = _norm_label(row.get("human_label", ""))
        if not human:
            continue
        if row.get("human_label_2", "").strip():
            h1.append(human)
            h2.append(_norm_label(row["human_label_2"]))
        ver = key[row["claim_id"]]
        n += 1
        agree += human == ver
        h_bad, v_bad = human == "unsupported", ver == "unsupported"
        tp += h_bad and v_bad
        fp += (not h_bad) and v_bad
        fn += h_bad and not v_bad
        tn += (not h_bad) and not v_bad
    prec = round(tp / (tp + fp), 3) if tp + fp else None
    rec = round(tp / (tp + fn), 3) if tp + fn else None
    return {"labelled": n, "agreement_with_verifier": round(agree / n, 3) if n else None,
            "unsupported_precision": prec, "unsupported_recall": rec,
            "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
            "double_labelled": len(h1), "inter_rater_agreement": round(sum(x == y for x, y in zip(h1, h2)) / len(h1), 3) if h1 else None,
            "cohens_kappa": _kappa(h1, h2)}
