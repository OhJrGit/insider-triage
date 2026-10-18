"""Deep-learning detectors (PyTorch; uses the GPU when available).

  torch_ae     Deep autoencoder on the session feature vectors; anomaly score = reconstruction error.
  transformer  Causal Transformer over each session's sequence of event tokens (logon, USB, file,
               email, flagged web visits, each tagged with a time-of-day bucket). Trained to predict
               the next event on training-period sessions; anomaly score = mean negative
               log-likelihood of the session's events, i.e. how "surprising" the sequence is.
               This follows the user-behaviour-sequence line of work (Transformer encoders over
               user activity sequences) rather than per-session counts.

Install PyTorch first (CUDA build for an NVIDIA GPU), e.g.:
  pip install torch --index-url https://download.pytorch.org/whl/cu124
"""
from __future__ import annotations

import math

import numpy as np

from . import config

MAX_LEN = 128

TOKEN_EXPR = f"""
    CASE
      WHEN source = 'logon'  THEN lower(activity)
      WHEN source = 'device' THEN CASE WHEN activity = 'Connect' THEN 'usb_on' ELSE 'usb_off' END
      WHEN source = 'file'   THEN CASE WHEN regexp_matches(lower(coalesce(detail,'')), '\\.(exe|zip|rar|7z|bat|ps1)$')
                                       THEN 'file_exec' ELSE 'file_doc' END
      WHEN source = 'email'  THEN 'email_'
            || CASE WHEN contains(replace(lower(coalesce(detail,'') || ';' || coalesce(email_cc,'') || ';' ||
                                  coalesce(email_bcc,'')), '@{config.ORG_DOMAIN.lower()}', ''), '@') THEN 'ext' ELSE 'int' END
            || CASE WHEN coalesce(email_attachments, 0) > 0 THEN '_att' ELSE '' END
      WHEN source = 'http'   THEN CASE
            WHEN regexp_matches(lower(coalesce(detail,'')), '{config.LEAK_URL_PATTERN}') THEN 'web_leak'
            WHEN regexp_matches(lower(coalesce(detail,'')), '{config.JOB_URL_PATTERN}')  THEN 'web_job'
            WHEN regexp_matches(lower(coalesce(detail,'')), '{config.HACK_URL_PATTERN}') THEN 'web_hack'
            ELSE 'web' END
      ELSE source END
    || '|' || CASE WHEN isodow(ts) >= 6 THEN 'wkend'
                   WHEN hour(ts) < 6 THEN 'night' WHEN hour(ts) < 12 THEN 'am'
                   WHEN hour(ts) < 18 THEN 'pm' ELSE 'eve' END
"""

# Consecutive repeats are collapsed and each session truncated to MAX_LEN - 1 tokens inside
# DuckDB, so only short lists ever reach Python (building full lists of all ~32M events first
# exhausted memory on a 16-32 GB PC).
TOKEN_SQL = f"""
WITH t AS (
  SELECT session_id, ts, id, {TOKEN_EXPR} AS tok FROM events WHERE session_id IS NOT NULL
), d AS (
  SELECT session_id, ts, id, tok,
         lag(tok) OVER (PARTITION BY session_id ORDER BY ts, id) AS prev FROM t
), k AS (
  SELECT session_id, ts, id, tok,
         row_number() OVER (PARTITION BY session_id ORDER BY ts, id) AS rn
  FROM d WHERE prev IS NULL OR prev <> tok
)
SELECT session_id, list(tok ORDER BY ts, id) AS toks FROM k WHERE rn < {MAX_LEN} GROUP BY session_id
"""


def _torch():
    try:
        import torch
    except ImportError as e:
        raise RuntimeError("PyTorch is not installed. See triage/deep.py for the install command.") from e
    return torch


def _device(torch):
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ------------------------------------------------------------------------------- autoencoder
def torch_ae_scores(Xtr: np.ndarray, Xte: np.ndarray, seed: int = 42, epochs: int = 30,
                    max_train: int = 200_000, batch: int = 1024) -> np.ndarray:
    torch = _torch()
    nn = torch.nn
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    dev = _device(torch)
    idx = rng.choice(len(Xtr), min(len(Xtr), max_train), replace=False)
    X = torch.tensor(Xtr[idx], dtype=torch.float32)
    n_val = max(1, len(X) // 10)
    Xv, Xt = X[:n_val].to(dev), X[n_val:]
    d = X.shape[1]
    model = nn.Sequential(nn.Linear(d, 64), nn.ReLU(), nn.Linear(64, 16), nn.ReLU(),
                          nn.Linear(16, 64), nn.ReLU(), nn.Linear(64, d)).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    best, best_state, patience = math.inf, None, 0
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), batch):
            xb = Xt[perm[i:i + batch]].to(dev)
            loss = ((model(xb) - xb) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            v = ((model(Xv) - Xv) ** 2).mean().item()
        if v < best - 1e-5:
            best, best_state, patience = v, {k: t.clone() for k, t in model.state_dict().items()}, 0
        else:
            patience += 1
            if patience >= 4:
                break
    model.load_state_dict(best_state)
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(Xte), 8192):
            xb = torch.tensor(Xte[i:i + 8192], dtype=torch.float32, device=dev)
            out.append(((model(xb) - xb) ** 2).mean(dim=1).cpu().numpy())
    return np.concatenate(out)


# ------------------------------------------------------------------------------- sequences
def load_sequences(con, memory_limit: str | None = None, n_buckets: int | None = None) -> dict[str, list[str]]:
    """session_id -> token list (consecutive repeats collapsed, truncated to MAX_LEN - 1).

    The per-session list() aggregate cannot spill to disk, so doing all ~470k sessions in one query needs
    several GB. Instead the sessions are split into `n_buckets` groups by hash(session_id) and each group is
    built by its own query, which keeps peak memory at roughly 1/n_buckets of the single-query figure.
    The result is identical for any n_buckets.

    Overrides (environment variables): TRIAGE_DUCKDB_MEMORY (default 6GB), TRIAGE_SEQ_BUCKETS (default 16),
    TRIAGE_DUCKDB_THREADS (default: DuckDB's own choice)."""
    import os
    memory_limit = memory_limit or os.environ.get("TRIAGE_DUCKDB_MEMORY", "6GB")
    n_buckets = int(n_buckets or os.environ.get("TRIAGE_SEQ_BUCKETS", 16))
    tmp = config.WORK_DIR / "duckdb_tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    con.execute(f"SET memory_limit = '{memory_limit}'")
    con.execute(f"SET temp_directory = '{tmp.as_posix()}'")
    con.execute("SET preserve_insertion_order = false")
    if os.environ.get("TRIAGE_DUCKDB_THREADS"):
        con.execute(f"SET threads = {int(os.environ['TRIAGE_DUCKDB_THREADS'])}")
    marker = "FROM events WHERE session_id IS NOT NULL"
    assert marker in TOKEN_SQL
    seqs = {}
    for bkt in range(n_buckets):
        sql = TOKEN_SQL if n_buckets == 1 else TOKEN_SQL.replace(
            marker, f"{marker} AND hash(session_id) % {n_buckets} = {bkt}")
        cur = con.execute(sql)
        while True:
            rows = cur.fetchmany(50_000)
            if not rows:
                break
            for sid, toks in rows:
                seqs[sid] = list(toks)
        if n_buckets > 1:
            print(f"    sequences: bucket {bkt + 1}/{n_buckets} done ({len(seqs):,} sessions so far)", flush=True)
    return seqs


def transformer_scores(seqs: dict[str, list[str]], train_ids, test_ids, seed: int = 42, epochs: int = 3,
                       max_train: int = 80_000, batch: int = 256, d_model: int = 64) -> np.ndarray:
    torch = _torch()
    nn = torch.nn
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    dev = _device(torch)
    vocab = {"<pad>": 0, "<bos>": 1, "<unk>": 2}
    for sid in train_ids:
        for t in seqs.get(sid, []):
            vocab.setdefault(t, len(vocab))

    def encode(sid):
        return [1] + [vocab.get(t, 2) for t in seqs.get(sid, [])]

    tr = [encode(s) for s in rng.choice(np.array(train_ids), min(len(train_ids), max_train), replace=False)]

    class Model(nn.Module):
        def __init__(self, v):
            super().__init__()
            self.emb = nn.Embedding(v, d_model, padding_idx=0)
            self.pos = nn.Embedding(MAX_LEN, d_model)
            layer = nn.TransformerEncoderLayer(d_model, 4, 128, dropout=0.1, batch_first=True)
            self.enc = nn.TransformerEncoder(layer, 2)
            self.out = nn.Linear(d_model, v)

        def forward(self, x):
            L = x.shape[1]
            h = self.emb(x) + self.pos(torch.arange(L, device=x.device))[None]
            causal = torch.triu(torch.ones((L, L), dtype=torch.bool, device=x.device), diagonal=1)
            h = self.enc(h, mask=causal, src_key_padding_mask=(x == 0))
            return self.out(h)

    def pad(batch_seqs):
        L = max(len(s) for s in batch_seqs)
        x = torch.zeros((len(batch_seqs), L), dtype=torch.long)
        for i, s in enumerate(batch_seqs):
            x[i, :len(s)] = torch.tensor(s)
        return x.to(dev)

    model = Model(len(vocab)).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    lossf = nn.CrossEntropyLoss(ignore_index=0)
    tr.sort(key=len)                                   # bucket by length -> less padding
    batches = [tr[i:i + batch] for i in range(0, len(tr), batch)]
    for _ in range(epochs):
        model.train()
        for j in rng.permutation(len(batches)):
            x = pad(batches[j])
            if x.shape[1] < 2:
                continue
            logits = model(x[:, :-1])
            loss = lossf(logits.reshape(-1, logits.shape[-1]), x[:, 1:].reshape(-1))
            opt.zero_grad()
            loss.backward()
            opt.step()

    model.eval()
    te = [(i, encode(s)) for i, s in enumerate(test_ids)]
    te.sort(key=lambda t: len(t[1]))
    scores = np.zeros(len(test_ids))
    nll = nn.CrossEntropyLoss(ignore_index=0, reduction="none")
    with torch.no_grad():
        for k in range(0, len(te), 1024):
            chunk = te[k:k + 1024]
            x = pad([c[1] for c in chunk])
            if x.shape[1] < 2:
                continue
            logits = model(x[:, :-1])
            tgt = x[:, 1:]
            per_tok = nll(logits.reshape(-1, logits.shape[-1]), tgt.reshape(-1)).reshape(tgt.shape)
            n = (tgt != 0).sum(dim=1).clamp(min=1)
            mean_nll = (per_tok.sum(dim=1) / n).cpu().numpy()
            for (i, _), s in zip(chunk, mean_nll):
                scores[i] = s
    return scores
