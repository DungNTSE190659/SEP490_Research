"""A small, self-contained DKT-style KT model in PyTorch.

Why this exists
---------------
The project's primary path is to use pyKT for the baselines. But pyKT's CLI
and data format can differ between versions, which can block you for hours.
This reference model lets the WHOLE pipeline (CV -> metrics -> sequencing ->
tables) run end-to-end on the standardised CSV immediately, so you can verify
the plumbing and the report tables before wrestling with pyKT.

It also natively supports the KC channel so the ablations in the report are runnable here.

This is deliberately simple (a GRU over interaction embeddings). It is NOT a
re-implementation of AKT/simpleKT; for the strong baselines use pyKT. Treat
its numbers as a working reference, and report pyKT numbers as the baselines
in the paper.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


# ----------------------------- data encoding ------------------------------

class Vocab:
    def __init__(self) -> None:
        self.p2i: Dict[str, int] = {}
        self.k2i: Dict[str, int] = {}

    def fit(self, df: pd.DataFrame) -> "Vocab":
        for p in sorted(df["problem_id"].unique()):
            self.p2i.setdefault(p, len(self.p2i) + 1)  # 0 = padding
        kcs = set()
        for s in df["kc_ids"]:
            for k in str(s).split(","):
                if k:
                    kcs.add(k)
        for k in sorted(kcs):
            self.k2i.setdefault(k, len(self.k2i) + 1)
        return self

    @property
    def n_problems(self) -> int:
        return len(self.p2i) + 1

    @property
    def n_kcs(self) -> int:
        return len(self.k2i) + 1


def _kc_index(kc_ids: str, vocab: Vocab) -> int:
    # use the first KC tag as the primary skill id (0 if none)
    for k in str(kc_ids).split(","):
        if k in vocab.k2i:
            return vocab.k2i[k]
    return 0


def build_sequences(
    df: pd.DataFrame, vocab: Vocab, seq_len: int = 200
) -> List[dict]:
    """Group standardised rows into per-student sequences of tensors."""
    seqs = []
    for sid, g in df.sort_values(["student_id", "order"]).groupby("student_id"):
        probs = [vocab.p2i.get(p, 0) for p in g["problem_id"]]
        kcs = [_kc_index(k, vocab) for k in g["kc_ids"]]
        resp = g["correct"].astype(int).tolist()
        
        for i in range(0, len(probs), seq_len):
            p_chunk = probs[i:i+seq_len]
            if len(p_chunk) < 2:
                continue
            seqs.append(
                {
                    "problem": torch.tensor(p_chunk, dtype=torch.long),
                    "kc": torch.tensor(kcs[i:i+seq_len], dtype=torch.long),
                    "resp": torch.tensor(resp[i:i+seq_len], dtype=torch.long),
                }
            )
    return seqs


# ------------------------------- the model --------------------------------

class ReferenceKT(nn.Module):
    def __init__(
        self,
        n_problems: int,
        n_kcs: int,
        emb_size: int = 100,
        hidden_size: int = 100,
        use_kc: bool = True,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.use_kc = use_kc

        # interaction embedding: (problem, correctness) pair
        self.inter_emb = nn.Embedding(2 * n_problems + 1, emb_size, padding_idx=0)
        in_dim = emb_size
        if use_kc:
            self.kc_emb = nn.Embedding(n_kcs, emb_size, padding_idx=0)
            in_dim += emb_size

        self.rnn = nn.GRU(in_dim, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(hidden_size, n_problems)  # score per problem

    def forward(self, problem, kc, resp, target_problem=None):
        # interaction id encodes both the problem and whether it was correct
        inter = problem + resp * (self.out.out_features)
        x = self.inter_emb(inter)
        parts = [x]
        if self.use_kc:
            parts.append(self.kc_emb(kc))
        h = torch.cat(parts, dim=-1) if len(parts) > 1 else parts[0]
        h, _ = self.rnn(h)
        h = self.dropout(h)
        
        if target_problem is not None:
            h_next = h[:, :-1, :]
            W = self.out.weight[target_problem]
            b = self.out.bias[target_problem]
            logits = (h_next * W).sum(dim=-1) + b
            return logits
        else:
            logits = self.out(h)  # (B, T, n_problems)
            return logits

def _gather_next(logits, next_problem):
    # Backward compatibility if target_problem is not provided
    if logits.dim() == 2:
        return logits # already gathered
    idx = next_problem.unsqueeze(-1)
    return torch.gather(logits, -1, idx).squeeze(-1)


def train_and_predict(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cfg: dict,
) -> Dict[str, list]:
    """Train the reference model on train_df, return test predictions.

    Returns dict with 'y_true' and 'y_prob' aligned lists (next-step targets).
    """
    torch.manual_seed(int(cfg.get("seed", 42)))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.backends.cudnn.benchmark = True

    vocab = Vocab().fit(pd.concat([train_df, test_df]))
    seq_len = int(cfg.get("seq_len", 200))
    train_seqs = build_sequences(train_df, vocab, seq_len=seq_len)
    test_seqs = build_sequences(test_df, vocab, seq_len=seq_len)

    model = ReferenceKT(
        n_problems=vocab.n_problems,
        n_kcs=vocab.n_kcs,
        emb_size=int(cfg.get("emb_size", 100)),
        hidden_size=int(cfg.get("hidden_size", 100)),
        use_kc=bool(cfg.get("use_kc", True)),
        dropout=float(cfg.get("dropout", 0.2)),
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=float(cfg.get("learning_rate", 1e-3)))
    bce = nn.BCEWithLogitsLoss(reduction='none')
    epochs = int(cfg.get("num_epochs", 10))

    batch_size = int(cfg.get("batch_size", 64))

    def batch_iter(seqs):
        from torch.nn.utils.rnn import pad_sequence
        for i in range(0, len(seqs), batch_size):
            batch_seqs = seqs[i : i + batch_size]
            yield {
                "problem": pad_sequence([s["problem"] for s in batch_seqs], batch_first=True, padding_value=0).to(device),
                "kc": pad_sequence([s["kc"] for s in batch_seqs], batch_first=True, padding_value=0).to(device),
                "resp": pad_sequence([s["resp"] for s in batch_seqs], batch_first=True, padding_value=0).to(device),
            }

    scaler = torch.amp.GradScaler('cuda', enabled=(device == "cuda"))
    model.train()
    from tqdm import tqdm
    for ep in range(epochs):
        for b in tqdm(batch_iter(train_seqs), desc=f"Epoch {ep+1}/{epochs}", leave=False):
            next_problem = b["problem"][:, 1:]
            
            with torch.amp.autocast('cuda', enabled=(device == "cuda")):
                pred = model(b["problem"], b["kc"], b["resp"], target_problem=next_problem)
                target = b["resp"][:, 1:].float()
                
                mask = (next_problem != 0)
                
                loss_all = bce(pred, target)
                loss = (loss_all * mask).sum() / (mask.sum() + 1e-8)
            
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

    model.eval()
    y_true: List[int] = []
    y_prob: List[float] = []
    with torch.no_grad():
        for b in batch_iter(test_seqs):
            next_problem = b["problem"][:, 1:]
            pred_logits = model(b["problem"], b["kc"], b["resp"], target_problem=next_problem)
            pred = torch.sigmoid(pred_logits)
            target = b["resp"][:, 1:]
            
            mask = (next_problem != 0)
            
            y_prob.extend(pred[mask].cpu().tolist())
            y_true.extend(target[mask].cpu().tolist())

    return {"y_true": y_true, "y_prob": y_prob, "vocab": vocab}
