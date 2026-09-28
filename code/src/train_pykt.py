"""Train one config with student-level 5-fold CV and save metrics.

Two backends:
  * backend="reference" (default): uses the self-contained model in
    reference_model.py. Runs immediately on the standardised CSV. Good for
    verifying the pipeline and for the KC/code ablations.
  * backend="pykt": Uses the pyKT-toolkit PyTorch models directly inside our
    own training loop, bypassing pykt's broken CLI and file formats!

Output: results/<dataset>__<config>.json with per-fold and aggregated
metrics (AUC/RMSE/ACC).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
import yaml

sys.path.append(os.path.dirname(__file__))
from common_schema import read_processed          # noqa: E402
from cross_validation import student_level_folds  # noqa: E402
from metrics import aggregate_folds, compute_metrics  # noqa: E402
from reference_model import build_sequences, Vocab, train_and_predict # noqa: E402

# Wrapper for PyKT models to work with our unified training loop
class PyKTWrapper(nn.Module):
    def __init__(self, cfg: dict, n_problems: int, n_kcs: int):
        super().__init__()
        model_name = cfg.get("model_name", "dkt").lower()
        emb_size = int(cfg.get("emb_size", 100))
        dropout = float(cfg.get("dropout", 0.2))
        
        self.model_name = model_name
        
        # In PyKT, num_c is the number of skills/concepts
        if model_name == "dkt":
            from pykt.models.dkt import DKT
            self.model = DKT(num_c=n_kcs, emb_size=emb_size, dropout=dropout)
        elif model_name == "akt":
            from pykt.models.akt import AKT
            self.model = AKT(n_question=n_kcs, n_pid=n_problems, d_model=emb_size, n_blocks=1, dropout=dropout, num_attn_heads=4)
        elif model_name == "sakt":
            from pykt.models.sakt import SAKT
            self.model = SAKT(num_c=n_kcs, seq_len=int(cfg.get("seq_len", 200)), emb_size=emb_size, num_attn_heads=4, dropout=dropout)
        else:
            from pykt.models.dkt import DKT
            self.model = DKT(num_c=n_kcs, emb_size=emb_size, dropout=dropout)

    def forward(self, problem, kc, resp, target_problem=None):
        q = kc
        r = resp
        
        # We assume `problem`, `kc`, `resp` are full sequences of length T.
        # If target_problem is provided, we are predicting the NEXT step, so we need to shift.
        if target_problem is not None:
            cshft = target_problem # This is the next KC we want to predict
            
            if self.model_name in ["dkt", "bkt", "simplekt"]:
                logits = self.model(q, r)
                # logits is (B, T, num_c). Gather the prediction for cshft.
                h_next = logits[:, :-1, :]
                idx = cshft.unsqueeze(-1)
                pred = torch.gather(h_next, -1, idx).squeeze(-1)
                return pred
                
            elif self.model_name == "akt":
                # AKT in pykt returns (B, T). It predicts the response for the CURRENT concept given previous.
                # So we pass the full sequence, and the prediction for step t is at index t.
                logits, _ = self.model(q, r, problem)
                # To match reference_model, pred at t should predict t+1.
                pred = logits[:, 1:]
                return pred
                
            elif self.model_name == "sakt":
                # SAKT requires the query (cshft) as the 3rd argument.
                # We want to predict step t+1, so we pass cshft which is shifted by 1.
                # SAKT will return (B, T-1)
                # q[:, :-1] is the history up to t.
                # r[:, :-1] is the response up to t.
                logits = self.model(q[:, :-1], r[:, :-1], cshft)
                return logits
        else:
            # Used for testing/simulation (1 step)
            if self.model_name in ["dkt", "bkt", "simplekt"]:
                return self.model(q, r)
            elif self.model_name == "akt":
                logits, _ = self.model(q, r, problem)
                return logits
            elif self.model_name == "sakt":
                # For simulation, we assume qry is the same as q
                logits = self.model(q, r, q)
                return logits

def train_pykt_direct(train_df, test_df, cfg):
    """Direct PyTorch loop avoiding pyKT's broken CLI."""
    torch.manual_seed(int(cfg.get("seed", 42)))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.backends.cudnn.benchmark = True

    vocab = Vocab().fit(pd.concat([train_df, test_df]))
    seq_len = int(cfg.get("seq_len", 200))
    train_seqs = build_sequences(train_df, vocab, seq_len=seq_len)
    test_seqs = build_sequences(test_df, vocab, seq_len=seq_len)

    model = PyKTWrapper(cfg, vocab.n_problems, vocab.n_kcs).to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    opt = torch.optim.Adam(model.parameters(), lr=float(cfg.get("learning_rate", 1e-3)))
    bce = nn.BCELoss(reduction='none')
    epochs = int(cfg.get("num_epochs", 10))
    batch_size = int(cfg.get("batch_size", 64)) * max(1, torch.cuda.device_count() * 4)

    from torch.utils.data import DataLoader, Dataset
    from torch.nn.utils.rnn import pad_sequence

    class SeqDataset(Dataset):
        def __init__(self, seqs):
            self.seqs = seqs
        def __len__(self): return len(self.seqs)
        def __getitem__(self, idx): return self.seqs[idx]

    def collate_fn(batch_seqs):
        return {
            "problem": pad_sequence([s["problem"] for s in batch_seqs], batch_first=True, padding_value=0),
            "kc": pad_sequence([s["kc"] for s in batch_seqs], batch_first=True, padding_value=0),
            "resp": pad_sequence([s["resp"] for s in batch_seqs], batch_first=True, padding_value=0),
        }

    train_loader = DataLoader(SeqDataset(train_seqs), batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=2, pin_memory=True)
    test_loader = DataLoader(SeqDataset(test_seqs), batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=2, pin_memory=True)

    scaler = torch.amp.GradScaler('cuda', enabled=(device == "cuda"))
    model.train()
    for ep in range(epochs):
        for b in tqdm(train_loader, desc=f"Epoch {ep+1}/{epochs}", leave=False):
            b_problem = b["problem"].to(device, non_blocking=True)
            b_kc = b["kc"].to(device, non_blocking=True)
            b_resp = b["resp"].to(device, non_blocking=True)
            
            next_kc = b_kc[:, 1:]
            
            with torch.amp.autocast('cuda', enabled=(device == "cuda")):
                pred = model(b_problem, b_kc, b_resp, target_problem=next_kc)
                pred = torch.clamp(pred, 1e-6, 1.0 - 1e-6)
                target = b_resp[:, 1:].float()
                
            mask = (next_kc != 0)
            
            loss_all = bce(pred.float(), target)
            loss = (loss_all * mask).sum() / (mask.sum() + 1e-8)
            
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

    model.eval()
    y_true: List[int] = []
    y_prob: List[float] = []
    with torch.no_grad():
        for b in test_loader:
            b_problem = b["problem"].to(device, non_blocking=True)
            b_kc = b["kc"].to(device, non_blocking=True)
            b_resp = b["resp"].to(device, non_blocking=True)
            
            next_kc = b_kc[:, 1:]
            pred = model(b_problem, b_kc, b_resp, target_problem=next_kc)
            target = b_resp[:, 1:]
            
            mask = (next_kc != 0)
            
            y_prob.extend(pred[mask].cpu().tolist())
            y_true.extend(target[mask].cpu().tolist())

    return {"y_true": y_true, "y_prob": y_prob, "vocab": vocab, "model": model}

def run_pykt(df, cfg, out_path):
    fold_metrics = []
    best_auc = 0.0
    for train_df, test_df, fold in student_level_folds(
        df, n_splits=cfg.get("n_splits", 5), seed=cfg.get("seed", 42)
    ):
        pred = train_pykt_direct(train_df, test_df, cfg)
        m = compute_metrics(pred["y_true"], pred["y_prob"])
        print(f"[fold {fold}] auc={m['auc']:.4f} rmse={m['rmse']:.4f} acc={m['acc']:.4f}")
        fold_metrics.append(m)
        if m["auc"] > best_auc:
            best_auc = m["auc"]
            model_path = out_path.replace(".json", ".pt")
            state_dict = pred["model"].module.state_dict() if isinstance(pred["model"], nn.DataParallel) else pred["model"].state_dict()
            torch.save({
                "model_state_dict": state_dict,
                "vocab": pred["vocab"],
                "cfg": cfg
            }, model_path)
    return fold_metrics


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="processed CSV path")
    ap.add_argument("--config", required=True, help="config YAML path")
    ap.add_argument("--out", required=True, help="output json path")
    ap.add_argument("--backend", choices=["reference", "pykt"], default="reference")
    ap.add_argument("--dataset", default="dataset", help="dataset name (for pykt)")
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    df = read_processed(args.data)
    config_name = os.path.splitext(os.path.basename(args.config))[0]

    if args.backend == "reference":
        from reference_model import train_and_predict
        fold_metrics = []
        for train_df, test_df, fold in student_level_folds(
            df, n_splits=cfg.get("n_splits", 5), seed=cfg.get("seed", 42)
        ):
            pred = train_and_predict(train_df, test_df, cfg)
            m = compute_metrics(pred["y_true"], pred["y_prob"])
            print(f"[fold {fold}] auc={m['auc']:.4f} rmse={m['rmse']:.4f} acc={m['acc']:.4f}")
            fold_metrics.append(m)
    else:
        fold_metrics = run_pykt(df, cfg, args.out)

    agg = aggregate_folds(fold_metrics)
    result = {
        "config": config_name,
        "model_name": cfg.get("model_name"),
        "use_kc": cfg.get("use_kc"),
        "backend": args.backend,
        "per_fold": fold_metrics,
        "aggregate": agg,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"[train_pykt] wrote {args.out}")
    print(f"  AUC {agg['auc']['mean']:.4f} +/- {agg['auc']['std']:.4f}")


if __name__ == "__main__":
    main()
