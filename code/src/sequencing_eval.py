"""Offline sequencing-utility evaluation and accuracy-vs-utility correlation.

Implements Algorithm 1 of the report: a trained KT model acts as a learner
simulator; a policy picks the next problem; we measure expected mastery gain
over a horizon H. We then correlate each model's predictive AUC with its
sequencing utility (research question (b) / hypothesis H3).

Utility metric here
-------------------
  U = mean over simulated learners of ( mastery_after - mastery_before ),
where mastery is the model's mean predicted P(correct) across all problems.
Higher U = the policy drives the simulated learner to a higher-mastery state.

This uses the reference model as the simulator so the whole pipeline runs.
If you train pyKT baselines, expose a `predict_next(state, problem)` for each
and pass it in; the algorithm below is model-agnostic.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import Callable, Dict, List

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

sys.path.append(os.path.dirname(__file__))
from common_schema import read_processed  # noqa: E402


# ------------------------- simulator + policy -----------------------------

def random_policy(mastery_per_problem: np.ndarray) -> int:
    """Pick a problem uniformly at random."""
    return int(np.random.randint(0, len(mastery_per_problem)))

def mastery_threshold_policy(mastery_per_problem: np.ndarray, threshold: float = 0.85) -> int:
    """Pick the first problem that hasn't reached the threshold.
    If all reached, pick the one with the lowest mastery."""
    below_thresh = np.where(mastery_per_problem < threshold)[0]
    if len(below_thresh) > 0:
        return int(below_thresh[0])
    return int(np.argmin(mastery_per_problem))

def greedy_policy(mastery_per_problem: np.ndarray) -> int:
    """Pick the problem with mastery closest to 0.5 (zone of proximal
    development: neither too easy nor too hard). Returns a problem index."""
    return int(np.argmin(np.abs(mastery_per_problem - 0.5)))


def simulate_utility(
    model_predict_fn: Callable[[List[int], List[int]], np.ndarray],
    n_problems: int,
    policy_fn: Callable[[np.ndarray], int],
    horizon: int,
    n_learners: int,
    seed: int = 42,
) -> float:
    """Algorithm 1: expected mastery gain over the horizon.
    
    model_predict_fn(history_q, history_r) -> array of P(correct) per problem, 
    estimated by the trained KT model.
    """
    rng = np.random.default_rng(seed)
    gains: List[float] = []
    for _ in range(n_learners):
        # start state: random low-ish mastery per problem (TRUE hidden state of student)
        true_state = rng.uniform(0.1, 0.4, size=n_problems)
        mastery_before = float(true_state.mean())
        
        history_q = []
        history_r = []
        
        for _h in range(horizon):
            # Model estimates probabilities from observed history
            model_probs = model_predict_fn(history_q, history_r)
            
            # Policy chooses next problem based on model's estimates
            q = policy_fn(model_probs)
            
            # Student attempts problem based on their true hidden state
            outcome = int(rng.random() < true_state[q])
            
            # Practising a problem nudges its true mastery up (bounded)
            delta = 0.08 if outcome else 0.03
            true_state[q] = min(1.0, true_state[q] + delta)
            
            history_q.append(q)
            history_r.append(outcome)
            
        # We use sum instead of mean to avoid dilution by n_problems
        gains.append(float(true_state.sum()) - (mastery_before * n_problems))
    return float(np.mean(gains))


def make_pykt_predict_fn(model_pt_path: str):
    import torch
    import sys
    import os
    sys.path.append(os.path.dirname(__file__))
    from train_pykt import PyKTWrapper
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(model_pt_path, map_location=device, weights_only=False)
    cfg = checkpoint["cfg"]
    vocab = checkpoint["vocab"]
    
    model = PyKTWrapper(cfg, vocab.n_problems, vocab.n_kcs).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    def predict_fn(history_q: List[int], history_r: List[int]) -> np.ndarray:
        if not history_q:
            return np.full(vocab.n_kcs, 0.5)
            
        with torch.no_grad():
            T = len(history_q)
            # Create a batch of size n_kcs, each having history + 1 candidate next query
            # history_q: [q1, q2, ... qT]
            # Next queries: 0, 1, ... n_kcs-1
            # We construct sequences of length T+1
            
            base_q = history_q + [0]
            base_r = history_r + [0] # dummy response for the next query
            
            # shape: (n_kcs, T+1)
            q_batch = torch.tensor(base_q, dtype=torch.long).unsqueeze(0).repeat(vocab.n_kcs, 1).to(device)
            r_batch = torch.tensor(base_r, dtype=torch.long).unsqueeze(0).repeat(vocab.n_kcs, 1).to(device)
            
            # fill the last timestep with the candidate KC queries
            cand_kcs = torch.arange(vocab.n_kcs, dtype=torch.long).to(device)
            q_batch[:, -1] = cand_kcs
            
            p_batch = q_batch
            
            with torch.amp.autocast('cuda', enabled=(device == "cuda")):
                if model.model_name in ["dkt", "bkt", "simplekt"]:
                    # DKT predicts all KCs at once. We only need batch size 1, length T
                    q_single = torch.tensor([history_q], dtype=torch.long).to(device)
                    r_single = torch.tensor([history_r], dtype=torch.long).to(device)
                    p_single = q_single
                    
                    logits = model(p_single, q_single, r_single, target_problem=None)
                    last_logits = logits[0, -1, :] # (n_kcs,)
                else:
                    # AKT / SAKT
                    logits = model(p_batch, q_batch, r_batch, target_problem=None)
                    # logits is (n_kcs, T+1)
                    last_logits = logits[:, -1] # (n_kcs,)
                
                probs = last_logits.cpu().numpy()
                return probs

    return predict_fn, vocab.n_kcs


# --------------------------- correlation ----------------------------------

def accuracy_utility_correlation(rows: List[Dict]) -> Dict:
    """rows: [{'model','auc','utility'}...] -> correlation summary."""
    auc = np.array([r["auc"] for r in rows], dtype=float)
    util = np.array([r["utility"] for r in rows], dtype=float)
    out: Dict[str, object] = {"n_models": len(rows)}
    if len(rows) >= 3 and np.std(auc) > 0 and np.std(util) > 0:
        pr, pp = pearsonr(auc, util)
        sr, sp = spearmanr(auc, util)
        out.update(
            pearson_r=float(pr), pearson_p=float(pp),
            spearman_r=float(sr), spearman_p=float(sp),
        )
    else:
        out["note"] = "Need >=3 models with variance for a meaningful correlation."
    return out


# ------------------------------- CLI --------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="processed CSV (for n_problems)")
    ap.add_argument("--metrics-dir", required=True,
                    help="folder with <dataset>__<config>.json prediction files")
    ap.add_argument("--out", required=True, help="output json path")
    ap.add_argument("--horizon", type=int, default=20)
    ap.add_argument("--n-learners", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    
    policies = {
        "random": random_policy,
        "mastery_threshold": mastery_threshold_policy,
        "greedy": greedy_policy
    }

    # one utility number per prediction-metrics file found
    rows = []
    for jf in sorted(glob.glob(os.path.join(args.metrics_dir, "*.json"))):
        with open(jf, "r", encoding="utf-8") as f:
            res = json.load(f)
        if "aggregate" not in res:
            continue
            
        pt_file = jf.replace(".json", ".pt")
        if not os.path.exists(pt_file):
            print(f"Skipping {jf} because {pt_file} not found. Please re-run train_pykt.py.")
            continue
            
        predict_fn, n_problems = make_pykt_predict_fn(pt_file)
        
        row = {
            "model": res.get("config", os.path.basename(jf)),
            "auc": res["aggregate"]["auc"]["mean"]
        }
        
        for p_name, p_fn in policies.items():
            util = simulate_utility(
                predict_fn, n_problems, p_fn, args.horizon, args.n_learners, args.seed
            )
            row[f"utility_{p_name}"] = util
        rows.append(row)

    corr_per_policy = {}
    for p_name in policies.keys():
        p_rows = [{"model": r["model"], "auc": r["auc"], "utility": r[f"utility_{p_name}"]} for r in rows]
        corr_per_policy[p_name] = accuracy_utility_correlation(p_rows)

    out = {"horizon": args.horizon, "n_learners": args.n_learners,
           "per_model": rows, "correlation": corr_per_policy}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"[sequencing_eval] wrote {args.out}")
    print(json.dumps(corr_per_policy, indent=2))


if __name__ == "__main__":
    main()
