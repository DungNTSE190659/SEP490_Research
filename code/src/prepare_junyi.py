"""Standardise the Junyi Academy dataset into the shared schema.

Usage:
    python src/prepare_junyi.py --raw data/raw/junyi/junyi_ProblemLog_original.csv \
        --out data/processed/junyi.csv
"""
from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

sys.path.append(os.path.dirname(__file__))
from common_schema import write_processed

def standardise(raw: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame()
    
    # Map columns from Junyi Academy
    # Junyi usually has 'user_id' or 'uuid', 'exercise' or 'problem_id', 'correct'
    user_col = "uuid" if "uuid" in raw.columns else "user_id" if "user_id" in raw.columns else raw.columns[0]
    prob_col = "exercise" if "exercise" in raw.columns else "problem_id" if "problem_id" in raw.columns else raw.columns[1]
    
    out["student_id"] = raw[user_col].astype(str)
    out["problem_id"] = raw[prob_col].astype(str)
    
    if "correct" in raw.columns:
        # Junyi 'correct' is usually boolean or 1/0
        out["correct"] = raw["correct"].astype(int)
    else:
        # Fallback to column 2 if named differently
        out["correct"] = raw.iloc[:, 2].astype(int)

    out["code"] = ""
    
    if "topic" in raw.columns:
        out["kc_ids"] = raw["topic"].astype(str)
    elif "skill_id" in raw.columns:
        out["kc_ids"] = raw["skill_id"].astype(str)
    else:
        out["kc_ids"] = out["problem_id"]
        
    # Ordering
    time_col = "time_done" if "time_done" in raw.columns else "timestamp" if "timestamp" in raw.columns else None
    
    if time_col and time_col in raw.columns:
        raw = raw.sort_values([user_col, time_col])
        out["order"] = raw.groupby(user_col).cumcount() + 1
    else:
        out["order"] = raw.groupby(out["student_id"]).cumcount() + 1

    # Ensure per-student 1..T ordering is sequential and clean
    out = out.sort_values(["student_id", "order"])
    out["order"] = out.groupby("student_id").cumcount() + 1
    
    return out

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, help="path to raw Junyi csv file")
    ap.add_argument("--out", required=True, help="output CSV path")
    args = ap.parse_args()

    raw = pd.read_csv(args.raw)
    std = standardise(raw)
    write_processed(std, args.out)

if __name__ == "__main__":
    main()
