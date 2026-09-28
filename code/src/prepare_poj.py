"""Standardise the POJ (Peking Online Judge) dataset into the shared schema.

Usage:
    python src/prepare_poj.py --raw data/raw/poj/poj_log.csv \
        --out data/processed/poj.csv
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
    # TODO: adjust these column names to match the exact headers in your downloaded POJ dataset
    out["student_id"] = raw["user_id"].astype(str) if "user_id" in raw.columns else raw.iloc[:, 0].astype(str)
    out["problem_id"] = raw["problem_id"].astype(str) if "problem_id" in raw.columns else raw.iloc[:, 1].astype(str)
    
    if "correct" in raw.columns:
        out["correct"] = raw["correct"].astype(int)
    elif "score" in raw.columns:
        out["correct"] = (raw["score"] > 0).astype(int)
    else:
        out["correct"] = raw.iloc[:, 2].astype(int)

    out["code"] = ""
    out["kc_ids"] = out["problem_id"]
    
    # Ordering
    if "submit_time" in raw.columns:
        raw = raw.sort_values(["user_id", "submit_time"])
        out["order"] = raw.groupby("user_id").cumcount() + 1
    else:
        out["order"] = raw.groupby(out["student_id"]).cumcount() + 1

    # Ensure per-student 1..T ordering
    out = out.sort_values(["student_id", "order"])
    out["order"] = out.groupby("student_id").cumcount() + 1
    
    return out

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, help="path to raw POJ csv file")
    ap.add_argument("--out", required=True, help="output CSV path")
    args = ap.parse_args()

    raw = pd.read_csv(args.raw)
    std = standardise(raw)
    write_processed(std, args.out)

if __name__ == "__main__":
    main()
