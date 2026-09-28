"""Standardise the ASSISTments2009 dataset into the shared schema.

Usage:
    python src/prepare_assistments2009.py --raw data/raw/assistments2009/skill_builder_data.csv \
        --out data/processed/assistments2009.csv
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
    
    # Map columns from ASSISTments2009
    # ASSISTments2009 typically has 'user_id', 'problem_id', 'skill_id', 'correct', 'order_id'
    out["student_id"] = raw["user_id"].astype(str) if "user_id" in raw.columns else raw.iloc[:, 0].astype(str)
    out["problem_id"] = raw["problem_id"].astype(str) if "problem_id" in raw.columns else raw.iloc[:, 1].astype(str)
    
    if "correct" in raw.columns:
        out["correct"] = raw["correct"].astype(int)
    else:
        out["correct"] = raw.iloc[:, 3].astype(int)

    out["code"] = ""
    
    # Map KC (Knowledge Component / Skill)
    if "skill_id" in raw.columns:
        out["kc_ids"] = raw["skill_id"].astype(str)
    elif "skill_name" in raw.columns:
        out["kc_ids"] = raw["skill_name"].astype(str)
    else:
        out["kc_ids"] = out["problem_id"]

    # Handle Ordering
    if "order_id" in raw.columns:
        raw = raw.sort_values(["user_id", "order_id"] if "user_id" in raw.columns else [raw.columns[0], "order_id"])
        out["order"] = raw.groupby("user_id" if "user_id" in raw.columns else raw.columns[0]).cumcount() + 1
    else:
        out["order"] = raw.groupby(out["student_id"]).cumcount() + 1

    # Ensure per-student 1..T ordering
    out = out.sort_values(["student_id", "order"])
    out["order"] = out.groupby("student_id").cumcount() + 1
    
    return out

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, help="path to raw assistments2009 csv file")
    ap.add_argument("--out", required=True, help="output CSV path")
    args = ap.parse_args()

    # Try reading the file (could be large, ASSISTments is ~300k-500k rows, pandas can handle this easily)
    raw = pd.read_csv(args.raw, encoding='latin1', low_memory=False)
    
    # Optional: drop rows with missing values in critical columns
    if "skill_id" in raw.columns and "correct" in raw.columns:
        raw = raw.dropna(subset=["skill_id", "correct"])

    std = standardise(raw)
    write_processed(std, args.out)

if __name__ == "__main__":
    main()
