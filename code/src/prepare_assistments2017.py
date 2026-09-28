"""Standardise the ASSISTments2017 dataset into the shared schema.

Usage:
    python src/prepare_assistments2017.py --raw data/raw/assistments2017/anonymized_full_release_competition_dataset.csv \
        --out data/processed/assistments2017.csv
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
    
    # ASSISTments 2017 typical columns: studentId, skill, problemId, correct
    out["student_id"] = raw["studentId"].astype(str) if "studentId" in raw.columns else raw.iloc[:, 0].astype(str)
    out["problem_id"] = raw["problemId"].astype(str) if "problemId" in raw.columns else raw.iloc[:, 2].astype(str)
    
    if "correct" in raw.columns:
        out["correct"] = raw["correct"].astype(int)
    else:
        out["correct"] = raw.iloc[:, 3].astype(int)

    out["code"] = ""
    
    if "skill" in raw.columns:
        out["kc_ids"] = raw["skill"].astype(str)
    else:
        out["kc_ids"] = out["problem_id"]

    # Ordering
    # ASSISTments2017 is usually already time-ordered or has startTime
    if "startTime" in raw.columns:
        raw = raw.sort_values(["studentId" if "studentId" in raw.columns else raw.columns[0], "startTime"])
        out["order"] = raw.groupby("studentId" if "studentId" in raw.columns else raw.columns[0]).cumcount() + 1
    else:
        out["order"] = raw.groupby(out["student_id"]).cumcount() + 1

    # Ensure per-student 1..T ordering
    out = out.sort_values(["student_id", "order"])
    out["order"] = out.groupby("student_id").cumcount() + 1
    
    return out

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, help="path to raw assistments2017 csv file")
    ap.add_argument("--out", required=True, help="output CSV path")
    args = ap.parse_args()

    raw = pd.read_csv(args.raw, encoding='latin1', low_memory=False)
    
    if "skill" in raw.columns and "correct" in raw.columns:
        raw = raw.dropna(subset=["skill", "correct"])

    std = standardise(raw)
    write_processed(std, args.out)

if __name__ == "__main__":
    main()
