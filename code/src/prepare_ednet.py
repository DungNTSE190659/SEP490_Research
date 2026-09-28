"""Standardise the EdNet-KT1 dataset into the shared schema.

Usage:
    # If your data is a single large CSV:
    python src/prepare_ednet.py --raw data/raw/ednet/merged_kt1.csv \
        --out data/processed/ednet.csv

    # If your data is a folder of user CSVs (u1.csv, u2.csv, ...):
    python src/prepare_ednet.py --raw data/raw/ednet/KT1 \
        --out data/processed/ednet.csv
"""
from __future__ import annotations

import argparse
import os
import sys
import glob

import pandas as pd

sys.path.append(os.path.dirname(__file__))
from common_schema import write_processed

def standardise(raw: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame()
    
    # Map columns from EdNet KT1
    # EdNet typically has 'user_id', 'timestamp', 'item_id', 'user_answer', 'user_correct'
    out["student_id"] = raw["user_id"].astype(str) if "user_id" in raw.columns else raw.iloc[:, 0].astype(str)
    
    if "item_id" in raw.columns:
        out["problem_id"] = raw["item_id"].astype(str) 
    elif "question_id" in raw.columns:
        out["problem_id"] = raw["question_id"].astype(str)
    else:
        out["problem_id"] = raw.iloc[:, 2].astype(str) # assuming 3rd column is item
    
    if "user_correct" in raw.columns:
        out["correct"] = raw["user_correct"].astype(int)
    elif "correct" in raw.columns:
        out["correct"] = raw["correct"].astype(int)
    else:
        # If the column name is completely different, try to infer or fallback
        out["correct"] = raw.iloc[:, 4].astype(int) if raw.shape[1] > 4 else 0

    out["code"] = ""
    out["kc_ids"] = out["problem_id"] # EdNet usually uses problem ID as KC
    
    # Ordering
    if "timestamp" in raw.columns:
        raw = raw.sort_values(["user_id" if "user_id" in raw.columns else raw.columns[0], "timestamp"])
        out["order"] = raw.groupby("user_id" if "user_id" in raw.columns else raw.columns[0]).cumcount() + 1
    else:
        out["order"] = raw.groupby(out["student_id"]).cumcount() + 1

    # Ensure per-student 1..T ordering
    out = out.sort_values(["student_id", "order"])
    out["order"] = out.groupby("student_id").cumcount() + 1
    
    return out

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, help="path to raw EdNet csv file or folder")
    ap.add_argument("--out", required=True, help="output CSV path")
    args = ap.parse_args()

    # EdNet is huge, so we need to handle it properly.
    if os.path.isdir(args.raw):
        # Folder of CSVs (KT1 standard distribution: u1.csv, u2.csv)
        print(f"Reading folder of CSVs at {args.raw}. This may take a while...")
        all_files = glob.glob(os.path.join(args.raw, "*.csv"))
        
        from tqdm import tqdm
        df_list = []
        for f in tqdm(all_files, desc="Đọc file CSV"):
            # KT1 individual files don't have user_id, it's inferred from the filename (e.g. u1.csv)
            user_id = os.path.basename(f).replace(".csv", "").replace("u", "")
            try:
                temp_df = pd.read_csv(f)
                temp_df["user_id"] = user_id
                df_list.append(temp_df)
            except pd.errors.EmptyDataError:
                continue
                
        print("Đang gộp tất cả dữ liệu lại, xin vui lòng đợi...")
        raw = pd.concat(df_list, ignore_index=True)
    else:
        # Single large CSV
        print(f"Reading single CSV at {args.raw}. This may take a while...")
        raw = pd.read_csv(args.raw, low_memory=False)

    print("Standardizing data...")
    std = standardise(raw)
    write_processed(std, args.out)

if __name__ == "__main__":
    main()
