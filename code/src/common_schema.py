"""Shared interaction schema and small IO helpers.

Every dataset is standardised into ONE tabular format so the rest of the
pipeline never has to know which dataset it came from.

Schema (one row = one attempt), ordered per student by ``order``:

    student_id : str   -- anonymised learner id
    order      : int   -- 1..T, the step index within that student's history
    problem_id : str   -- problem/exercise id
    kc_ids     : str   -- knowledge-component tags, comma-separated (may be "")
    correct    : int   -- 1 if the attempt was correct, else 0
    code       : str   -- submitted source code (may be "" for math datasets)

Keeping this contract in one place means prepare_*.py scripts only have to
produce these columns, and train/eval code only has to read them.
"""
from __future__ import annotations

import pandas as pd

SCHEMA_COLUMNS = ["student_id", "order", "problem_id", "kc_ids", "correct", "code"]


def validate(df: pd.DataFrame) -> pd.DataFrame:
    """Check that a dataframe matches the shared schema; raise if not.

    Returns the dataframe with columns in canonical order and dtypes fixed.
    """
    missing = [c for c in SCHEMA_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Standardised data is missing required columns: {missing}. "
            f"Got columns: {list(df.columns)}"
        )
    df = df.copy()
    df["student_id"] = df["student_id"].astype(str)
    df["order"] = df["order"].astype(int)
    df["problem_id"] = df["problem_id"].astype(str)
    df["kc_ids"] = df["kc_ids"].fillna("").astype(str)
    if not df["correct"].isin([0, 1]).all():
        bad = sorted(set(df["correct"].unique()) - {0, 1})
        raise ValueError(f"'correct' must be 0/1; found other values: {bad}")
    df["correct"] = df["correct"].astype(int)
    df["code"] = df["code"].fillna("").astype(str)
    return df[SCHEMA_COLUMNS].sort_values(["student_id", "order"]).reset_index(drop=True)


def write_processed(df: pd.DataFrame, out_path: str) -> None:
    df = validate(df)
    df.to_csv(out_path, index=False)
    n_students = df["student_id"].nunique()
    n_problems = df["problem_id"].nunique()
    print(
        f"[common_schema] wrote {len(df)} interactions "
        f"({n_students} students, {n_problems} problems) -> {out_path}"
    )


def read_processed(path: str) -> pd.DataFrame:
    return validate(pd.read_csv(path))
