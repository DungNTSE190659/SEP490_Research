"""Student-level k-fold splitting.

Crucial for KT: a student must never appear in both train and test, or the
model can leak within-student information and inflate AUC (see report,
Threats to Validity). We split on unique student_id, not on rows.
"""
from __future__ import annotations

from typing import Iterator, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold


def student_level_folds(
    df: pd.DataFrame, n_splits: int = 5, seed: int = 42
) -> Iterator[Tuple[pd.DataFrame, pd.DataFrame, int]]:
    """Yield (train_df, test_df, fold_index) with disjoint students."""
    students = np.array(sorted(df["student_id"].unique()))
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold, (train_idx, test_idx) in enumerate(kf.split(students)):
        train_students = set(students[train_idx])
        test_students = set(students[test_idx])
        train_df = df[df["student_id"].isin(train_students)].copy()
        test_df = df[df["student_id"].isin(test_students)].copy()
        yield train_df, test_df, fold


def summarise_split(df: pd.DataFrame, n_splits: int = 5, seed: int = 42) -> List[dict]:
    info = []
    for train_df, test_df, fold in student_level_folds(df, n_splits, seed):
        info.append(
            {
                "fold": fold,
                "train_students": train_df["student_id"].nunique(),
                "test_students": test_df["student_id"].nunique(),
                "train_rows": len(train_df),
                "test_rows": len(test_df),
            }
        )
    return info
