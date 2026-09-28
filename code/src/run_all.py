"""Orchestrate the whole pipeline for one dataset.

Steps:
  1. train every config in --config-dir with student-level 5-fold CV
  2. run the offline sequencing evaluation over those results
  3. generate LaTeX tables into the report

Assumes data/processed/<dataset>.csv already exists (run the prepare_*.py
script first -- see README section 4).

Example:
  python src/run_all.py --dataset poj --config-dir configs \
      --out results --backend pykt
"""
from __future__ import annotations

import argparse
import glob
import os
import subprocess
import sys

HERE = os.path.dirname(__file__)
PY = sys.executable
if os.path.exists(".venv/Scripts/python.exe"):
    PY = ".venv/Scripts/python.exe"
elif os.path.exists(".venv/bin/python"):
    PY = ".venv/bin/python"


def run(cmd: list) -> None:
    print("\n$ " + " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="e.g. assistments2009")
    ap.add_argument("--config-dir", default="configs")
    ap.add_argument("--out", default="results")
    ap.add_argument("--backend", choices=["reference", "pykt"], default="reference")
    ap.add_argument("--report-out", default=os.path.join("..", "report", "sections", "_generated"))
    args = ap.parse_args()

    data_csv = os.path.join("data", "processed", f"{args.dataset}.csv")
    if not os.path.exists(data_csv):
        raise FileNotFoundError(
            f"{data_csv} not found. Run the prepare_{args.dataset}.py script "
            f"first (see README section 4)."
        )

    os.makedirs(args.out, exist_ok=True)

    # 1. train each config
    for cfg in sorted(glob.glob(os.path.join(args.config_dir, "*.yaml"))):
        name = os.path.splitext(os.path.basename(cfg))[0]
        out_json = os.path.join(args.out, f"{args.dataset}__{name}.json")
        run([
            PY, os.path.join(HERE, "train_pykt.py"),
            "--data", data_csv, "--config", cfg,
            "--out", out_json, "--backend", args.backend,
            "--dataset", args.dataset,
        ])

    # 2. sequencing evaluation
    seq_json = os.path.join(args.out, f"{args.dataset}__sequencing.json")
    run([
        PY, os.path.join(HERE, "sequencing_eval.py"),
        "--data", data_csv, "--metrics-dir", args.out, "--out", seq_json,
    ])

    # 3. LaTeX tables
    run([
        PY, os.path.join(HERE, "make_tables.py"),
        "--results", args.out, "--out", args.report_out,
        "--sequencing", seq_json,
    ])

    print("\n[run_all] done. LaTeX tables in:", args.report_out)


if __name__ == "__main__":
    main()
