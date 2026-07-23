"""
offline_ragas_eval.py
======================
Offline RAGAS evaluation over department sample JSON files, using the
project's own RagasEvaluator (src/ragas_evaluation/ragas_evaluator.py)
so results match production scoring exactly (same LLM, embeddings,
and weighted overall_score formula).

Run this from the project root (role_based_rag/) so the `src`, `config`,
and `utils` imports resolve correctly.

Expected input format per file (list of records), matching your dataset schema:
[
  {
    "id": 2,
    "domain": "general",
    "question": "When was FinSolve Technologies founded?",
    "reference": "FinSolve Technologies was founded in 2015.",
    "relevant_contexts": ["Company Overview: FinSolve Technologies was established in 2015."],
    "expected_tools": ["retrieve_documents"],
    "actual_response": "...",       # <- must be filled in by running your RAG pipeline first
    "actual_contexts": ["..."],     # <- must be filled in by running your RAG pipeline first
    "actual_tools_called": []
  },
  ...
]

Field mapping used for RAGAS:
    question      <- question
    answer        <- actual_response
    contexts      <- actual_contexts
    ground_truth  <- reference

Records with empty actual_response or actual_contexts are skipped (nothing to
score yet) -- run your RAG pipeline over the questions first to populate those
two fields (ask me for the companion "populate" script if you don't have one).

Usage:
    python offline_ragas_eval.py --data-dir ./eval_data --out-dir ./eval_results

Expects files named like: hr.json, finance.json, marketing.json, engineering.json
in --data-dir. Any *.json file in that folder is treated as one department
(file name without extension = department name).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from src.ragas_evaluation.ragas_evaluator import RagasEvaluator, RagasResult

REQUIRED_SOURCE_FIELDS = ("question", "reference", "actual_response", "actual_contexts")


def load_department_file(path: Path) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(
            f"{path.name}: expected a JSON list of records, got {type(data).__name__}"
        )

    cleaned = []
    skipped_empty = 0
    for i, record in enumerate(data):
        rec_id = record.get("id", i)
        missing = [field for field in REQUIRED_SOURCE_FIELDS if field not in record]
        if missing:
            print(f"  [WARN] {path.name} id={rec_id}: missing fields {missing}, skipping")
            continue

        answer = record["actual_response"]
        contexts = record["actual_contexts"]

        if not answer or not contexts:
            skipped_empty += 1
            continue

        if not isinstance(contexts, list):
            print(f"  [WARN] {path.name} id={rec_id}: 'actual_contexts' is not a list, skipping")
            continue

        cleaned.append(
            {
                "id": rec_id,
                "question": record["question"],
                "answer": answer,
                "contexts": contexts,
                "ground_truth": record["reference"],
            }
        )

    if skipped_empty:
        print(
            f"  [INFO] {path.name}: skipped {skipped_empty} record(s) with empty "
            f"actual_response/actual_contexts -- run your RAG pipeline first to fill these in"
        )

    return cleaned


def results_to_dataframe(
    department: str,
    records: list[dict[str, Any]],
    results: list[RagasResult | None],
) -> pd.DataFrame:
    """Build a DataFrame from records/results pairs, skipping any record whose
    evaluation failed (result is None) so a failure never shifts the pairing
    for records that come after it."""
    rows = []
    skipped_failed = 0
    for record, result in zip(records, results):
        if result is None:
            skipped_failed += 1
            continue
        rows.append(
            {
                "department": department,
                "id": record["id"],
                "question": result.question,
                "answer": result.answer,
                "faithfulness": result.faithfulness,
                "answer_relevancy": result.answer_relevancy,
                "context_precision": result.context_precision,
                "context_recall": result.context_recall,
                "overall_score": result.overall_score,
                "pass": result.overall_score >= 0.5,
            }
        )
    if skipped_failed:
        print(f"  [INFO] {department}: {skipped_failed} record(s) failed evaluation, excluded from results")
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline RAGAS evaluation per department")
    parser.add_argument("--data-dir", type=Path, required=True, help="Folder with *.json files, one per department")
    parser.add_argument("--out-dir", type=Path, default=Path("./eval_results"), help="Where to write results")
    args = parser.parse_args()

    if not args.data_dir.is_dir():
        print(f"Data dir not found: {args.data_dir}")
        sys.exit(1)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    json_files = sorted(args.data_dir.glob("*.json"))
    if not json_files:
        print(f"No .json files found in {args.data_dir}")
        sys.exit(1)

    evaluator = RagasEvaluator()

    all_frames: list[pd.DataFrame] = []
    summary_rows: list[dict[str, Any]] = []

    for path in json_files:
        department = path.stem
        print(f"\n=== {department} ===")
        try:
            records = load_department_file(path)
        except ValueError as e:
            print(f"  [ERROR] {e}")
            continue

        if not records:
            print(f"  [SKIP] {department}: no valid records")
            continue

        print(f"  Running RAGAS on {len(records)} records for '{department}'...")
        results = evaluator.evaluate_batch(records)

        if not results or not any(r is not None for r in results):
            print(f"  [SKIP] {department}: all evaluations failed")
            continue

        df = results_to_dataframe(department, records, results)
        if df.empty:
            print(f"  [SKIP] {department}: no scoreable records after filtering")
            continue

        all_frames.append(df)

        per_dept_path = args.out_dir / f"{department}_detailed.csv"
        df.to_csv(per_dept_path, index=False)
        print(f"  Saved detailed results -> {per_dept_path}")

        successful_results = [r for r in results if r is not None]
        dept_summary = evaluator.summary(successful_results)
        dept_summary["department"] = department
        summary_rows.append(dept_summary)

    if not all_frames:
        print("\nNo departments produced results. Nothing to summarize.")
        sys.exit(1)

    combined = pd.concat(all_frames, ignore_index=True)
    combined_path = args.out_dir / "all_departments_detailed.csv"
    combined.to_csv(combined_path, index=False)

    summary_df = pd.DataFrame(summary_rows)
    cols = ["department"] + [c for c in summary_df.columns if c != "department"]
    summary_df = summary_df[cols]
    summary_path = args.out_dir / "summary_by_department.csv"
    summary_df.to_csv(summary_path, index=False)

    print("\n=== Summary (per department) ===")
    print(summary_df.to_string(index=False))
    print(f"\nDetailed (all): {combined_path}")
    print(f"Summary:        {summary_path}")


if __name__ == "__main__":
    main()