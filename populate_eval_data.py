"""
populate_eval_data.py
======================
Runs every question in each department JSON file through the project's
RAGChain, and writes the resulting answer/contexts back into the same
file's actual_response / actual_contexts fields. Run this BEFORE
offline_ragas_eval.py, since RAGAS needs real answers/contexts to score.

Run from the project root (role_based_rag/) so `src` imports resolve.

Usage:
    python populate_eval_data.py --data-dir src/ragas_evaluation/datasets

Options:
    --data-dir     Folder with *.json files, one per department (required)
    --overwrite    Re-run and overwrite records that already have a
                    non-empty actual_response (default: skip them)
    --top-k        k passed to RAGChain.invoke (default: 5)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from src.rag_chain.chain_pipeline import RAGChain
from utils.logger_exceptions import get_logger

logger = get_logger(__name__)


def load_records(path: Path) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8-sig") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path.name}: expected a JSON list, got {type(data).__name__}")
    return data


def save_records(path: Path, records: list[dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)


def populate_file(chain: RAGChain, path: Path, top_k: int, overwrite: bool) -> tuple[int, int]:
    records = load_records(path)
    department = path.stem

    updated = 0
    skipped = 0

    for record in records:
        rec_id = record.get("id", "?")
        question = record.get("question")

        if not question:
            logger.warning(f"{path.name} id={rec_id}: no question field, skipping")
            skipped += 1
            continue

        if not overwrite and record.get("actual_response"):
            skipped += 1
            continue

        print(f"  [{department}] id={rec_id}: {question[:70]}")

        try:
            result = chain.invoke(
                question=question,
                department=department,
                k=top_k,
            )
        except Exception:
            logger.exception(f"{path.name} id={rec_id}: RAGChain.invoke failed, leaving empty")
            skipped += 1
            continue

        answer = result.get("answer", "") or ""
        contexts = [s.get("chunk_text", "") for s in result.get("sources", []) if s.get("chunk_text")]
        tools_called = result.get("tools_called", [])

        record["actual_response"] = answer
        record["actual_contexts"] = contexts
        record["actual_tools_called"] = tools_called

        updated += 1

    save_records(path, records)
    return updated, skipped


def main() -> None:
    parser = argparse.ArgumentParser(description="Populate actual_response/actual_contexts via RAGChain")
    parser.add_argument("--data-dir", type=Path, required=True, help="Folder with *.json files, one per department")
    parser.add_argument("--overwrite", action="store_true", help="Re-run records that already have an answer")
    parser.add_argument("--top-k", type=int, default=5, help="k passed to RAGChain.invoke")
    args = parser.parse_args()

    if not args.data_dir.is_dir():
        print(f"Data dir not found: {args.data_dir}")
        sys.exit(1)

    json_files = sorted(args.data_dir.glob("*.json"))
    if not json_files:
        print(f"No .json files found in {args.data_dir}")
        sys.exit(1)

    print("Initializing RAGChain (this loads the LLM/embeddings, may take a moment)...")
    chain = RAGChain()

    total_updated = 0
    total_skipped = 0

    for path in json_files:
        print(f"\n=== {path.stem} ===")
        try:
            updated, skipped = populate_file(chain, path, args.top_k, args.overwrite)
        except ValueError as e:
            print(f"  [ERROR] {e}")
            continue

        print(f"  Updated: {updated} | Skipped: {skipped}")
        total_updated += updated
        total_skipped += skipped

    print(f"\nDone. Total updated: {total_updated} | Total skipped: {total_skipped}")
    print("Now run offline_ragas_eval.py to score these records.")


if __name__ == "__main__":
    main()