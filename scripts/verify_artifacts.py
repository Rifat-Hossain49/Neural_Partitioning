from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def row_count(path: Path) -> int:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def main() -> None:
    training = read_json(ARTIFACTS / "model" / "FINAL_TRAINING_STATE.json")
    decision = read_json(ARTIFACTS / "ablation" / "FINAL_DECISION.json")
    evaluation = read_json(ARTIFACTS / "final_evaluation" / "FINAL_STATE.json")
    checkpoint = ARTIFACTS / "model" / "selected.pt"

    assert training["status"] == "COMPLETE"
    assert decision["status"] == "COMPLETE"
    assert evaluation["status"] == "COMPLETE"
    assert decision["selection_data"] == "validation_only"
    assert decision["test_or_final_queries_read"] is False
    assert training["final_queries_read"] is False
    assert decision["selected_list_weight"] == 0.75
    assert decision["selected_classification_weight"] == 0.25
    assert training["selected_member"] == evaluation["selected_member"] == 2
    assert sha256(checkpoint) == training["selected_checkpoint_sha256"]
    assert evaluation["checkpoint_sha256"] == training["selected_checkpoint_sha256"]
    assert evaluation["cells"] == 9
    assert evaluation["queries_per_cell"] == 6000
    assert evaluation["correctness_mismatches"] == 0
    assert evaluation["baseline_rebuilds"] == 0
    assert row_count(ARTIFACTS / "final_evaluation" / "ALL_PAIRWISE_SUMMARY.csv") == 36
    assert len(list((ARTIFACTS / "construction").glob("*_CONSTRUCTION_ALL.json"))) == 3
    assert len(list((ARTIFACTS / "tree_structure").glob("*_TREE_STRUCTURE.json"))) == 9

    print("Artifact verification passed")
    print("  selected loss weights: 0.75 listwise / 0.25 classification")
    print("  selected member: 2")
    print("  locked evaluation: 9 cells x 6,000 queries, 0 mismatches")


if __name__ == "__main__":
    main()
