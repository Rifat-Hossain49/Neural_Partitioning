"""Rebuild one locked-evaluation cell and compare it with the recorded artifact.

The frozen checkpoint, the committed construction workload, and the committed
final queries are used exactly as in ``scripts/kaggle/run_locked_eval.py``.
Only the raw dataset must be supplied (see ``docs/DATASETS.md``).

Example:

    python scripts/reproduce_locked_cell.py --dataset arizona --capacity 512 \
        --data /path/to/arizona_buildings_n1464257_epsg26912.npy
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from realtrain.data import load_arizona, load_point_csv  # noqa: E402
from realtrain.eval import tree_metrics  # noqa: E402
from realtrain.pipeline import _evaluate_python_method  # noqa: E402
from realtrain.train import load_model  # noqa: E402
from realtrain.tree import build_neural_tree  # noqa: E402
from realtrain.utils import sha256_array, sha256_file  # noqa: E402
from realtrain.workload import load_query_suite  # noqa: E402

ARTIFACTS = ROOT / "artifacts"
CHECKPOINT = ARTIFACTS / "model" / "selected.pt"
METHOD = "NewNeural_075_025"
HIST_BINS = 12
QUERY_CAP = 128


def load_dataset(dataset: str, path: Path) -> np.ndarray:
    if dataset == "arizona":
        rects, _ = load_arizona(path, 0)
    else:
        rects, _ = load_point_csv(path, 0)
    return rects


def recorded_totals(dataset: str, capacity: int) -> tuple[int, dict[str, int]]:
    total = None
    with (ARTIFACTS / "final_evaluation" / "ALL_PAIRWISE_SUMMARY.csv").open(newline="") as f:
        for row in csv.DictReader(f):
            if (row["dataset"], int(row["capacity"]), row["left_method"], row["right_method"]) == (
                dataset, capacity, METHOD, "PLATON"):
                total = int(row["left_total_access"])
    by_workload: dict[str, int] = {}
    with (ARTIFACTS / "final_evaluation" / "ALL_PAIRWISE_BY_WORKLOAD.csv").open(newline="") as f:
        for row in csv.DictReader(f):
            if (row["dataset"], int(row["capacity"]), row["right_method"]) == (dataset, capacity, "PLATON"):
                by_workload[row["workload"]] = int(row["left_total_access"])
    if total is None:
        raise RuntimeError(f"No recorded result for {dataset} B={capacity}")
    return total, by_workload


def recorded_construction(dataset: str, capacity: int) -> dict:
    records = json.loads((ARTIFACTS / "construction" / f"{dataset}_CONSTRUCTION_ALL.json").read_text())
    return next(r for r in records if int(r["capacity"]) == capacity)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, choices=("arizona", "crimes", "twitter"))
    parser.add_argument("--capacity", required=True, type=int, choices=(128, 256, 512))
    parser.add_argument("--data", required=True, type=Path, help="Raw dataset file (see docs/DATASETS.md)")
    parser.add_argument("--device", default="cpu", help="Torch device for policy inference (default: cpu)")
    parser.add_argument("--output", type=Path, help="Optional JSON report path")
    args = parser.parse_args()

    workload = ARTIFACTS / "workloads" / args.dataset
    dataset_record = json.loads((workload / "DATASET.json").read_text())
    expected_ckpt = json.loads((ARTIFACTS / "model" / "FINAL_TRAINING_STATE.json").read_text())[
        "selected_checkpoint_sha256"]
    if sha256_file(CHECKPOINT) != expected_ckpt:
        raise RuntimeError("Checkpoint hash does not match FINAL_TRAINING_STATE.json")

    print(f"Loading {args.dataset} from {args.data}", flush=True)
    rects = load_dataset(args.dataset, args.data)
    data_hash = sha256_array(rects)
    if data_hash != dataset_record["normalized_object_sha256"] or len(rects) != int(dataset_record["objects"]):
        raise RuntimeError(
            f"Normalized dataset differs from the evaluated snapshot: {len(rects):,} objects, "
            f"sha256 {data_hash}; expected {int(dataset_record['objects']):,} objects, "
            f"sha256 {dataset_record['normalized_object_sha256']}")

    model = load_model(CHECKPOINT, args.device)
    construction_queries = np.load(workload / "construction_boxes.npy")
    suite = load_query_suite(workload, "final")
    print(f"Building WAHARP tree: {len(rects):,} objects, B={args.capacity}", flush=True)
    tree, build = build_neural_tree(rects, construction_queries, model, args.device,
                                    args.capacity, HIST_BINS, QUERY_CAP)
    print(f"Built in {build['build_seconds']:.1f}s; evaluating 6,000 final queries", flush=True)
    started = time.perf_counter()
    rows = _evaluate_python_method(tree, suite, len(rects), args.capacity, METHOD)
    eval_seconds = time.perf_counter() - started

    total = sum(int(r["total_node_accesses"]) for r in rows)
    by_workload: dict[str, int] = defaultdict(int)
    for r in rows:
        by_workload[str(r["workload"])] += int(r["total_node_accesses"])
    structure = tree_metrics(tree, args.capacity)

    rec_total, rec_by_workload = recorded_totals(args.dataset, args.capacity)
    rec_build = recorded_construction(args.dataset, args.capacity)
    rec_structure = json.loads(
        (ARTIFACTS / "tree_structure" / f"{args.dataset}_B{args.capacity}_TREE_STRUCTURE.json").read_text())

    checks = {
        "total_logical_node_accesses": (total, rec_total),
        "accesses_by_workload": (dict(sorted(by_workload.items())), dict(sorted(rec_by_workload.items()))),
        "neural_decision_count": (build["neural_decision_count"], rec_build["neural_decision_count"]),
        "action_counts": (build["action_counts"], rec_build["action_counts"]),
        "leaf_pages": (build["leaf_pages"], rec_build["leaf_pages"]),
        "tree_structure": (structure, rec_structure),
    }
    results = {name: {"reproduced": got, "recorded": want, "match": got == want}
               for name, (got, want) in checks.items()}
    passed = all(v["match"] for v in results.values())

    print()
    for name, v in results.items():
        shown = (f"{v['reproduced']:,} vs {v['recorded']:,}" if isinstance(v["reproduced"], int)
                 else "identical" if v["match"] else "DIFFERENT")
        print(f"  {'OK  ' if v['match'] else 'FAIL'} {name}: {shown}")
    print(f"\n{args.dataset} B={args.capacity}: {'REPRODUCED' if passed else 'MISMATCH'} "
          f"(build {build['build_seconds']:.1f}s, evaluation {eval_seconds:.1f}s)")

    if args.output:
        report = {
            "dataset": args.dataset, "capacity": args.capacity, "passed": passed,
            "checkpoint_sha256": expected_ckpt, "normalized_object_sha256": data_hash,
            "build_seconds": build["build_seconds"], "evaluation_seconds": eval_seconds,
            "torch": torch.__version__, "numpy": np.__version__,
            "query_types": dict(Counter(str(r["query_type"]) for r in rows)),
            "checks": results,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, default=str) + "\n")
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
