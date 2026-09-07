from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch


PROTOCOL_ID = "waharp_loss075_locked_eval_v1_20260903"
TRAINING_PROTOCOL_ID = "waharp_loss075_final_train_v1_20260903"
BASELINE_CAPACITY_PROTOCOL_ID = "waharp_final_capacity_sweep_member2_v1_20260901"
B128_PROTOCOL_ID = "waharp_final_fullscale_b128_member2_20260831"
QUERY_NAMESPACE = "waharp_final_fullscale_b128_member2_93d3c2545e44b9e21269a1b4"
OLD_CHECKPOINT_SHA256 = "8709a7f9025292636f0526ccf7b702eda8ce0bce953c02cc1c05245bb13d7e1e"
CAPACITIES = (128, 256, 512)
EXPECTED_QUERY_TYPES = Counter({"range": 4200, "point": 300, "knn": 1500})
INPUT_ROOT = Path("/kaggle/input")


def json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=json_default),
        encoding="utf-8",
    )
    tmp.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def locate_capacity_code() -> Path:
    matches: list[Path] = []
    for marker in INPUT_ROOT.rglob("WAHARP_FINAL_FULLSCALE_BUNDLE_MARKER.json"):
        try:
            raw = json.loads(marker.read_text(encoding="utf-8"))
        except Exception:
            continue
        if raw.get("protocol_id") == BASELINE_CAPACITY_PROTOCOL_ID:
            matches.append(marker.parent)
    if len(matches) != 1:
        raise RuntimeError(f"Expected one final capacity code bundle, found {matches}")
    return matches[0]


CODE_ROOT = locate_capacity_code()
sys.path.insert(0, str(CODE_ROOT))

from realtrain.data import (  # noqa: E402
    load_arizona,
    load_point_csv,
    locate_arizona_npy,
    locate_crimes_csv,
    locate_twitter_csv,
)
from realtrain.eval import summarize_rows, tree_metrics  # noqa: E402
from realtrain.pipeline import _evaluate_python_method  # noqa: E402
from realtrain.train import load_model  # noqa: E402
from realtrain.tree import build_neural_tree  # noqa: E402
from realtrain.utils import derive_seed, read_csv, sha256_array, write_csv  # noqa: E402
from realtrain.workload import load_query_suite  # noqa: E402


def locate_training() -> tuple[Path, dict[str, Any]]:
    matches: list[tuple[Path, dict[str, Any]]] = []
    for state_path in INPUT_ROOT.rglob("FINAL_TRAINING_STATE.json"):
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if state.get("protocol_id") == TRAINING_PROTOCOL_ID and state.get("status") == "COMPLETE":
            matches.append((state_path.parent, state))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one completed final training output, found {[str(x[0]) for x in matches]}")
    root, state = matches[0]
    checkpoint = root / state["selected_checkpoint"]
    actual_hash = sha256_file(checkpoint)
    if actual_hash != state["selected_checkpoint_sha256"]:
        raise RuntimeError(f"Selected checkpoint hash mismatch: {actual_hash} != {state['selected_checkpoint_sha256']}")
    return checkpoint, state


def locate_prior_root(dataset: str, capacity: int) -> tuple[Path, dict[str, Any]]:
    expected_protocol = B128_PROTOCOL_ID if capacity == 128 else BASELINE_CAPACITY_PROTOCOL_ID
    matches: list[tuple[Path, dict[str, Any]]] = []
    for state_path in INPUT_ROOT.rglob("FULLSCALE_STATE.json"):
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        namespace = state.get("fresh_final_namespace", state.get("paired_query_namespace"))
        if (
            state.get("protocol_id") == expected_protocol
            and state.get("dataset") == dataset
            and int(state.get("capacity", 0)) == capacity
            and str(state.get("status", "")).startswith("COMPLETE")
            and namespace == QUERY_NAMESPACE
            and state.get("checkpoint_sha256") == OLD_CHECKPOINT_SHA256
        ):
            matches.append((state_path.parent, state))
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one prior result for {dataset}/B{capacity}, found {[str(x[0]) for x in matches]}"
        )
    root, state = matches[0]
    required = [
        root / "workload" / "WORKLOAD_MANIFEST.json",
        root / "DATASET.json",
        root / "methods" / "Neural" / "PER_QUERY.csv",
        root / "methods" / "PLATON" / "PER_QUERY.csv",
        root / "methods" / "STR" / "PER_QUERY.csv",
        root / "methods" / "TGS" / "PER_QUERY.csv",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Prior result is incomplete: {missing}")
    return root, state


def load_dataset(dataset: str) -> tuple[np.ndarray, dict[str, Any]]:
    if dataset == "twitter":
        return load_point_csv(locate_twitter_csv(INPUT_ROOT), 0)
    if dataset == "crimes":
        return load_point_csv(locate_crimes_csv(INPUT_ROOT), 0)
    if dataset == "arizona":
        return load_arizona(locate_arizona_npy(INPUT_ROOT), 0)
    raise ValueError(dataset)


def workload_fingerprint(root: Path) -> dict[str, str]:
    files = sorted((root / "workload").glob("final_*.npy"))
    files.append(root / "workload" / "construction_boxes.npy")
    files.append(root / "workload" / "WORKLOAD_MANIFEST.json")
    return {path.name: sha256_file(path) for path in files}


def normalized_uid(row: dict[str, Any]) -> str:
    uid = str(row["query_uid"])
    if str(row["query_type"]) == "knn":
        if uid.startswith("knn|knn_k"):
            return uid
        if uid.startswith("knn|k"):
            return "knn|knn_k" + uid[len("knn|k"):]
    return uid


def correctness_signature(row: dict[str, Any]) -> tuple[int, int, int]:
    return int(row["result_count"]), int(row["id_sum"]), int(row["id_xor"])


def bootstrap_ratio(left: np.ndarray, right: np.ndarray, seed: int, draws: int = 2000) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    count = len(left)
    estimates = np.empty(draws, dtype=np.float64)
    for draw in range(draws):
        index = rng.integers(0, count, size=count)
        estimates[draw] = float(left[index].sum() / right[index].sum())
    return float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))


def pairwise(
    left_rows: list[dict[str, Any]],
    right_rows: list[dict[str, Any]],
    dataset: str,
    capacity: int,
    right_method: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    left = {normalized_uid(row): row for row in left_rows}
    right = {normalized_uid(row): row for row in right_rows}
    common = sorted(set(left).intersection(right))
    if len(common) != 6000:
        raise RuntimeError(f"Canonical join NewNeural/{right_method}: {len(common)} != 6000")
    query_types = Counter(str(left[key]["query_type"]) for key in common)
    if query_types != EXPECTED_QUERY_TYPES:
        raise RuntimeError(f"Query type gate failed: {query_types}")
    details: list[dict[str, Any]] = []
    mismatches = 0
    for key in common:
        new_row = left[key]
        old_row = right[key]
        if new_row["query_type"] == "knn":
            agrees = math.isclose(
                float(new_row["kth_distance"]),
                float(old_row["kth_distance"]),
                rel_tol=2e-6,
                abs_tol=2e-6,
            )
        else:
            agrees = correctness_signature(new_row) == correctness_signature(old_row)
        mismatches += int(not agrees)
        details.append(
            {
                "dataset": dataset,
                "capacity": capacity,
                "left_method": "NewNeural_075_025",
                "right_method": right_method,
                "query_uid": key,
                "query_type": new_row["query_type"],
                "workload": new_row["workload"],
                "left_access": int(new_row["total_node_accesses"]),
                "right_access": int(old_row["total_node_accesses"]),
                "correctness_agrees": agrees,
            }
        )
    if mismatches:
        raise RuntimeError(f"Correctness mismatch NewNeural/{right_method}: {mismatches}")
    left_access = np.asarray([row["left_access"] for row in details], dtype=np.float64)
    right_access = np.asarray([row["right_access"] for row in details], dtype=np.float64)
    ci_low, ci_high = bootstrap_ratio(
        left_access,
        right_access,
        derive_seed(PROTOCOL_ID, capacity, f"bootstrap|{dataset}|{right_method}"),
    )
    return {
        "dataset": dataset,
        "capacity": capacity,
        "left_method": "NewNeural_075_025",
        "right_method": right_method,
        "queries": len(details),
        "left_total_access": int(left_access.sum()),
        "right_total_access": int(right_access.sum()),
        "ratio_left_over_right": float(left_access.sum() / right_access.sum()),
        "bootstrap_ci_low": ci_low,
        "bootstrap_ci_high": ci_high,
        "strict_win_rate": float(np.mean(left_access < right_access)),
        "tie_rate": float(np.mean(left_access == right_access)),
        "loss_rate": float(np.mean(left_access > right_access)),
        "correctness_mismatches": 0,
    }, details


def pairwise_by_workload(details: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in details:
        key = (
            row["dataset"],
            row["capacity"],
            row["right_method"],
            row["query_type"],
            row["workload"],
        )
        groups[key].append(row)
    output: list[dict[str, Any]] = []
    for key in sorted(groups):
        rows = groups[key]
        left = np.asarray([row["left_access"] for row in rows], dtype=np.float64)
        right = np.asarray([row["right_access"] for row in rows], dtype=np.float64)
        output.append(
            {
                "dataset": key[0],
                "capacity": key[1],
                "right_method": key[2],
                "query_type": key[3],
                "workload": key[4],
                "queries": len(rows),
                "left_total_access": int(left.sum()),
                "right_total_access": int(right.sum()),
                "ratio_left_over_right": float(left.sum() / right.sum()),
                "strict_win_rate": float(np.mean(left < right)),
                "tie_rate": float(np.mean(left == right)),
                "loss_rate": float(np.mean(left > right)),
            }
        )
    return output


def write_report(
    output: Path,
    dataset: str,
    training_state: dict[str, Any],
    comparisons: list[dict[str, Any]],
    construction: list[dict[str, Any]],
) -> None:
    lines = [
        f"# WAHARP 0.75/0.25 Locked Evaluation: {dataset.title()}",
        "",
        f"Protocol: `{PROTOCOL_ID}`",
        f"Checkpoint: `{training_state['selected_checkpoint_sha256']}`",
        f"Locked query namespace: `{QUERY_NAMESPACE}`",
        "",
        "The new neural checkpoint is the only method rebuilt. All controls are reused from their immutable per-query outputs, and every comparison uses the same 6,000 locked final queries.",
        "",
        "| Capacity | Control | New/control | 95% CI | New accesses | Control accesses | Win rate |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in comparisons:
        lines.append(
            f"| {row['capacity']} | {row['right_method']} | {row['ratio_left_over_right']:.6f} | "
            f"[{row['bootstrap_ci_low']:.6f}, {row['bootstrap_ci_high']:.6f}] | "
            f"{row['left_total_access']:,} | {row['right_total_access']:,} | "
            f"{100.0 * row['strict_win_rate']:.2f}% |"
        )
    lines += [
        "",
        "## Construction",
        "",
        "| Capacity | Seconds | Nodes | Height | Neural decisions |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in construction:
        lines.append(
            f"| {row['capacity']} | {row['build_seconds']:.3f} | {row['nodes']:,} | "
            f"{row['height']} | {row['neural_decision_count']:,} |"
        )
    lines += [
        "",
        "Correctness gate: zero range/point signature mismatches and zero kNN kth-distance mismatches.",
        "Baseline reconstructions performed: `0`.",
    ]
    (output / "LOCKED_EVALUATION_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=("twitter", "crimes", "arizona"))
    args = parser.parse_args()
    dataset = args.dataset
    started = time.perf_counter()
    output = Path(f"/kaggle/working/WAHARP_LOSS075_LOCKED_{dataset.upper()}")
    output.mkdir(parents=True, exist_ok=True)
    write_json(
        output / "EVALUATION_STATE.json",
        {"protocol_id": PROTOCOL_ID, "dataset": dataset, "status": "RUNNING"},
    )

    checkpoint, training_state = locate_training()
    prior: dict[int, Path] = {}
    prior_states: dict[int, dict[str, Any]] = {}
    fingerprints: dict[int, dict[str, str]] = {}
    for capacity in CAPACITIES:
        prior[capacity], prior_states[capacity] = locate_prior_root(dataset, capacity)
        fingerprints[capacity] = workload_fingerprint(prior[capacity])
    if fingerprints[128] != fingerprints[256] or fingerprints[128] != fingerprints[512]:
        raise RuntimeError("Locked workload files differ across capacities")

    rects, dataset_metadata = load_dataset(dataset)
    actual_dataset_hash = sha256_array(rects)
    for capacity in CAPACITIES:
        old_metadata = json.loads((prior[capacity] / "DATASET.json").read_text(encoding="utf-8"))
        if old_metadata["normalized_object_sha256"] != actual_dataset_hash:
            raise RuntimeError(f"Dataset hash mismatch for B{capacity}")
        if int(old_metadata["objects"]) != len(rects):
            raise RuntimeError(f"Dataset size mismatch for B{capacity}")

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = load_model(checkpoint, device)
    all_comparisons: list[dict[str, Any]] = []
    all_details: list[dict[str, Any]] = []
    all_workloads: list[dict[str, Any]] = []
    all_construction: list[dict[str, Any]] = []
    control_map = {
        "OldNeural_060_040": "Neural",
        "PLATON": "PLATON",
        "STR": "STR",
        "TGS": "TGS",
    }

    for capacity in CAPACITIES:
        capacity_started = time.perf_counter()
        old_root = prior[capacity]
        workload_root = old_root / "workload"
        suite = load_query_suite(workload_root, "final")
        construction_queries = np.load(workload_root / "construction_boxes.npy")
        print(f"Building {dataset} B{capacity} with {len(rects):,} objects", flush=True)
        tree, build = build_neural_tree(
            rects,
            construction_queries,
            model,
            device,
            capacity,
            12,
            128,
        )
        rows = _evaluate_python_method(
            tree,
            suite,
            len(rects),
            capacity,
            "NewNeural_075_025",
        )
        if len(rows) != 6000:
            raise RuntimeError(f"Expected 6000 final queries, got {len(rows)}")
        query_types = Counter(str(row["query_type"]) for row in rows)
        if query_types != EXPECTED_QUERY_TYPES:
            raise RuntimeError(f"Unexpected query suite: {query_types}")

        method_root = output / f"B{capacity}" / "methods" / "NewNeural_075_025"
        write_csv(method_root / "PER_QUERY.csv", rows)
        write_csv(method_root / "WORKLOAD_METRICS.csv", summarize_rows(rows, "NewNeural_075_025", dataset))
        structure = tree_metrics(tree, capacity)
        construction = {
            **build,
            "dataset": dataset,
            "capacity": capacity,
            "checkpoint_sha256": training_state["selected_checkpoint_sha256"],
            "list_weight": 0.75,
            "classification_weight": 0.25,
            "training_performed_in_evaluation": False,
            "baseline_rebuilds": 0,
            "phase_wall_seconds": time.perf_counter() - capacity_started,
        }
        write_json(method_root / "CONSTRUCTION.json", construction)
        write_json(method_root / "TREE_STRUCTURE.json", structure)
        all_construction.append(
            {
                **construction,
                "nodes": int(structure["total_nodes"]),
                "height": int(structure["height"]),
            }
        )

        for label, directory in control_map.items():
            control_rows = read_csv(old_root / "methods" / directory / "PER_QUERY.csv")
            summary, details = pairwise(rows, control_rows, dataset, capacity, label)
            all_comparisons.append(summary)
            all_details.extend(details)
            all_workloads.extend(pairwise_by_workload(details))
            print(
                f"{dataset} B{capacity} vs {label}: ratio={summary['ratio_left_over_right']:.6f}",
                flush=True,
            )

        write_json(
            output / "EVALUATION_STATE.json",
            {
                "protocol_id": PROTOCOL_ID,
                "dataset": dataset,
                "status": "RUNNING",
                "completed_capacities": [value for value in CAPACITIES if value <= capacity],
                "checkpoint_sha256": training_state["selected_checkpoint_sha256"],
                "baseline_rebuilds": 0,
                "elapsed_seconds": time.perf_counter() - started,
            },
        )
        del tree, rows, suite, construction_queries
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    write_csv(output / "PAIRWISE_SUMMARY.csv", all_comparisons)
    write_csv(output / "PAIRWISE_PER_QUERY.csv", all_details)
    write_csv(output / "PAIRWISE_BY_WORKLOAD.csv", all_workloads)
    write_json(output / "CONSTRUCTION_ALL.json", all_construction)
    provenance = {
        "protocol_id": PROTOCOL_ID,
        "training_protocol_id": TRAINING_PROTOCOL_ID,
        "dataset": dataset,
        "dataset_metadata": dataset_metadata,
        "dataset_sha256": actual_dataset_hash,
        "objects": len(rects),
        "capacities": list(CAPACITIES),
        "checkpoint_sha256": training_state["selected_checkpoint_sha256"],
        "query_namespace": QUERY_NAMESPACE,
        "workload_fingerprint": fingerprints[128],
        "prior_roots": {str(capacity): str(path) for capacity, path in prior.items()},
        "prior_states": prior_states,
        "baseline_rebuilds": 0,
        "final_queries_used_only_after_training_and_member_selection": True,
        "correctness_mismatches": 0,
        "elapsed_seconds": time.perf_counter() - started,
    }
    write_json(output / "PROVENANCE.json", provenance)
    write_report(output, dataset, training_state, all_comparisons, all_construction)
    final_state = {
        "protocol_id": PROTOCOL_ID,
        "dataset": dataset,
        "status": "COMPLETE",
        "completed_capacities": list(CAPACITIES),
        "checkpoint_sha256": training_state["selected_checkpoint_sha256"],
        "queries_per_capacity": 6000,
        "correctness_mismatches": 0,
        "baseline_rebuilds": 0,
        "elapsed_seconds": time.perf_counter() - started,
    }
    write_json(output / "EVALUATION_STATE.json", final_state)
    print(json.dumps(final_state, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
