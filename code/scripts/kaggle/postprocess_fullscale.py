from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


METHODS = ("Neural", "STR", "TGS")
EXPECTED_TYPES = Counter({"range": 4200, "point": 300, "knn": 1500})


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = []
    seen = set()
    for row in rows:
        for field in row:
            if field not in seen:
                seen.add(field)
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def atomic_json(path: Path, payload) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def uid(row: dict) -> str:
    value = str(row["query_uid"])
    if row["query_type"] == "knn":
        if value.startswith("knn|knn_k"):
            return value
        if value.startswith("knn|k"):
            return "knn|knn_k" + value[len("knn|k"):]
    return value


def exact_signature(row: dict) -> tuple[int, int, int]:
    return int(row["result_count"]), int(row["id_sum"]), int(row["id_xor"])


def compare(root: Path, dataset: str, method: str) -> tuple[dict, list[dict], list[dict]]:
    left = {uid(row): row for row in read_csv(root / "methods" / method / "PER_QUERY.csv")}
    right = {uid(row): row for row in read_csv(root / "methods" / "PLATON" / "PER_QUERY.csv")}
    common = sorted(set(left).intersection(right))
    if len(common) != 6000:
        raise RuntimeError(f"{method}/PLATON join has {len(common)} queries")
    if Counter(left[key]["query_type"] for key in common) != EXPECTED_TYPES:
        raise RuntimeError(f"{method}/PLATON query-type composition failed")
    detail = []
    mismatches = Counter()
    for key in common:
        a, b = left[key], right[key]
        if a["query_type"] == "knn":
            agrees = math.isclose(
                float(a["kth_distance"]),
                float(b["kth_distance"]),
                rel_tol=2e-6,
                abs_tol=2e-6,
            )
            gate = "kth_distance"
        else:
            agrees = exact_signature(a) == exact_signature(b)
            gate = "exact_id_signature"
        mismatches[a["query_type"]] += int(not agrees)
        detail.append({
            "dataset": dataset,
            "left_method": method,
            "right_method": "PLATON",
            "query_uid": key,
            "query_type": a["query_type"],
            "workload": a["workload"],
            "correctness_gate": gate,
            "correctness_agrees": int(agrees),
            "left_access": int(a["total_node_accesses"]),
            "right_access": int(b["total_node_accesses"]),
        })
    if sum(mismatches.values()):
        raise RuntimeError(f"{method}/PLATON correctness mismatches: {dict(mismatches)}")
    by_workload = []
    groups = defaultdict(list)
    for row in detail:
        groups[(row["query_type"], row["workload"])].append(row)
    for (query_type, workload), rows in sorted(groups.items()):
        lv = np.asarray([row["left_access"] for row in rows], dtype=np.float64)
        rv = np.asarray([row["right_access"] for row in rows], dtype=np.float64)
        by_workload.append({
            "dataset": dataset,
            "left_method": method,
            "right_method": "PLATON",
            "query_type": query_type,
            "workload": workload,
            "queries": len(rows),
            "ratio_left_over_right": float(lv.sum() / rv.sum()),
            "strict_win_rate": float(np.mean(lv < rv)),
            "tie_rate": float(np.mean(lv == rv)),
            "loss_rate": float(np.mean(lv > rv)),
        })
    lv = np.asarray([row["left_access"] for row in detail], dtype=np.float64)
    rv = np.asarray([row["right_access"] for row in detail], dtype=np.float64)
    summary = {
        "dataset": dataset,
        "left_method": method,
        "right_method": "PLATON",
        "queries": len(detail),
        "left_total_access": int(lv.sum()),
        "right_total_access": int(rv.sum()),
        "ratio_left_over_right": float(lv.sum() / rv.sum()),
        "strict_win_rate": float(np.mean(lv < rv)),
        "tie_rate": float(np.mean(lv == rv)),
        "loss_rate": float(np.mean(lv > rv)),
        "range_mismatches": int(mismatches["range"]),
        "point_mismatches": int(mismatches["point"]),
        "knn_distance_mismatches": int(mismatches["knn"]),
    }
    return summary, detail, by_workload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--dataset", required=True)
    args = parser.parse_args()
    summaries = []
    details = []
    workloads = []
    for method in METHODS:
        summary, method_detail, method_workloads = compare(args.root, args.dataset, method)
        summaries.append(summary)
        details.extend(method_detail)
        workloads.extend(method_workloads)
    write_csv(args.root / "PAIRWISE_VS_PLATON.csv", summaries)
    write_csv(args.root / "PAIRWISE_PER_QUERY.csv", details)
    write_csv(args.root / "PAIRWISE_BY_WORKLOAD.csv", workloads)
    neural = next(row for row in summaries if row["left_method"] == "Neural")
    state_path = args.root / "FULLSCALE_STATE.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state.update({
        "status": "COMPLETE_POSTPROCESSED",
        "completed_methods": ["Neural", "STR", "TGS", "PLATON"],
        "durable_units": 4,
        "pairwise": summaries,
        "correctness_protocol": {
            "range_point": "exact result count, ID sum, and ID XOR",
            "knn": "equivalent kth distance; libspatialindex expands boundary ties",
        },
    })
    atomic_json(state_path, state)
    capacity = int(state["capacity"])
    lines = [
        f"# {args.dataset.title()} Full-Scale B={capacity} Result",
        "",
        f"Objects: {json.loads((args.root / 'DATASET.json').read_text())['objects']:,}",
        "",
        "| Method vs PLATON | Node-access ratio | Strict query win rate |",
        "|---|---:|---:|",
    ]
    for row in summaries:
        lines.append(f"| {row['left_method']} | {row['ratio_left_over_right']:.6f} | {row['strict_win_rate']:.4f} |")
    lines += [
        "",
        "All 4,200 range and 300 point queries passed exact ID-signature checks.",
        "All 1,500 kNN queries passed kth-distance equivalence checks.",
        "The neural method uses the B=128-trained frozen member-2 checkpoint without deterministic fallback.",
        "This is a frozen cross-capacity generalization result; the model was not retrained for this capacity.",
        "",
        f"Neural headline ratio: {neural['ratio_left_over_right']:.6f}.",
    ]
    (args.root / "SCIENTIFIC_SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": "COMPLETE_POSTPROCESSED", "pairwise": summaries}, indent=2))


if __name__ == "__main__":
    main()
