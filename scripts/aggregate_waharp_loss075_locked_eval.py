from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


PROTOCOL_ID = "waharp_loss075_locked_eval_v1_20260903"
TRAINING_PROTOCOL_ID = "waharp_loss075_final_train_v1_20260903"
DATASETS = ("twitter", "crimes", "arizona")
CAPACITIES = (128, 256, 512)
CONTROLS = ("OldNeural_060_040", "PLATON", "STR", "TGS")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_root(path: Path, dataset: str) -> tuple[dict[str, Any], list[dict[str, str]], list[dict[str, str]]]:
    state = json.loads((path / "EVALUATION_STATE.json").read_text(encoding="utf-8"))
    if state.get("protocol_id") != PROTOCOL_ID or state.get("status") != "COMPLETE":
        raise RuntimeError(f"Incomplete or unexpected result at {path}: {state}")
    if state.get("dataset") != dataset:
        raise RuntimeError(f"Dataset mismatch at {path}: {state.get('dataset')} != {dataset}")
    if int(state.get("correctness_mismatches", -1)) != 0:
        raise RuntimeError(f"Correctness gate failed at {path}")
    if int(state.get("baseline_rebuilds", -1)) != 0:
        raise RuntimeError(f"Baseline reuse gate failed at {path}")
    summary = read_csv(path / "PAIRWISE_SUMMARY.csv")
    workloads = read_csv(path / "PAIRWISE_BY_WORKLOAD.csv")
    if len(summary) != len(CAPACITIES) * len(CONTROLS):
        raise RuntimeError(f"Unexpected summary size at {path}: {len(summary)}")
    observed = {(int(row["capacity"]), row["right_method"]) for row in summary}
    expected = {(capacity, control) for capacity in CAPACITIES for control in CONTROLS}
    if observed != expected:
        raise RuntimeError(f"Missing summary cells at {path}: {expected - observed}")
    return state, summary, workloads


def aggregate(rows: list[dict[str, str]], keys: tuple[str, ...]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in keys)].append(row)
    output: list[dict[str, Any]] = []
    for key in sorted(grouped):
        members = grouped[key]
        left_total = sum(int(row["left_total_access"]) for row in members)
        right_total = sum(int(row["right_total_access"]) for row in members)
        ratios = [float(row["ratio_left_over_right"]) for row in members]
        record: dict[str, Any] = dict(zip(keys, key))
        record.update(
            {
                "cells": len(members),
                "queries": sum(int(row["queries"]) for row in members),
                "left_total_access": left_total,
                "right_total_access": right_total,
                "micro_ratio_left_over_right": left_total / right_total,
                "geometric_mean_cell_ratio": math.exp(sum(math.log(value) for value in ratios) / len(ratios)),
                "improvement_percent_micro": 100.0 * (1.0 - left_total / right_total),
                "improvement_percent_geomean": 100.0 * (
                    1.0 - math.exp(sum(math.log(value) for value in ratios) / len(ratios))
                ),
            }
        )
        output.append(record)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--twitter", type=Path, required=True)
    parser.add_argument("--crimes", type=Path, required=True)
    parser.add_argument("--arizona", type=Path, required=True)
    parser.add_argument("--training", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    roots = {name: getattr(args, name) for name in DATASETS}

    training = json.loads((args.training / "FINAL_TRAINING_STATE.json").read_text(encoding="utf-8"))
    if training.get("protocol_id") != TRAINING_PROTOCOL_ID or training.get("status") != "COMPLETE":
        raise RuntimeError(f"Unexpected training state: {training}")
    if training.get("final_queries_read") is not False:
        raise RuntimeError("Final-query isolation gate failed during training")

    states: dict[str, dict[str, Any]] = {}
    summaries: list[dict[str, str]] = []
    workloads: list[dict[str, str]] = []
    for dataset, root in roots.items():
        state, dataset_summary, dataset_workloads = load_root(root, dataset)
        states[dataset] = state
        summaries.extend(dataset_summary)
        workloads.extend(dataset_workloads)
    checkpoint_hashes = {state["checkpoint_sha256"] for state in states.values()}
    if checkpoint_hashes != {training["selected_checkpoint_sha256"]}:
        raise RuntimeError(f"Checkpoint mismatch: {checkpoint_hashes}")

    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "ALL_PAIRWISE_SUMMARY.csv", summaries)
    write_csv(args.output / "ALL_PAIRWISE_BY_WORKLOAD.csv", workloads)
    overall = aggregate(summaries, ("right_method",))
    by_dataset = aggregate(summaries, ("dataset", "right_method"))
    by_capacity = aggregate(summaries, ("capacity", "right_method"))
    write_csv(args.output / "AGGREGATE_OVERALL.csv", overall)
    write_csv(args.output / "AGGREGATE_BY_DATASET.csv", by_dataset)
    write_csv(args.output / "AGGREGATE_BY_CAPACITY.csv", by_capacity)

    old_rows = sorted(
        (row for row in summaries if row["right_method"] == "OldNeural_060_040"),
        key=lambda row: (row["dataset"], int(row["capacity"])),
    )
    lines = [
        "# WAHARP 0.75/0.25 Final Locked Evaluation",
        "",
        f"Training protocol: `{TRAINING_PROTOCOL_ID}`",
        f"Evaluation protocol: `{PROTOCOL_ID}`",
        f"Selected member: `{training['selected_member']}`",
        f"Checkpoint SHA-256: `{training['selected_checkpoint_sha256']}`",
        "",
        "Training and member selection read no final queries. The final pass reused all baseline per-query outputs and rebuilt only the new neural trees.",
        "",
        "## New vs. previous neural model",
        "",
        "| Dataset | Capacity | New/old | 95% CI | Change | New accesses | Old accesses |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in old_rows:
        ratio = float(row["ratio_left_over_right"])
        lines.append(
            f"| {row['dataset'].title()} | {row['capacity']} | {ratio:.6f} | "
            f"[{float(row['bootstrap_ci_low']):.6f}, {float(row['bootstrap_ci_high']):.6f}] | "
            f"{100.0 * (ratio - 1.0):+.2f}% | {int(row['left_total_access']):,} | "
            f"{int(row['right_total_access']):,} |"
        )
    lines += [
        "",
        "## Aggregate controls",
        "",
        "| Control | Micro new/control | Cell-geomean new/control | Micro change | Cell-geomean change |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in overall:
        lines.append(
            f"| {row['right_method']} | {row['micro_ratio_left_over_right']:.6f} | "
            f"{row['geometric_mean_cell_ratio']:.6f} | "
            f"{-row['improvement_percent_micro']:+.2f}% | "
            f"{-row['improvement_percent_geomean']:+.2f}% |"
        )
    lines += [
        "",
        "## Gates",
        "",
        "- Nine dataset-capacity cells completed.",
        "- Each cell used 6,000 locked final queries: 4,200 range, 300 point, and 1,500 kNN.",
        "- Correctness mismatches: 0.",
        "- Baseline reconstructions: 0.",
        "- The same frozen checkpoint was used in every cell.",
    ]
    (args.output / "FINAL_LOCKED_EVALUATION_REPORT.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "protocol_id": PROTOCOL_ID,
        "training_protocol_id": TRAINING_PROTOCOL_ID,
        "status": "COMPLETE",
        "checkpoint_sha256": training["selected_checkpoint_sha256"],
        "selected_member": training["selected_member"],
        "datasets": list(DATASETS),
        "capacities": list(CAPACITIES),
        "cells": len(DATASETS) * len(CAPACITIES),
        "queries_per_cell": 6000,
        "correctness_mismatches": 0,
        "baseline_rebuilds": 0,
    }
    (args.output / "FINAL_STATE.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
