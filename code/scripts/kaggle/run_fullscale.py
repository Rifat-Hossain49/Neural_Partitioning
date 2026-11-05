from __future__ import annotations

import argparse
import gc
import json
import math
import os
import shutil
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch


PROTOCOL = "waharp_final_capacity_sweep_member2_v1_20260901"
# Reuse the locked B=128 query namespace so every capacity is evaluated on
# identical queries. No model selection or training is performed in this sweep.
QUERY_NAMESPACE = "waharp_final_fullscale_b128_member2_93d3c2545e44b9e21269a1b4"
CHECKPOINT_SHA256 = "8709a7f9025292636f0526ccf7b702eda8ce0bce953c02cc1c05245bb13d7e1e"
CAPACITY = 0
SEED = 2026083159
SOFT_BUDGET_SECONDS = 10.5 * 3600
PACKAGE_RESERVE_SECONDS = 45 * 60


def find_bundle(inputs: Path) -> Path:
    for marker in inputs.rglob("WAHARP_FINAL_FULLSCALE_BUNDLE_MARKER.json"):
        raw = json.loads(marker.read_text(encoding="utf-8"))
        if raw.get("protocol_id") == PROTOCOL:
            return marker.parent
    raise FileNotFoundError("Final full-scale code bundle was not attached")


def restore_state(inputs: Path, output: Path, dataset: str) -> None:
    if output.exists():
        return
    choices = []
    for marker in inputs.rglob("FULLSCALE_STATE.json"):
        try:
            raw = json.loads(marker.read_text(encoding="utf-8"))
        except Exception:
            continue
        if (
            raw.get("protocol_id") == PROTOCOL
            and raw.get("dataset") == dataset
            and int(raw.get("capacity", 0)) == CAPACITY
        ):
            choices.append((int(raw.get("durable_units", 0)), int(raw.get("platon_actions", 0)), marker.parent))
    if choices:
        source = max(choices, key=lambda row: (row[0], row[1], str(row[2])))[2]
        shutil.copytree(source, output)
        print("restored state:", source, flush=True)
    else:
        output.mkdir(parents=True, exist_ok=True)


INPUT = Path("/kaggle/input")
BUNDLE = find_bundle(INPUT)
sys.path.insert(0, str(BUNDLE))

from realtrain.baselines import build_str_tree, build_tgs
from realtrain.data import (
    load_arizona,
    load_point_csv,
    locate_arizona_npy,
    locate_crimes_csv,
    locate_twitter_csv,
)
from realtrain.eval import summarize_rows, tree_metrics
from realtrain.native import (
    compile_native,
    discover_platon_support,
    eval_native_tree,
    native_env,
    run_cmd,
    write_platon_array,
    write_records,
)
from realtrain.pipeline import _evaluate_python_method
from realtrain.train import load_model
from realtrain.tree import build_neural_tree
from realtrain.utils import atomic_json, derive_seed, read_csv, sha256_array, sha256_file, write_csv
from realtrain.workload import create_dataset_workloads, load_query_suite


def load_dataset(dataset: str) -> tuple[np.ndarray, dict]:
    if dataset == "twitter":
        return load_point_csv(locate_twitter_csv(INPUT), 0)
    if dataset == "crimes":
        return load_point_csv(locate_crimes_csv(INPUT), 0)
    if dataset == "arizona":
        return load_arizona(locate_arizona_npy(INPUT), 0)
    raise ValueError(dataset)


def normalized_uid(row: dict) -> str:
    uid = str(row["query_uid"])
    if str(row["query_type"]) == "knn":
        if uid.startswith("knn|knn_k"):
            return uid
        if uid.startswith("knn|k"):
            return "knn|knn_k" + uid[len("knn|k"):]
    return uid


def correctness_signature(row: dict):
    if row["query_type"] in ("range", "point"):
        return int(row["result_count"]), int(row["id_sum"]), int(row["id_xor"])
    return int(row["result_count"]), int(row["id_sum"]), int(row["id_xor"])


def pairwise(left_rows: list[dict], right_rows: list[dict], dataset: str, left: str, right: str) -> tuple[dict, list[dict]]:
    a = {normalized_uid(row): row for row in left_rows}
    b = {normalized_uid(row): row for row in right_rows}
    common = sorted(set(a).intersection(b))
    if len(common) != 6000:
        raise RuntimeError(f"canonical join {left}/{right}: {len(common)} != 6000")
    types = Counter(a[key]["query_type"] for key in common)
    if types != Counter({"range": 4200, "point": 300, "knn": 1500}):
        raise RuntimeError(f"query type gate failed: {types}")
    detail = []
    mismatch = 0
    for key in common:
        la, rb = a[key], b[key]
        if la["query_type"] == "knn":
            # libspatialindex returns every object tied at the kth boundary,
            # while the Python evaluator retains exactly k IDs after exploring
            # that same boundary. The semantic invariant is the kth distance.
            agrees = math.isclose(
                float(la["kth_distance"]),
                float(rb["kth_distance"]),
                rel_tol=2e-6,
                abs_tol=2e-6,
            )
        else:
            agrees = correctness_signature(la) == correctness_signature(rb)
        mismatch += int(not agrees)
        detail.append({
            "dataset": dataset,
            "left_method": left,
            "right_method": right,
            "query_uid": key,
            "query_type": la["query_type"],
            "workload": la["workload"],
            "left_access": int(la["total_node_accesses"]),
            "right_access": int(rb["total_node_accesses"]),
        })
    if mismatch:
        raise RuntimeError(f"correctness mismatch {left}/{right}: {mismatch}")
    lv = np.asarray([row["left_access"] for row in detail], dtype=np.float64)
    rv = np.asarray([row["right_access"] for row in detail], dtype=np.float64)
    return {
        "dataset": dataset,
        "left_method": left,
        "right_method": right,
        "queries": len(detail),
        "left_total_access": int(lv.sum()),
        "right_total_access": int(rv.sum()),
        "ratio_left_over_right": float(lv.sum() / rv.sum()),
        "strict_win_rate": float(np.mean(lv < rv)),
        "tie_rate": float(np.mean(lv == rv)),
        "loss_rate": float(np.mean(lv > rv)),
        "correctness_mismatches": 0,
    }, detail


def depth_rows(rows: list[dict], dataset: str, method: str) -> list[dict]:
    totals = defaultdict(float)
    counts = defaultdict(int)
    for row in rows:
        raw = row.get("depth_accesses_json", "")
        if not raw:
            continue
        values = json.loads(raw)
        for depth, accesses in values.items():
            key = (row["query_type"], row["workload"], int(depth))
            totals[key] += float(accesses)
            counts[key] += 1
    return [
        {
            "dataset": dataset,
            "method": method,
            "query_type": key[0],
            "workload": key[1],
            "depth": key[2],
            "queries": counts[key],
            "mean_node_accesses": totals[key] / counts[key],
        }
        for key in sorted(totals)
    ]


def state_payload(dataset: str, status: str, output: Path, started: float, **extra) -> dict:
    methods = sorted(path.parent.name for path in (output / "methods").glob("*/PER_QUERY.csv"))
    progress_path = output / "platon" / "PROGRESS.json"
    progress = json.loads(progress_path.read_text(encoding="utf-8")) if progress_path.is_file() else {}
    return {
        "protocol_id": PROTOCOL,
        "paired_query_namespace": QUERY_NAMESPACE,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "dataset": dataset,
        "capacity": CAPACITY,
        "status": status,
        "elapsed_seconds_this_run": time.monotonic() - started,
        "durable_units": len(methods),
        "completed_methods": methods,
        "platon_actions": int(progress.get("actions_completed", 0)),
        **extra,
    }


def save_method(output: Path, dataset: str, method: str, rows: list[dict], construction: dict, structure: dict | None) -> None:
    root = output / "methods" / method
    write_csv(root / "PER_QUERY.csv", rows)
    write_csv(root / "WORKLOAD_METRICS.csv", summarize_rows(rows, method, dataset))
    write_csv(root / "NODE_ACCESSES_BY_LEVEL.csv", depth_rows(rows, dataset, method))
    atomic_json(root / "CONSTRUCTION.json", construction)
    if structure is not None:
        atomic_json(root / "TREE_STRUCTURE.json", structure)


def run_python_methods(dataset: str, rects: np.ndarray, workroot: Path, output: Path, started: float) -> None:
    suite = load_query_suite(workroot, "final")
    construction_queries = np.load(workroot / "construction_boxes.npy")
    specs = ("Neural", "STR", "TGS")
    for method in specs:
        result_path = output / "methods" / method / "PER_QUERY.csv"
        if result_path.is_file():
            continue
        method_started = time.perf_counter()
        if method == "Neural":
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
            checkpoint = BUNDLE / "frozen_model" / "member_2.pt"
            if sha256_file(checkpoint) != CHECKPOINT_SHA256:
                raise RuntimeError("Frozen checkpoint hash mismatch")
            model = load_model(checkpoint, device)
            tree, build = build_neural_tree(rects, construction_queries, model, device, CAPACITY, 12, 128)
            build.update({"checkpoint_sha256": CHECKPOINT_SHA256, "fallbacks_used": False, "training_performed": False})
        elif method == "STR":
            tree, build = build_str_tree(rects, CAPACITY)
        else:
            tree, build = build_tgs(rects, CAPACITY)
        rows = _evaluate_python_method(tree, suite, len(rects), CAPACITY, method)
        build["phase_wall_seconds"] = time.perf_counter() - method_started
        save_method(output, dataset, method, rows, build, tree_metrics(tree, CAPACITY))
        atomic_json(output / "FULLSCALE_STATE.json", state_payload(dataset, "RUNNING", output, started, last_completed=method))
        print(method, "complete", build, flush=True)
        del tree, rows
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def run_platon(dataset: str, rects: np.ndarray, workroot: Path, output: Path, started: float) -> bool:
    result_path = output / "methods" / "PLATON" / "PER_QUERY.csv"
    if result_path.is_file():
        return True
    remaining = SOFT_BUDGET_SECONDS - (time.monotonic() - started)
    if remaining <= PACKAGE_RESERVE_SECONDS + 600:
        return False
    native = compile_native(INPUT, output, BUNDLE)
    support, author = discover_platon_support(INPUT)
    scratch = Path("/kaggle/working") / f"waharp_fullscale_scratch_{dataset}"
    scratch.mkdir(parents=True, exist_ok=True)
    data_path = scratch / "data.npy"
    query_path = scratch / "construction.npy"
    records_path = scratch / "records.txt"
    write_platon_array(data_path, rects)
    construction = np.load(workroot / "construction_boxes.npy")
    write_platon_array(query_path, construction)
    proot = output / "platon"
    proot.mkdir(parents=True, exist_ok=True)
    cutlist = proot / "author_cuts.txt"
    progress = proot / "PROGRESS.json"
    checkpoint = proot / "actions.sqlite"
    if not cutlist.is_file():
        budget = max(60.0, remaining - PACKAGE_RESERVE_SECONDS)
        cmd = [
            sys.executable,
            BUNDLE / "scripts" / "platon_resumable_cutlist.py",
            "--author-root", author,
            "--data", data_path,
            "--queries", query_path,
            "--output-cutlist", cutlist,
            "--checkpoint-db", checkpoint,
            "--progress-json", progress,
            "--branch", str(CAPACITY),
            "--seed", str(derive_seed(QUERY_NAMESPACE, SEED, dataset) & 0xFFFFFFFF),
            "--rollouts", "25",
            "--simulation-steps", "100",
            "--budget-seconds", str(budget),
            "--decision-reserve-seconds", "300",
            "--progress-every-actions", "25",
        ]
        cp = subprocess.run([str(x) for x in cmd], text=True)
        if cp.returncode:
            raise RuntimeError(f"resumable PLATON cut-list process failed: {cp.returncode}")
    meta = json.loads(progress.read_text(encoding="utf-8"))
    if not meta.get("complete"):
        atomic_json(output / "FULLSCALE_STATE.json", state_payload(dataset, "PAUSED_RESUME_REQUIRED", output, started, reason="platon_cutlist"))
        return False

    write_records(records_path, rects)
    tree_base = scratch / f"{dataset}_PLATON_B{CAPACITY}"
    construction_path = proot / "CONSTRUCTION.json"
    tree_files_exist = all(Path(str(tree_base) + suffix).is_file() for suffix in (".idx", ".dat"))
    if not construction_path.is_file() or not tree_files_exist:
        env = native_env(native)
        bulk_started = time.perf_counter()
        utilization = 0.8
        nominal_capacity = int(round(CAPACITY / utilization))
        text = run_cmd(
            [native["bulk"], records_path, tree_base, str(nominal_capacity), str(utilization), cutlist, "4096"],
            proot / "bulk.log",
            env,
        )
        bulk_seconds = time.perf_counter() - bulk_started
        lines = [line for line in text.splitlines() if line.startswith("PLATON_BUILD_RESULT")]
        if not lines:
            raise RuntimeError("missing PLATON_BUILD_RESULT")
        values = dict(item.split("=", 1) for item in lines[-1].split(",")[1:])
        if values.get("valid_tree") != "1":
            raise RuntimeError(f"invalid PLATON tree: {values}")
        construction_meta = {
            "method": "author_PLATON",
            "tree_base": str(tree_base),
            "index_identifier": int(values["index_identifier"]),
            "policy_seconds": float(meta.get("decision_time_seconds_sum", 0.0)),
            "bulk_seconds": bulk_seconds,
            "total_construction_seconds": float(meta.get("decision_time_seconds_sum", 0.0)) + bulk_seconds,
            "cut_count": int(meta["cut_count"]),
            "cut_sha256": sha256_file(cutlist),
            "capacity": CAPACITY,
            "nominal_capacity": nominal_capacity,
            "utilization": utilization,
        }
        atomic_json(construction_path, construction_meta)
    construction_meta = json.loads(construction_path.read_text(encoding="utf-8"))
    suite = load_query_suite(workroot, "final")
    rows = eval_native_tree(tree_base, int(construction_meta["index_identifier"]), suite, len(rects), CAPACITY, proot / "eval", native, "PLATON")
    save_method(output, dataset, "PLATON", rows, construction_meta, None)
    for suffix in (".idx", ".dat"):
        path = Path(str(tree_base) + suffix)
        if path.exists():
            path.unlink()
    for path in (records_path, data_path, query_path):
        if path.exists():
            path.unlink()
    return True


def finish_report(dataset: str, output: Path, started: float) -> None:
    methods = {}
    for method in ("Neural", "STR", "TGS", "PLATON"):
        path = output / "methods" / method / "PER_QUERY.csv"
        if path.is_file():
            methods[method] = read_csv(path)
    if "PLATON" not in methods:
        return
    summaries = []
    details = []
    for method in ("Neural", "STR", "TGS"):
        summary, detail = pairwise(methods[method], methods["PLATON"], dataset, method, "PLATON")
        summaries.append(summary)
        details.extend(detail)
    write_csv(output / "PAIRWISE_VS_PLATON.csv", summaries)
    write_csv(output / "PAIRWISE_PER_QUERY.csv", details)
    atomic_json(output / "FULLSCALE_STATE.json", state_payload(dataset, "COMPLETE", output, started, pairwise=summaries))


def main() -> None:
    global CAPACITY
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("twitter", "crimes", "arizona"), required=True)
    parser.add_argument("--capacity", type=int, choices=(256, 512), required=True)
    args = parser.parse_args()
    CAPACITY = args.capacity
    started = time.monotonic()
    output = Path("/kaggle/working") / f"WAHARP_FINAL_FULLSCALE_{args.dataset.upper()}_B{CAPACITY}"
    restore_state(INPUT, output, args.dataset)
    atomic_json(output / "FULLSCALE_STATE.json", state_payload(args.dataset, "RUNNING", output, started))
    rects, metadata = load_dataset(args.dataset)
    atomic_json(output / "DATASET.json", metadata)
    workroot = output / "workload"
    create_dataset_workloads(
        rects,
        args.dataset,
        workroot,
        QUERY_NAMESPACE,
        derive_seed(QUERY_NAMESPACE, SEED, args.dataset),
        2800,
        100,
        100,
        100,
        300,
        300,
        300,
    )
    run_python_methods(args.dataset, rects, workroot, output, started)
    if not run_platon(args.dataset, rects, workroot, output, started):
        print("PAUSED_RESUME_REQUIRED", flush=True)
        return
    finish_report(args.dataset, output, started)
    print("COMPLETE", output, flush=True)


if __name__ == "__main__":
    main()
