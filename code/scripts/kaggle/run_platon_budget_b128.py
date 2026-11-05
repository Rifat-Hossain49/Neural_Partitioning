from __future__ import annotations

import csv
import gc
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch


PROTOCOL_ID = "waharp_platon_budget_b128_v1_20260907"
FRESH_EVAL_NAMESPACE = "waharp_platon_budget_b128_fresh_eval_v1_20260907"
PAPER_CONSTRUCTION_NAMESPACE = "waharp_final_fullscale_b128_member2_93d3c2545e44b9e21269a1b4"
PAPER_B128_PROTOCOL = "waharp_final_fullscale_b128_member2_20260831"
CODE_PROTOCOL = "waharp_final_capacity_sweep_member2_v1_20260901"
TRAINING_PROTOCOL = "waharp_loss075_final_train_v1_20260903"

CAPACITY = 128
ROLLOUTS = (1, 5, 10, 25)
SIMULATION_STEPS = 100
DATASETS = ("arizona", "crimes", "twitter")
SEED = 2026090717
CONSTRUCTION_QUERY_COUNT = 2800
FINAL_RANGE_PER_CONDITION = 300
FINAL_POINT_COUNT = 300
FINAL_KNN_POINTS = 300
EXPECTED_QUERY_TYPES = Counter({"range": 4200, "point": 300, "knn": 1500})
SOFT_BUDGET_SECONDS = 10.5 * 3600
PACKAGE_RESERVE_SECONDS = 45 * 60

INPUT = Path("/kaggle/input")
OUTPUT = Path("/kaggle/working/WAHARP_PLATON_BUDGET_B128")
SCRATCH = Path("/tmp/waharp_platon_budget_b128")
OUTPUT.mkdir(parents=True, exist_ok=True)
SCRATCH.mkdir(parents=True, exist_ok=True)
RUN_STARTED = time.monotonic()


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def json_default(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=json_default),
        encoding="utf-8",
    )
    os.replace(tmp, path)


def atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    if not rows:
        tmp.write_text("", encoding="utf-8")
    else:
        fields: list[str] = []
        seen: set[str] = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    fields.append(key)
                    seen.add(key)
        with tmp.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    os.replace(tmp, path)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def find_code_bundle() -> Path:
    matches: list[Path] = []
    for marker in INPUT.rglob("WAHARP_FINAL_FULLSCALE_BUNDLE_MARKER.json"):
        try:
            raw = json.loads(marker.read_text(encoding="utf-8"))
        except Exception:
            continue
        if raw.get("protocol_id") == CODE_PROTOCOL:
            matches.append(marker.parent)
    if len(matches) != 1:
        raise RuntimeError(f"Expected one code bundle, found {matches}")
    return matches[0]


CODE_ROOT = find_code_bundle()
sys.path.insert(0, str(CODE_ROOT))

from realtrain.data import (  # noqa: E402
    load_arizona,
    load_point_csv,
    locate_arizona_npy,
    locate_crimes_csv,
    locate_twitter_csv,
)
from realtrain.native import (  # noqa: E402
    compile_native,
    discover_platon_support,
    eval_native_tree,
    native_env,
    run_cmd,
    write_platon_array,
    write_records,
)
from realtrain.pipeline import _evaluate_python_method  # noqa: E402
from realtrain.train import load_model  # noqa: E402
from realtrain.tree import build_neural_tree  # noqa: E402
from realtrain.utils import derive_seed, sha256_array  # noqa: E402
from realtrain.workload import (  # noqa: E402
    KNN_K,
    load_query_suite,
    make_points,
    make_range_conditions,
)


def restore_previous_output() -> None:
    if (OUTPUT / "experiment_state.json").is_file():
        return
    candidates: list[tuple[int, Path]] = []
    for state_path in INPUT.rglob("experiment_state.json"):
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if state.get("protocol_id") != PROTOCOL_ID:
            continue
        cells = len(state.get("completed_platon_cells", []))
        candidates.append((cells, state_path.parent))
    if candidates:
        source = max(candidates, key=lambda item: (item[0], str(item[1])))[1]
        shutil.copytree(source, OUTPUT, dirs_exist_ok=True)
        print(f"Restored prior experiment output from {source}", flush=True)


def completed_cells() -> tuple[list[str], list[str]]:
    waharp = []
    platon = []
    for dataset in DATASETS:
        if (OUTPUT / "datasets" / dataset / "WAHARP" / "PER_QUERY.csv").is_file():
            waharp.append(dataset)
        for rollout in ROLLOUTS:
            if (OUTPUT / "datasets" / dataset / f"PLATON_{rollout}" / "PER_QUERY.csv").is_file():
                platon.append(f"{dataset}/r{rollout}")
    return waharp, platon


def save_state(status: str, **extra: Any) -> None:
    waharp, platon = completed_cells()
    state = {
        "protocol_id": PROTOCOL_ID,
        "status": status,
        "capacity": CAPACITY,
        "rollouts": list(ROLLOUTS),
        "simulation_steps": SIMULATION_STEPS,
        "datasets": list(DATASETS),
        "fresh_evaluation_namespace": FRESH_EVAL_NAMESPACE,
        "paper_construction_namespace": PAPER_CONSTRUCTION_NAMESPACE,
        "queries_per_cell": 6000,
        "query_mix": dict(EXPECTED_QUERY_TYPES),
        "gpu_available": bool(torch.cuda.is_available()),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "waharp_retrained": False,
        "completed_waharp_datasets": waharp,
        "completed_platon_cells": platon,
        "elapsed_seconds_this_session": time.monotonic() - RUN_STARTED,
        "updated_utc": now_utc(),
        **extra,
    }
    atomic_json(OUTPUT / "experiment_state.json", state)


def remaining_seconds() -> float:
    return SOFT_BUDGET_SECONDS - (time.monotonic() - RUN_STARTED)


def locate_checkpoint() -> tuple[Path, dict[str, Any]]:
    matches: list[tuple[Path, dict[str, Any]]] = []
    for state_path in INPUT.rglob("FINAL_TRAINING_STATE.json"):
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if state.get("protocol_id") == TRAINING_PROTOCOL and state.get("status") == "COMPLETE":
            matches.append((state_path.parent, state))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one frozen WAHARP training output, found {matches}")
    root, state = matches[0]
    checkpoint = root / state["selected_checkpoint"]
    if sha256_file(checkpoint) != state["selected_checkpoint_sha256"]:
        raise RuntimeError("Frozen WAHARP checkpoint hash mismatch")
    if state.get("final_queries_read") is not False:
        raise RuntimeError("Frozen model does not satisfy final-query isolation gate")
    return checkpoint, state


def locate_paper_b128(dataset: str) -> Path:
    matches: list[Path] = []
    for state_path in INPUT.rglob("FULLSCALE_STATE.json"):
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        namespace = state.get("fresh_final_namespace", state.get("paired_query_namespace"))
        root = state_path.parent
        cut_meta = root / "platon" / "author_cuts.json"
        cut_list = root / "platon" / "author_cuts.txt"
        if (
            state.get("protocol_id") == PAPER_B128_PROTOCOL
            and state.get("dataset") == dataset
            and int(state.get("capacity", 0)) == CAPACITY
            and str(state.get("status", "")).startswith("COMPLETE")
            and namespace == PAPER_CONSTRUCTION_NAMESPACE
            and cut_meta.is_file()
            and cut_list.is_file()
        ):
            matches.append(root)
    if len(matches) != 1:
        raise RuntimeError(f"Expected one reusable paper B=128 root for {dataset}, found {matches}")
    return matches[0]


def load_dataset(dataset: str) -> tuple[np.ndarray, dict[str, Any]]:
    if dataset == "twitter":
        return load_point_csv(locate_twitter_csv(INPUT), 0)
    if dataset == "crimes":
        return load_point_csv(locate_crimes_csv(INPUT), 0)
    if dataset == "arizona":
        return load_arizona(locate_arizona_npy(INPUT), 0)
    raise ValueError(dataset)


def create_fresh_workload(dataset: str, rects: np.ndarray, paper_root: Path) -> Path:
    root = OUTPUT / "datasets" / dataset / "workload"
    manifest_path = root / "WORKLOAD_MANIFEST.json"
    if manifest_path.is_file():
        return root
    root.mkdir(parents=True, exist_ok=True)

    paper_manifest = json.loads(
        (paper_root / "workload" / "WORKLOAD_MANIFEST.json").read_text(encoding="utf-8")
    )
    construction = np.load(paper_root / "workload" / "construction_boxes.npy")
    if len(construction) != CONSTRUCTION_QUERY_COUNT:
        raise RuntimeError(f"Unexpected construction-query count for {dataset}")
    if sha256_array(construction) != paper_manifest["construction"]["sha256"]:
        raise RuntimeError(f"Paper construction-query hash mismatch for {dataset}")
    np.save(root / "construction_boxes.npy", construction)

    def ds(label: str) -> int:
        return derive_seed(FRESH_EVAL_NAMESPACE, SEED, f"{dataset}|{label}")

    ranges, specs = make_range_conditions(FINAL_RANGE_PER_CONDITION, ds("final_ranges"))
    for name, values in ranges.items():
        np.save(root / f"final_range_{name}.npy", values)
    points = make_points(rects, FINAL_POINT_COUNT, ds("final_points"))
    knn_points = make_points(rects, FINAL_KNN_POINTS, ds("final_knn"))
    np.save(root / "final_points.npy", points)
    np.save(root / "final_knn_points.npy", knn_points)

    manifest = {
        "dataset": dataset,
        "namespace": FRESH_EVAL_NAMESPACE,
        "seed": SEED,
        "construction": {
            **paper_manifest["construction"],
            "source_namespace": PAPER_CONSTRUCTION_NAMESPACE,
            "reuse_reason": "same paper construction-query protocol; evaluation queries are fresh",
        },
        "final_range_specs": specs,
        "final_points": {"count": len(points), "sha256": sha256_array(points)},
        "final_knn": {
            "count_per_k": len(knn_points),
            "k_values": list(KNN_K),
            "sha256": sha256_array(knn_points),
        },
        "query_mix": dict(EXPECTED_QUERY_TYPES),
        "leakage_rule": "fresh final queries are generated after model selection and never used for construction",
    }
    atomic_json(manifest_path, manifest)
    suite = load_query_suite(root, "final")
    observed = Counter(
        {"range": sum(len(v) for v in suite["ranges"].values()),
         "point": len(suite["points"]),
         "knn": len(suite["knn_points"]) * len(KNN_K)}
    )
    if observed != EXPECTED_QUERY_TYPES:
        raise RuntimeError(f"Fresh query mix mismatch for {dataset}: {observed}")
    return root


def prepare_scratch(dataset: str, rects: np.ndarray, workroot: Path) -> dict[str, Path]:
    root = SCRATCH / dataset
    root.mkdir(parents=True, exist_ok=True)
    data_path = root / "data.npy"
    query_path = root / "construction.npy"
    records_path = root / "records.txt"
    write_platon_array(data_path, rects)
    write_platon_array(query_path, np.load(workroot / "construction_boxes.npy"))
    write_records(records_path, rects)
    return {"root": root, "data": data_path, "queries": query_path, "records": records_path}


def run_waharp(
    dataset: str,
    rects: np.ndarray,
    workroot: Path,
    checkpoint: Path,
    checkpoint_state: dict[str, Any],
) -> None:
    root = OUTPUT / "datasets" / dataset / "WAHARP"
    result_path = root / "PER_QUERY.csv"
    construction_path = root / "CONSTRUCTION.json"
    if result_path.is_file() and construction_path.is_file():
        return
    if result_path.is_file():
        result_path.unlink()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = load_model(checkpoint, device)
    construction = np.load(workroot / "construction_boxes.npy")
    started = time.perf_counter()
    tree, diagnostics = build_neural_tree(
        rects,
        construction,
        model,
        device,
        CAPACITY,
        hist_bins=12,
        query_cap=128,
    )
    build_seconds = time.perf_counter() - started
    suite = load_query_suite(workroot, "final")
    rows = _evaluate_python_method(tree, suite, len(rects), CAPACITY, "WAHARP")
    if len(rows) != 6000:
        raise RuntimeError(f"WAHARP query count mismatch for {dataset}: {len(rows)}")
    construction_meta = {
        "dataset": dataset,
        "method": "WAHARP",
        "capacity": CAPACITY,
        "construction_seconds": build_seconds,
        "gpu_used": device.startswith("cuda"),
        "checkpoint_sha256": checkpoint_state["selected_checkpoint_sha256"],
        "selected_member": checkpoint_state["selected_member"],
        "training_performed": False,
        "diagnostics": diagnostics,
    }
    atomic_json(construction_path, construction_meta)
    atomic_csv(result_path, rows)
    save_state("RUNNING", last_completed=f"{dataset}/WAHARP")
    del rows, tree, model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def verify_reused_cutlist(
    dataset: str,
    paper_root: Path,
    scratch: dict[str, Path],
    destination: Path,
) -> dict[str, Any]:
    source_cut = paper_root / "platon" / "author_cuts.txt"
    source_meta = json.loads(
        (paper_root / "platon" / "author_cuts.json").read_text(encoding="utf-8")
    )
    gates = {
        "complete": source_meta.get("complete") is True,
        "branch": int(source_meta.get("branch", 0)) == CAPACITY,
        "rollouts": int(source_meta.get("rollouts", 0)) == 25,
        "simulation_steps": int(source_meta.get("simulation_steps", 0)) == SIMULATION_STEPS,
        "data_hash": source_meta.get("data_file_sha256") == sha256_file(scratch["data"]),
        "query_hash": source_meta.get("query_file_sha256") == sha256_file(scratch["queries"]),
    }
    if not all(gates.values()):
        raise RuntimeError(f"Unsafe PLATON-25 reuse for {dataset}: {gates}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_cut, destination)
    reused = {
        **source_meta,
        "reused": True,
        "reuse_gates": gates,
        "source_protocol": PAPER_B128_PROTOCOL,
        "source_cut_sha256": sha256_file(source_cut),
    }
    atomic_json(destination.with_suffix(".json"), reused)
    return reused


def generate_cutlist(
    dataset: str,
    rollout: int,
    scratch: dict[str, Path],
    author: Path,
    paper_root: Path,
) -> tuple[bool, dict[str, Any]]:
    root = OUTPUT / "datasets" / dataset / f"PLATON_{rollout}"
    root.mkdir(parents=True, exist_ok=True)
    cutlist = root / "author_cuts.txt"
    progress = root / "PROGRESS.json"
    checkpoint_db = root / "actions.sqlite"
    if cutlist.is_file() and progress.is_file():
        meta = json.loads(progress.read_text(encoding="utf-8"))
        if meta.get("complete"):
            return True, meta
    if rollout == 25:
        meta = verify_reused_cutlist(dataset, paper_root, scratch, cutlist)
        atomic_json(progress, meta)
        return True, meta

    remaining = remaining_seconds()
    if remaining <= PACKAGE_RESERVE_SECONDS + 600:
        return False, {"complete": False, "reason": "global_time_reserve"}
    driver = CODE_ROOT / "scripts" / "platon_resumable_cutlist.py"
    cmd = [
        sys.executable,
        driver,
        "--author-root", author,
        "--data", scratch["data"],
        "--queries", scratch["queries"],
        "--output-cutlist", cutlist,
        "--checkpoint-db", checkpoint_db,
        "--progress-json", progress,
        "--branch", str(CAPACITY),
        "--seed", str(derive_seed(PAPER_CONSTRUCTION_NAMESPACE, 2026083159, dataset) & 0xFFFFFFFF),
        "--rollouts", str(rollout),
        "--simulation-steps", str(SIMULATION_STEPS),
        "--budget-seconds", str(max(60.0, remaining - PACKAGE_RESERVE_SECONDS)),
        "--decision-reserve-seconds", "300",
        "--progress-every-actions", "25",
    ]
    completed = subprocess.run([str(value) for value in cmd], check=False).returncode
    if completed:
        raise RuntimeError(f"PLATON cut-list process failed for {dataset}/r{rollout}")
    meta = json.loads(progress.read_text(encoding="utf-8"))
    return bool(meta.get("complete")), meta


def run_platon(
    dataset: str,
    rollout: int,
    scratch: dict[str, Path],
    workroot: Path,
    object_count: int,
    native: dict[str, str],
    author: Path,
    paper_root: Path,
) -> bool:
    root = OUTPUT / "datasets" / dataset / f"PLATON_{rollout}"
    result_path = root / "PER_QUERY.csv"
    construction_path = root / "CONSTRUCTION.json"
    if result_path.is_file() and construction_path.is_file():
        return True
    if result_path.is_file():
        result_path.unlink()
    complete, cut_meta = generate_cutlist(dataset, rollout, scratch, author, paper_root)
    if not complete:
        save_state(
            "PAUSED_RESUME_REQUIRED",
            paused_cell=f"{dataset}/r{rollout}",
            pause_reason=cut_meta.get("reason", "platon_cutlist"),
        )
        return False

    tree_base = scratch["root"] / f"{dataset}_PLATON_r{rollout}_B128"
    for suffix in (".idx", ".dat"):
        stale = Path(str(tree_base) + suffix)
        if stale.exists():
            stale.unlink()
    nominal_capacity = int(round(CAPACITY / 0.8))
    started = time.perf_counter()
    text = run_cmd(
        [
            native["bulk"],
            scratch["records"],
            tree_base,
            str(nominal_capacity),
            "0.8",
            root / "author_cuts.txt",
            "4096",
        ],
        root / "bulk.log",
        native_env(native),
    )
    native_build_seconds = time.perf_counter() - started
    lines = [line for line in text.splitlines() if line.startswith("PLATON_BUILD_RESULT")]
    if not lines:
        raise RuntimeError(f"Missing PLATON build result for {dataset}/r{rollout}")
    values = dict(item.split("=", 1) for item in lines[-1].split(",")[1:])
    if values.get("valid_tree") != "1":
        raise RuntimeError(f"Invalid PLATON tree for {dataset}/r{rollout}: {values}")

    mcts_seconds = float(cut_meta.get("decision_time_seconds_sum", 0.0))
    construction = {
        "dataset": dataset,
        "method": f"PLATON-{rollout}",
        "capacity": CAPACITY,
        "mcts_rollouts": rollout,
        "mcts_simulation_steps": SIMULATION_STEPS,
        "mcts_seconds": mcts_seconds,
        "native_build_seconds": native_build_seconds,
        "construction_seconds": mcts_seconds + native_build_seconds,
        "cut_count": int(cut_meta["cut_count"]),
        "cut_sha256": sha256_file(root / "author_cuts.txt"),
        "reused_mcts": rollout == 25,
        "valid_tree": True,
        "index_identifier": int(values["index_identifier"]),
    }
    suite = load_query_suite(workroot, "final")
    eval_root = root / "eval"
    if eval_root.exists():
        shutil.rmtree(eval_root)
    rows = eval_native_tree(
        tree_base,
        construction["index_identifier"],
        suite,
        object_count,
        CAPACITY,
        eval_root,
        native,
        f"PLATON-{rollout}",
    )
    if len(rows) != 6000:
        raise RuntimeError(f"PLATON query count mismatch for {dataset}/r{rollout}: {len(rows)}")
    atomic_json(construction_path, construction)
    atomic_csv(result_path, rows)
    save_state("RUNNING", last_completed=f"{dataset}/PLATON-{rollout}")
    for suffix in (".idx", ".dat"):
        path = Path(str(tree_base) + suffix)
        if path.exists():
            path.unlink()
    return True


def normalized_uid(row: dict[str, Any]) -> str:
    uid = str(row["query_uid"])
    if str(row["query_type"]) == "knn":
        if uid.startswith("knn|knn_k"):
            return uid
        if uid.startswith("knn|k"):
            return "knn|knn_k" + uid[len("knn|k"):]
    return uid


def compare_rows(
    dataset: str,
    rollout: int,
    waharp_rows: list[dict[str, Any]],
    platon_rows: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    left = {normalized_uid(row): row for row in waharp_rows}
    right = {normalized_uid(row): row for row in platon_rows}
    if set(left) != set(right) or len(left) != 6000:
        raise RuntimeError(f"Query join mismatch for {dataset}/r{rollout}")
    by_family: list[dict[str, Any]] = []
    correctness: list[dict[str, Any]] = []
    all_left: list[int] = []
    all_right: list[int] = []
    total_mismatches = 0

    for family in ("range", "point", "knn"):
        keys = sorted(key for key, row in left.items() if row["query_type"] == family)
        lv = np.asarray([int(left[key]["total_node_accesses"]) for key in keys])
        rv = np.asarray([int(right[key]["total_node_accesses"]) for key in keys])
        mismatches = 0
        for key in keys:
            a = left[key]
            b = right[key]
            if family == "knn":
                agrees = math.isclose(
                    float(a["kth_distance"]),
                    float(b["kth_distance"]),
                    rel_tol=2e-6,
                    abs_tol=2e-6,
                )
                check = "equivalent kth distance; native search may expand boundary ties"
            else:
                agrees = (
                    int(a["result_count"]), int(a["id_sum"]), int(a["id_xor"])
                ) == (
                    int(b["result_count"]), int(b["id_sum"]), int(b["id_xor"])
                )
                check = "exact result count, ID sum, and ID XOR"
            mismatches += int(not agrees)
        total_mismatches += mismatches
        by_family.append({
            "dataset": dataset,
            "capacity": CAPACITY,
            "platon_rollouts": rollout,
            "query_family": family,
            "queries": len(keys),
            "waharp_logical_node_accesses": int(lv.sum()),
            "platon_logical_node_accesses": int(rv.sum()),
            "waharp_over_platon_ratio": float(lv.sum() / rv.sum()),
            "waharp_strict_wins": int(np.sum(lv < rv)),
            "ties": int(np.sum(lv == rv)),
            "waharp_losses": int(np.sum(lv > rv)),
        })
        correctness.append({
            "dataset": dataset,
            "capacity": CAPACITY,
            "platon_rollouts": rollout,
            "query_family": family,
            "queries_checked": len(keys),
            "mismatches": mismatches,
            "passed": mismatches == 0,
            "correctness_check": check,
        })
        all_left.extend(lv.tolist())
        all_right.extend(rv.tolist())

    lv = np.asarray(all_left)
    rv = np.asarray(all_right)
    overall = {
        "dataset": dataset,
        "capacity": CAPACITY,
        "platon_rollouts": rollout,
        "platon_simulation_steps": SIMULATION_STEPS,
        "queries": len(lv),
        "waharp_logical_node_accesses": int(lv.sum()),
        "platon_logical_node_accesses": int(rv.sum()),
        "waharp_over_platon_ratio": float(lv.sum() / rv.sum()),
        "waharp_strict_wins": int(np.sum(lv < rv)),
        "ties": int(np.sum(lv == rv)),
        "waharp_losses": int(np.sum(lv > rv)),
        "correctness_mismatches": total_mismatches,
    }
    return overall, by_family, correctness


def regenerate_outputs() -> tuple[list[dict[str, Any]], int]:
    results: list[dict[str, Any]] = []
    families: list[dict[str, Any]] = []
    correctness: list[dict[str, Any]] = []
    construction: list[dict[str, Any]] = []

    for dataset in DATASETS:
        waharp_root = OUTPUT / "datasets" / dataset / "WAHARP"
        waharp_query = waharp_root / "PER_QUERY.csv"
        waharp_build = waharp_root / "CONSTRUCTION.json"
        if waharp_build.is_file():
            raw = json.loads(waharp_build.read_text(encoding="utf-8"))
            construction.append({
                "dataset": dataset,
                "method": "WAHARP",
                "capacity": CAPACITY,
                "platon_rollouts": "",
                "mcts_seconds": "",
                "native_build_seconds": "",
                "construction_seconds": raw["construction_seconds"],
                "reused_mcts": False,
            })
        if not waharp_query.is_file():
            continue
        waharp_rows = read_csv(waharp_query)
        for rollout in ROLLOUTS:
            root = OUTPUT / "datasets" / dataset / f"PLATON_{rollout}"
            query_path = root / "PER_QUERY.csv"
            build_path = root / "CONSTRUCTION.json"
            if build_path.is_file():
                raw = json.loads(build_path.read_text(encoding="utf-8"))
                construction.append({
                    "dataset": dataset,
                    "method": f"PLATON-{rollout}",
                    "capacity": CAPACITY,
                    "platon_rollouts": rollout,
                    "mcts_seconds": raw["mcts_seconds"],
                    "native_build_seconds": raw["native_build_seconds"],
                    "construction_seconds": raw["construction_seconds"],
                    "reused_mcts": raw["reused_mcts"],
                })
            if not query_path.is_file():
                continue
            overall, family_rows, correctness_rows = compare_rows(
                dataset, rollout, waharp_rows, read_csv(query_path)
            )
            results.append(overall)
            families.extend(family_rows)
            correctness.extend(correctness_rows)

    atomic_csv(OUTPUT / "RESULTS.csv", results)
    atomic_csv(OUTPUT / "QUERY_FAMILY_RESULTS.csv", families)
    atomic_csv(OUTPUT / "CORRECTNESS.csv", correctness)
    atomic_csv(OUTPUT / "CONSTRUCTION_TIMES.csv", construction)
    mismatches = sum(int(row["mismatches"]) for row in correctness)
    write_report(results, construction, mismatches)
    if len(results) == len(DATASETS) * len(ROLLOUTS):
        write_plot(results, construction)
    return results, mismatches


def write_report(
    results: list[dict[str, Any]],
    construction: list[dict[str, Any]],
    mismatches: int,
) -> None:
    lines = [
        "# WAHARP vs. PLATON MCTS Budget at B=128",
        "",
        f"Protocol: `{PROTOCOL_ID}`",
        "",
        "WAHARP used the existing frozen model; no training was performed. "
        "PLATON used the paper's author implementation with 100 simulation steps. "
        "The PLATON-25 cut lists were reused only after protocol, data-hash, and "
        "construction-query-hash verification. Evaluation used a fresh held-out "
        "6,000-query workload per dataset, shared by every method.",
        "",
        "## Logical node accesses",
        "",
        "| Dataset | Rollouts | WAHARP | PLATON | WAHARP / PLATON | W / T / L |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(results, key=lambda r: (r["dataset"], r["platon_rollouts"])):
        lines.append(
            f"| {row['dataset']} | {row['platon_rollouts']} | "
            f"{row['waharp_logical_node_accesses']:,} | "
            f"{row['platon_logical_node_accesses']:,} | "
            f"{row['waharp_over_platon_ratio']:.6f} | "
            f"{row['waharp_strict_wins']} / {row['ties']} / {row['waharp_losses']} |"
        )
    lines += [
        "",
        "## Construction time",
        "",
        "| Dataset | Method | MCTS s | Native build s | Total s |",
        "|---|---|---:|---:|---:|",
    ]
    for row in sorted(construction, key=lambda r: (r["dataset"], str(r["method"]))):
        mcts = "-" if row["mcts_seconds"] == "" else f"{float(row['mcts_seconds']):.3f}"
        native = "-" if row["native_build_seconds"] == "" else f"{float(row['native_build_seconds']):.3f}"
        lines.append(
            f"| {row['dataset']} | {row['method']} | {mcts} | {native} | "
            f"{float(row['construction_seconds']):.3f} |"
        )
    lines += [
        "",
        "## Correctness",
        "",
        f"Total semantic mismatches across completed comparisons: **{mismatches}**.",
        "Range and point queries require exact result count, ID sum, and ID XOR. "
        "kNN queries require equivalent kth distance because the native engine may "
        "return additional boundary ties.",
        "",
        "Detailed query-family and correctness results are in "
        "`QUERY_FAMILY_RESULTS.csv` and `CORRECTNESS.csv`.",
    ]
    tmp = OUTPUT / "FINAL_REPORT.md.tmp"
    tmp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.replace(tmp, OUTPUT / "FINAL_REPORT.md")


def write_plot(results: list[dict[str, Any]], construction: list[dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    result_map = {(row["dataset"], int(row["platon_rollouts"])): row for row in results}
    time_map = {(row["dataset"], row["method"]): float(row["construction_seconds"]) for row in construction}
    colors = {
        "WAHARP": "#0072B2",
        "PLATON-1": "#009E73",
        "PLATON-5": "#E69F00",
        "PLATON-10": "#CC79A7",
        "PLATON-25": "#D55E00",
    }
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), constrained_layout=True)
    for axis, dataset in zip(axes, DATASETS):
        baseline = float(result_map[(dataset, 25)]["platon_logical_node_accesses"])
        waharp_access = float(result_map[(dataset, 25)]["waharp_logical_node_accesses"])
        points = [("WAHARP", time_map[(dataset, "WAHARP")], waharp_access / baseline)]
        for rollout in ROLLOUTS:
            name = f"PLATON-{rollout}"
            access = float(result_map[(dataset, rollout)]["platon_logical_node_accesses"])
            points.append((name, time_map[(dataset, name)], access / baseline))
        for name, x_value, y_value in points:
            axis.scatter(x_value, y_value, s=65, color=colors[name], label=name, zorder=3)
            axis.annotate(name, (x_value, y_value), xytext=(4, 5), textcoords="offset points", fontsize=8)
        axis.axhline(1.0, color="#777777", linewidth=1, linestyle="--")
        axis.set_xscale("log")
        axis.grid(True, which="both", alpha=0.22)
        axis.set_title(dataset.title())
        axis.set_xlabel("Construction time (seconds, log scale)")
        axis.set_ylabel("Logical accesses / PLATON-25")
    fig.suptitle("WAHARP and PLATON quality versus construction cost at B=128")
    fig.savefig(OUTPUT / "QUALITY_VS_CONSTRUCTION_COST.png", dpi=220)
    plt.close(fig)


def ensure_dataset_context(dataset: str) -> tuple[Path, dict[str, Path], int, Path]:
    paper_root = locate_paper_b128(dataset)
    rects, metadata = load_dataset(dataset)
    paper_metadata = json.loads((paper_root / "DATASET.json").read_text(encoding="utf-8"))
    if metadata["normalized_object_sha256"] != paper_metadata["normalized_object_sha256"]:
        raise RuntimeError(f"Normalized dataset hash mismatch for {dataset}")
    atomic_json(OUTPUT / "datasets" / dataset / "DATASET.json", metadata)
    workroot = create_fresh_workload(dataset, rects, paper_root)
    scratch = prepare_scratch(dataset, rects, workroot)
    object_count = len(rects)
    return workroot, scratch, object_count, paper_root


def main() -> None:
    restore_previous_output()
    checkpoint, checkpoint_state = locate_checkpoint()
    save_state(
        "RUNNING",
        checkpoint_sha256=checkpoint_state["selected_checkpoint_sha256"],
        selected_member=checkpoint_state["selected_member"],
        started_utc=now_utc(),
    )
    native = compile_native(INPUT, OUTPUT, CODE_ROOT)
    _, author = discover_platon_support(INPUT)

    contexts: dict[str, tuple[Path, dict[str, Path], int, Path]] = {}
    for dataset in DATASETS:
        print(f"Preparing {dataset}", flush=True)
        paper_root = locate_paper_b128(dataset)
        rects, metadata = load_dataset(dataset)
        paper_metadata = json.loads((paper_root / "DATASET.json").read_text(encoding="utf-8"))
        if metadata["normalized_object_sha256"] != paper_metadata["normalized_object_sha256"]:
            raise RuntimeError(f"Normalized dataset hash mismatch for {dataset}")
        atomic_json(OUTPUT / "datasets" / dataset / "DATASET.json", metadata)
        workroot = create_fresh_workload(dataset, rects, paper_root)
        scratch = prepare_scratch(dataset, rects, workroot)
        contexts[dataset] = (workroot, scratch, len(rects), paper_root)
        run_waharp(dataset, rects, workroot, checkpoint, checkpoint_state)
        del rects
        gc.collect()
        regenerate_outputs()

    # Materialize the verified paper-budget control first; its MCTS is reused.
    for dataset in DATASETS:
        workroot, scratch, object_count, paper_root = contexts[dataset]
        if not run_platon(dataset, 25, scratch, workroot, object_count, native, author, paper_root):
            regenerate_outputs()
            return
        regenerate_outputs()

    # Reduced budgets are ordered from cheapest to most expensive across all datasets.
    for rollout in (1, 5, 10):
        for dataset in DATASETS:
            workroot, scratch, object_count, paper_root = contexts[dataset]
            if not run_platon(dataset, rollout, scratch, workroot, object_count, native, author, paper_root):
                regenerate_outputs()
                return
            regenerate_outputs()

    results, mismatches = regenerate_outputs()
    expected = len(DATASETS) * len(ROLLOUTS)
    if len(results) != expected:
        raise RuntimeError(f"Incomplete final result table: {len(results)} != {expected}")
    if mismatches:
        save_state("FAILED_CORRECTNESS", correctness_mismatches=mismatches)
        raise RuntimeError(f"Correctness mismatches detected: {mismatches}")
    save_state(
        "COMPLETE",
        correctness_mismatches=0,
        result_rows=len(results),
        finished_utc=now_utc(),
    )
    print(f"COMPLETE: {OUTPUT}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        save_state("FAILED", error=f"{type(exc).__name__}: {exc}")
        regenerate_outputs()
        raise
