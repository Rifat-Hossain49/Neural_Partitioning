"""Locked-model simple-policy audit and fresh same-instance PLATON timing."""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import time
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

# Set thread limits before importing the numerical libraries.
for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from realtrain.data import (load_arizona, load_point_csv, locate_arizona_npy,
                            locate_crimes_csv, locate_twitter_csv)
from realtrain.eval import evaluate_knn_suite, evaluate_range_suite, tree_metrics
from realtrain.native import (compile_native, eval_native_tree, native_env,
                              write_platon_array, write_records)
from realtrain.train import load_model
from realtrain.tree import (SimplePolicy, build_neural_tree, build_policy_tree,
                            load_tree, save_tree)
from realtrain.utils import atomic_json, derive_seed, read_csv, sha256_array, sha256_file, write_csv
from realtrain.workload import KNN_K, load_query_suite

PROTOCOL = "waharp_reviewer_checks_v1_20261003"
CHECKPOINT = "c158a7e365dc5fbad01de6126e8665a99fc942f13e362d359cb49be301f6b4f7"
QUERY_NAMESPACE = "waharp_final_fullscale_b128_member2_93d3c2545e44b9e21269a1b4"
DATASETS = ("arizona", "crimes", "twitter")
CAPACITIES = (128, 256, 512)
METHODS = ("WAHARP", "Fixed-X-Quarter", "Widest-Axis-Quarter")
MIX = Counter(range=4200, point=300, knn=1500)


def timing_tasks(manifest=None):
    if manifest is None:
        return [dict(dataset=dataset, capacity=capacity, repetition=rep)
                for rep in range(3) for capacity in CAPACITIES for dataset in DATASETS]
    if manifest.get("protocol") != PROTOCOL or not manifest.get("shard_id"):
        raise ValueError("Invalid timing shard identity")
    tasks = manifest.get("tasks", [])
    if not tasks:
        raise ValueError("A timing shard must contain at least one task")
    seen = set()
    for task in tasks:
        key = (task["dataset"], task["capacity"], task["repetition"])
        if (key[0] not in DATASETS or type(key[1]) is not int or type(key[2]) is not int
                or key[1] not in CAPACITIES or key[2] not in range(3)
                or key in seen):
            raise ValueError("Duplicate or unsupported timing shard task")
        estimate = task.get("admission_seconds", 0)
        if not isinstance(estimate, (int, float)) or not math.isfinite(estimate) or estimate < 0:
            raise ValueError("Invalid timing admission estimate")
        seen.add(key)
    return tasks


class ResumeRequired(RuntimeError):
    pass


def atomic_csv(path, rows):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    write_csv(temporary, rows)
    temporary.replace(path)


def uid(row):
    value = str(row["query_uid"])
    if value.startswith("knn|k") and not value.startswith("knn|knn_k"):
        return "knn|knn_k" + value[len("knn|k"):]
    return value


def compare(left_rows, right_rows, dataset, capacity, right_method):
    left = {uid(r): r for r in left_rows}
    right = {uid(r): r for r in right_rows}
    if set(left) != set(right) or len(left) != 6000:
        raise RuntimeError("Incomplete or mismatched paired query IDs")
    if Counter(r["query_type"] for r in left.values()) != MIX:
        raise RuntimeError("Unexpected locked query mix")
    summaries = []
    checks = []
    for family in ("all", "range", "point", "knn"):
        keys = sorted(k for k, r in left.items() if family == "all" or r["query_type"] == family)
        a = np.array([int(left[k]["total_node_accesses"]) for k in keys])
        b = np.array([int(right[k]["total_node_accesses"]) for k in keys])
        mismatches = 0
        for key in keys:
            x, y = left[key], right[key]
            if x["query_type"] == "knn":
                agrees = math.isclose(float(x["kth_distance"]), float(y["kth_distance"]),
                                     rel_tol=2e-6, abs_tol=2e-6)
            else:
                agrees = all(int(x[t]) == int(y[t]) for t in ("result_count", "id_sum", "id_xor"))
            mismatches += not agrees
        rng = np.random.default_rng(derive_seed(PROTOCOL, 20261003,
                                                f"{dataset}/{capacity}/{right_method}/{family}"))
        ratios = np.empty(2000)
        for draw in range(len(ratios)):
            indices = rng.integers(0, len(keys), size=len(keys))
            ratios[draw] = a[indices].sum() / b[indices].sum()
        common = dict(dataset=dataset, capacity=capacity, right_method=right_method,
                      query_family=family, queries=len(keys))
        summaries.append({**common, "waharp_accesses": int(a.sum()),
                          "baseline_accesses": int(b.sum()), "waharp_over_baseline": float(a.sum()/b.sum()),
                          "ci_low": float(np.quantile(ratios, .025)),
                          "ci_high": float(np.quantile(ratios, .975)),
                          "waharp_strict_wins": int((a < b).sum()),
                          "ties": int((a == b).sum()), "waharp_losses": int((a > b).sum())})
        if family != "all":
            checks.append({**common, "mismatches": int(mismatches), "passed": mismatches == 0})
        if mismatches:
            raise RuntimeError(f"Correctness failure: {dataset}/B{capacity}/{right_method}/{family}")
    return summaries, checks


def depth_audit(tree):
    pending = [(tree.root, 0)]
    depths = set()
    while pending:
        node, depth = pending.pop()
        if node.is_leaf:
            depths.add(depth)
        else:
            pending.extend((e.child, depth + 1) for e in node.entries)
    if len(depths) != 1:
        raise RuntimeError("Leaves have unequal depths")
    return dict(equal_leaf_depth=True, leaf_depth=next(iter(depths)))


def query_batches(suite, size=25):
    for name, queries in suite["ranges"].items():
        for offset in range(0, len(queries), size):
            yield "range", name, offset, queries[offset:offset+size], None
    points = suite["points"]
    boxes = np.column_stack([points, points]).astype(np.float32)
    for offset in range(0, len(boxes), size):
        yield "point", "point", offset, boxes[offset:offset+size], None
    for k in KNN_K:
        for offset in range(0, len(suite["knn_points"]), size):
            yield "knn", f"knn_k{k}", offset, suite["knn_points"][offset:offset+size], k


class Experiment:
    def __init__(self, args):
        self.args = args
        self.output = args.output
        self.output.mkdir(parents=True, exist_ok=True)
        self.started = time.monotonic()
        self.session = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
        self.model = None
        self.native = None
        task_path = getattr(args, "timing_tasks", None)
        self.shard = json.loads(task_path.read_text()) if task_path else None
        self.tasks = timing_tasks(self.shard)
        self.restore()
        if self.shard:
            if len(list(self.output.glob("cells/*/B*/methods/*/COMPLETE.json"))) != 27:
                raise RuntimeError("Timing shards require the complete frozen quality checkpoint")
            atomic_json(self.output/"TIMING_SHARD.json", self.shard)
        expected = dict(protocol=PROTOCOL, checkpoint=CHECKPOINT, capacities=list(CAPACITIES),
                        methods=list(METHODS), datasets=list(DATASETS), query_namespace=QUERY_NAMESPACE,
                        repetitions=3, platon_rollouts=25, platon_simulation_steps=100,
                        inference_device="cpu", threads=1, hist_bins=12, local_query_cap=128)
        frozen = self.output / "PROTOCOL.json"
        if frozen.exists() and json.loads(frozen.read_text()) != expected:
            raise RuntimeError("Resume protocol mismatch")
        atomic_json(frozen, expected)
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        hardware = dict(session=self.session, utc=datetime.now(timezone.utc).isoformat(),
                        platform=platform.platform(), python=sys.version, numpy=np.__version__,
                        pytorch=torch.__version__, allocated_cpus=os.cpu_count(),
                        thread_limits={k: os.environ[k] for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")},
                        gpu_available=torch.cuda.is_available(),
                        gpus=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
                        machine_cpu=Path("/proc/cpuinfo").read_text() if Path("/proc/cpuinfo").exists() else platform.processor(),
                        memory=Path("/proc/meminfo").read_text() if Path("/proc/meminfo").exists() else None)
        for command in (["g++", "--version"], ["nvidia-smi"]):
            try:
                cp = subprocess.run(command, capture_output=True, text=True, timeout=15)
                hardware[command[0]] = cp.stdout
            except (OSError, subprocess.TimeoutExpired):
                hardware[command[0]] = "unavailable"
        atomic_json(self.output / "sessions" / self.session / "HARDWARE.json", hardware)

    def restore(self):
        expected = getattr(self.args, "expected_resume_state_sha256", None)
        if (self.output / "experiment_state.json").exists():
            if expected and sha256_file(self.output/"experiment_state.json") != expected:
                raise RuntimeError("Local resume state is not the requested checkpoint")
            return
        candidates = []
        for path in self.args.inputs.rglob("experiment_state.json"):
            try:
                state = json.loads(path.read_text())
                if state.get("protocol_id") == PROTOCOL and (not expected or sha256_file(path) == expected):
                    candidates.append((state.get("completed_units", 0), path.parent))
            except (OSError, ValueError):
                continue
        if candidates:
            source = max(candidates, key=lambda r: (r[0], str(r[1])))[1]
            shutil.copytree(source, self.output, dirs_exist_ok=True)
            if expected:
                state = json.loads((self.output/"experiment_state.json").read_text())
                qualities = list(self.output.glob("cells/*/B*/methods/*/COMPLETE.json"))
                pairs = list(self.output.glob("cells/*/B*/timing/rep_*/COMPLETE.json"))
                if (len(qualities) != state["completed_quality_cells"]
                        or len(pairs) != state["completed_timing_pairs"]):
                    raise RuntimeError("Mounted resume files disagree with the requested checkpoint")
                print("Verified exact resume counts:", len(qualities), len(pairs), flush=True)
            print("Restored", source, flush=True)
        elif expected:
            raise RuntimeError("Requested checkpoint is not mounted; refusing to repeat completed work")

    def remaining(self):
        return self.args.hours * 3600 - (time.monotonic() - self.started)

    def guard(self, reserve=900):
        if self.remaining() < reserve:
            raise ResumeRequired("Session safety budget reached")

    def state(self, status, **extra):
        qualities = list(self.output.glob("cells/*/B*/methods/*/COMPLETE.json"))
        pairs = list(self.output.glob("cells/*/B*/timing/rep_*/COMPLETE.json"))
        shard = getattr(self, "shard", None)
        if shard:
            done = sum(self.task_complete(task) for task in self.tasks)
            extra.update(shard_id=shard["shard_id"], shard_target_tasks=len(self.tasks),
                         shard_completed_tasks=done)
        atomic_json(self.output / "experiment_state.json",
                    dict(protocol_id=PROTOCOL, status=status, session=self.session,
                         completed_units=len(qualities)+len(pairs), completed_quality_cells=len(qualities),
                         completed_timing_pairs=len(pairs), target_quality_cells=27, target_timing_pairs=27,
                         elapsed_session_seconds=time.monotonic()-self.started,
                         remaining_session_seconds=self.remaining(), retrained=False, **extra))

    def task_complete(self, task):
        cell = self.output/"cells"/task["dataset"]/f"B{task['capacity']}"
        rep = task["repetition"]
        return ((cell/"timing"/f"rep_{rep}"/"COMPLETE.json").exists()
                and (rep != 0 or (cell/"PLATON"/"PAIRWISE.csv").exists()))

    def context(self, dataset):
        candidates = []
        for path in self.args.inputs.rglob("FULLSCALE_STATE.json"):
            try:
                state = json.loads(path.read_text())
                namespace = state.get("fresh_final_namespace", state.get("paired_query_namespace"))
                if (state.get("dataset") == dataset and int(state.get("capacity", 0)) == 128
                        and namespace == QUERY_NAMESPACE and str(state.get("status", "")).startswith("COMPLETE")
                        and state.get("protocol_id") == "waharp_final_fullscale_b128_member2_20260831"):
                    candidates.append(path.parent)
            except (ValueError, OSError):
                continue
        if not candidates:
            raise RuntimeError(f"Missing locked B128 workload for {dataset}")
        roots = sorted(candidates, key=str)
        for other in roots[1:]:
            if sha256_file(other/"workload"/"construction_boxes.npy") != sha256_file(roots[0]/"workload"/"construction_boxes.npy"):
                raise RuntimeError("Conflicting locked workloads")
        prior = roots[0]
        base = self.output / "datasets" / dataset
        base.mkdir(parents=True, exist_ok=True)
        metadata = json.loads((prior/"DATASET.json").read_text())
        array_path = base/"rectangles.npy"
        if not array_path.exists():
            if dataset == "arizona":
                rects, loaded = load_arizona(locate_arizona_npy(self.args.inputs), 0)
            else:
                locator = locate_crimes_csv if dataset == "crimes" else locate_twitter_csv
                rects, loaded = load_point_csv(locator(self.args.inputs), 0)
            if loaded["normalized_object_sha256"] != metadata["normalized_object_sha256"]:
                raise RuntimeError(f"Paper dataset identity mismatch: {dataset}")
            temporary = array_path.with_suffix(".tmp.npy")
            np.save(temporary, rects)
            temporary.replace(array_path)
        rects = np.load(array_path, mmap_mode="r")
        if sha256_array(rects) != metadata["normalized_object_sha256"]:
            raise RuntimeError("Cached input identity mismatch")
        atomic_json(base/"DATASET.json", metadata)
        workload = base/"workload"
        workload.mkdir(exist_ok=True)
        fingerprints = {}
        for path in sorted((prior/"workload").glob("*")):
            if path.name.startswith("final_") or path.name in ("construction_boxes.npy", "WORKLOAD_MANIFEST.json"):
                fingerprints[path.name] = sha256_file(path)
                target = workload/path.name
                if not target.exists():
                    shutil.copy2(path, target)
                if sha256_file(target) != fingerprints[path.name]:
                    raise RuntimeError("Cached workload identity mismatch")
        atomic_json(base/"WORKLOAD_IDENTITIES.json", fingerprints)
        return rects, np.load(workload/"construction_boxes.npy"), load_query_suite(workload, "final"), base

    def load_model(self):
        path = ROOT/"artifacts"/"model"/"selected.pt"
        if sha256_file(path) != CHECKPOINT:
            raise RuntimeError("Wrong frozen checkpoint")
        training = json.loads((path.parent/"FINAL_TRAINING_STATE.json").read_text())
        if training.get("selected_checkpoint_sha256") != CHECKPOINT or training.get("final_queries_read") is not False:
            raise RuntimeError("Frozen model provenance gate failed")
        self.model = load_model(path, "cpu")
        with torch.no_grad():
            self.model(torch.zeros((1, 498)))
        atomic_json(self.output/"MODEL_PROVENANCE.json", training)

    def build(self, method, rects, construction, capacity):
        started = time.perf_counter()
        if method == "WAHARP":
            tree, diag = build_neural_tree(rects, construction, self.model, "cpu", capacity, 12, 128)
        else:
            tree, diag = build_policy_tree(rects, SimplePolicy(method, capacity), 128)
        seconds = time.perf_counter()-started
        return tree, {**diag, "method": method, "construction_seconds": seconds,
                      "validation_included": True, "device": "cpu", "session": self.session,
                      "validation_seconds_approx": max(0, seconds-diag["build_seconds"])}

    def evaluate(self, tree, suite, nobj, capacity, method, root):
        complete = root/"PER_QUERY.csv"
        if complete.exists():
            return read_csv(complete)
        all_rows = []
        for batch, (family, name, offset, queries, k) in enumerate(query_batches(suite)):
            path = root/"chunks"/f"{batch:04d}.csv"
            if path.exists():
                rows = read_csv(path)
            else:
                self.guard()
                if family == "knn":
                    rows = evaluate_knn_suite(tree, queries, k, name)
                else:
                    rows = evaluate_range_suite(tree, queries, nobj, capacity, family, name)
                for i, row in enumerate(rows):
                    row.update(query_index=offset+i, query_uid=f"{family}|{name}|{offset+i:05d}",
                               query_type=family, workload=name, method=method)
                atomic_csv(path, rows)
                if batch % 12 == 0:
                    print(method, capacity, "query batch", batch, flush=True)
                    self.state("RUNNING", active_method=method, capacity=capacity, query_batch=batch)
            all_rows.extend(rows)
        if len(all_rows) != 6000 or len({uid(r) for r in all_rows}) != 6000:
            raise RuntimeError("Incomplete final evaluation")
        atomic_csv(complete, all_rows)
        return all_rows

    def quality(self, dataset, capacity, method, context):
        rects, construction, suite, _ = context
        root = self.output/"cells"/dataset/f"B{capacity}"/"methods"/method
        if (root/"COMPLETE.json").exists():
            return
        self.guard(1800)
        root.mkdir(parents=True, exist_ok=True)
        cache = root/"tree.pkl.gz"
        if cache.exists():
            tree, diag = load_tree(cache)
        else:
            tree, diag = self.build(method, rects, construction, capacity)
            temporary = cache.with_suffix(".tmp")
            save_tree(temporary, tree, diag)
            temporary.replace(cache)
            atomic_json(root/"CONSTRUCTION.json", diag)
        structure = {**tree_metrics(tree, capacity), **depth_audit(tree)}
        atomic_json(root/"TREE_STRUCTURE.json", structure)
        self.evaluate(tree, suite, len(rects), capacity, method, root)
        atomic_json(root/"COMPLETE.json", dict(session=self.session, queries=6000, method=method))
        cache.unlink(missing_ok=True)
        del tree
        gc.collect()
        self.state("RUNNING", last_completed=f"{dataset}/B{capacity}/{method}")

    def command(self, command, log):
        log.parent.mkdir(parents=True, exist_ok=True)
        print("$", " ".join(map(str, command)), flush=True)
        with log.open("w", encoding="utf-8") as handle:
            cp = subprocess.run(list(map(str, command)), stdout=handle, stderr=subprocess.STDOUT,
                                env=native_env(self.native), timeout=max(1, self.remaining()-120))
        if cp.returncode:
            raise RuntimeError(f"Command failed; see {log}")
        return log.read_text()

    def timing(self, dataset, capacity, rep, context):
        rects, construction, suite, base = context
        root = self.output/"cells"/dataset/f"B{capacity}"/"timing"/f"rep_{rep}"
        marker = root/"COMPLETE.json"
        if marker.exists():
            if rep == 0:
                self.platon_quality(dataset, capacity, context, json.loads(marker.read_text()))
            return
        self.guard(2100)
        # A new session creates a new attempt: interrupted timings are never pooled.
        attempt = root/"attempts"/self.session
        attempt.mkdir(parents=True, exist_ok=True)
        if self.native is None:
            self.native = compile_native(self.args.inputs, self.output, ROOT)
            atomic_json(self.output/"sessions"/self.session/"NATIVE.json", self.native)
        records, data, queries = base/"records.txt", base/"platon.npy", base/"platon_queries.npy"
        if not records.exists():
            write_records(records, rects)
        if not data.exists():
            write_platon_array(data, rects)
        if not queries.exists():
            write_platon_array(queries, construction)
        seed = derive_seed(QUERY_NAMESPACE, 2026083159, dataset) & 0xffffffff
        pair = dict(dataset=dataset, capacity=capacity, repetition=rep, session=self.session,
                    checkpoint=CHECKPOINT, dataset_hash=sha256_array(rects),
                    construction_hash=sha256_array(construction), seed=seed,
                    fresh_pair=True, validation_included_both_methods=True)
        order = ("WAHARP", "PLATON") if rep % 2 == 0 else ("PLATON", "WAHARP")
        pair["method_order"] = list(order)
        for method in order:
            if method == "WAHARP":
                tree, diag = self.build(method, rects, construction, capacity)
                pair["waharp_seconds"] = diag["construction_seconds"]
                atomic_json(attempt/"WAHARP.json", diag)
                del tree
                gc.collect()
            else:
                cut = attempt/"author_cuts.txt"
                driver = ROOT/"scripts"/"kaggle"/"platon_resumable_cutlist.py"
                self.command([sys.executable, driver, "--author-root", self.native["author"],
                              "--data", data, "--queries", queries, "--output-cutlist", cut,
                              "--checkpoint-db", attempt/"actions.sqlite", "--progress-json", attempt/"PROGRESS.json",
                              "--branch", capacity, "--seed", seed, "--rollouts", 25,
                              "--simulation-steps", 100, "--budget-seconds", max(1, self.remaining()-1200),
                              "--decision-reserve-seconds", 300, "--progress-every-actions", 25], attempt/"mcts.log")
                meta = json.loads((attempt/"PROGRESS.json").read_text())
                if not meta.get("complete"):
                    raise ResumeRequired("MCTS reached safety stop; actions saved, timing pair incomplete")
                policy_seconds = meta.get("fresh_continuous_construction_seconds")
                if policy_seconds is None or meta.get("replayed_action_count") != 0:
                    raise RuntimeError("Replayed MCTS cannot supply a fresh matched timing")
                tree_base = attempt/"platon_tree"
                started = time.perf_counter()
                text = self.command([self.native["bulk"], records, tree_base,
                                     int(round(capacity/.8)), .8, cut, 4096], attempt/"native_build.log")
                native_seconds = time.perf_counter()-started
                line = next(line for line in text.splitlines() if line.startswith("PLATON_BUILD_RESULT"))
                values = dict(item.split("=", 1) for item in line.split(",")[1:])
                if values.get("valid_tree") != "1":
                    raise RuntimeError("Invalid PLATON tree")
                pair.update(platon_policy_seconds=policy_seconds,
                            platon_mcts_decision_seconds=meta["decision_time_seconds_sum"],
                            platon_native_seconds=native_seconds,
                            platon_seconds=policy_seconds+native_seconds,
                            platon_tree=str(tree_base), index_identifier=int(values["index_identifier"]))
                atomic_json(attempt/"PLATON.json", pair)
        pair["platon_over_waharp_seconds"] = pair["platon_seconds"]/pair["waharp_seconds"]
        atomic_json(marker, pair)
        self.state("RUNNING", last_completed=f"{dataset}/B{capacity}/timing/{rep}")
        if rep == 0:
            self.platon_quality(dataset, capacity, context, pair)
        for suffix in (".idx", ".dat"):
            Path(pair["platon_tree"]+suffix).unlink(missing_ok=True)

    def platon_quality(self, dataset, capacity, context, pair):
        proot = self.output/"cells"/dataset/f"B{capacity}"/"PLATON"
        if (proot/"PAIRWISE.csv").exists():
            return
        self.guard(1200)
        rects, _, suite, base = context
        if self.native is None:
            self.native = compile_native(self.args.inputs, self.output, ROOT)
        tree_base = Path(pair["platon_tree"])
        index = pair["index_identifier"]
        if not all(Path(str(tree_base)+suffix).exists() for suffix in (".idx", ".dat")):
            # Recreating a tree for evaluation never changes an already saved timing.
            text = self.command([self.native["bulk"], base/"records.txt", tree_base,
                                 int(round(capacity/.8)), .8, tree_base.parent/"author_cuts.txt", 4096],
                                proot/"quality_rebuild.log")
            line = next(line for line in text.splitlines() if line.startswith("PLATON_BUILD_RESULT"))
            values = dict(item.split("=", 1) for item in line.split(",")[1:])
            if values.get("valid_tree") != "1":
                raise RuntimeError("Invalid PLATON quality rebuild")
            index = int(values["index_identifier"])
        rows = eval_native_tree(tree_base, index, suite, len(rects), capacity,
                                proot/"eval", self.native, "PLATON")
        atomic_csv(proot/"PER_QUERY.csv", rows)
        waharp = read_csv(self.output/"cells"/dataset/f"B{capacity}"/"methods"/"WAHARP"/"PER_QUERY.csv")
        summary, checks = compare(waharp, rows, dataset, capacity, "PLATON")
        atomic_csv(proot/"PAIRWISE.csv", summary)
        atomic_csv(proot/"CORRECTNESS.csv", checks)
        for suffix in (".idx", ".dat"):
            Path(str(tree_base)+suffix).unlink(missing_ok=True)

    def outputs(self):
        results, correctness, times, structures, build_times = [], [], [], [], []
        for dataset in DATASETS:
            for capacity in CAPACITIES:
                cell = self.output/"cells"/dataset/f"B{capacity}"
                reference = cell/"methods"/"WAHARP"/"PER_QUERY.csv"
                if reference.exists():
                    for method in METHODS[1:]:
                        root = cell/"methods"/method
                        if (root/"COMPLETE.json").exists():
                            summary, checks = compare(read_csv(reference), read_csv(root/"PER_QUERY.csv"), dataset, capacity, method)
                            atomic_csv(root/"PAIRWISE.csv", summary)
                            atomic_csv(root/"CORRECTNESS.csv", checks)
                            results.extend(summary)
                            correctness.extend(checks)
                platon = cell/"PLATON"
                if (platon/"PAIRWISE.csv").exists():
                    results.extend(read_csv(platon/"PAIRWISE.csv"))
                    correctness.extend(read_csv(platon/"CORRECTNESS.csv"))
                for root in (cell/"methods").glob("*"):
                    if (root/"CONSTRUCTION.json").exists():
                        build_times.append({**json.loads((root/"CONSTRUCTION.json").read_text()),
                                            "dataset": dataset, "capacity": capacity})
                    if (root/"TREE_STRUCTURE.json").exists():
                        structures.append({**json.loads((root/"TREE_STRUCTURE.json").read_text()),
                                           "dataset": dataset, "capacity": capacity, "method": root.name})
                for marker in (cell/"timing").glob("rep_*/COMPLETE.json"):
                    times.append(json.loads(marker.read_text()))
        atomic_csv(self.output/"RESULTS.csv", [r for r in results if r["query_family"] == "all"])
        atomic_csv(self.output/"QUERY_FAMILY_RESULTS.csv", [r for r in results if r["query_family"] != "all"])
        atomic_csv(self.output/"CORRECTNESS.csv", correctness)
        atomic_csv(self.output/"CONSTRUCTION_TIMES.csv", times)
        atomic_csv(self.output/"SIMPLE_POLICY_BUILD_TIMES.csv", build_times)
        atomic_json(self.output/"TREE_STRUCTURES.json", {"records": structures})
        aggregates = []
        for dataset in DATASETS:
            for capacity in CAPACITIES:
                rows = [r for r in times if r["dataset"] == dataset and int(r["capacity"]) == capacity]
                if rows:
                    a = np.array([r["waharp_seconds"] for r in rows])
                    b = np.array([r["platon_seconds"] for r in rows])
                    aggregates.append(dict(dataset=dataset, capacity=capacity, paired_repetitions=len(rows),
                                           waharp_median=float(np.median(a)), platon_median=float(np.median(b)),
                                           waharp_iqr=float(np.quantile(a,.75)-np.quantile(a,.25)),
                                           platon_iqr=float(np.quantile(b,.75)-np.quantile(b,.25)),
                                           median_paired_speedup=float(np.median(b/a)),
                                           sessions=len({r["session"] for r in rows})))
        atomic_csv(self.output/"TIMING_SUMMARY.csv", aggregates)
        lines = ["# WAHARP Reviewer Checks", "",
                 "Frozen model; no retraining or parameter selection. Locked paper query arrays reused.",
                 "Primary timing uses one CPU thread for both methods on the same instance within each pair.",
                 "Both construction measurements include their existing tree-validity checks; neither includes final queries.",
                 "PLATON policy wall time includes hierarchy execution and action checkpoint writes; pure MCTS-decision time is separate.",
                 "Offline training, input loading/conversion, model loading, warm-up, compilation, and idle time are excluded.",
                 "GPU availability is audited, but CPU is used to make the primary comparison explicit.", "",
                 "| Dataset | B | Baseline | WAHARP/baseline accesses | Win / tie / loss |",
                 "|---|---:|---|---:|---:|"]
        for r in results:
            if r["query_family"] == "all":
                lines.append(f"| {r['dataset']} | {r['capacity']} | {r['right_method']} | "
                             f"{float(r['waharp_over_baseline']):.6f} | "
                             f"{r['waharp_strict_wins']} / {r['ties']} / {r['waharp_losses']} |")
        lines += ["", "| Dataset | B | Timing pairs | WAHARP median s | PLATON median s | Median paired ratio |",
                  "|---|---:|---:|---:|---:|---:|"]
        for r in aggregates:
            lines.append(f"| {r['dataset']} | {r['capacity']} | {r['paired_repetitions']} | "
                         f"{r['waharp_median']:.3f} | {r['platon_median']:.3f} | {r['median_paired_speedup']:.3f} |")
        lines += ["", f"Completed timing pairs: {len(times)}/27.",
                  "Incomplete settings/repetitions must not be presented as a completed experiment.",
                  "Correctness: range/point count, 64-bit ID sum and XOR; kNN kth-distance tolerance 2e-6.",
                  "Interpret simple-policy wins/ties honestly; results do not guarantee publication acceptance."]
        temporary = self.output/"FINAL_REPORT.md.tmp"
        temporary.write_text("\n".join(lines)+"\n", encoding="utf-8")
        temporary.replace(self.output/"FINAL_REPORT.md")

    def run(self):
        self.load_model()
        self.state("RUNNING")
        # Complete the inexpensive controls before spending the session on MCTS.
        for dataset in DATASETS:
            if all((self.output/"cells"/dataset/f"B{capacity}"/"methods"/method/"COMPLETE.json").exists()
                   for capacity in CAPACITIES for method in METHODS):
                print("Skipping completed quality dataset", dataset, flush=True)
                continue
            context = self.context(dataset)
            for capacity in CAPACITIES:
                for method in METHODS:
                    self.quality(dataset, capacity, method, context)
                self.outputs()
            del context
            gc.collect()
        # Sharding changes scheduling only, never either method's timed execution.
        for task in getattr(self, "tasks", timing_tasks()):
            dataset, capacity, rep = task["dataset"], task["capacity"], task["repetition"]
            if self.task_complete(task):
                print("Skipping completed timing pair", dataset, capacity, rep, flush=True)
                continue
            marker = self.output/"cells"/dataset/f"B{capacity}"/"timing"/f"rep_{rep}"/"COMPLETE.json"
            if not marker.exists():
                self.guard(max(2100, task.get("admission_seconds", 0)+900))
            context = self.context(dataset)
            self.timing(dataset, capacity, rep, context)
            self.outputs()
            del context
            gc.collect()
        self.outputs()
        self.state("SHARD_COMPLETE" if getattr(self, "shard", None) else "COMPLETE")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, default=Path("/kaggle/input"))
    parser.add_argument("--output", type=Path, default=Path("/kaggle/working/WAHARP_REVIEWER_CHECKS"))
    parser.add_argument("--hours", type=float, default=10.5)
    parser.add_argument("--expected-resume-state-sha256")
    parser.add_argument("--timing-tasks", type=Path, help="Explicit non-overlapping timing shard manifest")
    args = parser.parse_args()
    experiment = Experiment(args)
    try:
        experiment.run()
    except (ResumeRequired, subprocess.TimeoutExpired) as exc:
        experiment.outputs()
        experiment.state("PAUSED_RESUME_REQUIRED", reason=str(exc))
        print("Checkpoint saved; resume required:", exc, flush=True)
    except Exception as exc:
        experiment.state("FAILED", error=f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    main()
