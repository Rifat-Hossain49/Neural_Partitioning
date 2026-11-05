#!/usr/bin/env python3
"""Crash/preemption-resumable driver around the user's author PLATON checkout.

This module does not reimplement or alter PLATON's MCTS reward/search. It imports
RtreeEnv/MCTS/Node from the attached author checkout and checkpoints every
completed partition decision in SQLite. On resume, committed actions are
replayed deterministically, the saved Python/NumPy RNG state is restored, and
MCTS continues at the first unseen decision.
"""
from __future__ import annotations

import argparse
import importlib
import hashlib
import json
import math
import os
import pickle
import random
import signal
import sqlite3
import sys
import time
import types
from collections import deque
from pathlib import Path

import numpy as np


def optional_import_stubs() -> None:
    def ensure(name: str):
        if name in sys.modules:
            return sys.modules[name]
        try:
            return importlib.import_module(name)
        except ImportError:
            module = types.ModuleType(name)
            sys.modules[name] = module
            return module

    ensure("pandas")
    ensure("geopandas")
    ensure("geoplot")
    matplotlib = ensure("matplotlib")
    pyplot = ensure("matplotlib.pyplot")
    setattr(matplotlib, "pyplot", pyplot)
    pil = ensure("PIL")
    image = ensure("PIL.Image")
    setattr(pil, "Image", image)
    shapely = ensure("shapely")
    geometry = ensure("shapely.geometry")
    setattr(shapely, "geometry", geometry)


def load_author(author_root: Path):
    packing = Path(author_root) / "learned-packing"
    if not (packing / "env.py").is_file() or not (packing / "mcts.py").is_file():
        raise FileNotFoundError(f"Not an author PLATON checkout: {author_root}")
    optional_import_stubs()
    sys.path.insert(0, str(packing))
    try:
        env = importlib.import_module("env")
        mcts = importlib.import_module("mcts")
        return env.RtreeEnv, env.getMBR, mcts.MCTS, mcts.Node
    finally:
        if sys.path and sys.path[0] == str(packing):
            sys.path.pop(0)


def tree_levels(n: int, branch: int) -> int:
    levels, covered = 1, int(branch)
    while covered < n:
        covered *= int(branch)
        levels += 1
    return levels


def file_sha256(path: Path, block_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


class CheckpointDB:
    def __init__(self, path: Path, signature: dict):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS actions (seq INTEGER PRIMARY KEY, axis INTEGER NOT NULL, relpos INTEGER NOT NULL, decision_seconds REAL NOT NULL)"
        )
        self.db.execute("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value BLOB NOT NULL)")
        encoded = json.dumps(signature, sort_keys=True).encode("utf-8")
        existing = self.db.execute("SELECT value FROM meta WHERE key='signature'").fetchone()
        if existing is None:
            with self.db:
                self.db.execute("INSERT INTO meta(key,value) VALUES('signature',?)", (encoded,))
        elif bytes(existing[0]) != encoded:
            raise RuntimeError("PLATON resume DB signature mismatch; refusing to mix experiments")

    def actions(self):
        return [
            (int(row[0]), int(row[1]), float(row[2]))
            for row in self.db.execute("SELECT axis,relpos,decision_seconds FROM actions ORDER BY seq")
        ]

    def rng_state(self):
        row = self.db.execute("SELECT value FROM meta WHERE key='rng_state'").fetchone()
        return pickle.loads(bytes(row[0])) if row is not None else None

    def commit_action(self, axis: int, relpos: int, decision_seconds: float) -> None:
        rng_blob = sqlite3.Binary(pickle.dumps((random.getstate(), np.random.get_state()), protocol=4))
        with self.db:
            seq = int(self.db.execute("SELECT COALESCE(MAX(seq),-1)+1 FROM actions").fetchone()[0])
            self.db.execute(
                "INSERT INTO actions(seq,axis,relpos,decision_seconds) VALUES(?,?,?,?)",
                (seq, int(axis), int(relpos), float(decision_seconds)),
            )
            self.db.execute(
                "INSERT OR REPLACE INTO meta(key,value) VALUES('rng_state',?)", (rng_blob,)
            )

    def close(self):
        self.db.close()


STOP = False


def handle_signal(signum, frame):  # pragma: no cover
    global STOP
    STOP = True
    print(f"PLATON_PREEMPTION_SIGNAL_RECEIVED signal={signum}", flush=True)


def generate(args) -> dict:
    global STOP
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            signal.signal(sig, handle_signal)
        except Exception:
            pass

    data = np.load(args.data, mmap_mode="r")
    queries = np.load(args.queries, mmap_mode="r")
    if data.ndim != 2 or data.shape[1] != 4 or queries.ndim != 2 or queries.shape[1] != 4:
        raise ValueError("PLATON data and queries must both use shape (n,4), [xmin,xmax,ymin,ymax]")
    if len(queries) == 0:
        raise ValueError("PLATON requires non-empty construction workload")

    data_array = np.asarray(data, dtype=np.float64)
    query_array = np.asarray(queries, dtype=np.float64)
    levels = tree_levels(len(data_array), args.branch)
    signature = {
        "format": "platon-resume-v1",
        "object_count": int(len(data_array)),
        "query_count": int(len(query_array)),
        "branch": int(args.branch),
        "levels": int(levels),
        "rollouts": int(args.rollouts),
        "simulation_steps": int(args.simulation_steps),
        "seed": int(args.seed),
        "data_path_name": Path(args.data).name,
        "query_path_name": Path(args.queries).name,
        "data_file_sha256": file_sha256(args.data),
        "query_file_sha256": file_sha256(args.queries),
    }
    db = CheckpointDB(args.checkpoint_db, signature)
    saved = db.actions()
    saved_rng = db.rng_state()

    random.seed(args.seed)
    np.random.seed(args.seed)
    RtreeEnv, get_mbr, MCTS, Node = load_author(args.author_root)
    env = RtreeEnv(data_array, get_mbr(data_array), args.branch, levels, query_array)
    queue = deque([env])
    action_cursor = 0
    rng_restored = False
    policy_nodes = 0
    run_started = time.monotonic()
    max_recent_decision = max([row[2] for row in saved[-20:]] or [1.0])

    def remaining() -> float:
        return max(0.0, float(args.budget_seconds) - (time.monotonic() - run_started))

    while queue:
        current = queue.popleft()
        while True:
            index, partition = current.getPartition()
            if index == -1:
                break

            if action_cursor < len(saved):
                axis_id, relpos, _seconds = saved[action_cursor]
                dim = 0 if axis_id == 0 else 2
                pos_start, pos_end = current.partitionList[index][3]
                absolute = int(pos_start + relpos - 1)
                if not (pos_start <= absolute <= pos_end):
                    raise RuntimeError(
                        f"Saved PLATON action {action_cursor} is not applicable: rel={relpos}, range={pos_start,pos_end}"
                    )
                current.cutPartitionGreedyTest(index, dim, absolute, [])
                action_cursor += 1
                continue

            if not rng_restored and saved_rng is not None:
                py_state, np_state = saved_rng
                random.setstate(py_state)
                np.random.set_state(np_state)
                rng_restored = True

            reserve = max(float(args.decision_reserve_seconds), 2.0 * max_recent_decision + 15.0)
            if STOP or remaining() <= reserve:
                metadata = {
                    **signature,
                    "complete": False,
                    "resume_required": True,
                    "actions_completed": int(action_cursor),
                    "policy_nodes_completed": int(policy_nodes),
                    "run_elapsed_seconds": float(time.monotonic() - run_started),
                    "remaining_budget_seconds": float(remaining()),
                    "reason": "signal" if STOP else "time_budget",
                }
                atomic_json(args.progress_json, metadata)
                db.close()
                print(json.dumps(metadata, indent=2, sort_keys=True), flush=True)
                return metadata

            decision_started = time.monotonic()
            node_queries = current.partitionList[index][4]
            if len(node_queries) == 0:
                dim, absolute, _ = current.getRandomAction(index)
            else:
                position_range = current.partitionList[index][3]
                normalization = (
                    current.childPageSize
                    * max(1, len(current.partitionList[index][5]) // current.childSize)
                    * len(node_queries)
                )
                state_env = RtreeEnv(
                    current.partitionList[index][5],
                    current.partitionList[index][1],
                    current.branch,
                    current.level,
                    current.sampleQueryList,
                    posRange=position_range,
                    sampleRate=1,
                )
                state = Node(
                    state_env,
                    0,
                    [],
                    max(1, len(node_queries)),
                    None,
                    args.simulation_steps,
                    max(1, normalization),
                )
                search = MCTS()
                for _ in range(args.rollouts):
                    search.do_rollout(state)
                selected = search.choose(state)
                dim, absolute = selected.history[-1]

            pos_start, pos_end = current.partitionList[index][3]
            relpos = int(absolute - pos_start + 1)
            current.cutPartitionGreedyTest(index, dim, absolute, [])
            decision_seconds = float(time.monotonic() - decision_started)
            max_recent_decision = max(decision_seconds, max_recent_decision * 0.98)
            db.commit_action(0 if dim == 0 else 1, relpos, decision_seconds)
            action_cursor += 1

            if action_cursor % max(1, args.progress_every_actions) == 0:
                metadata = {
                    **signature,
                    "complete": False,
                    "resume_required": True,
                    "actions_completed": int(action_cursor),
                    "policy_nodes_completed": int(policy_nodes),
                    "run_elapsed_seconds": float(time.monotonic() - run_started),
                    "remaining_budget_seconds": float(remaining()),
                    "last_decision_seconds": decision_seconds,
                }
                atomic_json(args.progress_json, metadata)
                print(
                    f"PLATON actions={action_cursor} policy_nodes={policy_nodes} queued={len(queue)} "
                    f"last={decision_seconds:.2f}s remaining={remaining():.1f}s",
                    flush=True,
                )

        if current.level >= 3:
            queue.extend(current.getChildEnv())
        policy_nodes += 1

    if action_cursor != len(db.actions()):
        raise AssertionError("PLATON action cursor/database length mismatch")
    actions = db.actions()
    args.output_cutlist.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output_cutlist.with_suffix(args.output_cutlist.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as handle:
        for axis, relpos, _seconds in actions:
            handle.write(f"{axis} {relpos}\n")
    os.replace(tmp, args.output_cutlist)
    total_decision_seconds = float(sum(row[2] for row in actions))
    metadata = {
        **signature,
        "complete": True,
        "resume_required": False,
        "actions_completed": len(actions),
        "cut_count": len(actions),
        "policy_nodes_completed": int(policy_nodes),
        "decision_time_seconds_sum": total_decision_seconds,
        "run_elapsed_seconds": float(time.monotonic() - run_started),
        "author_root": str(args.author_root),
        "checkpoint_db": str(args.checkpoint_db),
    }
    atomic_json(args.output_cutlist.with_suffix(".json"), metadata)
    atomic_json(args.progress_json, metadata)
    db.close()
    print(json.dumps(metadata, indent=2, sort_keys=True), flush=True)
    return metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--author-root", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--output-cutlist", type=Path, required=True)
    parser.add_argument("--checkpoint-db", type=Path, required=True)
    parser.add_argument("--progress-json", type=Path, required=True)
    parser.add_argument("--branch", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--rollouts", type=int, default=25)
    parser.add_argument("--simulation-steps", type=int, default=100)
    parser.add_argument("--budget-seconds", type=float, required=True)
    parser.add_argument("--decision-reserve-seconds", type=float, default=120.0)
    parser.add_argument("--progress-every-actions", type=int, default=10)
    args = parser.parse_args()
    if args.branch < 2 or args.rollouts < 1 or args.simulation_steps < 1 or args.budget_seconds <= 0:
        raise ValueError("Invalid PLATON parameters")
    generate(args)


if __name__ == "__main__":
    main()
