from __future__ import annotations

import csv
import gc
import hashlib
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


PROTOCOL_ID = "waharp_teacher_cost_ablation_v1_20260926"
SOURCE_PROTOCOL_ID = "waharp_realtrain_v1_three_domain_pageio_20260829"
CODE_PROTOCOL_ID = "waharp_teacher_cost_ablation_code_v1_20260926"
BASE_SEED = 2026082951
DOMAINS = ("twitter", "crimes", "arizona")
CURRENT_OVERLAP = 1e-4
CURRENT_MARGIN = 2e-5
OVERLAP_VALUES = (0.0, 1e-5, 1e-4, 1e-3)
MARGIN_VALUES = (0.0, 2e-6, 2e-5, 2e-4)
LIST_WEIGHT = 0.75
CLASSIFICATION_WEIGHT = 0.25
REGRESSION_WEIGHT = 1.0
TEMPERATURE = 0.025
EPOCHS = 28
BATCH_SIZE = 256
LEARNING_RATE = 6e-4
WEIGHT_DECAY = 1e-4
PATIENCE = 6
VAL_FRACTION = 0.20
EXPECTED_STATES_PER_DOMAIN = 10_000
EXPECTED_ACTIONS = 80
MAX_WORK_SECONDS = 10.5 * 3600
INPUT_ROOT = Path("/kaggle/input")
OUTPUT_ROOT = Path("/kaggle/working/WAHARP_TEACHER_COST_ABLATION")
STARTED = time.time()
os.environ["PYTHONUNBUFFERED"] = "1"


class GracefulPause(RuntimeError):
    pass


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
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=json_default),
        encoding="utf-8",
    )
    temporary.replace(path)


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def sha256_array(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.shape).encode("ascii"))
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(array.view(np.uint8))
    return digest.hexdigest()


def elapsed() -> float:
    return time.time() - STARTED


def require_time(seconds: float, label: str) -> None:
    if elapsed() + seconds > MAX_WORK_SECONDS:
        raise GracefulPause(
            f"Insufficient safe runtime for {label}: elapsed={elapsed():.1f}s"
        )


def update_state(status: str, **extra: Any) -> None:
    write_json(
        OUTPUT_ROOT / "EXPERIMENT_STATE.json",
        {
            "protocol_id": PROTOCOL_ID,
            "status": status,
            "elapsed_seconds": elapsed(),
            "updated_unix": time.time(),
            "gpu_names": [
                torch.cuda.get_device_name(index)
                for index in range(torch.cuda.device_count())
            ],
            **extra,
        },
    )


def restore_resume() -> None:
    candidates: list[tuple[int, Path]] = []
    for marker in INPUT_ROOT.rglob("EXPERIMENT_STATE.json"):
        try:
            state = json.loads(marker.read_text(encoding="utf-8"))
        except Exception:
            continue
        if state.get("protocol_id") != PROTOCOL_ID or state.get("status") == "COMPLETE":
            continue
        score = len(list(marker.parent.rglob("*.npz"))) * 1000
        score += len(list(marker.parent.rglob("shadow.json"))) * 100
        score += len(list(marker.parent.rglob("*.pt")))
        candidates.append((score, marker.parent))
    if candidates:
        _, source = max(candidates, key=lambda item: item[0])
        print(f"Restoring resumable state from {source}", flush=True)
        shutil.copytree(source, OUTPUT_ROOT, dirs_exist_ok=True)


def locate_code_root() -> Path:
    matches: list[Path] = []
    for marker_path in INPUT_ROOT.rglob("WAHARP_TEACHER_COST_ABLATION_CODE.json"):
        try:
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if marker.get("protocol_id") == CODE_PROTOCOL_ID:
            matches.append(marker_path.parent)
    if len(matches) != 1:
        raise RuntimeError(f"Expected one ablation code bundle, found {matches}")
    return matches[0]


CODE_ROOT = locate_code_root()
sys.path.insert(0, str(CODE_ROOT))

from realtrain.actions import (  # noqa: E402
    ACTION_COUNT,
    candidate_action_ids,
    page_count,
)
from realtrain.config import DEFAULT_CONFIG  # noqa: E402
from realtrain.data import load_all_domains  # noqa: E402
from realtrain.features import local_queries_for_state, state_feature  # noqa: E402
from realtrain.geometry import centers, morton_codes  # noqa: E402
from realtrain.model import StateQNet, unwrap_model  # noqa: E402
from realtrain.pipeline import _evaluate_python_method  # noqa: E402
from realtrain.teacher import (  # noqa: E402
    action_cost_components,
    preliminary_levels,
)
from realtrain.train import load_model  # noqa: E402
from realtrain.tree import build_neural_tree  # noqa: E402
from realtrain.utils import derive_seed, seed_everything  # noqa: E402
from realtrain.workload import load_query_suite  # noqa: E402


def locate_source_root() -> Path:
    matches: list[Path] = []
    for marker_path in INPUT_ROOT.rglob("STATE.json"):
        try:
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        root = marker_path.parent
        if marker.get("protocol_id") != SOURCE_PROTOCOL_ID:
            continue
        required = [root / "models" / "bootstrap.pt"]
        required += [root / "teacher" / f"initial_{name}.npz" for name in DOMAINS]
        required += [root / "teacher" / f"dagger_{name}.npz" for name in DOMAINS]
        required += [root / "workloads" / name / "construction_boxes.npy" for name in DOMAINS]
        if all(path.is_file() for path in required):
            matches.append(root)
    unique = {str(path.resolve()): path for path in matches}
    if len(unique) != 1:
        raise RuntimeError(f"Expected one complete source root, found {list(unique)}")
    return next(iter(unique.values()))


def component_record(
    entries: np.ndarray,
    queries: np.ndarray,
    domain: int,
    level: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    if page_count(len(entries), DEFAULT_CONFIG.capacity) < 2:
        raise ValueError("component state must span at least two pages")
    action_ids = candidate_action_ids(rng, EXPECTED_ACTIONS)
    if len(action_ids) != ACTION_COUNT or len(np.unique(action_ids)) != ACTION_COUNT:
        raise RuntimeError("Expected all 80 actions exactly once")
    hit = np.empty(ACTION_COUNT, dtype=np.float32)
    overlap = np.empty(ACTION_COUNT, dtype=np.float32)
    margin = np.empty(ACTION_COUNT, dtype=np.float32)
    for action_id in action_ids:
        values = action_cost_components(
            entries, queries, int(action_id), DEFAULT_CONFIG.capacity
        )
        hit[int(action_id)], overlap[int(action_id)], margin[int(action_id)] = values
    return {
        "x": state_feature(
            entries,
            queries,
            DEFAULT_CONFIG.capacity,
            DEFAULT_CONFIG.hist_bins,
            level,
        ),
        "hit": hit,
        "overlap": overlap,
        "margin": margin,
        "domain": int(domain),
        "level": int(level),
        "entry_count": int(len(entries)),
        "query_count": int(len(queries)),
    }


def sample_component_states(
    rects: np.ndarray,
    construction_queries: np.ndarray,
    domain_id: int,
    count: int,
    seed: int,
    progress_prefix: str,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    levels = preliminary_levels(rects, DEFAULT_CONFIG.capacity)
    level_orders = [
        np.argsort(morton_codes(centers(level)), kind="mergesort")
        for level in levels
    ]
    records: list[dict[str, Any]] = []
    attempts = 0
    while len(records) < count:
        attempts += 1
        level = int(rng.integers(0, len(levels)))
        array = levels[level]
        order = level_orders[level]
        if len(array) <= DEFAULT_CONFIG.capacity:
            continue
        maximum_pages = min(
            DEFAULT_CONFIG.state_max_pages,
            page_count(len(array), DEFAULT_CONFIG.capacity),
        )
        if maximum_pages < DEFAULT_CONFIG.state_min_pages:
            level = 0
            array = levels[0]
            order = level_orders[0]
            maximum_pages = min(
                DEFAULT_CONFIG.state_max_pages,
                page_count(len(array), DEFAULT_CONFIG.capacity),
            )
        pages = int(
            rng.integers(DEFAULT_CONFIG.state_min_pages, maximum_pages + 1)
        )
        lower = max((pages - 1) * DEFAULT_CONFIG.capacity + 1, 2)
        target = min(
            len(array),
            int(rng.integers(lower, pages * DEFAULT_CONFIG.capacity + 1)),
        )
        if target <= DEFAULT_CONFIG.capacity:
            continue
        start = int(rng.integers(0, max(1, len(array) - target + 1)))
        positions = order[start : start + target]
        entries = np.asarray(array[positions], dtype=np.float32)
        queries = local_queries_for_state(
            entries,
            construction_queries,
            DEFAULT_CONFIG.local_query_cap,
            rng,
        )
        try:
            record = component_record(entries, queries, domain_id, level, rng)
        except Exception:
            if attempts > count * 20:
                raise
            continue
        records.append(record)
        if len(records) % 100 == 0:
            print(f"{progress_prefix}{len(records)}/{count}", flush=True)
    return records


def save_component_records(path: Path, records: list[dict[str, Any]]) -> None:
    if not records:
        raise ValueError("No component records")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporary,
        x=np.stack([record["x"] for record in records]).astype(np.float32),
        hit=np.stack([record["hit"] for record in records]).astype(np.float32),
        overlap=np.stack([record["overlap"] for record in records]).astype(np.float32),
        margin=np.stack([record["margin"] for record in records]).astype(np.float32),
        domain=np.asarray([record["domain"] for record in records], dtype=np.int8),
        level=np.asarray([record["level"] for record in records], dtype=np.int8),
        entry_count=np.asarray(
            [record["entry_count"] for record in records], dtype=np.int32
        ),
        query_count=np.asarray(
            [record["query_count"] for record in records], dtype=np.int16
        ),
    )
    temporary.replace(path)


def verify_against_source(component_path: Path, source_path: Path) -> dict[str, Any]:
    component = np.load(component_path, allow_pickle=False)
    source = np.load(source_path, allow_pickle=False)
    current_cost = (
        component["hit"]
        + CURRENT_OVERLAP * component["overlap"]
        + CURRENT_MARGIN * component["margin"]
    ).astype(np.float32)
    best = np.min(current_cost, axis=1, keepdims=True)
    current_regret = (current_cost - best) / np.maximum(best, 1.0)
    checks = {
        "record_count": int(len(component["x"])),
        "x_exact": bool(np.array_equal(component["x"], source["x"])),
        "domain_exact": bool(np.array_equal(component["domain"], source["domain"])),
        "level_exact": bool(np.array_equal(component["level"], source["level"])),
        "entry_count_exact": bool(
            np.array_equal(component["entry_count"], source["entry_count"])
        ),
        "query_count_exact": bool(
            np.array_equal(component["query_count"], source["query_count"])
        ),
        "raw_cost_max_abs_delta": float(
            np.max(np.abs(current_cost - source["raw_costs"]))
        ),
        "regret_max_abs_delta": float(
            np.max(np.abs(current_regret - source["regrets"]))
        ),
    }
    checks["passed"] = bool(
        checks["x_exact"]
        and checks["domain_exact"]
        and checks["level_exact"]
        and checks["entry_count_exact"]
        and checks["query_count_exact"]
        and checks["raw_cost_max_abs_delta"] <= 2e-5
        and checks["regret_max_abs_delta"] <= 2e-5
    )
    if not checks["passed"]:
        raise RuntimeError(
            f"Regenerated component states do not reproduce {source_path}: {checks}"
        )
    return checks


def generate_component_bank(
    domains: dict[str, np.ndarray], source_root: Path
) -> list[Path]:
    component_root = OUTPUT_ROOT / "components"
    component_root.mkdir(parents=True, exist_ok=True)
    checks: dict[str, Any] = {}
    paths: list[Path] = []

    for domain_id, name in enumerate(DOMAINS):
        require_time(1800, f"initial component states for {name}")
        path = component_root / f"initial_{name}.npz"
        source = source_root / "teacher" / f"initial_{name}.npz"
        if not path.is_file():
            construction = np.load(
                source_root / "workloads" / name / "construction_boxes.npy"
            )
            records = sample_component_states(
                domains[name],
                construction,
                domain_id,
                DEFAULT_CONFIG.initial_states_per_domain,
                derive_seed(
                    SOURCE_PROTOCOL_ID,
                    BASE_SEED,
                    f"teacher_initial|{name}",
                ),
                f"[{name}/initial-components] ",
            )
            save_component_records(path, records)
        checks[f"initial_{name}"] = verify_against_source(path, source)
        paths.append(path)
        update_state("RUNNING", phase="component_bank", completed=str(path.name))

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    bootstrap = load_model(source_root / "models" / "bootstrap.pt", device)
    for domain_id, name in enumerate(DOMAINS):
        require_time(1800, f"DAgger component states for {name}")
        path = component_root / f"dagger_{name}.npz"
        source = source_root / "teacher" / f"dagger_{name}.npz"
        if not path.is_file():
            rng = np.random.default_rng(
                derive_seed(SOURCE_PROTOCOL_ID, BASE_SEED, f"dagger|{name}")
            )
            construction = np.load(
                source_root / "workloads" / name / "construction_boxes.npy"
            )
            count = min(DEFAULT_CONFIG.shadow_rows_per_domain, len(domains[name]))
            if count < len(domains[name]):
                indices = rng.choice(len(domains[name]), size=count, replace=False)
            else:
                indices = np.arange(count)
            subset = domains[name][indices]
            target = DEFAULT_CONFIG.dagger_states_per_domain
            expected = max(1, math.ceil(count / DEFAULT_CONFIG.capacity))
            probability = min(0.85, max(0.10, target / expected * 0.75))
            records: list[dict[str, Any]] = []

            def capture(entries: np.ndarray, queries: np.ndarray, level: int) -> None:
                if len(records) >= target or rng.random() > probability:
                    return
                try:
                    records.append(
                        component_record(entries, queries, domain_id, level, rng)
                    )
                except Exception:
                    return
                if len(records) % 100 == 0:
                    print(
                        f"[{name}/dagger-components] {len(records)}/{target}",
                        flush=True,
                    )

            build_neural_tree(
                subset,
                construction,
                bootstrap,
                device,
                DEFAULT_CONFIG.capacity,
                DEFAULT_CONFIG.hist_bins,
                DEFAULT_CONFIG.local_query_cap,
                capture,
            )
            if len(records) < target:
                records.extend(
                    sample_component_states(
                        domains[name],
                        construction,
                        domain_id,
                        target - len(records),
                        derive_seed(
                            SOURCE_PROTOCOL_ID,
                            BASE_SEED,
                            f"dagger_fill|{name}",
                        ),
                        f"[{name}/dagger-fill-components] ",
                    )
                )
            save_component_records(path, records[:target])
        checks[f"dagger_{name}"] = verify_against_source(path, source)
        paths.append(path)
        update_state("RUNNING", phase="component_bank", completed=str(path.name))

    write_json(
        component_root / "SOURCE_REPRODUCTION.json",
        {
            "protocol_id": PROTOCOL_ID,
            "source_protocol_id": SOURCE_PROTOCOL_ID,
            "all_checks_passed": all(row["passed"] for row in checks.values()),
            "checks": checks,
        },
    )
    return paths


def load_component_bank(paths: list[Path]) -> dict[str, np.ndarray]:
    parts = [np.load(path, allow_pickle=False) for path in paths]
    keys = (
        "x",
        "hit",
        "overlap",
        "margin",
        "domain",
        "level",
        "entry_count",
        "query_count",
    )
    arrays = {key: np.concatenate([part[key] for part in parts], axis=0) for key in keys}
    if len(arrays["x"]) != len(DOMAINS) * EXPECTED_STATES_PER_DOMAIN:
        raise RuntimeError(f"Unexpected component state count: {len(arrays['x'])}")
    return arrays


def arrays_for_weights(
    components: dict[str, np.ndarray], overlap_weight: float, margin_weight: float
) -> dict[str, np.ndarray]:
    costs = (
        components["hit"]
        + overlap_weight * components["overlap"]
        + margin_weight * components["margin"]
    ).astype(np.float32)
    best = np.min(costs, axis=1, keepdims=True)
    regrets = ((costs - best) / np.maximum(best, 1.0)).astype(np.float32)
    return {
        "x": components["x"],
        "regrets": regrets,
        "domain": components["domain"],
    }


class StateDataset(Dataset):
    def __init__(self, arrays: dict[str, np.ndarray], indices: np.ndarray):
        self.arrays = arrays
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int):
        row = self.indices[index]
        return (
            torch.from_numpy(self.arrays["x"][row]),
            torch.from_numpy(self.arrays["regrets"][row]),
            torch.tensor(int(self.arrays["domain"][row]), dtype=torch.long),
        )


def split_indices(
    arrays: dict[str, np.ndarray], seed: int
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    training: list[int] = []
    validation: list[int] = []
    domains = np.asarray(arrays["domain"])
    for domain in sorted(np.unique(domains)):
        indices = np.flatnonzero(domains == domain)
        rng.shuffle(indices)
        validation_count = max(1, int(round(len(indices) * VAL_FRACTION)))
        validation.extend(indices[:validation_count].tolist())
        training.extend(indices[validation_count:].tolist())
    return np.asarray(training, dtype=np.int64), np.asarray(validation, dtype=np.int64)


def loss_batch(
    prediction: torch.Tensor, target: torch.Tensor
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    mask = torch.isfinite(target)
    safe = torch.nan_to_num(target, nan=0.0)
    regression = nn.functional.smooth_l1_loss(prediction[mask], safe[mask])
    big = torch.tensor(1e6, device=prediction.device, dtype=prediction.dtype)
    prediction_masked = torch.where(mask, prediction, big)
    target_logits = torch.where(mask, -safe / TEMPERATURE, -big)
    target_probability = torch.softmax(target_logits, dim=1)
    prediction_log_probability = torch.log_softmax(-prediction_masked, dim=1)
    listwise = -(target_probability * prediction_log_probability).sum(dim=1).mean()
    best = torch.argmin(torch.where(mask, safe, big), dim=1)
    classification = nn.functional.cross_entropy(-prediction_masked, best)
    total = (
        REGRESSION_WEIGHT * regression
        + LIST_WEIGHT * listwise
        + CLASSIFICATION_WEIGHT * classification
    )
    return total, {
        "regression": regression,
        "listwise": listwise,
        "classification": classification,
    }


@torch.no_grad()
def evaluate_teacher_fit(
    model: nn.Module, loader: DataLoader, device: str
) -> dict[str, Any]:
    model.eval()
    losses: list[float] = []
    selected_regrets: list[float] = []
    exact_best: list[float] = []
    by_domain: dict[int, list[float]] = {}
    for features, target, domain in loader:
        features = features.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        prediction = model(features)
        loss, _ = loss_batch(prediction, target)
        losses.append(float(loss))
        chosen = torch.argmin(prediction, dim=1)
        best = torch.argmin(target, dim=1)
        selected = target.gather(1, chosen[:, None]).squeeze(1).cpu().numpy()
        exact_best.extend((chosen == best).float().cpu().numpy().tolist())
        selected_regrets.extend(selected.tolist())
        for domain_id, regret in zip(domain.numpy(), selected):
            by_domain.setdefault(int(domain_id), []).append(float(regret))
    domain_means = {
        str(domain): float(np.mean(values))
        for domain, values in by_domain.items()
    }
    mean_regret = float(np.mean(selected_regrets))
    worst_domain = max(domain_means.values())
    return {
        "loss": float(np.mean(losses)),
        "mean_selected_regret": mean_regret,
        "p95_selected_regret": float(np.quantile(selected_regrets, 0.95)),
        "worst_domain_regret": worst_domain,
        "teacher_selection_score": worst_domain + 0.25 * mean_regret,
        "exact_best_action_accuracy": float(np.mean(exact_best)),
        "domain_mean_selected_regret": domain_means,
    }


def train_one(
    arrays: dict[str, np.ndarray], output_path: Path, seed: int
) -> dict[str, Any]:
    seed_everything(seed)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    training_indices, validation_indices = split_indices(arrays, seed)
    generator = torch.Generator()
    generator.manual_seed(seed)
    training_loader = DataLoader(
        StateDataset(arrays, training_indices),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        generator=generator,
    )
    validation_loader = DataLoader(
        StateDataset(arrays, validation_indices),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )
    base = StateQNet(int(arrays["x"].shape[1]), ACTION_COUNT).to(device)
    gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    model: nn.Module = base
    if gpu_count >= 2:
        model = nn.DataParallel(base, device_ids=list(range(gpu_count)))
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
    best_score = float("inf")
    best_epoch = -1
    best_validation: dict[str, Any] | None = None
    stale = 0
    history: list[dict[str, Any]] = []
    started = time.perf_counter()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        training_losses: list[float] = []
        for features, target, _ in training_loader:
            features = features.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                prediction = model(features)
                loss, _ = loss_batch(prediction, target)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            training_losses.append(float(loss.detach()))
        validation = evaluate_teacher_fit(model, validation_loader, device)
        score = float(validation["teacher_selection_score"])
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(training_losses)),
                **validation,
            }
        )
        print(
            f"model={output_path.parent.name} seed={seed} epoch={epoch:02d} "
            f"train={np.mean(training_losses):.6f} teacher_score={score:.6f}",
            flush=True,
        )
        if score < best_score - 1e-6:
            best_score = score
            best_epoch = epoch
            best_validation = validation
            stale = 0
            raw = unwrap_model(model)
            torch.save(
                {
                    "state_dict": raw.state_dict(),
                    "input_dim": raw.input_dim,
                    "action_count": raw.action_count,
                    "seed": seed,
                    "epoch": epoch,
                    "validation": validation,
                },
                output_path,
            )
        else:
            stale += 1
            if stale >= PATIENCE:
                break
    assert best_validation is not None
    metadata = {
        "seed": seed,
        "best_epoch": best_epoch,
        "epochs_completed": len(history),
        "seconds": time.perf_counter() - started,
        "train_states": int(len(training_indices)),
        "validation_states": int(len(validation_indices)),
        "gpu_count": gpu_count,
        "data_parallel": bool(gpu_count >= 2),
        **best_validation,
        "history": history,
    }
    write_json(output_path.with_suffix(".training.json"), metadata)
    return metadata


def prepare_shadow_data(
    domains: dict[str, np.ndarray], source_root: Path
) -> dict[str, dict[str, Any]]:
    prepared: dict[str, dict[str, Any]] = {}
    for name in DOMAINS:
        rng = np.random.default_rng(
            derive_seed(SOURCE_PROTOCOL_ID, BASE_SEED, f"shadow_objects|{name}")
        )
        count = min(DEFAULT_CONFIG.shadow_rows_per_domain, len(domains[name]))
        if count < len(domains[name]):
            indices = rng.choice(len(domains[name]), size=count, replace=False)
        else:
            indices = np.arange(count)
        prepared[name] = {
            "rects": domains[name][indices],
            "construction": np.load(
                source_root / "workloads" / name / "construction_boxes.npy"
            ),
            "suite": load_query_suite(source_root / "workloads" / name, "validation"),
        }
    return prepared


def evaluate_shadow(
    checkpoint: Path, shadow_data: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = load_model(checkpoint, device)
    per_domain: dict[str, Any] = {}
    for name in DOMAINS:
        data = shadow_data[name]
        tree, diagnostics = build_neural_tree(
            data["rects"],
            data["construction"],
            model,
            device,
            DEFAULT_CONFIG.capacity,
            DEFAULT_CONFIG.hist_bins,
            DEFAULT_CONFIG.local_query_cap,
        )
        rows = _evaluate_python_method(
            tree,
            data["suite"],
            len(data["rects"]),
            DEFAULT_CONFIG.capacity,
            "teacher_cost_ablation",
        )
        totals: dict[str, float] = {}
        counts: dict[str, int] = {}
        for row in rows:
            family = str(row["query_type"])
            totals[family] = totals.get(family, 0.0) + float(
                row["total_node_accesses"]
            )
            counts[family] = counts.get(family, 0) + 1
        total_accesses = float(
            sum(float(row["total_node_accesses"]) for row in rows)
        )
        per_domain[name] = {
            "queries": len(rows),
            "total_accesses": total_accesses,
            "mean_accesses": total_accesses / max(len(rows), 1),
            "family_total_accesses": totals,
            "family_query_counts": counts,
            "build_seconds": float(diagnostics["build_seconds"]),
            "tree_validation": diagnostics["validation"],
        }
        del tree
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {"per_domain": per_domain}


def config_name(overlap_weight: float, margin_weight: float) -> str:
    overlap_multiplier = overlap_weight / CURRENT_OVERLAP
    margin_multiplier = margin_weight / CURRENT_MARGIN
    return f"overlap_{overlap_multiplier:g}x_margin_{margin_multiplier:g}x"


def training_seeds() -> list[int]:
    return [
        derive_seed(SOURCE_PROTOCOL_ID, BASE_SEED, f"final_member|{member}")
        for member in range(3)
    ]


def run_models(
    components: dict[str, np.ndarray], shadow_data: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    for overlap_weight in OVERLAP_VALUES:
        for margin_weight in MARGIN_VALUES:
            arrays = arrays_for_weights(components, overlap_weight, margin_weight)
            name = config_name(overlap_weight, margin_weight)
            for seed in training_seeds():
                require_time(900, f"training and shadow evaluation for {name}/{seed}")
                run_root = OUTPUT_ROOT / "runs" / name / f"seed_{seed}"
                checkpoint = run_root / "model.pt"
                training_path = checkpoint.with_suffix(".training.json")
                shadow_path = run_root / "shadow.json"
                if checkpoint.is_file() and training_path.is_file():
                    training = json.loads(training_path.read_text(encoding="utf-8"))
                else:
                    training = train_one(arrays, checkpoint, seed)
                if shadow_path.is_file():
                    shadow = json.loads(shadow_path.read_text(encoding="utf-8"))
                else:
                    shadow = evaluate_shadow(checkpoint, shadow_data)
                    write_json(shadow_path, shadow)
                run = {
                    "config": name,
                    "overlap_weight": overlap_weight,
                    "margin_weight": margin_weight,
                    "overlap_multiplier": overlap_weight / CURRENT_OVERLAP,
                    "margin_multiplier": margin_weight / CURRENT_MARGIN,
                    "seed": seed,
                    "checkpoint": str(checkpoint),
                    "training": training,
                    "shadow": shadow,
                }
                runs.append(run)
                update_state(
                    "RUNNING",
                    phase="models",
                    completed_config=name,
                    completed_seed=seed,
                    completed_runs=len(runs),
                    total_runs=len(OVERLAP_VALUES) * len(MARGIN_VALUES) * 3,
                )
    return runs


def aggregate_runs(runs: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_seed_domain: dict[tuple[int, str], float] = {}
    for seed in training_seeds():
        for domain in DOMAINS:
            by_seed_domain[(seed, domain)] = min(
                float(run["shadow"]["per_domain"][domain]["total_accesses"])
                for run in runs
                if int(run["seed"]) == seed
            )

    flat_runs: list[dict[str, Any]] = []
    for run in runs:
        ratios = {
            domain: float(run["shadow"]["per_domain"][domain]["total_accesses"])
            / by_seed_domain[(int(run["seed"]), domain)]
            for domain in DOMAINS
        }
        validation_score = max(ratios.values()) + 0.25 * float(
            np.mean(list(ratios.values()))
        )
        flat_runs.append(
            {
                "config": run["config"],
                "overlap_weight": run["overlap_weight"],
                "margin_weight": run["margin_weight"],
                "overlap_multiplier": run["overlap_multiplier"],
                "margin_multiplier": run["margin_multiplier"],
                "seed": run["seed"],
                "validation_score": validation_score,
                "twitter_accesses": run["shadow"]["per_domain"]["twitter"]["total_accesses"],
                "crimes_accesses": run["shadow"]["per_domain"]["crimes"]["total_accesses"],
                "arizona_accesses": run["shadow"]["per_domain"]["arizona"]["total_accesses"],
                "twitter_ratio": ratios["twitter"],
                "crimes_ratio": ratios["crimes"],
                "arizona_ratio": ratios["arizona"],
                "teacher_selection_score": run["training"]["teacher_selection_score"],
                "exact_best_action_accuracy": run["training"]["exact_best_action_accuracy"],
                "best_epoch": run["training"]["best_epoch"],
                "training_seconds": run["training"]["seconds"],
            }
        )

    summaries: list[dict[str, Any]] = []
    for overlap_weight in OVERLAP_VALUES:
        for margin_weight in MARGIN_VALUES:
            name = config_name(overlap_weight, margin_weight)
            selected = [row for row in flat_runs if row["config"] == name]
            scores = np.asarray(
                [float(row["validation_score"]) for row in selected], dtype=np.float64
            )
            summary: dict[str, Any] = {
                "config": name,
                "overlap_weight": overlap_weight,
                "margin_weight": margin_weight,
                "overlap_multiplier": overlap_weight / CURRENT_OVERLAP,
                "margin_multiplier": margin_weight / CURRENT_MARGIN,
                "seeds": len(selected),
                "validation_score_mean": float(np.mean(scores)),
                "validation_score_std": float(np.std(scores, ddof=1)),
            }
            for domain in DOMAINS:
                values = np.asarray(
                    [float(row[f"{domain}_accesses"]) for row in selected],
                    dtype=np.float64,
                )
                ratios = np.asarray(
                    [float(row[f"{domain}_ratio"]) for row in selected],
                    dtype=np.float64,
                )
                summary[f"{domain}_accesses_mean"] = float(np.mean(values))
                summary[f"{domain}_ratio_mean"] = float(np.mean(ratios))
            summaries.append(summary)
    return flat_runs, summaries


def label_sensitivity(components: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    hit_only = np.argmin(components["hit"], axis=1)
    current_cost = (
        components["hit"]
        + CURRENT_OVERLAP * components["overlap"]
        + CURRENT_MARGIN * components["margin"]
    )
    current = np.argmin(current_cost, axis=1)
    rows: list[dict[str, Any]] = []
    for overlap_weight in OVERLAP_VALUES:
        for margin_weight in MARGIN_VALUES:
            costs = (
                components["hit"]
                + overlap_weight * components["overlap"]
                + margin_weight * components["margin"]
            )
            selected = np.argmin(costs, axis=1)
            rows.append(
                {
                    "config": config_name(overlap_weight, margin_weight),
                    "overlap_weight": overlap_weight,
                    "margin_weight": margin_weight,
                    "fraction_best_action_changed_vs_hit_only": float(
                        np.mean(selected != hit_only)
                    ),
                    "fraction_best_action_changed_vs_current": float(
                        np.mean(selected != current)
                    ),
                }
            )
    return rows


def write_heatmap(summaries: list[dict[str, Any]]) -> None:
    import matplotlib.pyplot as plt

    matrix = np.empty((len(OVERLAP_VALUES), len(MARGIN_VALUES)), dtype=np.float64)
    for row in summaries:
        i = OVERLAP_VALUES.index(float(row["overlap_weight"]))
        j = MARGIN_VALUES.index(float(row["margin_weight"]))
        matrix[i, j] = float(row["validation_score_mean"])
    figure, axis = plt.subplots(figsize=(7.4, 5.4))
    image = axis.imshow(matrix, cmap="viridis_r", aspect="auto")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            axis.text(j, i, f"{matrix[i, j]:.4f}", ha="center", va="center")
    axis.set_xticks(range(len(MARGIN_VALUES)))
    axis.set_xticklabels([f"{value:.0e}" if value else "0" for value in MARGIN_VALUES])
    axis.set_yticks(range(len(OVERLAP_VALUES)))
    axis.set_yticklabels([f"{value:.0e}" if value else "0" for value in OVERLAP_VALUES])
    axis.set_xlabel("Margin coefficient")
    axis.set_ylabel("Overlap coefficient")
    axis.set_title("Validation-only teacher-cost coefficient ablation")
    figure.colorbar(image, ax=axis, label="Validation score (lower is better)")
    figure.tight_layout()
    figure.savefig(OUTPUT_ROOT / "VALIDATION_SCORE_HEATMAP.png", dpi=180)
    plt.close(figure)


def write_report(
    summaries: list[dict[str, Any]],
    flat_runs: list[dict[str, Any]],
    sensitivity: list[dict[str, Any]],
    winner: dict[str, Any],
) -> None:
    current_name = config_name(CURRENT_OVERLAP, CURRENT_MARGIN)
    current = next(row for row in summaries if row["config"] == current_name)
    paired = []
    for seed in training_seeds():
        winning_run = next(
            row
            for row in flat_runs
            if row["config"] == winner["config"] and int(row["seed"]) == seed
        )
        current_run = next(
            row
            for row in flat_runs
            if row["config"] == current_name and int(row["seed"]) == seed
        )
        paired.append(
            float(winning_run["validation_score"])
            - float(current_run["validation_score"])
        )
    lines = [
        "# WAHARP Teacher-Cost Coefficient Ablation",
        "",
        f"Protocol: `{PROTOCOL_ID}`",
        "",
        "This is a validation-only, paired ablation. It does not read final queries or PLATON results.",
        "All configurations use the same 30,000 states, three training seeds, 0.75/0.25 listwise/best-action loss, shadow objects, and validation queries.",
        "The initial states and historical-policy DAgger trajectories are held fixed across coefficient pairs; only the teacher costs and resulting targets are recomputed.",
        "",
        "## Grid",
        "",
        f"- Overlap coefficients: `{OVERLAP_VALUES}`",
        f"- Margin coefficients: `{MARGIN_VALUES}`",
        f"- Historical pair: `{CURRENT_OVERLAP}` / `{CURRENT_MARGIN}`",
        "",
        "## Validation results",
        "",
        "| Overlap | Margin | Score | Twitter ratio | Crimes ratio | Arizona ratio |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(summaries, key=lambda item: float(item["validation_score_mean"])):
        lines.append(
            f"| {row['overlap_weight']:.0e} | {row['margin_weight']:.0e} | "
            f"{row['validation_score_mean']:.6f} +/- {row['validation_score_std']:.6f} | "
            f"{row['twitter_ratio_mean']:.6f} | {row['crimes_ratio_mean']:.6f} | "
            f"{row['arizona_ratio_mean']:.6f} |"
        )
    lines += [
        "",
        "## Selection",
        "",
        f"Selected overlap coefficient: **{winner['overlap_weight']:.0e}**.",
        f"Selected margin coefficient: **{winner['margin_weight']:.0e}**.",
        f"Selected validation score: **{winner['validation_score_mean']:.6f} +/- {winner['validation_score_std']:.6f}**.",
        f"Historical validation score: **{current['validation_score_mean']:.6f} +/- {current['validation_score_std']:.6f}**.",
        f"Paired selected-minus-historical score differences: `{paired}`.",
        "",
        "Selection minimizes the mean across three paired seeds of the worst-domain validation-access ratio plus 0.25 times the mean domain ratio. Ratios are normalized per seed and domain against the best grid configuration.",
        "",
        "## Interpretation boundary",
        "",
        "This experiment selects teacher-cost coefficients using only validation query node accesses. It does not support a final-test claim by itself.",
        "Because the paired study uses a fixed historical-policy trajectory bank, a selected non-historical pair must be rerun through its own full DAgger/final-training pipeline and locked final evaluation before changing the paper's main results.",
        "",
        "## Label sensitivity",
        "",
        "| Overlap | Margin | Changed vs hit-only | Changed vs historical |",
        "|---:|---:|---:|---:|",
    ]
    for row in sensitivity:
        lines.append(
            f"| {row['overlap_weight']:.0e} | {row['margin_weight']:.0e} | "
            f"{row['fraction_best_action_changed_vs_hit_only']:.4%} | "
            f"{row['fraction_best_action_changed_vs_current']:.4%} |"
        )
    (OUTPUT_ROOT / "FINAL_REPORT.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    restore_resume()
    update_state("RUNNING", phase="initialization")
    try:
        source_root = locate_source_root()
        domains, metadata = load_all_domains(INPUT_ROOT, 1_000_000, 1_000_000, 0)
        write_json(OUTPUT_ROOT / "DATASETS.json", metadata)
        component_paths = generate_component_bank(domains, source_root)
        components = load_component_bank(component_paths)
        provenance = {
            "protocol_id": PROTOCOL_ID,
            "source_protocol_id": SOURCE_PROTOCOL_ID,
            "code_protocol_id": CODE_PROTOCOL_ID,
            "source_root": str(source_root),
            "states": int(len(components["x"])),
            "actions": int(components["hit"].shape[1]),
            "x_sha256": sha256_array(components["x"]),
            "hit_sha256": sha256_array(components["hit"]),
            "overlap_sha256": sha256_array(components["overlap"]),
            "margin_sha256": sha256_array(components["margin"]),
            "overlap_values": OVERLAP_VALUES,
            "margin_values": MARGIN_VALUES,
            "list_weight": LIST_WEIGHT,
            "classification_weight": CLASSIFICATION_WEIGHT,
            "trajectory_bank": "fixed initial states plus DAgger states captured by the historical bootstrap policy",
            "final_or_test_queries_read": False,
            "platon_results_read": False,
        }
        write_json(OUTPUT_ROOT / "PROVENANCE.json", provenance)
        shadow_data = prepare_shadow_data(domains, source_root)
        runs = run_models(components, shadow_data)
        flat_runs, summaries = aggregate_runs(runs)
        sensitivity = label_sensitivity(components)
        winner = min(
            summaries,
            key=lambda row: (
                float(row["validation_score_mean"]),
                float(row["validation_score_std"]),
                abs(float(row["overlap_multiplier"]) - 1.0)
                + abs(float(row["margin_multiplier"]) - 1.0),
                float(row["overlap_weight"]) + float(row["margin_weight"]),
            ),
        )
        write_rows(OUTPUT_ROOT / "ABLATION_RUNS.csv", flat_runs)
        write_rows(OUTPUT_ROOT / "ABLATION_SUMMARY.csv", summaries)
        write_rows(OUTPUT_ROOT / "LABEL_SENSITIVITY.csv", sensitivity)
        write_heatmap(summaries)
        current_selected = bool(
            math.isclose(float(winner["overlap_weight"]), CURRENT_OVERLAP)
            and math.isclose(float(winner["margin_weight"]), CURRENT_MARGIN)
        )
        decision = {
            "protocol_id": PROTOCOL_ID,
            "status": "COMPLETE",
            "selection_data": "validation_queries_only",
            "selection_rule": "minimum three-seed mean of worst-domain validation-access ratio plus 0.25 mean domain ratio",
            "selected_overlap_weight": winner["overlap_weight"],
            "selected_margin_weight": winner["margin_weight"],
            "historical_pair_selected": current_selected,
            "winner": winner,
            "requires_full_retraining_and_locked_evaluation": not current_selected,
            "final_or_test_queries_read": False,
            "platon_results_read": False,
            "elapsed_seconds": elapsed(),
        }
        write_json(OUTPUT_ROOT / "FINAL_DECISION.json", decision)
        write_report(summaries, flat_runs, sensitivity, winner)
        update_state("COMPLETE", decision=decision)
        print(json.dumps(decision, indent=2, sort_keys=True, default=json_default))
    except GracefulPause as error:
        update_state(
            "PAUSED_RESUME_REQUIRED",
            reason=str(error),
            resume_instruction="Attach this kernel output as a private dataset and rerun unchanged",
        )
        print(str(error), flush=True)


if __name__ == "__main__":
    main()
