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
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


PROTOCOL_ID = "waharp_lodo_zero_shot_v1_20260919"
SOURCE_PROTOCOL_ID = "waharp_realtrain_v1_three_domain_pageio_20260829"
CODE_PROTOCOL_ID = "waharp_final_capacity_sweep_member2_v1_20260901"
ALL_DOMAIN_PROTOCOL_ID = "waharp_loss075_locked_eval_v1_20260903"
QUERY_NAMESPACE = "waharp_final_fullscale_b128_member2_93d3c2545e44b9e21269a1b4"
BASE_SEED = 2026082951
DOMAINS = ("twitter", "crimes", "arizona")
CAPACITIES = (128, 256, 512)
EXPECTED_QUERIES = 6000
KNN_QUERIES_PER_K = 300
EXPECTED_ACTIONS = 80
EXPECTED_INITIAL_STATES = 8000
EXPECTED_DAGGER_STATES = 2000
LIST_WEIGHT = 0.75
CLASSIFICATION_WEIGHT = 0.25
REGRESSION_WEIGHT = 1.0
TEMPERATURE = 0.025
FINAL_EPOCHS = 28
BATCH_SIZE = 256
LEARNING_RATE = 6e-4
WEIGHT_DECAY = 1e-4
PATIENCE = 6
VAL_FRACTION = 0.20
MAX_WORK_SECONDS = 10.5 * 3600
INPUT_ROOT = Path("/kaggle/input")
OUTPUT_ROOT = Path("/kaggle/working/WAHARP_LODO_GENERALIZATION")
INITIAL_HASHES = {
    "arizona": "415eaedb90641bd378ee2bb2c2a8959c98452ba4cc6277033da5712cbaf2d3e0",
    "crimes": "b01ec93ba1bd50859b01123e73248fa83becc4f6d21381952fa6b790d79bc399",
    "twitter": "4df13f143b5206cd6e26895b92e819f7dd7ac8f6320810e229388d382f4d29bd",
}
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
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=json_default), encoding="utf-8")
    tmp.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def elapsed() -> float:
    return time.time() - STARTED


def require_time(seconds: float, label: str) -> None:
    if elapsed() + seconds > MAX_WORK_SECONDS:
        raise GracefulPause(f"Insufficient safe runtime for {label}: elapsed={elapsed():.1f}s")


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def update_state(status: str, **extra: Any) -> None:
    payload = {
        "protocol_id": PROTOCOL_ID,
        "status": status,
        "elapsed_seconds": elapsed(),
        "gpu_names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        "updated_unix": time.time(),
        **extra,
    }
    write_json(OUTPUT_ROOT / "LODO_STATE.json", payload)


def restore_resume() -> None:
    candidates: list[tuple[int, Path, dict[str, Any]]] = []
    for marker in INPUT_ROOT.rglob("LODO_STATE.json"):
        try:
            state = json.loads(marker.read_text(encoding="utf-8"))
        except Exception:
            continue
        if state.get("protocol_id") != PROTOCOL_ID or state.get("status") == "COMPLETE":
            continue
        score = len(list(marker.parent.rglob("*.pt"))) * 100 + len(list(marker.parent.rglob("PER_QUERY.csv")))
        candidates.append((score, marker.parent, state))
    if not candidates:
        extract_root = Path("/tmp/waharp_lodo_resume")
        for archive in INPUT_ROOT.rglob("WAHARP_LODO_GENERALIZATION.zip"):
            with zipfile.ZipFile(archive) as zipped:
                members = zipped.namelist()
                if not any(name.endswith("LODO_STATE.json") for name in members):
                    continue
                base = extract_root.resolve()
                for name in members:
                    target = (extract_root / name).resolve()
                    if base != target and base not in target.parents:
                        raise RuntimeError(f"Unsafe resume archive member: {name}")
                if extract_root.exists():
                    shutil.rmtree(extract_root)
                zipped.extractall(extract_root)
            for marker in extract_root.rglob("LODO_STATE.json"):
                state = json.loads(marker.read_text(encoding="utf-8"))
                if state.get("protocol_id") != PROTOCOL_ID or state.get("status") == "COMPLETE":
                    continue
                score = len(list(marker.parent.rglob("*.pt"))) * 100 + len(list(marker.parent.rglob("PER_QUERY.csv")))
                candidates.append((score, marker.parent, state))
    if candidates:
        _, source, state = max(candidates, key=lambda item: item[0])
        print(f"Restoring resumable state from {source} ({state.get('status')})", flush=True)
        shutil.copytree(source, OUTPUT_ROOT, dirs_exist_ok=True)


def locate_code_root() -> Path:
    matches = []
    for marker in INPUT_ROOT.rglob("WAHARP_FINAL_FULLSCALE_BUNDLE_MARKER.json"):
        try:
            raw = json.loads(marker.read_text(encoding="utf-8"))
        except Exception:
            continue
        if raw.get("protocol_id") == CODE_PROTOCOL_ID:
            matches.append(marker.parent)
    if len(matches) != 1:
        raise RuntimeError(f"Expected one code bundle, found {matches}")
    return matches[0]


CODE_ROOT = locate_code_root()
sys.path.insert(0, str(CODE_ROOT))

from realtrain.config import DEFAULT_CONFIG  # noqa: E402
from realtrain.data import (  # noqa: E402
    load_arizona,
    load_point_csv,
    locate_arizona_npy,
    locate_crimes_csv,
    locate_twitter_csv,
)
from realtrain.model import StateQNet, unwrap_model  # noqa: E402
from realtrain.pipeline import _evaluate_python_method  # noqa: E402
from realtrain.teacher import label_state, load_records, sample_states_for_domain, save_records  # noqa: E402
from realtrain.train import load_model, train_one as train_bootstrap  # noqa: E402
from realtrain.tree import build_neural_tree  # noqa: E402
from realtrain.utils import derive_seed, seed_everything, sha256_array  # noqa: E402
from realtrain.workload import load_query_suite  # noqa: E402


def load_training_domain(name: str) -> tuple[np.ndarray, dict[str, Any]]:
    if name == "twitter":
        return load_point_csv(locate_twitter_csv(INPUT_ROOT), 1_000_000)
    if name == "crimes":
        return load_point_csv(locate_crimes_csv(INPUT_ROOT), 1_000_000)
    if name == "arizona":
        return load_arizona(locate_arizona_npy(INPUT_ROOT), 0)
    raise KeyError(name)


def load_evaluation_domain(name: str) -> tuple[np.ndarray, dict[str, Any]]:
    if name == "twitter":
        return load_point_csv(locate_twitter_csv(INPUT_ROOT), 0)
    if name == "crimes":
        return load_point_csv(locate_crimes_csv(INPUT_ROOT), 0)
    if name == "arizona":
        return load_arizona(locate_arizona_npy(INPUT_ROOT), 0)
    raise KeyError(name)


def locate_teacher_root() -> tuple[Path, dict[str, Path]]:
    found: dict[str, list[Path]] = {name: [] for name in DOMAINS}
    for path in INPUT_ROOT.rglob("initial_*.npz"):
        name = path.stem.removeprefix("initial_")
        if name in found and sha256_file(path) == INITIAL_HASHES[name]:
            found[name].append(path)
    if any(len(paths) != 1 for paths in found.values()):
        raise RuntimeError(f"Initial teacher-state discovery failed: {found}")
    paths = {name: values[0] for name, values in found.items()}
    roots = {path.parent.parent for path in paths.values()}
    if len(roots) != 1:
        raise RuntimeError(f"Initial teacher states do not share one root: {roots}")
    root = next(iter(roots))
    marker = json.loads((root / "STATE.json").read_text(encoding="utf-8"))
    if marker.get("protocol_id") != SOURCE_PROTOCOL_ID:
        raise RuntimeError(f"Unexpected teacher protocol: {marker}")
    for name in DOMAINS:
        manifest = json.loads((root / "workloads" / name / "WORKLOAD_MANIFEST.json").read_text(encoding="utf-8"))
        if manifest.get("namespace") != SOURCE_PROTOCOL_ID:
            raise RuntimeError(f"Unexpected source validation workload for {name}")
    return root, paths


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


def split_indices(arrays: dict[str, np.ndarray], seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    train: list[int] = []
    validation: list[int] = []
    domains = np.asarray(arrays["domain"])
    for domain in sorted(np.unique(domains)):
        indices = np.flatnonzero(domains == domain)
        rng.shuffle(indices)
        count = max(1, int(round(len(indices) * VAL_FRACTION)))
        validation.extend(indices[:count].tolist())
        train.extend(indices[count:].tolist())
    return np.asarray(train, dtype=np.int64), np.asarray(validation, dtype=np.int64)


def loss_batch(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    mask = torch.isfinite(target)
    safe = torch.nan_to_num(target, nan=0.0)
    regression = nn.functional.smooth_l1_loss(prediction[mask], safe[mask])
    big = torch.tensor(1e6, device=prediction.device, dtype=prediction.dtype)
    pred_masked = torch.where(mask, prediction, big)
    target_prob = torch.softmax(torch.where(mask, -safe / TEMPERATURE, -big), dim=1)
    listwise = -(target_prob * torch.log_softmax(-pred_masked, dim=1)).sum(dim=1).mean()
    best = torch.argmin(torch.where(mask, safe, big), dim=1)
    classification = nn.functional.cross_entropy(-pred_masked, best)
    return REGRESSION_WEIGHT * regression + LIST_WEIGHT * listwise + CLASSIFICATION_WEIGHT * classification


@torch.no_grad()
def evaluate_loss(model: nn.Module, loader: DataLoader, device: str) -> dict[str, float]:
    model.eval()
    losses: list[float] = []
    regrets: list[float] = []
    by_domain: dict[int, list[float]] = {}
    for features, target, domain in loader:
        features = features.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        prediction = model(features)
        losses.append(float(loss_batch(prediction, target)))
        mask = torch.isfinite(target)
        big = torch.tensor(1e6, device=prediction.device, dtype=prediction.dtype)
        chosen = torch.argmin(torch.where(mask, prediction, big), dim=1)
        selected = target.gather(1, chosen[:, None]).squeeze(1).cpu().numpy()
        regrets.extend(selected.tolist())
        for domain_id, regret in zip(domain.numpy(), selected):
            by_domain.setdefault(int(domain_id), []).append(float(regret))
    means = {key: float(np.mean(values)) for key, values in by_domain.items()}
    mean = float(np.mean(regrets))
    worst = max(means.values())
    return {"loss": float(np.mean(losses)), "mean_selected_regret": mean, "worst_domain_regret": worst,
            "selection_score": worst + 0.25 * mean}


def train_final(arrays: dict[str, np.ndarray], path: Path, seed: int, member: int) -> dict[str, Any]:
    if path.is_file() and path.with_suffix(".training.json").is_file():
        return json.loads(path.with_suffix(".training.json").read_text(encoding="utf-8"))
    require_time(900, f"final member {member}")
    seed_everything(seed)
    device = "cuda:0"
    train_idx, val_idx = split_indices(arrays, seed)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(StateDataset(arrays, train_idx), batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=2, pin_memory=True, generator=generator)
    val_loader = DataLoader(StateDataset(arrays, val_idx), batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=2, pin_memory=True)
    base = StateQNet(int(arrays["x"].shape[1]), EXPECTED_ACTIONS).to(device)
    model: nn.Module = nn.DataParallel(base, device_ids=[0, 1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    best_score = float("inf")
    best_epoch = -1
    stale = 0
    history: list[dict[str, Any]] = []
    started = time.perf_counter()
    path.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, FINAL_EPOCHS + 1):
        model.train()
        epoch_losses = []
        for features, target, _ in train_loader:
            features = features.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=True):
                prediction = model(features)
                loss = loss_batch(prediction, target)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            epoch_losses.append(float(loss.detach()))
        validation = evaluate_loss(model, val_loader, device)
        score = validation["selection_score"]
        history.append({"epoch": epoch, "train_loss": float(np.mean(epoch_losses)), **validation})
        print(f"fold-member={member} epoch={epoch:02d} score={score:.6f}", flush=True)
        if score < best_score - 1e-6:
            best_score = score
            best_epoch = epoch
            stale = 0
            raw = unwrap_model(model)
            torch.save({"state_dict": raw.state_dict(), "input_dim": raw.input_dim,
                        "action_count": raw.action_count, "seed": seed, "epoch": epoch,
                        "validation": validation, "loss": {"regression_weight": REGRESSION_WEIGHT,
                        "list_weight": LIST_WEIGHT, "classification_weight": CLASSIFICATION_WEIGHT,
                        "temperature": TEMPERATURE}}, path)
        else:
            stale += 1
            if stale >= PATIENCE:
                break
    metadata = {"member": member, "seed": seed, "best_epoch": best_epoch,
                "epochs_completed": len(history), "best_score": best_score, "history": history,
                "seconds": time.perf_counter() - started, "train_states": len(train_idx),
                "validation_states": len(val_idx), "gpu_count": 2, "data_parallel": True,
                "checkpoint_sha256": sha256_file(path)}
    write_json(path.with_suffix(".training.json"), metadata)
    return metadata


def make_dagger_domain(name: str, domain_id: int, rects: np.ndarray, workroot: Path,
                       bootstrap_path: Path, output_path: Path, fold_seed: int) -> None:
    if output_path.is_file():
        records = load_records([output_path])
        if len(records["x"]) != EXPECTED_DAGGER_STATES:
            raise RuntimeError(f"Bad resumed DAgger file: {output_path}")
        return
    require_time(1200, f"DAgger {name}")
    device = "cuda:0"
    model = load_model(bootstrap_path, device)
    rng = np.random.default_rng(derive_seed(PROTOCOL_ID, fold_seed, f"dagger|{name}"))
    records = []
    construction = np.load(workroot / "construction_boxes.npy")
    n = min(DEFAULT_CONFIG.shadow_rows_per_domain, len(rects))
    idx = rng.choice(len(rects), size=n, replace=False) if n < len(rects) else np.arange(n)
    subset = rects[idx]
    expected = max(1, math.ceil(n / DEFAULT_CONFIG.capacity))
    probability = min(0.85, max(0.10, EXPECTED_DAGGER_STATES / expected * 0.75))

    def capture(entries: np.ndarray, queries: np.ndarray, level: int) -> None:
        if len(records) >= EXPECTED_DAGGER_STATES or rng.random() > probability:
            return
        try:
            records.append(label_state(entries, queries, domain_id, level, DEFAULT_CONFIG.capacity,
                                       DEFAULT_CONFIG.hist_bins, DEFAULT_CONFIG.candidates_per_state, rng,
                                       DEFAULT_CONFIG.teacher_overlap_weight, DEFAULT_CONFIG.teacher_margin_weight))
        except Exception:
            return
        if len(records) % 100 == 0:
            print(f"[{name}] source-only DAgger {len(records)}/{EXPECTED_DAGGER_STATES}", flush=True)

    build_neural_tree(subset, construction, model, device, DEFAULT_CONFIG.capacity,
                      DEFAULT_CONFIG.hist_bins, DEFAULT_CONFIG.local_query_cap, capture)
    if len(records) < EXPECTED_DAGGER_STATES:
        extra = sample_states_for_domain(
            rects, construction, domain_id, EXPECTED_DAGGER_STATES - len(records),
            DEFAULT_CONFIG.capacity, DEFAULT_CONFIG.state_min_pages, DEFAULT_CONFIG.state_max_pages,
            DEFAULT_CONFIG.local_query_cap, DEFAULT_CONFIG.hist_bins, DEFAULT_CONFIG.candidates_per_state,
            derive_seed(PROTOCOL_ID, fold_seed, f"dagger_fill|{name}"),
            DEFAULT_CONFIG.teacher_overlap_weight, DEFAULT_CONFIG.teacher_margin_weight,
            progress_prefix=f"[{name}/fill] ")
        records.extend(extra)
    save_records(output_path, records[:EXPECTED_DAGGER_STATES])
    del model
    gc.collect()
    torch.cuda.empty_cache()


def train_fold(heldout: str, teacher_root: Path, initial_paths: dict[str, Path]) -> dict[str, Any]:
    fold = OUTPUT_ROOT / "folds" / f"heldout_{heldout}"
    frozen = fold / "frozen_model" / "selected.pt"
    fold_state = fold / "FOLD_STATE.json"
    if frozen.is_file() and fold_state.is_file():
        state = json.loads(fold_state.read_text(encoding="utf-8"))
        if state.get("status") == "FROZEN" and state.get("heldout_data_loaded_before_freeze") is False:
            return state
    sources = [name for name in DOMAINS if name != heldout]
    fold_seed = derive_seed(PROTOCOL_ID, BASE_SEED, f"fold|heldout_{heldout}")
    update_state("RUNNING", stage="training", heldout=heldout, source_domains=sources)
    source_data: dict[str, np.ndarray] = {}
    source_meta: dict[str, Any] = {}
    for name in sources:
        source_data[name], source_meta[name] = load_training_domain(name)
    initial = load_records([initial_paths[name] for name in sources])
    if len(initial["x"]) != EXPECTED_INITIAL_STATES * 2:
        raise RuntimeError(f"Unexpected initial state count for heldout {heldout}: {len(initial['x'])}")
    bootstrap = fold / "models" / "bootstrap.pt"
    if not bootstrap.is_file():
        require_time(900, f"bootstrap heldout {heldout}")
        train_bootstrap(initial, bootstrap, derive_seed(PROTOCOL_ID, fold_seed, "bootstrap_model"),
                        DEFAULT_CONFIG.bootstrap_epochs, BATCH_SIZE, LEARNING_RATE, WEIGHT_DECAY,
                        PATIENCE, True)
    for name in sources:
        make_dagger_domain(name, DOMAINS.index(name), source_data[name], teacher_root / "workloads" / name,
                           bootstrap, fold / "teacher" / f"dagger_{name}.npz", fold_seed)
        update_state("RUNNING", stage="dagger", heldout=heldout, completed_source=name)
    combined = load_records([initial_paths[name] for name in sources] +
                            [fold / "teacher" / f"dagger_{name}.npz" for name in sources])
    if len(combined["x"]) != (EXPECTED_INITIAL_STATES + EXPECTED_DAGGER_STATES) * 2:
        raise RuntimeError("Unexpected combined source-only teacher-state count")
    member_paths = []
    training = []
    for member in range(3):
        path = fold / "models" / f"member_{member}.pt"
        seed = derive_seed(PROTOCOL_ID, fold_seed, f"final_member|{member}")
        training.append(train_final(combined, path, seed, member))
        member_paths.append(path)
        update_state("RUNNING", stage="members", heldout=heldout, completed_member=member)

    shadow_path = fold / "models" / "SHADOW_SELECTION.json"
    if shadow_path.is_file():
        shadow = json.loads(shadow_path.read_text(encoding="utf-8"))
    else:
        results = []
        for member, path in enumerate(member_paths):
            model = load_model(path, "cuda:0")
            means = {}
            for name in sources:
                per_query = fold / "shadow" / f"member_{member}_{name}_PER_QUERY.csv"
                if per_query.is_file():
                    rows = read_rows(per_query)
                else:
                    require_time(900, f"shadow member {member} source {name}")
                    rng = np.random.default_rng(derive_seed(PROTOCOL_ID, fold_seed, f"shadow_objects|{name}"))
                    n = min(DEFAULT_CONFIG.shadow_rows_per_domain, len(source_data[name]))
                    idx = rng.choice(len(source_data[name]), size=n, replace=False) if n < len(source_data[name]) else np.arange(n)
                    subset = source_data[name][idx]
                    workroot = teacher_root / "workloads" / name
                    tree, _ = build_neural_tree(subset, np.load(workroot / "construction_boxes.npy"), model,
                                                 "cuda:0", 128, DEFAULT_CONFIG.hist_bins,
                                                 DEFAULT_CONFIG.local_query_cap)
                    rows = _evaluate_python_method(tree, load_query_suite(workroot, "validation"), n, 128,
                                                   f"member_{member}")
                    write_rows(per_query, rows)
                    del tree
                means[name] = float(np.mean([float(row["total_node_accesses"]) for row in rows]))
                update_state("RUNNING", stage="shadow", heldout=heldout, member=member, source=name)
            results.append(means)
            del model
            torch.cuda.empty_cache()
        best = {name: min(result[name] for result in results) for name in sources}
        scores = []
        for member, result in enumerate(results):
            ratios = [result[name] / best[name] for name in sources]
            scores.append({"member": member, "means": result, "source_domain_ratios": dict(zip(sources, ratios)),
                           "score": max(ratios) + 0.25 * float(np.mean(ratios))})
        selected = min(scores, key=lambda item: item["score"])["member"]
        shadow = {"selected_member": selected, "members": scores,
                  "selection_rule": "source-domain validation only; held-out data, PLATON, and final queries excluded"}
        write_json(shadow_path, shadow)
    selected = int(shadow["selected_member"])
    frozen.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(member_paths[selected], frozen)
    state = {
        "protocol_id": PROTOCOL_ID, "status": "FROZEN", "heldout": heldout,
        "source_domains": sources, "source_dataset_metadata": source_meta,
        "initial_state_hashes": {name: INITIAL_HASHES[name] for name in sources},
        "reused_all_domain_dagger_states": False, "fold_specific_source_only_dagger": True,
        "teacher_states": int(len(combined["x"])), "selected_member": selected,
        "selected_checkpoint_sha256": sha256_file(frozen), "training": training,
        "selection": shadow, "heldout_data_loaded_before_freeze": False,
        "heldout_final_queries_loaded_before_freeze": False, "frozen_unix": time.time(),
    }
    write_json(fold_state, state)
    del source_data, initial, combined
    gc.collect()
    torch.cuda.empty_cache()
    update_state("RUNNING", stage="fold_frozen", heldout=heldout, checkpoint_sha256=state["selected_checkpoint_sha256"])
    return state


def locate_baseline_root(dataset: str, capacity: int) -> Path:
    matches = []
    for marker in INPUT_ROOT.rglob("FULLSCALE_STATE.json"):
        try:
            raw = json.loads(marker.read_text(encoding="utf-8"))
        except Exception:
            continue
        root = marker.parent
        if raw.get("dataset") == dataset and int(raw.get("capacity", -1)) == capacity and \
                all((root / "methods" / method / "PER_QUERY.csv").is_file() for method in ("PLATON", "STR", "TGS")):
            matches.append(root)
    if len(matches) != 1:
        raise RuntimeError(f"Expected one baseline root for {dataset} B{capacity}, found {matches}")
    return matches[0]


def locate_all_domain_root(dataset: str) -> Path:
    matches = []
    for marker in INPUT_ROOT.rglob("EVALUATION_STATE.json"):
        try:
            raw = json.loads(marker.read_text(encoding="utf-8"))
        except Exception:
            continue
        if raw.get("protocol_id") == ALL_DOMAIN_PROTOCOL_ID and raw.get("dataset") == dataset and raw.get("status") == "COMPLETE":
            matches.append(marker.parent)
    if len(matches) != 1:
        raise RuntimeError(f"Expected one all-domain evaluation root for {dataset}, found {matches}")
    return matches[0]


def validate_reference_artifacts(root: Path, dataset: str, capacity: int, object_hash: str) -> dict[str, list[dict[str, str]]]:
    state = json.loads((root / "FULLSCALE_STATE.json").read_text(encoding="utf-8"))
    metadata = json.loads((root / "DATASET.json").read_text(encoding="utf-8"))
    manifest = json.loads((root / "workload" / "WORKLOAD_MANIFEST.json").read_text(encoding="utf-8"))
    if state.get("dataset") != dataset or int(state.get("capacity", -1)) != capacity:
        raise RuntimeError("Baseline dataset/capacity mismatch")
    if metadata.get("normalized_object_sha256") != object_hash:
        raise RuntimeError(f"Object hash mismatch for {dataset} B{capacity}")
    if manifest.get("namespace") != QUERY_NAMESPACE:
        raise RuntimeError(f"Query namespace mismatch for {dataset} B{capacity}")
    result = {}
    for method in ("PLATON", "STR", "TGS"):
        rows = read_rows(root / "methods" / method / "PER_QUERY.csv")
        if len(rows) != EXPECTED_QUERIES or len({row["query_uid"] for row in rows}) != EXPECTED_QUERIES:
            raise RuntimeError(f"Bad {method} trace for {dataset} B{capacity}")
        result[method] = rows
    return result


def construction_seconds(path: Path) -> float:
    raw = json.loads(path.read_text(encoding="utf-8"))
    for key in ("total_construction_seconds", "seconds", "build_seconds"):
        if key in raw:
            return float(raw[key])
    return 0.0


def access_map(rows: list[dict[str, Any]]) -> dict[str, float]:
    return {query_key(row): float(row["total_node_accesses"]) for row in rows}


def query_key(row: dict[str, Any]) -> str:
    if str(row["query_type"]) == "knn":
        within_k = int(row["query_index"]) % KNN_QUERIES_PER_K
        return f"knn|k{int(float(row['k']))}|{within_k:05d}"
    return str(row["query_uid"])


def bootstrap_ratio(left: np.ndarray, right: np.ndarray, seed: int, draws: int = 500) -> tuple[float, float, float]:
    ratio = float(left.sum() / right.sum())
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(draws):
        idx = rng.integers(0, len(left), size=len(left))
        values.append(float(left[idx].sum() / right[idx].sum()))
    return ratio, float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def compare_rows(left_rows: list[dict[str, Any]], right_rows: list[dict[str, Any]], dataset: str,
                 capacity: int, right_method: str, family: str = "all") -> dict[str, Any]:
    left = {query_key(row): row for row in left_rows if family == "all" or row["query_type"] == family}
    right = {query_key(row): row for row in right_rows if family == "all" or row["query_type"] == family}
    if set(left) != set(right):
        raise RuntimeError(f"Unpaired query traces for {dataset} B{capacity} {right_method} {family}")
    keys = sorted(left)
    la = np.asarray([float(left[key]["total_node_accesses"]) for key in keys])
    ra = np.asarray([float(right[key]["total_node_accesses"]) for key in keys])
    ratio, low, high = bootstrap_ratio(la, ra, derive_seed(PROTOCOL_ID, BASE_SEED,
                                                          f"ci|{dataset}|{capacity}|{right_method}|{family}"))
    return {"dataset": dataset, "capacity": capacity, "query_family": family,
            "left_method": "WAHARP_LODO", "right_method": right_method, "queries": len(keys),
            "left_total_accesses": int(la.sum()), "right_total_accesses": int(ra.sum()),
            "access_ratio": ratio, "bootstrap_ci_low": low, "bootstrap_ci_high": high,
            "strict_wins": int(np.sum(la < ra)), "ties": int(np.sum(la == ra)),
            "losses": int(np.sum(la > ra)), "strict_win_rate": float(np.mean(la < ra))}


def correctness(left_rows: list[dict[str, Any]], reference_rows: list[dict[str, Any]]) -> dict[str, int | bool]:
    ref = {query_key(row): row for row in reference_rows}
    rp = 0
    knn = 0
    for row in left_rows:
        other = ref[query_key(row)]
        if row["query_type"] in ("range", "point"):
            got = (int(row["result_count"]), int(row["id_sum"]), int(row["id_xor"]))
            expected = (int(other["result_count"]), int(other["id_sum"]), int(other["id_xor"]))
            rp += int(got != expected)
        else:
            got = float(row["kth_distance"])
            expected = float(other["kth_distance"])
            knn += int(not math.isclose(got, expected, rel_tol=2e-6, abs_tol=2e-6))
    return {"range_point_mismatches": rp, "knn_distance_mismatches": knn, "passed": rp == 0 and knn == 0}


def evaluate_cell(dataset: str, capacity: int, rects: np.ndarray, metadata: dict[str, Any],
                  checkpoint: Path) -> dict[str, Any]:
    cell = OUTPUT_ROOT / "evaluation" / dataset / f"B{capacity}"
    per_query_path = cell / "WAHARP_LODO_PER_QUERY.csv"
    construction_path = cell / "WAHARP_LODO_CONSTRUCTION.json"
    if per_query_path.is_file() and construction_path.is_file():
        rows = read_rows(per_query_path)
        if len(rows) != EXPECTED_QUERIES:
            raise RuntimeError(f"Bad resumed LODO trace: {per_query_path}")
        return {"rows": rows, "construction": json.loads(construction_path.read_text(encoding="utf-8"))}
    require_time(1200, f"evaluation {dataset} B{capacity}")
    baseline_root = locate_baseline_root(dataset, capacity)
    validate_reference_artifacts(baseline_root, dataset, capacity, metadata["normalized_object_sha256"])
    workroot = baseline_root / "workload"
    model = load_model(checkpoint, "cuda:0")
    tree, diagnostics = build_neural_tree(rects, np.load(workroot / "construction_boxes.npy"), model,
                                          "cuda:0", capacity, DEFAULT_CONFIG.hist_bins,
                                          DEFAULT_CONFIG.local_query_cap)
    rows = _evaluate_python_method(tree, load_query_suite(workroot, "final"), len(rects), capacity, "WAHARP_LODO")
    if len(rows) != EXPECTED_QUERIES:
        raise RuntimeError(f"Expected {EXPECTED_QUERIES} queries, got {len(rows)}")
    cell.mkdir(parents=True, exist_ok=True)
    write_rows(per_query_path, rows)
    diagnostics.update({"method": "WAHARP_LODO", "heldout_dataset": dataset,
                        "capacity": capacity, "checkpoint_sha256": sha256_file(checkpoint)})
    write_json(construction_path, diagnostics)
    del tree, model
    gc.collect()
    torch.cuda.empty_cache()
    return {"rows": rows, "construction": diagnostics}


def make_plot(results: list[dict[str, Any]], construction: list[dict[str, Any]]) -> None:
    import matplotlib.pyplot as plt

    ratios = {(row["dataset"], int(row["capacity"]), row["right_method"]): float(row["access_ratio"])
              for row in results if row["query_family"] == "all"}
    times = {(row["dataset"], int(row["capacity"]), row["method"]): float(row["seconds"])
             for row in construction}
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
    for axis, dataset in zip(axes, DOMAINS):
        x = [times[(dataset, cap, "WAHARP_LODO")] for cap in CAPACITIES]
        y = [ratios[(dataset, cap, "PLATON")] for cap in CAPACITIES]
        axis.plot(x, y, "o-", label="WAHARP LODO", linewidth=2)
        x_all = [times[(dataset, cap, "WAHARP_ALL_DOMAIN")] for cap in CAPACITIES]
        y_all = [ratios[(dataset, cap, "PLATON")] /
                 ratios[(dataset, cap, "WAHARP_ALL_DOMAIN")] for cap in CAPACITIES]
        axis.plot(x_all, y_all, "s--", label="WAHARP all-domain")
        x_platon = [times[(dataset, cap, "PLATON")] for cap in CAPACITIES]
        axis.plot(x_platon, [1.0] * 3, "^:", label="PLATON")
        axis.axhline(1.0, color="black", linewidth=0.8)
        axis.set_title(dataset.title())
        axis.set_xlabel("Construction time (s)")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Logical accesses / PLATON")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUTPUT_ROOT / "ZERO_SHOT_GENERALIZATION.png", dpi=180)
    plt.close(fig)


def write_report(results: list[dict[str, Any]], correctness_rows: list[dict[str, Any]],
                 fold_states: list[dict[str, Any]]) -> None:
    lines = ["# WAHARP Leave-One-Dataset-Out Generalization", "",
             f"Protocol: `{PROTOCOL_ID}`", "",
             "Three source-only models were trained. For each fold, the held-out dataset was excluded from initial states, "
             "bootstrap training, fold-specific DAgger, final training, and member selection. The checkpoint was frozen "
             "before held-out data and final queries were loaded.", "", "## Overall logical node accesses", "",
             "| Held-out dataset | B | LODO / PLATON | 95% CI | LODO / all-domain | Wins / ties / losses vs PLATON |",
             "|---|---:|---:|---:|---:|---:|"]
    all_rows = [row for row in results if row["query_family"] == "all"]
    for dataset in DOMAINS:
        for capacity in CAPACITIES:
            platon = next(row for row in all_rows if row["dataset"] == dataset and row["capacity"] == capacity and row["right_method"] == "PLATON")
            full = next(row for row in all_rows if row["dataset"] == dataset and row["capacity"] == capacity and row["right_method"] == "WAHARP_ALL_DOMAIN")
            lines.append(f"| {dataset.title()} | {capacity} | {platon['access_ratio']:.4f} | "
                         f"[{platon['bootstrap_ci_low']:.4f}, {platon['bootstrap_ci_high']:.4f}] | "
                         f"{full['access_ratio']:.4f} | {platon['strict_wins']} / {platon['ties']} / {platon['losses']} |")
    lines += ["", "## Integrity", "",
              f"- Dual-T4 DataParallel was enforced for training.",
              f"- Reused all-domain DAgger states: **No**.",
              f"- Correctness mismatches across all cells: **{sum(int(r['range_point_mismatches']) + int(r['knn_distance_mismatches']) for r in correctness_rows)}**.",
              f"- Frozen folds: **{len(fold_states)}/3**.",
              "- Each cell uses the exact paper workload of 6,000 paired queries and the existing baseline trace.", ""]
    (OUTPUT_ROOT / "FINAL_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    restore_resume()
    names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    if torch.cuda.device_count() != 2 or any("T4" not in name.upper() for name in names):
        raise RuntimeError(f"This protocol requires exactly two NVIDIA T4 GPUs; found {names}")
    seed_everything(BASE_SEED)
    update_state("RUNNING", stage="preflight")
    teacher_root, initial_paths = locate_teacher_root()

    fold_states = []
    for heldout in DOMAINS:
        fold_states.append(train_fold(heldout, teacher_root, initial_paths))
    if len(fold_states) != 3 or any(state.get("heldout_data_loaded_before_freeze") is not False for state in fold_states):
        raise RuntimeError("Not all leakage-free folds were frozen")
    update_state("RUNNING", stage="all_folds_frozen", all_folds_frozen_before_final_evaluation=True)

    results: list[dict[str, Any]] = []
    family_results: list[dict[str, Any]] = []
    correctness_rows: list[dict[str, Any]] = []
    construction_rows: list[dict[str, Any]] = []
    for dataset in DOMAINS:
        rects, metadata = load_evaluation_domain(dataset)
        checkpoint = OUTPUT_ROOT / "folds" / f"heldout_{dataset}" / "frozen_model" / "selected.pt"
        all_domain_root = locate_all_domain_root(dataset)
        for capacity in CAPACITIES:
            baseline_root = locate_baseline_root(dataset, capacity)
            baselines = validate_reference_artifacts(baseline_root, dataset, capacity,
                                                     metadata["normalized_object_sha256"])
            all_domain_path = all_domain_root / f"B{capacity}" / "methods" / "NewNeural_075_025" / "PER_QUERY.csv"
            all_domain = read_rows(all_domain_path)
            if len(all_domain) != EXPECTED_QUERIES:
                raise RuntimeError(f"Bad all-domain trace: {all_domain_path}")
            evaluated = evaluate_cell(dataset, capacity, rects, metadata, checkpoint)
            zero = evaluated["rows"]
            expected_uids = {query_key(row) for row in baselines["PLATON"]}
            if {query_key(row) for row in zero} != expected_uids or {query_key(row) for row in all_domain} != expected_uids:
                raise RuntimeError(f"Query pairing failure for {dataset} B{capacity}")
            references = {**baselines, "WAHARP_ALL_DOMAIN": all_domain}
            for method, rows in references.items():
                overall = compare_rows(zero, rows, dataset, capacity, method)
                results.append(overall)
                for family in ("range", "point", "knn"):
                    family_results.append(compare_rows(zero, rows, dataset, capacity, method, family))
            corr = correctness(zero, baselines["PLATON"])
            correctness_rows.append({"dataset": dataset, "capacity": capacity, "queries": EXPECTED_QUERIES, **corr})
            construction_rows.append({"dataset": dataset, "capacity": capacity, "method": "WAHARP_LODO",
                                      "seconds": float(evaluated["construction"]["build_seconds"])})
            construction_rows.append({"dataset": dataset, "capacity": capacity, "method": "WAHARP_ALL_DOMAIN",
                                      "seconds": construction_seconds(all_domain_root / f"B{capacity}" / "methods" / "NewNeural_075_025" / "CONSTRUCTION.json")})
            for method in ("PLATON", "STR", "TGS"):
                construction_rows.append({"dataset": dataset, "capacity": capacity, "method": method,
                                          "seconds": construction_seconds(baseline_root / "methods" / method / "CONSTRUCTION.json")})
            update_state("RUNNING", stage="evaluation", completed_dataset=dataset, completed_capacity=capacity)
        del rects
        gc.collect()
    if any(not bool(row["passed"]) for row in correctness_rows):
        raise RuntimeError(f"Correctness failure: {correctness_rows}")
    write_rows(OUTPUT_ROOT / "LODO_RESULTS.csv", results)
    write_rows(OUTPUT_ROOT / "LODO_BY_QUERY_FAMILY.csv", family_results)
    write_rows(OUTPUT_ROOT / "CORRECTNESS.csv", correctness_rows)
    write_rows(OUTPUT_ROOT / "CONSTRUCTION_TIMES.csv", construction_rows)
    fold_summary = [{key: state[key] for key in ("heldout", "source_domains", "teacher_states", "selected_member",
                                                  "selected_checkpoint_sha256", "heldout_data_loaded_before_freeze",
                                                  "heldout_final_queries_loaded_before_freeze")} for state in fold_states]
    write_rows(OUTPUT_ROOT / "FOLD_TRAINING.csv", fold_summary)
    make_plot(results, construction_rows)
    write_report(results, correctness_rows, fold_states)
    update_state("COMPLETE", stage="complete", all_folds_frozen_before_final_evaluation=True,
                 cells=9, queries_per_cell=EXPECTED_QUERIES, correctness_mismatches=0,
                 reused_all_domain_dagger_states=False)
    print((OUTPUT_ROOT / "FINAL_REPORT.md").read_text(encoding="utf-8"), flush=True)


if __name__ == "__main__":
    try:
        main()
    except GracefulPause as exc:
        OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
        update_state("PAUSED_RESUME_REQUIRED", reason=str(exc), resume_instruction="Attach this kernel output and rerun unchanged")
        print(f"GRACEFUL PAUSE: {exc}", flush=True)
    except Exception as exc:
        OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
        update_state("FAILED", error_type=type(exc).__name__, error=str(exc))
        raise
