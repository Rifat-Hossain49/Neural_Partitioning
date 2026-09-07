from __future__ import annotations

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


PROTOCOL_ID = "waharp_loss075_final_train_v1_20260903"
SOURCE_PROTOCOL_ID = "waharp_realtrain_v1_three_domain_pageio_20260829"
SOURCE_FINGERPRINT = "6cb1018bb2c849e76b7a8caa4a86c7c0e1eb7b08e19a9801bd53d899852ce388"
ABLATION_PROTOCOL_ID = "waharp_loss_weight_ablation_v1_20260902"
BASE_SEED = 2026082951
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
EXPECTED_STATES = 30_000
EXPECTED_ACTIONS = 80
EXPECTED_STATES_PER_DOMAIN = 10_000
INPUT_ROOT = Path("/kaggle/input")
OUTPUT_ROOT = Path("/kaggle/working/WAHARP_LOSS075_FINAL_TRAIN")
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
os.environ["PYTHONUNBUFFERED"] = "1"


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


def sha256_indices(indices: np.ndarray) -> str:
    values = np.ascontiguousarray(indices, dtype=np.int64)
    digest = hashlib.sha256()
    digest.update(str(values.shape).encode("ascii"))
    digest.update(values.view(np.uint8))
    return digest.hexdigest()


def locate_code_root() -> Path:
    matches: list[Path] = []
    for marker_path in INPUT_ROOT.rglob("WAHARP_REALTRAIN_V1_BUNDLE_MARKER.json"):
        try:
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if (
            marker.get("protocol_id") == SOURCE_PROTOCOL_ID
            and marker.get("code_fingerprint") == SOURCE_FINGERPRINT
        ):
            matches.append(marker_path.parent)
    if len(matches) != 1:
        raise RuntimeError(f"Expected one matching V1 code bundle, found {matches}")
    return matches[0]


CODE_ROOT = locate_code_root()
sys.path.insert(0, str(CODE_ROOT))

from realtrain.config import DEFAULT_CONFIG  # noqa: E402
from realtrain.data import load_all_domains  # noqa: E402
from realtrain.model import StateQNet, unwrap_model  # noqa: E402
from realtrain.pipeline import DOMAIN_NAMES, _select_member  # noqa: E402
from realtrain.teacher import load_records  # noqa: E402
from realtrain.utils import derive_seed, seed_everything  # noqa: E402


def locate_teacher_paths() -> list[Path]:
    expected = {
        f"{phase}_{domain}.npz"
        for phase in ("initial", "dagger")
        for domain in DOMAIN_NAMES
    }
    candidates: dict[str, list[Path]] = {name: [] for name in expected}
    for path in INPUT_ROOT.rglob("*.npz"):
        if path.name in candidates and CODE_ROOT not in path.parents:
            candidates[path.name].append(path)
    ambiguous = {name: paths for name, paths in candidates.items() if len(paths) != 1}
    if ambiguous:
        diagnostic = {name: [str(path) for path in paths] for name, paths in candidates.items()}
        raise RuntimeError(f"Teacher-state discovery failed: {diagnostic}")
    return [candidates[name][0] for name in sorted(expected)]


def locate_source_root(teacher_paths: list[Path]) -> Path:
    roots = {path.parent.parent for path in teacher_paths}
    if len(roots) != 1:
        raise RuntimeError(f"Teacher files do not share one source root: {roots}")
    root = next(iter(roots))
    marker = json.loads((root / "STATE.json").read_text(encoding="utf-8"))
    if marker.get("protocol_id") != SOURCE_PROTOCOL_ID:
        raise RuntimeError(f"Unexpected source state: {marker}")
    for domain in DOMAIN_NAMES:
        manifest = root / "workloads" / domain / "WORKLOAD_MANIFEST.json"
        if not manifest.is_file():
            raise FileNotFoundError(manifest)
    return root


def verify_ablation_decision() -> dict[str, Any]:
    matches = []
    for path in INPUT_ROOT.rglob("FINAL_DECISION.json"):
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if raw.get("protocol_id") == ABLATION_PROTOCOL_ID:
            matches.append((path, raw))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one ablation decision, found {[str(x[0]) for x in matches]}")
    path, decision = matches[0]
    if not math.isclose(float(decision["selected_list_weight"]), LIST_WEIGHT):
        raise RuntimeError(f"Ablation did not select listwise weight {LIST_WEIGHT}: {decision}")
    if not math.isclose(float(decision["selected_classification_weight"]), CLASSIFICATION_WEIGHT):
        raise RuntimeError(f"Ablation did not select classification weight {CLASSIFICATION_WEIGHT}: {decision}")
    return {"path": str(path), "sha256": sha256_file(path), "decision": decision}


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
        validation_count = max(1, int(round(len(indices) * VAL_FRACTION)))
        validation.extend(indices[:validation_count].tolist())
        train.extend(indices[validation_count:].tolist())
    return np.asarray(train, dtype=np.int64), np.asarray(validation, dtype=np.int64)


def loss_batch(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    mask = torch.isfinite(target)
    safe_target = torch.nan_to_num(target, nan=0.0)
    regression = nn.functional.smooth_l1_loss(prediction[mask], safe_target[mask])
    big = torch.tensor(1e6, device=prediction.device, dtype=prediction.dtype)
    prediction_masked = torch.where(mask, prediction, big)
    target_logits = torch.where(mask, -safe_target / TEMPERATURE, -big)
    target_probability = torch.softmax(target_logits, dim=1)
    prediction_log_probability = torch.log_softmax(-prediction_masked, dim=1)
    listwise = -(target_probability * prediction_log_probability).sum(dim=1).mean()
    best = torch.argmin(torch.where(mask, safe_target, big), dim=1)
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
def evaluate(model: nn.Module, loader: DataLoader, device: str) -> dict[str, Any]:
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
        mask = torch.isfinite(target)
        big = torch.tensor(1e6, device=prediction.device, dtype=prediction.dtype)
        chosen = torch.argmin(torch.where(mask, prediction, big), dim=1)
        best = torch.argmin(torch.where(mask, target, big), dim=1)
        selected = target.gather(1, chosen[:, None]).squeeze(1).detach().cpu().numpy()
        domains = domain.numpy()
        selected_regrets.extend(selected.tolist())
        exact_best.extend((chosen == best).float().detach().cpu().numpy().tolist())
        for domain_id, regret in zip(domains, selected):
            by_domain.setdefault(int(domain_id), []).append(float(regret))
    domain_means = {str(key): float(np.mean(values)) for key, values in by_domain.items()}
    mean_regret = float(np.mean(selected_regrets))
    worst_domain = max(domain_means.values())
    return {
        "loss": float(np.mean(losses)),
        "mean_selected_regret": mean_regret,
        "p95_selected_regret": float(np.quantile(selected_regrets, 0.95)),
        "worst_domain_regret": worst_domain,
        "selection_score": worst_domain + 0.25 * mean_regret,
        "exact_best_action_accuracy": float(np.mean(exact_best)),
        "domain_mean_selected_regret": domain_means,
    }


def train_one(
    arrays: dict[str, np.ndarray],
    output_path: Path,
    seed: int,
    member: int,
) -> dict[str, Any]:
    seed_everything(seed)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    train_indices, validation_indices = split_indices(arrays, seed)
    generator = torch.Generator()
    generator.manual_seed(seed)
    train_loader = DataLoader(
        StateDataset(arrays, train_indices),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
    )
    validation_loader = DataLoader(
        StateDataset(arrays, validation_indices),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )
    base = StateQNet(int(arrays["x"].shape[1]), EXPECTED_ACTIONS)
    model: nn.Module = base.to(device)
    gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if gpu_count >= 2:
        model = nn.DataParallel(model, device_ids=list(range(gpu_count)))
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
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
        for features, target, _ in train_loader:
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
        validation = evaluate(model, validation_loader, device)
        score = float(validation["selection_score"])
        record = {"epoch": epoch, "train_loss": float(np.mean(training_losses)), **validation}
        history.append(record)
        print(
            f"member={member} seed={seed} epoch={epoch:02d} "
            f"train={record['train_loss']:.6f} mean_regret={validation['mean_selected_regret']:.6f} "
            f"worst={validation['worst_domain_regret']:.6f} score={score:.6f}",
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
                    "loss": {
                        "regression_weight": REGRESSION_WEIGHT,
                        "list_weight": LIST_WEIGHT,
                        "classification_weight": CLASSIFICATION_WEIGHT,
                        "temperature": TEMPERATURE,
                    },
                },
                output_path,
            )
        else:
            stale += 1
            if stale >= PATIENCE:
                break
    if best_validation is None:
        raise RuntimeError("No checkpoint was selected")
    metadata = {
        "member": member,
        "seed": seed,
        "best_epoch": best_epoch,
        "epochs_completed": len(history),
        "best_score": best_score,
        "best_validation": best_validation,
        "history": history,
        "seconds": time.perf_counter() - started,
        "train_states": int(len(train_indices)),
        "validation_states": int(len(validation_indices)),
        "train_indices_sha256": sha256_indices(train_indices),
        "validation_indices_sha256": sha256_indices(validation_indices),
        "gpu_count": gpu_count,
        "data_parallel": bool(gpu_count >= 2),
        "checkpoint_sha256": sha256_file(output_path),
    }
    write_json(output_path.with_suffix(".training.json"), metadata)
    return metadata


def main() -> None:
    started = time.perf_counter()
    write_json(
        OUTPUT_ROOT / "FINAL_TRAINING_STATE.json",
        {"protocol_id": PROTOCOL_ID, "status": "RUNNING", "started_unix": time.time()},
    )
    ablation = verify_ablation_decision()
    teacher_paths = locate_teacher_paths()
    source_root = locate_source_root(teacher_paths)
    arrays = load_records(teacher_paths)
    domain_counts = {
        str(domain): int(np.sum(arrays["domain"] == domain))
        for domain in sorted(np.unique(arrays["domain"]))
    }
    if int(arrays["x"].shape[0]) != EXPECTED_STATES:
        raise RuntimeError(f"Expected {EXPECTED_STATES} states, got {arrays['x'].shape[0]}")
    if int(arrays["regrets"].shape[1]) != EXPECTED_ACTIONS:
        raise RuntimeError(f"Expected {EXPECTED_ACTIONS} actions, got {arrays['regrets'].shape[1]}")
    if any(count != EXPECTED_STATES_PER_DOMAIN for count in domain_counts.values()):
        raise RuntimeError(f"Unexpected per-domain counts: {domain_counts}")

    seeds = [
        derive_seed(SOURCE_PROTOCOL_ID, BASE_SEED, f"final_member|{member}")
        for member in range(3)
    ]
    member_paths: list[Path] = []
    training: list[dict[str, Any]] = []
    for member, seed in enumerate(seeds):
        checkpoint = OUTPUT_ROOT / "models" / f"member_{member}.pt"
        member_paths.append(checkpoint)
        training.append(train_one(arrays, checkpoint, seed, member))
    write_json(OUTPUT_ROOT / "models" / "ENSEMBLE_TRAINING.json", {"members": training})

    print("Loading real domains for locked shadow member selection", flush=True)
    domains, dataset_metadata = load_all_domains(INPUT_ROOT, 1_000_000, 1_000_000, 0)
    workroots = {name: source_root / "workloads" / name for name in DOMAIN_NAMES}
    selected = _select_member(member_paths, domains, workroots, OUTPUT_ROOT, DEFAULT_CONFIG)
    selected_source = member_paths[selected]
    selected_target = OUTPUT_ROOT / "frozen_model" / "selected.pt"
    selected_target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(selected_source, selected_target)
    selected_hash = sha256_file(selected_target)

    provenance = {
        "protocol_id": PROTOCOL_ID,
        "source_protocol_id": SOURCE_PROTOCOL_ID,
        "source_code_fingerprint": SOURCE_FINGERPRINT,
        "ablation": ablation,
        "teacher_files": [
            {"name": path.name, "sha256": sha256_file(path), "bytes": path.stat().st_size}
            for path in teacher_paths
        ],
        "state_count": int(arrays["x"].shape[0]),
        "feature_dim": int(arrays["x"].shape[1]),
        "action_count": int(arrays["regrets"].shape[1]),
        "domain_counts": domain_counts,
        "member_seeds": seeds,
        "loss": {
            "regression_weight": REGRESSION_WEIGHT,
            "list_weight": LIST_WEIGHT,
            "classification_weight": CLASSIFICATION_WEIGHT,
            "temperature": TEMPERATURE,
        },
        "training": {
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "patience": PATIENCE,
            "validation_fraction": VAL_FRACTION,
        },
        "selection": {
            "selected_member": selected,
            "checkpoint_sha256": selected_hash,
            "rule": "locked three-domain shadow validation access; PLATON and final queries excluded",
            "shadow_selection_sha256": sha256_file(OUTPUT_ROOT / "models" / "SHADOW_SELECTION.json"),
        },
        "datasets": dataset_metadata,
        "final_queries_read": False,
        "elapsed_seconds": time.perf_counter() - started,
    }
    write_json(OUTPUT_ROOT / "PROVENANCE.json", provenance)
    final_state = {
        "protocol_id": PROTOCOL_ID,
        "status": "COMPLETE",
        "selected_member": selected,
        "selected_checkpoint": "frozen_model/selected.pt",
        "selected_checkpoint_sha256": selected_hash,
        "list_weight": LIST_WEIGHT,
        "classification_weight": CLASSIFICATION_WEIGHT,
        "final_queries_read": False,
        "elapsed_seconds": time.perf_counter() - started,
    }
    write_json(OUTPUT_ROOT / "FINAL_TRAINING_STATE.json", final_state)
    print(json.dumps(final_state, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
