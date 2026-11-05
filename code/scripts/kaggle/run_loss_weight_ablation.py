from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


PROTOCOL_ID = "waharp_loss_weight_ablation_v1_20260902"
SOURCE_PROTOCOL_ID = "waharp_realtrain_v1_three_domain_pageio_20260829"
SOURCE_FINGERPRINT = "6cb1018bb2c849e76b7a8caa4a86c7c0e1eb7b08e19a9801bd53d899852ce388"
BASE_SEED = 2026082951
LIST_WEIGHTS = (0.00, 0.25, 0.50, 0.60, 0.75, 1.00)
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
OUTPUT_ROOT = Path("/kaggle/working/WAHARP_LOSS_WEIGHT_ABLATION_V1")
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
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=json_default),
        encoding="utf-8",
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
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

from realtrain.model import StateQNet, unwrap_model  # noqa: E402
from realtrain.teacher import load_records  # noqa: E402
from realtrain.utils import derive_seed, seed_everything  # noqa: E402


def locate_teacher_paths() -> list[Path]:
    expected = {
        f"{phase}_{domain}.npz"
        for phase in ("initial", "dagger")
        for domain in ("twitter", "crimes", "arizona")
    }
    candidates: dict[str, list[Path]] = {name: [] for name in expected}
    for path in INPUT_ROOT.rglob("*.npz"):
        if path.name in candidates and CODE_ROOT not in path.parents:
            candidates[path.name].append(path)
    missing = [name for name, paths in candidates.items() if not paths]
    ambiguous = {name: paths for name, paths in candidates.items() if len(paths) != 1}
    if missing or ambiguous:
        diagnostic = {name: [str(path) for path in paths] for name, paths in candidates.items()}
        raise RuntimeError(
            f"Teacher-state discovery failed; missing={missing}, ambiguous={ambiguous}, all={diagnostic}"
        )
    return [candidates[name][0] for name in sorted(expected)]


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
    list_weight: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    class_weight = 1.0 - list_weight
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
    total = regression + list_weight * listwise + class_weight * classification
    return total, {
        "regression": regression,
        "listwise": listwise,
        "classification": classification,
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    list_weight: float,
) -> dict[str, Any]:
    model.eval()
    losses: list[float] = []
    regressions: list[float] = []
    listwise_losses: list[float] = []
    classification_losses: list[float] = []
    selected_regrets: list[float] = []
    exact_best: list[float] = []
    zero_regret: list[float] = []
    by_domain: dict[int, list[float]] = {}

    for features, target, domain in loader:
        features = features.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        prediction = model(features)
        loss, components = loss_batch(prediction, target, list_weight)
        losses.append(float(loss))
        regressions.append(float(components["regression"]))
        listwise_losses.append(float(components["listwise"]))
        classification_losses.append(float(components["classification"]))

        mask = torch.isfinite(target)
        big = torch.tensor(1e6, device=prediction.device, dtype=prediction.dtype)
        chosen = torch.argmin(torch.where(mask, prediction, big), dim=1)
        best = torch.argmin(torch.where(mask, target, big), dim=1)
        selected = target.gather(1, chosen[:, None]).squeeze(1)
        selected_cpu = selected.detach().cpu().numpy()
        domain_cpu = domain.numpy()
        selected_regrets.extend(selected_cpu.tolist())
        exact_best.extend((chosen == best).float().detach().cpu().numpy().tolist())
        zero_regret.extend((selected <= 1e-7).float().detach().cpu().numpy().tolist())
        for domain_id, regret in zip(domain_cpu, selected_cpu):
            by_domain.setdefault(int(domain_id), []).append(float(regret))

    domain_means = {str(key): float(np.mean(values)) for key, values in by_domain.items()}
    mean_selected = float(np.mean(selected_regrets))
    worst_domain = max(domain_means.values())
    selection_score = worst_domain + 0.25 * mean_selected
    return {
        "loss": float(np.mean(losses)),
        "regression_loss": float(np.mean(regressions)),
        "listwise_loss": float(np.mean(listwise_losses)),
        "classification_loss": float(np.mean(classification_losses)),
        "mean_selected_regret": mean_selected,
        "p95_selected_regret": float(np.quantile(selected_regrets, 0.95)),
        "worst_domain_regret": worst_domain,
        "selection_score": selection_score,
        "exact_best_action_accuracy": float(np.mean(exact_best)),
        "optimal_action_rate": float(np.mean(zero_regret)),
        "domain_mean_selected_regret": domain_means,
    }


def train_one(
    arrays: dict[str, np.ndarray],
    list_weight: float,
    seed: int,
    run_index: int,
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

    base = StateQNet(int(arrays["x"].shape[1]), EXPECTED_ACTIONS)
    model: nn.Module = base.to(device)
    gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if gpu_count >= 2:
        model = nn.DataParallel(model, device_ids=list(range(gpu_count)))
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

    for epoch in range(1, EPOCHS + 1):
        model.train()
        training_losses: list[float] = []
        for features, target, _ in train_loader:
            features = features.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                prediction = model(features)
                loss, _ = loss_batch(prediction, target, list_weight)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            training_losses.append(float(loss.detach()))

        validation = evaluate(model, validation_loader, device, list_weight)
        record = {
            "epoch": epoch,
            "train_loss": float(np.mean(training_losses)),
            **validation,
        }
        history.append(record)
        score = float(validation["selection_score"])
        print(
            f"run={run_index:02d} lambda_list={list_weight:.2f} seed={seed} "
            f"epoch={epoch:02d} train={record['train_loss']:.6f} "
            f"mean_regret={validation['mean_selected_regret']:.6f} "
            f"worst={validation['worst_domain_regret']:.6f} "
            f"score={score:.6f}",
            flush=True,
        )
        if score < best_score - 1e-6:
            best_score = score
            best_epoch = epoch
            best_validation = validation
            stale = 0
        else:
            stale += 1
            if stale >= PATIENCE:
                break

    assert best_validation is not None
    result = {
        "protocol_id": PROTOCOL_ID,
        "source_protocol_id": SOURCE_PROTOCOL_ID,
        "run_index": run_index,
        "seed": seed,
        "list_weight": list_weight,
        "classification_weight": 1.0 - list_weight,
        "regression_weight": 1.0,
        "temperature": TEMPERATURE,
        "best_epoch": best_epoch,
        "epochs_completed": len(history),
        "seconds": time.perf_counter() - started,
        "train_states": int(len(train_indices)),
        "validation_states": int(len(validation_indices)),
        "train_indices_sha256": sha256_indices(train_indices),
        "validation_indices_sha256": sha256_indices(validation_indices),
        "gpu_count": gpu_count,
        "data_parallel": bool(gpu_count >= 2),
        **best_validation,
        "history": history,
    }
    return result


def mean_std(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    return float(np.mean(array)), float(np.std(array, ddof=1)) if len(array) > 1 else 0.0


def aggregate(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    metrics = (
        "selection_score",
        "mean_selected_regret",
        "worst_domain_regret",
        "p95_selected_regret",
        "exact_best_action_accuracy",
        "optimal_action_rate",
        "best_epoch",
        "seconds",
    )
    for list_weight in LIST_WEIGHTS:
        selected_runs = [run for run in runs if math.isclose(run["list_weight"], list_weight)]
        if len(selected_runs) != 3:
            raise RuntimeError(f"Expected three runs for lambda={list_weight}, found {len(selected_runs)}")
        row: dict[str, Any] = {
            "list_weight": list_weight,
            "classification_weight": 1.0 - list_weight,
            "seeds": len(selected_runs),
        }
        for metric in metrics:
            mean_value, std_value = mean_std([float(run[metric]) for run in selected_runs])
            row[f"{metric}_mean"] = mean_value
            row[f"{metric}_std"] = std_value
        for domain_id in range(3):
            values = [float(run["domain_mean_selected_regret"][str(domain_id)]) for run in selected_runs]
            mean_value, std_value = mean_std(values)
            row[f"domain_{domain_id}_regret_mean"] = mean_value
            row[f"domain_{domain_id}_regret_std"] = std_value
        rows.append(row)
    return rows


def paired_against_current(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key = {(float(run["list_weight"]), int(run["seed"])): run for run in runs}
    seeds = sorted({int(run["seed"]) for run in runs})
    rows: list[dict[str, Any]] = []
    for list_weight in LIST_WEIGHTS:
        deltas = [
            float(by_key[(list_weight, seed)]["selection_score"])
            - float(by_key[(0.60, seed)]["selection_score"])
            for seed in seeds
        ]
        delta_mean, delta_std = mean_std(deltas)
        rows.append(
            {
                "list_weight": list_weight,
                "classification_weight": 1.0 - list_weight,
                "paired_selection_score_delta_vs_0.60_mean": delta_mean,
                "paired_selection_score_delta_vs_0.60_std": delta_std,
                "wins_vs_0.60": int(sum(delta < 0 for delta in deltas)),
                "ties_vs_0.60": int(sum(math.isclose(delta, 0.0, abs_tol=1e-12) for delta in deltas)),
                "losses_vs_0.60": int(sum(delta > 0 for delta in deltas)),
            }
        )
    return rows


def write_report(
    summary: list[dict[str, Any]],
    paired: list[dict[str, Any]],
    winner: dict[str, Any],
    provenance: dict[str, Any],
) -> None:
    lines = [
        "# WAHARP Loss-Weight Ablation",
        "",
        f"Protocol: `{PROTOCOL_ID}`",
        f"Source protocol: `{SOURCE_PROTOCOL_ID}`",
        "",
        "This is a validation-only ablation. No final query workload or test metric is read by this job.",
        "The Smooth-L1 coefficient is fixed at 1.0 and the temperature is fixed at 0.025.",
        "For each listwise weight lambda, the best-action coefficient is 1-lambda.",
        "Each configuration uses the same three paired seeds and the original stratified 80/20 splits.",
        "",
        "## Aggregate validation results",
        "",
        "| Listwise | Best-action | Selection score | Mean selected regret | Worst-domain regret | Exact-best accuracy | Optimal-action rate |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| {row['list_weight']:.2f} | {row['classification_weight']:.2f} | "
            f"{row['selection_score_mean']:.6f} +/- {row['selection_score_std']:.6f} | "
            f"{row['mean_selected_regret_mean']:.6f} +/- {row['mean_selected_regret_std']:.6f} | "
            f"{row['worst_domain_regret_mean']:.6f} +/- {row['worst_domain_regret_std']:.6f} | "
            f"{row['exact_best_action_accuracy_mean']:.4f} +/- {row['exact_best_action_accuracy_std']:.4f} | "
            f"{row['optimal_action_rate_mean']:.4f} +/- {row['optimal_action_rate_std']:.4f} |"
        )
    lines += [
        "",
        "## Selection",
        "",
        f"The validation-selected mixture is **{winner['list_weight']:.2f} listwise / "
        f"{winner['classification_weight']:.2f} best-action**, with aggregate selection score "
        f"{winner['selection_score_mean']:.6f} +/- {winner['selection_score_std']:.6f}.",
        "",
        "The selection score is worst-domain selected regret plus 0.25 times mean selected regret; lower is better.",
        "",
        "## Paired comparison with 0.60/0.40",
        "",
        "| Listwise | Best-action | Paired score delta | W/T/L |",
        "|---:|---:|---:|---:|",
    ]
    for row in paired:
        lines.append(
            f"| {row['list_weight']:.2f} | {row['classification_weight']:.2f} | "
            f"{row['paired_selection_score_delta_vs_0.60_mean']:+.6f} +/- "
            f"{row['paired_selection_score_delta_vs_0.60_std']:.6f} | "
            f"{row['wins_vs_0.60']}/{row['ties_vs_0.60']}/{row['losses_vs_0.60']} |"
        )
    lines += [
        "",
        "## Provenance",
        "",
        f"Teacher states: `{provenance['state_count']}`",
        f"Actions per state: `{provenance['action_count']}`",
        f"Domain counts: `{provenance['domain_counts']}`",
        f"GPU count: `{provenance['gpu_count']}`",
        "",
        "The result supports only validation-based coefficient selection. End-to-end final-query claims require a separately locked test evaluation.",
    ]
    (OUTPUT_ROOT / "ABLATION_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    started = time.perf_counter()
    teacher_paths = locate_teacher_paths()
    print("CODE_ROOT:", CODE_ROOT, flush=True)
    print("Teacher files:", [str(path) for path in teacher_paths], flush=True)
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

    member_seeds = [
        derive_seed(SOURCE_PROTOCOL_ID, BASE_SEED, f"final_member|{member}")
        for member in range(3)
    ]
    provenance = {
        "protocol_id": PROTOCOL_ID,
        "source_protocol_id": SOURCE_PROTOCOL_ID,
        "source_code_fingerprint": SOURCE_FINGERPRINT,
        "code_root": str(CODE_ROOT),
        "teacher_files": [
            {"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size}
            for path in teacher_paths
        ],
        "state_count": int(arrays["x"].shape[0]),
        "feature_dim": int(arrays["x"].shape[1]),
        "action_count": int(arrays["regrets"].shape[1]),
        "domain_counts": domain_counts,
        "member_seeds": member_seeds,
        "list_weights": list(LIST_WEIGHTS),
        "classification_weights": [1.0 - weight for weight in LIST_WEIGHTS],
        "regression_weight": 1.0,
        "temperature": TEMPERATURE,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "patience": PATIENCE,
        "validation_fraction": VAL_FRACTION,
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "gpu_names": [
            torch.cuda.get_device_name(index)
            for index in range(torch.cuda.device_count())
        ] if torch.cuda.is_available() else [],
        "test_or_final_queries_read": False,
    }
    write_json(OUTPUT_ROOT / "PROVENANCE.json", provenance)

    runs: list[dict[str, Any]] = []
    run_index = 0
    for list_weight in LIST_WEIGHTS:
        for seed in member_seeds:
            run_index += 1
            run_path = OUTPUT_ROOT / "runs" / f"lambda_{list_weight:.2f}_seed_{seed}.json"
            if run_path.is_file():
                result = json.loads(run_path.read_text(encoding="utf-8"))
                print(f"Resuming completed run from {run_path}", flush=True)
            else:
                result = train_one(arrays, list_weight, seed, run_index)
                write_json(run_path, result)
            runs.append(result)

    summary = aggregate(runs)
    paired = paired_against_current(runs)
    winner = min(
        summary,
        key=lambda row: (
            float(row["selection_score_mean"]),
            float(row["selection_score_std"]),
            abs(float(row["list_weight"]) - 0.5),
        ),
    )
    result = {
        "protocol_id": PROTOCOL_ID,
        "status": "COMPLETE",
        "selection_data": "validation_only",
        "selection_rule": "minimum mean of worst-domain selected regret plus 0.25 mean selected regret across three paired seeds",
        "selected_list_weight": winner["list_weight"],
        "selected_classification_weight": winner["classification_weight"],
        "current_0.60_0.40_is_selected": bool(math.isclose(winner["list_weight"], 0.60)),
        "winner": winner,
        "elapsed_seconds": time.perf_counter() - started,
        "test_or_final_queries_read": False,
    }
    write_csv(OUTPUT_ROOT / "ABLATION_RUNS.csv", [
        {key: value for key, value in run.items() if key not in {"history", "domain_mean_selected_regret"}}
        | {
            "domain_0_regret": run["domain_mean_selected_regret"]["0"],
            "domain_1_regret": run["domain_mean_selected_regret"]["1"],
            "domain_2_regret": run["domain_mean_selected_regret"]["2"],
        }
        for run in runs
    ])
    write_csv(OUTPUT_ROOT / "ABLATION_SUMMARY.csv", summary)
    write_csv(OUTPUT_ROOT / "PAIRED_VS_060.csv", paired)
    write_json(OUTPUT_ROOT / "FINAL_DECISION.json", result)
    write_report(summary, paired, winner, provenance)
    print(json.dumps(result, indent=2, sort_keys=True, default=json_default), flush=True)


if __name__ == "__main__":
    main()
