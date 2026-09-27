from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from realtrain.baselines import build_str_tree, build_tgs  # noqa: E402
from realtrain.eval import evaluate_knn_suite, evaluate_range_suite  # noqa: E402
from realtrain.train import load_model  # noqa: E402
from realtrain.tree import build_neural_tree, rect_obj, validate_tree  # noqa: E402
from rtreelib.strategies.guttman import RTreeGuttman  # noqa: E402
from rtreelib.strategies.rstar import RStarTree  # noqa: E402


METHODS = ("WAHARP", "STR", "TGS", "Guttman", "RStar")
POINT_X_NAMES = ("x", "lon", "lng", "longitude")
POINT_Y_NAMES = ("y", "lat", "latitude")
BBOX_ALIASES = {
    "xmin": ("xmin", "minx", "x_min", "left"),
    "ymin": ("ymin", "miny", "y_min", "bottom"),
    "xmax": ("xmax", "maxx", "x_max", "right"),
    "ymax": ("ymax", "maxy", "y_max", "top"),
}


def _first_present(names: tuple[str, ...], fields: dict[str, str]) -> str | None:
    for name in names:
        if name in fields:
            return fields[name]
    return None


def _load_csv(path: Path) -> np.ndarray:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"CSV has no header: {path}")
        fields = {name.strip().lower(): name for name in reader.fieldnames}
        bbox_columns = {
            key: _first_present(aliases, fields) for key, aliases in BBOX_ALIASES.items()
        }
        if all(bbox_columns.values()):
            columns = [bbox_columns[key] for key in ("xmin", "ymin", "xmax", "ymax")]
        else:
            x_column = _first_present(POINT_X_NAMES, fields)
            y_column = _first_present(POINT_Y_NAMES, fields)
            if not x_column or not y_column:
                raise ValueError(
                    "CSV must contain x/y (or longitude/latitude) columns, or "
                    "xmin/ymin/xmax/ymax columns."
                )
            columns = [x_column, y_column]
        rows: list[list[float]] = []
        for row in reader:
            try:
                values = [float(row[column]) for column in columns]
            except (TypeError, ValueError):
                continue
            if np.isfinite(values).all():
                rows.append(values)
    if not rows:
        raise ValueError(f"No valid numeric spatial rows found in {path}")
    return np.asarray(rows, dtype=np.float64)


def load_spatial_data(path: Path | None, synthetic_objects: int, seed: int) -> np.ndarray:
    if path is None:
        rng = np.random.default_rng(seed)
        centers = np.vstack(
            [
                rng.normal((0.25, 0.30), (0.08, 0.05), size=(synthetic_objects // 2, 2)),
                rng.normal(
                    (0.72, 0.68),
                    (0.10, 0.07),
                    size=(synthetic_objects - synthetic_objects // 2, 2),
                ),
            ]
        )
        return np.clip(centers, 0.0, 1.0)
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".npy":
        values = np.load(path, allow_pickle=False)
    elif path.suffix.lower() in {".csv", ".txt"}:
        values = _load_csv(path)
    else:
        raise ValueError("Supported input formats are .npy, .csv, and .txt")
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] not in (2, 4):
        raise ValueError(f"Expected an N x 2 or N x 4 array, found {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("Input contains NaN or infinite coordinates")
    return values


def normalize_rectangles(values: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    values = np.asarray(values, dtype=np.float64)
    if values.shape[1] == 2:
        lower = values.copy()
        upper = values.copy()
        input_kind = "points"
    else:
        lower = values[:, :2].copy()
        upper = values[:, 2:].copy()
        if np.any(lower > upper):
            raise ValueError("Rectangle input must use xmin,ymin,xmax,ymax order")
        input_kind = "rectangles"
    global_lower = np.min(lower, axis=0)
    global_upper = np.max(upper, axis=0)
    span = global_upper - global_lower
    if np.any(span <= 0):
        raise ValueError("Each coordinate axis must have nonzero extent")
    lower = (lower - global_lower) / span
    upper = (upper - global_lower) / span
    if input_kind == "points":
        epsilon = 1e-6
        lower = np.maximum(0.0, lower - epsilon / 2.0)
        upper = np.minimum(1.0, upper + epsilon / 2.0)
    rectangles = np.column_stack([lower[:, 0], lower[:, 1], upper[:, 0], upper[:, 1]])
    return rectangles.astype(np.float32), {
        "input_kind": input_kind,
        "original_min": global_lower.tolist(),
        "original_max": global_upper.tolist(),
        "normalization": "independent min-max scaling to [0,1] on each axis",
    }


def make_boxes(rng: np.random.Generator, count: int, side: float) -> np.ndarray:
    width = min(max(float(side), 1e-6), 1.0)
    lower = rng.uniform(0.0, max(1.0 - width, 1e-9), size=(count, 2))
    upper = np.minimum(1.0, lower + width)
    return np.column_stack([lower[:, 0], lower[:, 1], upper[:, 0], upper[:, 1]]).astype(
        np.float32
    )


def make_workload(
    rectangles: np.ndarray,
    construction_count: int,
    range_count: int,
    point_count: int,
    knn_count: int,
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    rng = np.random.default_rng(seed)
    scales = (0.002, 0.01, 0.05)
    construction_parts = [
        make_boxes(rng, math.ceil(construction_count / len(scales)), scale)
        for scale in scales
    ]
    construction = np.concatenate(construction_parts, axis=0)[:construction_count]
    ranges = {
        f"side_{scale:g}": make_boxes(rng, range_count, scale) for scale in scales
    }
    centers = np.column_stack(
        [
            (rectangles[:, 0] + rectangles[:, 2]) * 0.5,
            (rectangles[:, 1] + rectangles[:, 3]) * 0.5,
        ]
    )
    point_indices = rng.integers(0, len(centers), size=point_count)
    points = centers[point_indices].astype(np.float32)
    knn_points = rng.uniform(0.0, 1.0, size=(knn_count, 2)).astype(np.float32)
    return construction, {"ranges": ranges, "points": points, "knn_points": knn_points}


def build_dynamic(method: str, rectangles: np.ndarray, capacity: int):
    started = time.perf_counter()
    if method == "Guttman":
        tree = RTreeGuttman(max_entries=capacity)
    elif method == "RStar":
        tree = RStarTree(max_entries=capacity, rstar_accelerator="numpy")
    else:
        raise ValueError(method)
    for identifier, row in enumerate(rectangles):
        tree.insert(identifier, rect_obj(row))
    validation = validate_tree(tree, len(rectangles), capacity)
    if not validation["passed"]:
        raise RuntimeError(f"{method} produced an invalid tree: {validation}")
    return tree, {
        "method": method,
        "build_seconds": time.perf_counter() - started,
        "validation": validation,
    }


def build_method(
    method: str,
    rectangles: np.ndarray,
    construction_queries: np.ndarray,
    capacity: int,
    checkpoint: Path,
    device: str,
):
    if method == "WAHARP":
        model = load_model(checkpoint, device)
        return build_neural_tree(
            rectangles,
            construction_queries,
            model,
            device,
            capacity,
            hist_bins=12,
            query_cap=96,
        )
    if method == "STR":
        return build_str_tree(rectangles, capacity)
    if method == "TGS":
        return build_tgs(rectangles, capacity)
    return build_dynamic(method, rectangles, capacity)


def evaluate_method(
    tree,
    suite: dict[str, Any],
    object_count: int,
    capacity: int,
    method: str,
    knn_values: tuple[int, ...],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name, queries in suite["ranges"].items():
        rows.extend(evaluate_range_suite(tree, queries, object_count, capacity, "range", name))
    points = suite["points"]
    point_boxes = np.column_stack([points[:, 0], points[:, 1], points[:, 0], points[:, 1]])
    rows.extend(
        evaluate_range_suite(tree, point_boxes, object_count, capacity, "point", "point")
    )
    for k in knn_values:
        rows.extend(evaluate_knn_suite(tree, suite["knn_points"], k, f"knn_k{k}"))
    for row in rows:
        row["method"] = method
    return rows


def query_key(row: dict[str, Any]) -> tuple[str, str, int]:
    return str(row["query_type"]), str(row["workload"]), int(row["query_index"])


def correctness_mismatches(
    reference: list[dict[str, Any]], candidate: list[dict[str, Any]]
) -> int:
    expected = {query_key(row): row for row in reference}
    actual = {query_key(row): row for row in candidate}
    if set(expected) != set(actual):
        return abs(len(expected) - len(actual)) + len(set(expected).symmetric_difference(actual))
    mismatches = 0
    for key, left in expected.items():
        right = actual[key]
        if key[0] == "knn":
            agrees = math.isclose(
                float(left["kth_distance"]),
                float(right["kth_distance"]),
                rel_tol=2e-6,
                abs_tol=2e-6,
            )
        else:
            agrees = (
                int(left["result_count"]),
                int(left["id_sum"]),
                int(left["id_xor"]),
            ) == (
                int(right["result_count"]),
                int(right["id_sum"]),
                int(right["id_xor"]),
            )
        mismatches += int(not agrees)
    return mismatches


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run_benchmark(args: argparse.Namespace) -> list[dict[str, Any]]:
    raw = load_spatial_data(args.data, args.synthetic_objects, args.seed)
    rectangles, normalization = normalize_rectangles(raw)
    if args.max_objects and len(rectangles) > args.max_objects:
        rng = np.random.default_rng(args.seed)
        indices = rng.choice(len(rectangles), size=args.max_objects, replace=False)
        rectangles = rectangles[np.sort(indices)]
    if len(rectangles) <= args.capacity:
        raise ValueError("Dataset must contain more objects than the selected capacity")
    construction, suite = make_workload(
        rectangles,
        args.construction_queries,
        args.range_queries,
        args.point_queries,
        args.knn_queries,
        args.seed + 1,
    )
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    methods = [name.strip() for name in args.methods.split(",") if name.strip()]
    unknown = [name for name in methods if name not in METHODS]
    if unknown:
        raise ValueError(f"Unknown methods {unknown}; choose from {METHODS}")
    device = args.device
    if device == "auto":
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    all_rows: dict[str, list[dict[str, Any]]] = {}
    builds: dict[str, dict[str, Any]] = {}
    for method in methods:
        print(f"Building {method} on {len(rectangles):,} objects...", flush=True)
        tree, build = build_method(
            method,
            rectangles,
            construction,
            args.capacity,
            checkpoint,
            device,
        )
        rows = evaluate_method(
            tree,
            suite,
            len(rectangles),
            args.capacity,
            method,
            tuple(args.knn_k),
        )
        builds[method] = build
        all_rows[method] = rows
        del tree

    reference_method = "WAHARP" if "WAHARP" in all_rows else methods[0]
    reference = all_rows[reference_method]
    reference_accesses = sum(int(row["total_node_accesses"]) for row in reference)
    summary: list[dict[str, Any]] = []
    family_rows: list[dict[str, Any]] = []
    per_query: list[dict[str, Any]] = []
    for method in methods:
        rows = all_rows[method]
        accesses = sum(int(row["total_node_accesses"]) for row in rows)
        metrics = tree_metrics_from_build(builds[method])
        summary.append(
            {
                "method": method,
                "objects": len(rectangles),
                "capacity": args.capacity,
                "queries": len(rows),
                "build_seconds": float(builds[method]["build_seconds"]),
                "total_node_accesses": accesses,
                "mean_node_accesses": accesses / len(rows),
                f"ratio_to_{reference_method.lower()}": accesses / reference_accesses,
                "correctness_mismatches_vs_reference": correctness_mismatches(reference, rows),
                **metrics,
            }
        )
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            grouped[str(row["query_type"])].append(row)
            per_query.append(row)
        for family, subset in sorted(grouped.items()):
            family_access = sum(int(row["total_node_accesses"]) for row in subset)
            family_rows.append(
                {
                    "method": method,
                    "query_family": family,
                    "queries": len(subset),
                    "total_node_accesses": family_access,
                    "mean_node_accesses": family_access / len(subset),
                }
            )
    write_csv(output / "SUMMARY.csv", summary)
    write_csv(output / "BY_QUERY_FAMILY.csv", family_rows)
    write_csv(output / "PER_QUERY.csv", per_query)
    (output / "RUN_CONFIG.json").write_text(
        json.dumps(
            {
                "data": str(args.data) if args.data else "synthetic_demo",
                "objects": len(rectangles),
                "capacity": args.capacity,
                "methods": methods,
                "reference_method": reference_method,
                "device": device,
                "seed": args.seed,
                "normalization": normalization,
                "correctness_gate_passed": all(
                    row["correctness_mismatches_vs_reference"] == 0 for row in summary
                ),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return summary


def tree_metrics_from_build(build: dict[str, Any]) -> dict[str, Any]:
    validation = build.get("validation", {})
    return {
        "tree_valid": bool(validation.get("passed", False)),
        "tree_nodes": int(validation.get("nodes", 0)),
        "tree_height": int(validation.get("height", 0)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build WAHARP and baseline R-trees on a custom 2D dataset."
    )
    parser.add_argument("--data", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("custom_benchmark_output"))
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "artifacts/model/selected.pt")
    parser.add_argument("--capacity", type=int, default=128)
    parser.add_argument("--methods", default="WAHARP,STR,TGS")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda:0"))
    parser.add_argument("--max-objects", type=int, default=0)
    parser.add_argument("--synthetic-objects", type=int, default=10_000)
    parser.add_argument("--construction-queries", type=int, default=512)
    parser.add_argument("--range-queries", type=int, default=100)
    parser.add_argument("--point-queries", type=int, default=100)
    parser.add_argument("--knn-queries", type=int, default=100)
    parser.add_argument("--knn-k", type=int, nargs="+", default=(1, 10, 100))
    parser.add_argument("--seed", type=int, default=20260927)
    args = parser.parse_args()
    if args.capacity < 2:
        parser.error("--capacity must be at least 2")
    if any(k < 1 for k in args.knn_k):
        parser.error("--knn-k values must be positive")
    return args


def main() -> None:
    summary = run_benchmark(parse_args())
    for row in summary:
        print(
            f"{row['method']:8s} build={row['build_seconds']:.3f}s "
            f"accesses={row['total_node_accesses']:,} "
            f"mismatches={row['correctness_mismatches_vs_reference']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
