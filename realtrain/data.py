from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any

import numpy as np

from .utils import sha256_array


def _candidate_column(header: list[str], names: tuple[str, ...]) -> int:
    normalized = [h.strip().lower().replace(" ", "_") for h in header]
    for name in names:
        name = name.lower()
        if name in normalized:
            return normalized.index(name)
    for i, h in enumerate(normalized):
        if any(name in h for name in names):
            return i
    raise KeyError(f"Could not locate any of columns {names} in header={header[:30]}")


def locate_twitter_csv(input_root: Path) -> Path:
    c = [p for p in Path(input_root).rglob("twitter.csv") if p.is_file()]
    if not c:
        c = [p for p in Path(input_root).rglob("*.csv") if p.is_file() and "twitter" in p.name.lower()]
    if not c:
        raise FileNotFoundError("Twitter CSV not found. Attach the existing Twitter-Dataset Kaggle dataset.")
    c.sort(key=lambda p: ("rifathosain" not in str(p).lower(), len(str(p)), str(p)))
    return c[0]


def locate_crimes_csv(input_root: Path) -> Path:
    names = ["Crimes_-_2001_to_Present.csv", "crimes.csv"]
    c: list[Path] = []
    for name in names:
        c.extend([p for p in Path(input_root).rglob(name) if p.is_file()])
    if not c:
        c = [p for p in Path(input_root).rglob("*.csv") if p.is_file() and "crime" in p.name.lower()]
    if not c:
        raise FileNotFoundError("Crimes CSV not found. Attach Crimes In Chicago (2001 to 2023).")
    c.sort(key=lambda p: ("utkarshx27" not in str(p).lower(), len(str(p)), str(p)))
    return c[0]


def locate_arizona_npy(input_root: Path) -> Path:
    exact = [p for p in Path(input_root).rglob("arizona_buildings_n1464257_epsg26912.npy") if p.is_file()]
    if exact:
        exact.sort(key=lambda p: (len(str(p)), str(p)))
        return exact[0]
    c = [p for p in Path(input_root).rglob("*.npy") if p.is_file() and "arizona" in str(p).lower()]
    c += [p for p in Path(input_root).rglob("*.npy") if p.is_file() and "building" in p.name.lower() and "1464257" in p.name]
    if not c:
        raise FileNotFoundError(
            "Arizona cache not found. Attach the prior Arizona benchmark dataset containing "
            "data/arizona_cache/arizona_buildings_n1464257_epsg26912.npy."
        )
    c = sorted(set(c), key=lambda p: (len(str(p)), str(p)))
    return c[0]


def _normalize_point_array(points: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    a = np.asarray(points, dtype=np.float64)
    xmin, ymin = np.min(a, axis=0)
    xmax, ymax = np.max(a, axis=0)
    dx = max(float(xmax - xmin), 1e-12)
    dy = max(float(ymax - ymin), 1e-12)
    x = np.clip((a[:, 0] - xmin) / dx, 0.0, 1.0)
    y = np.clip((a[:, 1] - ymin) / dy, 0.0, 1.0)
    eps = 1e-6
    low_x = np.clip(x - 0.5 * eps, 0.0, 1.0 - eps)
    low_y = np.clip(y - 0.5 * eps, 0.0, 1.0 - eps)
    rects = np.column_stack([low_x, low_y, np.minimum(low_x + eps, 1.0), np.minimum(low_y + eps, 1.0)]).astype(np.float32)
    return rects, {"x_min": float(xmin), "y_min": float(ymin), "x_max": float(xmax), "y_max": float(ymax)}


def load_point_csv(path: Path, max_rows: int) -> tuple[np.ndarray, dict[str, Any]]:
    path = Path(path)
    with path.open("r", newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f)
        header = next(reader)
        x_col = _candidate_column(header, ("longitude", "lon", "lng", "x"))
        y_col = _candidate_column(header, ("latitude", "lat", "y"))
        pts = []
        invalid = 0
        seen = 0
        for row in reader:
            seen += 1
            if len(row) <= max(x_col, y_col):
                invalid += 1
                continue
            try:
                x = float(row[x_col]); y = float(row[y_col])
            except (ValueError, TypeError):
                invalid += 1
                continue
            if not (math.isfinite(x) and math.isfinite(y)):
                invalid += 1
                continue
            pts.append((x, y))
            if max_rows > 0 and len(pts) >= max_rows:
                break
    if not pts:
        raise RuntimeError(f"No valid points loaded from {path}")
    rects, bounds = _normalize_point_array(np.asarray(pts, dtype=np.float64))
    return rects, {
        "source": str(path), "objects": int(len(rects)), "source_rows_seen": int(seen),
        "invalid_rows": int(invalid), "source_bounds": bounds, "kind": "point",
        "normalized_object_sha256": sha256_array(rects),
    }


def _valid_xyxy(a: np.ndarray) -> float:
    return float(np.mean((a[:, 0] <= a[:, 2]) & (a[:, 1] <= a[:, 3])))


def load_arizona(path: Path, max_rows: int = 0) -> tuple[np.ndarray, dict[str, Any]]:
    path = Path(path)
    raw = np.load(path, mmap_mode="r")
    if raw.ndim != 2 or raw.shape[1] < 4:
        raise RuntimeError(f"Arizona array must be Nx4+, got {raw.shape}")
    n = len(raw) if max_rows <= 0 else min(len(raw), max_rows)
    a = np.asarray(raw[:n, :4], dtype=np.float64)
    # Historical cache convention is xmin,xmax,ymin,ymax. Prefer that for the
    # exact known cache; otherwise select the convention with more valid boxes.
    hist = np.column_stack([a[:, 0], a[:, 2], a[:, 1], a[:, 3]])
    direct = a.copy()
    if "arizona_buildings_n1464257_epsg26912" in path.name:
        rects = hist
        source_order = "xmin,xmax,ymin,ymax"
    elif _valid_xyxy(hist) > _valid_xyxy(direct) + 0.05:
        rects = hist
        source_order = "xmin,xmax,ymin,ymax"
    else:
        rects = direct
        source_order = "xmin,ymin,xmax,ymax"
    finite = np.isfinite(rects).all(axis=1)
    valid = finite & (rects[:, 2] >= rects[:, 0]) & (rects[:, 3] >= rects[:, 1])
    rects = rects[valid]
    if not len(rects):
        raise RuntimeError("Arizona cache contains no valid rectangles after coordinate audit")
    xmin = float(rects[:, 0].min()); ymin = float(rects[:, 1].min())
    xmax = float(rects[:, 2].max()); ymax = float(rects[:, 3].max())
    dx = max(xmax - xmin, 1e-12); dy = max(ymax - ymin, 1e-12)
    out = rects.copy()
    out[:, [0, 2]] = (out[:, [0, 2]] - xmin) / dx
    out[:, [1, 3]] = (out[:, [1, 3]] - ymin) / dy
    out = np.clip(out, 0.0, 1.0).astype(np.float32)
    return out, {
        "source": str(path), "objects": int(len(out)), "source_rows": int(n),
        "invalid_rows": int(n - len(out)), "source_order": source_order,
        "source_bounds": {"x_min": xmin, "y_min": ymin, "x_max": xmax, "y_max": ymax},
        "kind": "rectangle", "normalized_object_sha256": sha256_array(out),
    }


def load_all_domains(input_root: Path, twitter_max_rows: int, crimes_max_rows: int, arizona_max_rows: int) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    twitter_path = locate_twitter_csv(input_root)
    crimes_path = locate_crimes_csv(input_root)
    arizona_path = locate_arizona_npy(input_root)
    tw, twm = load_point_csv(twitter_path, twitter_max_rows)
    cr, crm = load_point_csv(crimes_path, crimes_max_rows)
    az, azm = load_arizona(arizona_path, arizona_max_rows)
    return {"twitter": tw, "crimes": cr, "arizona": az}, {"twitter": twm, "crimes": crm, "arizona": azm}
