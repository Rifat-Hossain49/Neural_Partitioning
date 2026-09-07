from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import random
import shutil
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed & 0xFFFFFFFF)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def derive_seed(namespace: str, seed: int, label: str) -> int:
    payload = f"{namespace}|{seed}|{label}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & 0x7FFF_FFFF_FFFF_FFFF


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_array(arr: np.ndarray) -> str:
    a = np.ascontiguousarray(arr)
    h = hashlib.sha256()
    h.update(str(a.shape).encode("ascii"))
    h.update(str(a.dtype).encode("ascii"))
    h.update(a.view(np.uint8))
    return h.hexdigest()


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default), encoding="utf-8")
    tmp.replace(path)


def _json_default(x):
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, Path):
        return str(x)
    raise TypeError(type(x).__name__)


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fields.append(key)
    opener = path.open
    with opener("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def now_seconds() -> float:
    return time.perf_counter()


class Budget:
    def __init__(self, total_seconds: int, reserve_seconds: int = 1800):
        self.started = time.perf_counter()
        self.total_seconds = float(total_seconds)
        self.reserve_seconds = float(reserve_seconds)

    @property
    def elapsed(self) -> float:
        return time.perf_counter() - self.started

    @property
    def remaining(self) -> float:
        return self.total_seconds - self.elapsed

    def enough_for(self, estimated_seconds: float) -> bool:
        return self.remaining > self.reserve_seconds + estimated_seconds

    def require(self, estimated_seconds: float, label: str) -> None:
        if not self.enough_for(estimated_seconds):
            raise TimeBudgetStop(label, self.elapsed, self.remaining)


class TimeBudgetStop(RuntimeError):
    def __init__(self, label: str, elapsed: float, remaining: float):
        super().__init__(f"Time budget stop before {label}: elapsed={elapsed:.1f}s remaining={remaining:.1f}s")
        self.label = label
        self.elapsed = elapsed
        self.remaining = remaining


def copytree_merge(src: Path, dst: Path) -> None:
    src = Path(src)
    dst = Path(dst)
    if not src.exists():
        return
    dst.mkdir(parents=True, exist_ok=True)
    for p in src.rglob("*"):
        rel = p.relative_to(src)
        q = dst / rel
        if p.is_dir():
            q.mkdir(parents=True, exist_ok=True)
        else:
            q.parent.mkdir(parents=True, exist_ok=True)
            if not q.exists():
                shutil.copy2(p, q)


def geometric_mean(values: Iterable[float]) -> float:
    vals = np.asarray([float(v) for v in values if math.isfinite(float(v)) and float(v) > 0], dtype=np.float64)
    if len(vals) == 0:
        return float("nan")
    return float(np.exp(np.mean(np.log(vals))))
