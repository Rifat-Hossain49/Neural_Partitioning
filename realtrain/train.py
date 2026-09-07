from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from .actions import ACTION_COUNT
from .model import StateQNet, unwrap_model
from .utils import atomic_json, seed_everything, sha256_file


class StateDataset(Dataset):
    def __init__(self, arrays: dict[str, np.ndarray], indices: np.ndarray):
        self.a = arrays
        self.idx = np.asarray(indices, dtype=np.int64)
    def __len__(self): return len(self.idx)
    def __getitem__(self, i):
        j = self.idx[i]
        return (
            torch.from_numpy(self.a["x"][j]),
            torch.from_numpy(self.a["regrets"][j]),
            torch.tensor(int(self.a["domain"][j]), dtype=torch.long),
        )


def split_indices(arrays: dict[str, np.ndarray], seed: int, val_fraction: float = 0.20) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    train, val = [], []
    domains = np.asarray(arrays["domain"])
    for d in sorted(np.unique(domains)):
        idx = np.flatnonzero(domains == d)
        rng.shuffle(idx)
        nval = max(1, int(round(len(idx) * val_fraction)))
        val.extend(idx[:nval]); train.extend(idx[nval:])
    return np.asarray(train, dtype=np.int64), np.asarray(val, dtype=np.int64)


def _loss_batch(pred: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
    mask = torch.isfinite(target)
    safe_target = torch.nan_to_num(target, nan=0.0)
    # Regression over observed actions.
    reg_loss = nn.functional.smooth_l1_loss(pred[mask], safe_target[mask])
    # Masked listwise soft target over observed actions only.
    big = torch.tensor(1e6, device=pred.device, dtype=pred.dtype)
    pred_masked = torch.where(mask, pred, big)
    target_logits = torch.where(mask, -safe_target / 0.025, -big)
    target_prob = torch.softmax(target_logits, dim=1)
    logp = torch.log_softmax(-pred_masked, dim=1)
    list_loss = -(target_prob * logp).sum(dim=1).mean()
    # Best observed action classification.
    best = torch.argmin(torch.where(mask, safe_target, big), dim=1)
    cls_loss = nn.functional.cross_entropy(-pred_masked, best)
    total = reg_loss + 0.60 * list_loss + 0.40 * cls_loss
    return total, {"reg": float(reg_loss.detach()), "list": float(list_loss.detach()), "cls": float(cls_loss.detach())}


@torch.no_grad()
def evaluate(model, loader, device: str) -> dict[str, Any]:
    model.eval()
    losses = []
    selected_regrets = []
    by_domain: dict[int, list[float]] = {}
    for x, target, domain in loader:
        x = x.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        pred = model(x)
        loss, _ = _loss_batch(pred, target)
        losses.append(float(loss))
        mask = torch.isfinite(target)
        big = torch.tensor(1e6, device=device, dtype=pred.dtype)
        chosen = torch.argmin(torch.where(mask, pred, big), dim=1)
        selected = target.gather(1, chosen[:, None]).squeeze(1).detach().cpu().numpy()
        dom = domain.numpy()
        selected_regrets.extend(selected.tolist())
        for d, r in zip(dom, selected): by_domain.setdefault(int(d), []).append(float(r))
    dm = {str(k): float(np.mean(v)) for k, v in by_domain.items()}
    return {
        "loss": float(np.mean(losses)) if losses else float("nan"),
        "mean_selected_regret": float(np.mean(selected_regrets)) if selected_regrets else float("nan"),
        "p95_selected_regret": float(np.quantile(selected_regrets, 0.95)) if selected_regrets else float("nan"),
        "domain_mean_selected_regret": dm,
        "worst_domain_regret": max(dm.values()) if dm else float("nan"),
    }


def train_one(arrays: dict[str, np.ndarray], output_path: Path, seed: int, epochs: int, batch_size: int,
              lr: float, weight_decay: float, patience: int, use_multi_gpu: bool = True) -> dict[str, Any]:
    seed_everything(seed)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    train_idx, val_idx = split_indices(arrays, seed)
    train_loader = DataLoader(StateDataset(arrays, train_idx), batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(StateDataset(arrays, val_idx), batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    base = StateQNet(int(arrays["x"].shape[1]), ACTION_COUNT)
    model: nn.Module = base.to(device)
    gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if use_multi_gpu and gpu_count >= 2:
        model = nn.DataParallel(model, device_ids=list(range(gpu_count)))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
    best_score = float("inf"); best_epoch = -1; stale = 0; history = []
    started = time.perf_counter()
    output_path = Path(output_path); output_path.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, epochs + 1):
        model.train(); batch_losses=[]
        for x, target, _ in train_loader:
            x=x.to(device,non_blocking=True); target=target.to(device,non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                pred=model(x); loss,_=_loss_batch(pred,target)
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt); scaler.update(); batch_losses.append(float(loss.detach()))
        val = evaluate(model,val_loader,device)
        rec={"epoch":epoch,"train_loss":float(np.mean(batch_losses)),**val}; history.append(rec)
        score=float(val["worst_domain_regret"])+0.25*float(val["mean_selected_regret"])
        print(f"epoch {epoch:02d} train={rec['train_loss']:.5f} val_regret={val['mean_selected_regret']:.5f} worst={val['worst_domain_regret']:.5f}",flush=True)
        if score < best_score - 1e-6:
            best_score=score; best_epoch=epoch; stale=0
            raw=unwrap_model(model)
            torch.save({"state_dict":raw.state_dict(),"input_dim":raw.input_dim,"action_count":raw.action_count,"seed":seed,"epoch":epoch,"validation":val},output_path)
        else:
            stale+=1
            if stale>=patience: break
    meta={"seed":seed,"epochs_requested":epochs,"best_epoch":best_epoch,"best_score":best_score,"history":history,
          "seconds":time.perf_counter()-started,"train_states":int(len(train_idx)),"val_states":int(len(val_idx)),
          "gpu_count":gpu_count,"data_parallel":bool(use_multi_gpu and gpu_count>=2)}
    atomic_json(output_path.with_suffix(".training.json"),meta)
    meta["checkpoint_sha256"]=sha256_file(output_path)
    return meta


def load_model(path: Path, device: str = "cuda:0") -> StateQNet:
    ckpt=torch.load(Path(path),map_location="cpu")
    model=StateQNet(int(ckpt["input_dim"]),int(ckpt["action_count"]))
    model.load_state_dict(ckpt["state_dict"])
    model.to(device); model.eval(); return model
