from __future__ import annotations

import collections
import gzip
import pickle
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from rtreelib.models import Rect
from rtreelib.rtree import RTreeEntry, RTreeNode
from rtreelib.strategies.guttman import RTreeGuttman

from .actions import ACTION_COUNT, page_count, split_positions
from .features import local_queries_for_state, state_feature
from .geometry import bbox


def rect_obj(row) -> Rect:
    return Rect(float(row[0]), float(row[1]), float(row[2]), float(row[3]))


def rect_tuple(r: Rect) -> tuple[float, float, float, float]:
    return float(r.min_x), float(r.min_y), float(r.max_x), float(r.max_y)


class NeuralPolicy:
    def __init__(self, model, device: str, capacity: int, hist_bins: int, construction_queries: np.ndarray):
        self.model = model
        self.device = device
        self.capacity = capacity
        self.hist_bins = hist_bins
        self.construction_queries = np.asarray(construction_queries, dtype=np.float32)
        self.decisions = 0
        self.action_counts = collections.Counter()

    @torch.no_grad()
    def choose(self, entries: np.ndarray, level: int, query_cap: int = 96) -> tuple[int, np.ndarray, np.ndarray]:
        queries = local_queries_for_state(entries, self.construction_queries, query_cap, None)
        feat = state_feature(entries, queries, self.capacity, self.hist_bins, level)
        x = torch.from_numpy(feat[None]).to(self.device)
        scores = self.model(x)[0]
        action_id = int(torch.argmin(scores).item())
        self.decisions += 1
        self.action_counts[action_id] += 1
        return action_id, queries, feat


def neural_partition_groups(entries: np.ndarray, policy: NeuralPolicy, level: int,
                            capture: Callable[[np.ndarray, np.ndarray, int], None] | None = None,
                            query_cap: int = 96) -> list[np.ndarray]:
    r = np.asarray(entries, dtype=np.float32)
    groups: list[np.ndarray] = []

    def rec(pos: np.ndarray):
        if len(pos) <= policy.capacity:
            groups.append(pos.copy()); return
        sub = r[pos]
        action_id, q, _ = policy.choose(sub, level, query_cap=query_cap)
        if capture is not None:
            capture(sub, q, level)
        left_local, right_local = split_positions(sub, action_id, policy.capacity)
        rec(pos[left_local]); rec(pos[right_local])

    rec(np.arange(len(r), dtype=np.int64))
    expected = page_count(len(r), policy.capacity)
    if len(groups) != expected:
        raise RuntimeError(f"neural page count mismatch {len(groups)} != {expected}")
    if max(map(len, groups), default=0) > policy.capacity:
        raise RuntimeError("neural partition overflow")
    merged = np.concatenate(groups) if groups else np.empty(0, dtype=np.int64)
    if len(merged) != len(r) or len(np.unique(merged)) != len(r):
        raise RuntimeError("neural partition does not preserve one-to-one entry coverage")
    return groups


def _node_rects(nodes: list[RTreeNode]) -> np.ndarray:
    return np.asarray([rect_tuple(n.get_bounding_rect()) for n in nodes], dtype=np.float32)


def _make_leaf_nodes(tree: RTreeGuttman, rects: np.ndarray, groups: list[np.ndarray]) -> list[RTreeNode]:
    nodes=[]
    for g in groups:
        entries=[RTreeEntry(rect_obj(rects[int(i)]),data=int(i)) for i in g]
        nodes.append(RTreeNode(tree,is_leaf=True,entries=entries))
    return nodes


def _make_parent_nodes(tree: RTreeGuttman, children: list[RTreeNode], groups: list[np.ndarray]) -> list[RTreeNode]:
    parents=[]
    for g in groups:
        selected=[children[int(i)] for i in g]
        entries=[RTreeEntry(child.get_bounding_rect(),child=child) for child in selected]
        parent=RTreeNode(tree,is_leaf=False,entries=entries)
        for child in selected: child.parent=parent
        parents.append(parent)
    return parents


def build_neural_tree(rects: np.ndarray, construction_queries: np.ndarray, model, device: str,
                      capacity: int, hist_bins: int, query_cap: int = 96,
                      capture: Callable[[np.ndarray, np.ndarray, int], None] | None = None) -> tuple[RTreeGuttman, dict[str, Any]]:
    started=time.perf_counter()
    r=np.asarray(rects,dtype=np.float32)
    tree=RTreeGuttman(max_entries=capacity)
    policy=NeuralPolicy(model,device,capacity,hist_bins,construction_queries)
    leaf_groups=neural_partition_groups(r,policy,0,capture,query_cap)
    nodes=_make_leaf_nodes(tree,r,leaf_groups)
    level=1; level_counts=[len(nodes)]
    while len(nodes)>capacity:
        nr=_node_rects(nodes)
        groups=neural_partition_groups(nr,policy,level,capture,query_cap)
        nodes=_make_parent_nodes(tree,nodes,groups)
        level_counts.append(len(nodes)); level+=1
    if len(nodes)==1:
        tree.root=nodes[0]; tree.root.parent=None
    else:
        tree.grow_tree(nodes)
    diag={"build_seconds":time.perf_counter()-started,"capacity":capacity,"objects":len(r),
          "leaf_pages":len(leaf_groups),"stored_level_node_counts":level_counts,
          "neural_decision_count":policy.decisions,"action_counts":{str(k):int(v) for k,v in sorted(policy.action_counts.items())}}
    v=validate_tree(tree,len(r),capacity); diag["validation"]=v
    if not v["passed"]: raise RuntimeError(f"neural tree invalid: {v}")
    return tree,diag


def validate_tree(tree: RTreeGuttman, object_count: int, capacity: int) -> dict[str, Any]:
    stack=[(tree.root,0)]; ids=[]; over=[]; null_children=0; nodes=0; leaves=0; internal=0; max_depth=0
    while stack:
        node,d=stack.pop(); nodes+=1; max_depth=max(max_depth,d)
        if len(node.entries)>capacity: over.append((d,len(node.entries)))
        if node.is_leaf:
            leaves+=1; ids.extend(int(e.data) for e in node.entries)
        else:
            internal+=1
            for e in node.entries:
                if e.child is None: null_children+=1
                else: stack.append((e.child,d+1))
    unique=len(set(ids)); complete=(len(ids)==object_count and unique==object_count and (min(ids,default=0)>=0) and (max(ids,default=-1)<object_count))
    return {"passed":not over and null_children==0 and complete,"nodes":nodes,"leaves":leaves,"internal":internal,"height":max_depth+1,
            "object_entries":len(ids),"unique_object_ids":unique,"overflow_nodes":over[:10],"null_children":null_children}


def save_tree(path: Path, tree, diagnostics: dict[str, Any]) -> None:
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    with gzip.open(path,"wb",compresslevel=3) as f: pickle.dump((tree,diagnostics),f,pickle.HIGHEST_PROTOCOL)


def load_tree(path: Path):
    with gzip.open(Path(path),"rb") as f: return pickle.load(f)
