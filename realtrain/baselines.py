from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any

import numpy as np
from rtreelib.rtree import RTreeEntry,RTreeNode
from rtreelib.models import Rect
from rtreelib.strategies.guttman import RTreeGuttman
from rtreelib.strategies.tgs import build_tree_tgs

from .tree import rect_obj,rect_tuple,validate_tree
from .utils import atomic_json


def _str_groups(rects:np.ndarray,capacity:int)->list[np.ndarray]:
    r=np.asarray(rects,dtype=np.float32);n=len(r)
    if n<=capacity:return [np.arange(n,dtype=np.int64)]
    pages=math.ceil(n/capacity);slices=max(1,int(math.ceil(math.sqrt(pages))))
    c=np.column_stack(((r[:,0]+r[:,2])*.5,(r[:,1]+r[:,3])*.5))
    xorder=np.argsort(c[:,0],kind="mergesort");slice_size=math.ceil(n/slices);order=[]
    for s in range(slices):
        part=xorder[s*slice_size:min(n,(s+1)*slice_size)]
        if len(part):part=part[np.argsort(c[part,1],kind="mergesort")];order.extend(part.tolist())
    order=np.asarray(order,dtype=np.int64)
    return [order[i:i+capacity] for i in range(0,n,capacity)]


def _node_rects(nodes:list[RTreeNode])->np.ndarray:
    return np.asarray([rect_tuple(n.get_bounding_rect()) for n in nodes],dtype=np.float32)


def build_str_tree(rects:np.ndarray,capacity:int)->tuple[RTreeGuttman,dict[str,Any]]:
    t=time.perf_counter();r=np.asarray(rects,dtype=np.float32);tree=RTreeGuttman(max_entries=capacity)
    groups=_str_groups(r,capacity);nodes=[]
    for g in groups:
        entries=[RTreeEntry(rect_obj(r[int(i)]),data=int(i)) for i in g];nodes.append(RTreeNode(tree,is_leaf=True,entries=entries))
    level_counts=[len(nodes)]
    while len(nodes)>capacity:
        nr=_node_rects(nodes);pg=_str_groups(nr,capacity);new=[]
        for g in pg:
            ch=[nodes[int(i)] for i in g];parent=RTreeNode(tree,is_leaf=False,entries=[RTreeEntry(x.get_bounding_rect(),child=x) for x in ch])
            for x in ch:x.parent=parent
            new.append(parent)
        nodes=new;level_counts.append(len(nodes))
    if len(nodes)==1:tree.root=nodes[0];tree.root.parent=None
    else:tree.grow_tree(nodes)
    v=validate_tree(tree,len(r),capacity)
    return tree,{"method":"STR","build_seconds":time.perf_counter()-t,"level_counts":level_counts,"validation":v}


def build_tgs(rects:np.ndarray,capacity:int)->tuple[Any,dict[str,Any]]:
    t=time.perf_counter();data=np.arange(len(rects),dtype=np.int64)
    tree=build_tree_tgs(np.asarray(rects,dtype=np.float32),data,max_entries=capacity,cost="area",orderings=("center",))
    v=validate_tree(tree,len(rects),capacity)
    return tree,{"method":"TGS","build_seconds":time.perf_counter()-t,"configuration":{"cost":"area","orderings":["center"]},"validation":v}
