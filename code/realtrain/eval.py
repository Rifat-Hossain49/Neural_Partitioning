from __future__ import annotations

import heapq
import json
import math
import time
from collections import Counter, defaultdict
from typing import Any, Iterable

import numpy as np

from .geometry import intersects, min_dist_point_rect, normalized_overlap_pair_area


def _rtuple(rect) -> tuple[float,float,float,float]:
    return float(rect.min_x),float(rect.min_y),float(rect.max_x),float(rect.max_y)


def result_signature(values: Iterable[int]) -> tuple[int,int,int]:
    arr=np.asarray(list(values),dtype=np.uint64)
    if len(arr)==0: return 0,0,0
    return int(len(arr)),int(arr.sum(dtype=np.uint64)),int(np.bitwise_xor.reduce(arr))


def min_tree_levels(object_count:int,capacity:int)->int:
    if object_count<=capacity:return 1
    pages=math.ceil(object_count/capacity);levels=1
    while pages>1:
        pages=math.ceil(pages/capacity);levels+=1
    return levels


def capacity_io_lower_bound(result_count:int,object_count:int,capacity:int)->int:
    levels=min_tree_levels(object_count,capacity)
    if result_count<=0:return 1
    if levels==1:return 1
    current=math.ceil(result_count/capacity);total=current
    for _ in range(max(0,levels-2)):
        current=max(1,math.ceil(current/capacity));total+=current
    return int(total+1)


def evaluate_range_query(tree, query: np.ndarray, object_count:int, capacity:int) -> dict[str,Any]:
    q=tuple(map(float,query)); results=[]; total=indexv=leafv=0;depth=Counter();stack=[(tree.root,0)]
    started=time.perf_counter()
    while stack:
        node,d=stack.pop();total+=1;depth[d]+=1
        if node.is_leaf:leafv+=1
        else:indexv+=1
        br=node.get_bounding_rect()
        if br is not None and not intersects(_rtuple(br),q):continue
        if node.is_leaf:
            for e in node.entries:
                if intersects(_rtuple(e.rect),q):results.append(int(e.data))
        else:
            for e in node.entries:
                if intersects(_rtuple(e.rect),q):stack.append((e.child,d+1))
    us=(time.perf_counter()-started)*1e6
    cnt,s,x=result_signature(results);lower=capacity_io_lower_bound(cnt,object_count,capacity)
    return {"result_count":cnt,"id_sum":s,"id_xor":x,"index_node_accesses":indexv,"leaf_node_accesses":leafv,"total_node_accesses":total,
            "normalized_io":float(total/max(lower,1)),"io_lower_bound":lower,"latency_us":us,"depth_accesses_json":json.dumps(dict(sorted(depth.items())))}


def evaluate_range_suite(tree, queries: np.ndarray, object_count:int, capacity:int, query_type:str, workload:str)->list[dict[str,Any]]:
    rows=[]
    for i,q in enumerate(np.asarray(queries,dtype=np.float32)):
        r=evaluate_range_query(tree,q,object_count,capacity);r.update({"query_index":i,"query_type":query_type,"workload":workload,"query_uid":f"{query_type}|{workload}|{i:05d}"});rows.append(r)
    return rows


def _mindist(point:np.ndarray,rect)->float:
    x=float(point[0]); y=float(point[1])
    xmin=float(rect.min_x); ymin=float(rect.min_y); xmax=float(rect.max_x); ymax=float(rect.max_y)
    dx = xmin - x if x < xmin else (x - xmax if x > xmax else 0.0)
    dy = ymin - y if y < ymin else (y - ymax if y > ymax else 0.0)
    return math.hypot(dx, dy)


def evaluate_knn_query(tree, point:np.ndarray, k:int)->dict[str,Any]:
    p=np.asarray(point,dtype=np.float64);seq=0;queue=[];root=tree.root
    heapq.heappush(queue,(_mindist(p,root.get_bounding_rect()),seq,root,0));seq+=1
    best=[];total=indexv=leafv=0;depth=Counter();started=time.perf_counter()
    while queue:
        dmin,_,node,d=heapq.heappop(queue)
        worst=-best[0][0] if len(best)>=k else float("inf")
        if len(best)>=k and dmin>worst+1e-12:break
        total+=1;depth[d]+=1
        if node.is_leaf:
            leafv+=1
            for e in node.entries:
                dist=_mindist(p,e.rect);ident=int(e.data)
                item=(-dist,-ident,ident)
                if len(best)<k:heapq.heappush(best,item)
                else:
                    wd=-best[0][0];wid=-best[0][1]
                    if dist<wd-1e-12 or (abs(dist-wd)<=1e-12 and ident<wid):heapq.heapreplace(best,item)
        else:
            indexv+=1
            worst=-best[0][0] if len(best)>=k else float("inf")
            for e in node.entries:
                md=_mindist(p,e.rect)
                if md<=worst+1e-12:
                    heapq.heappush(queue,(md,seq,e.child,d+1));seq+=1
    us=(time.perf_counter()-started)*1e6
    vals=sorted([(-a,c) for a,_,c in best],key=lambda z:(z[0],z[1]))
    ids=[v[1] for v in vals[:k]];kth=float(vals[min(k,len(vals))-1][0]) if vals else float("inf")
    cnt,s,x=result_signature(ids)
    return {"k":int(k),"result_count":cnt,"id_sum":s,"id_xor":x,"kth_distance":kth,"index_node_accesses":indexv,"leaf_node_accesses":leafv,
            "total_node_accesses":total,"latency_us":us,"depth_accesses_json":json.dumps(dict(sorted(depth.items())))}


def evaluate_knn_suite(tree,points:np.ndarray,k:int,workload:str)->list[dict[str,Any]]:
    rows=[]
    for i,p in enumerate(np.asarray(points,dtype=np.float32)):
        r=evaluate_knn_query(tree,p,k);r.update({"query_index":i,"query_type":"knn","workload":workload,"query_uid":f"knn|{workload}|{i:05d}"});rows.append(r)
    return rows


def tree_metrics(tree,capacity:int)->dict[str,Any]:
    q=[(tree.root,0)];occ=[];by=defaultdict(list);sibling_overlap=[];nodes=leaves=internal=0
    while q:
        node,d=q.pop();nodes+=1;occ.append(len(node.entries));by[d].append(len(node.entries))
        if node.is_leaf:leaves+=1
        else:
            internal+=1
            boxes=np.asarray([_rtuple(e.rect) for e in node.entries],dtype=np.float32)
            sibling_overlap.append(normalized_overlap_pair_area(boxes))
            for e in node.entries:q.append((e.child,d+1))
    return {"height":1+max(by,default=0),"total_nodes":nodes,"leaf_nodes":leaves,"internal_nodes":internal,
            "mean_fill":float(np.mean(occ)/capacity) if occ else 0.0,"p50_fill":float(np.median(occ)/capacity) if occ else 0.0,
            "min_entries":int(min(occ,default=0)),"max_entries":int(max(occ,default=0)),"mean_sibling_overlap":float(np.mean(sibling_overlap)) if sibling_overlap else 0.0,
            "by_depth":{str(d):{"nodes":len(v),"mean_entries":float(np.mean(v)),"fill":float(np.mean(v)/capacity)} for d,v in sorted(by.items())}}


def summarize_rows(rows:list[dict[str,Any]],method:str,dataset:str)->list[dict[str,Any]]:
    out=[]
    keys=sorted(set((r["query_type"],r["workload"]) for r in rows))
    for qt,w in keys:
        sub=[r for r in rows if r["query_type"]==qt and r["workload"]==w]
        acc=np.asarray([float(r["total_node_accesses"]) for r in sub]);lat=np.asarray([float(r["latency_us"]) for r in sub])
        out.append({"dataset":dataset,"method":method,"query_type":qt,"workload":w,"queries":len(sub),"mean_node_accesses":float(acc.mean()),"p50_node_accesses":float(np.median(acc)),
                    "p95_node_accesses":float(np.quantile(acc,.95)),"mean_latency_us":float(lat.mean()),"p50_latency_us":float(np.median(lat)),"p95_latency_us":float(np.quantile(lat,.95))})
    return out
