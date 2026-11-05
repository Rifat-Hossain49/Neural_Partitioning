from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .eval import capacity_io_lower_bound
from .utils import atomic_json, derive_seed, read_csv, sha256_file, write_csv
from .workload import KNN_K


def run_cmd(cmd: Sequence[Any], log: Path, env: Mapping[str,str] | None = None) -> str:
    log=Path(log);log.parent.mkdir(parents=True,exist_ok=True)
    print("$"," ".join(map(str,cmd)),flush=True)
    cp=subprocess.run([str(x) for x in cmd],text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,env=dict(env) if env else None)
    log.write_text(cp.stdout,encoding="utf-8",errors="replace")
    if cp.returncode:
        raise RuntimeError(f"command failed ({cp.returncode}): {' '.join(map(str,cmd))}\n"+"\n".join(cp.stdout.splitlines()[-80:]))
    return cp.stdout


def discover_platon_support(input_root: Path) -> tuple[Path,Path]:
    matches=[]
    for g in Path(input_root).rglob("generate_author_cutlist.py"):
        if g.parent.name!="platon_native": continue
        support=g.parent.parent;author=support/"vendor"/"PLATON_upstream"
        req=[support/"platon_native"/"platon_bulk_load.cpp",support/"platon_native"/"platon_query_node_access.cpp",author/"learned-packing"/"mcts.py",author/"spatialindex-src-1.9.3"/"CMakeLists.txt"]
        if all(p.is_file() for p in req):matches.append((support,author))
    if not matches:
        raise FileNotFoundError("Attach thesis-full / author PLATON support dataset.")
    matches.sort(key=lambda x:(0 if "lom_npn_bestorder_teacher_pairrepack_b113_phased_v12" in str(x[0]) else 1,len(str(x[0])),str(x[0])))
    return matches[0]


def _find_library(author_root: Path, build_root: Path) -> Path:
    shipped=author_root/"spatialindex-src-1.9.3"/"bin"
    cand=[p for p in shipped.glob("libspatialindex.so*") if p.is_file() and p.stat().st_size>1_000_000]
    if cand:return max(cand,key=lambda p:p.stat().st_size).parent
    cand=[p for p in build_root.rglob("libspatialindex.so*") if p.is_file() and p.stat().st_size>1_000_000]
    if not cand:raise FileNotFoundError("libspatialindex not built")
    return max(cand,key=lambda p:p.stat().st_size).parent


def _prepare_lib(source_dir:Path,runtime_dir:Path)->Path:
    runtime_dir.mkdir(parents=True,exist_ok=True)
    cand=[p for p in source_dir.glob("libspatialindex.so*") if p.is_file() and p.stat().st_size>1_000_000]
    src=max(cand,key=lambda p:p.stat().st_size);dst=runtime_dir/src.name
    if not dst.is_file() or dst.stat().st_size!=src.stat().st_size:shutil.copy2(src,dst)
    for name in ("libspatialindex.so.6","libspatialindex.so"):
        a=runtime_dir/name
        if a.exists() or a.is_symlink():a.unlink()
        try:a.symlink_to(dst.name)
        except OSError:shutil.copy2(dst,a)
    return dst


def compile_native(input_root:Path,output_root:Path,code_root:Path)->dict[str,str]:
    nroot=Path(output_root)/"native_runtime";marker=nroot/"NATIVE.json"
    support,author=discover_platon_support(input_root)
    source_root=author/"spatialindex-src-1.9.3";include=source_root/"include";build=nroot/"build";logs=nroot/"logs";bins=nroot/"bin";bins.mkdir(parents=True,exist_ok=True)
    sources=[(support/"platon_native"/"platon_bulk_load.cpp","platon_bulk_load"),(support/"platon_native"/"platon_query_node_access.cpp","platon_range"),(code_root/"native"/"platon_knn_metrics.cpp","platon_knn"),(code_root/"native"/"native_dynamic_rtree_build.cpp","dynamic_build")]
    h=hashlib.sha256()
    for src,_ in sources:
        h.update(src.name.encode());h.update(b"\0");h.update(sha256_file(src).encode());h.update(b"\0")
    source_fingerprint=h.hexdigest()
    if marker.is_file():
        try:
            m=json.loads(marker.read_text())
            ok=all(Path(m[k]).is_file() for k in ("bulk","range","knn","dynamic")) and Path(m["libdir"]).is_dir()
            if ok and m.get("source_fingerprint")==source_fingerprint:return m
        except Exception:pass
    try:libsrc=_find_library(author,build)
    except FileNotFoundError:
        run_cmd(["cmake","-S",source_root,"-B",build,"-DCMAKE_BUILD_TYPE=Release","-DSIDX_BUILD_TESTS=OFF"],logs/"cmake_configure.log")
        run_cmd(["cmake","--build",build,"--parallel","2"],logs/"cmake_build.log");libsrc=_find_library(author,build)
    lib=_prepare_lib(libsrc,nroot/"lib");libdir=lib.parent
    env=dict(os.environ);env["LD_LIBRARY_PATH"]=f"{libdir}:{env.get('LD_LIBRARY_PATH','')}"
    for src,name in sources:
        run_cmd(["g++","-O3","-std=c++17",src,f"-I{include}",lib,f"-Wl,-rpath,{libdir}","-o",bins/name],logs/f"compile_{name}.log",env=env)
    m={"support":str(support),"author":str(author),"libdir":str(libdir),"bulk":str(bins/"platon_bulk_load"),"range":str(bins/"platon_range"),"knn":str(bins/"platon_knn"),"dynamic":str(bins/"dynamic_build"),"source_fingerprint":source_fingerprint}
    atomic_json(marker,m);return m


def native_env(native:Mapping[str,str])->dict[str,str]:
    env=dict(os.environ);env["LD_LIBRARY_PATH"]=f"{native['libdir']}:{env.get('LD_LIBRARY_PATH','')}";return env


def write_records(path:Path,rects:np.ndarray)->None:
    path=Path(path)
    if path.is_file() and path.stat().st_size>100:return
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",encoding="utf-8",newline="\n") as f:
        for i,(xmin,ymin,xmax,ymax) in enumerate(np.asarray(rects,dtype=np.float64)):
            f.write(f"1 {i} {xmin:.9f} {ymin:.9f} {xmax:.9f} {ymax:.9f}\n")


def write_queries(path:Path,queries:np.ndarray)->None:
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",encoding="utf-8",newline="\n") as f:
        for i,(xmin,ymin,xmax,ymax) in enumerate(np.asarray(queries,dtype=np.float64)):
            f.write(f"2 {i} {xmin:.9f} {ymin:.9f} {xmax:.9f} {ymax:.9f}\n")


def write_platon_array(path:Path,rows:np.ndarray)->None:
    r=np.asarray(rows,dtype=np.float32);np.save(path,np.column_stack([r[:,0],r[:,2],r[:,1],r[:,3]]))


def flatten_range_point(suite:dict[str,Any])->tuple[np.ndarray,list[str],list[str],list[str]]:
    arr=[];uids=[];types=[];works=[]
    for name,q in suite["ranges"].items():
        for i,row in enumerate(q):arr.append(row);uids.append(f"range|{name}|{i:05d}");types.append("range");works.append(name)
    for i,(x,y) in enumerate(suite["points"]):
        arr.append([x,y,x,y]);uids.append(f"point|point|{i:05d}");types.append("point");works.append("point")
    return np.asarray(arr,dtype=np.float32),uids,types,works


def parse_range(path:Path,uids:list[str],types:list[str],works:list[str],nobj:int,capacity:int)->list[dict[str,Any]]:
    raw=read_csv(path)
    if len(raw)!=len(uids):raise RuntimeError(f"native range count {len(raw)} != {len(uids)}")
    out=[]
    for i,r in enumerate(raw):
        cnt=int(r["result_count"]);tot=int(r["total_node_accesses"]);lower=capacity_io_lower_bound(cnt,nobj,capacity)
        out.append({"query_uid":uids[i],"query_type":types[i],"workload":works[i],"query_index":i,"result_count":cnt,"id_sum":int(r["id_sum"]),"id_xor":int(r["id_xor"]),"index_node_accesses":int(r["index_node_accesses"]),"leaf_node_accesses":int(r["leaf_node_accesses"]),"total_node_accesses":tot,"normalized_io":tot/max(lower,1),"io_lower_bound":lower,"latency_us":float(r["elapsed_us"])})
    return out


def eval_native_tree(tree_base:Path,index_id:int,suite:dict[str,Any],nobj:int,capacity:int,out_root:Path,native:Mapping[str,str],method:str)->list[dict[str,Any]]:
    out_root=Path(out_root);out_root.mkdir(parents=True,exist_ok=True);env=native_env(native)
    q,uids,types,works=flatten_range_point(suite);qtxt=out_root/"range_point.txt";qcsv=out_root/"range_point.csv";write_queries(qtxt,q)
    if not qcsv.is_file():run_cmd([native["range"],qtxt,tree_base,str(index_id),qcsv,"4096"],out_root/"range.log",env)
    rows=parse_range(qcsv,uids,types,works,nobj,capacity)
    ktxt=out_root/"knn.txt";kcsv=out_root/"knn.csv";kuids=[];kworks=[];kid=0
    with ktxt.open("w",encoding="utf-8") as f:
        for k in KNN_K:
            for i,(x,y) in enumerate(suite["knn_points"]):
                kuids.append(f"knn|k{k}|{i:05d}");kworks.append(f"knn_k{k}");f.write(f"{kid} {float(x):.9f} {float(y):.9f} {int(k)}\n");kid+=1
    if not kcsv.is_file():run_cmd([native["knn"],tree_base,str(index_id),ktxt,kcsv],out_root/"knn.log",env)
    raw=read_csv(kcsv)
    if len(raw)!=len(kuids):raise RuntimeError("native KNN row count mismatch")
    for i,r in enumerate(raw):
        rows.append({"query_uid":kuids[i],"query_type":"knn","workload":kworks[i],"query_index":i,"k":int(r["k"]),"result_count":int(r["result_count"]),"id_sum":int(r["id_sum"]),"id_xor":int(r["id_xor"]),"kth_distance":float(r.get("max_result_distance",r.get("returned_distance",0.0))),"index_node_accesses":int(r["index_node_accesses"]),"leaf_node_accesses":int(r["leaf_node_accesses"]),"total_node_accesses":int(r["total_node_accesses"]),"normalized_io":"","io_lower_bound":"","latency_us":float(r["elapsed_us"])})
    for r in rows:r["method"]=method
    write_csv(out_root/f"{method}_PER_QUERY.csv",rows);return rows


def build_platon(dataset:str,rects:np.ndarray,construction_boxes:np.ndarray,capacity:int,root:Path,native:Mapping[str,str],seed:int,rollouts:int,steps:int,utilization:float,page_bytes:int)->tuple[Path,int,dict[str,Any]]:
    root=Path(root);root.mkdir(parents=True,exist_ok=True);meta=root/"CONSTRUCTION.json"
    if meta.is_file():
        m=json.loads(meta.read_text());return Path(m["tree_base"]),int(m["index_identifier"]),m
    records=root/"records.txt";data=root/"data.npy";queries=root/"construction.npy";write_records(records,rects);write_platon_array(data,rects);write_platon_array(queries,construction_boxes)
    support=Path(native["support"]);author=Path(native["author"]);cut=root/f"{dataset}_B{capacity}_cuts.txt";seed32=derive_seed("platon",seed,dataset)&0xffffffff
    t=time.perf_counter();run_cmd([sys.executable,support/"platon_native"/"generate_author_cutlist.py","--author-root",author,"--data",data,"--queries",queries,"--output",cut,"--branch",str(capacity),"--seed",str(seed32),"--rollouts",str(rollouts),"--simulation-steps",str(steps)],root/"cutlist.log");policy=time.perf_counter()-t
    tree_base=root/f"{dataset}_PLATON_B{capacity}";nominal=int(round(capacity/utilization));env=native_env(native)
    t=time.perf_counter();txt=run_cmd([native["bulk"],records,tree_base,str(nominal),str(utilization),cut,str(page_bytes)],root/"bulk.log",env);build=time.perf_counter()-t
    lines=[x for x in txt.splitlines() if x.startswith("PLATON_BUILD_RESULT")]
    if not lines:raise RuntimeError("missing PLATON_BUILD_RESULT")
    vals=dict(item.split("=",1) for item in lines[-1].split(",")[1:])
    if vals.get("valid_tree")!="1":raise RuntimeError(f"invalid PLATON tree {vals}")
    idx=int(vals["index_identifier"]);m={"method":"PLATON","tree_base":str(tree_base),"index_identifier":idx,"policy_seconds":policy,"physical_build_seconds":build,"total_construction_seconds":policy+build,"cut_sha256":sha256_file(cut),"capacity":capacity,"rollouts":rollouts,"steps":steps}
    atomic_json(meta,m);return tree_base,idx,m


def build_dynamic_native(method:str,rects:np.ndarray,capacity:int,root:Path,native:Mapping[str,str],fill_factor:float,page_bytes:int)->tuple[Path,int,dict[str,Any]]:
    root=Path(root);root.mkdir(parents=True,exist_ok=True);meta=root/"CONSTRUCTION.json"
    if meta.is_file():m=json.loads(meta.read_text());return Path(m["tree_base"]),int(m["index_identifier"]),m
    variant="rstar" if method.lower().startswith("rstar") else "quadratic"
    effective_fill = float(fill_factor) if variant == "rstar" else min(float(fill_factor), 0.5)
    records=root/"records.txt";write_records(records,rects);tree_base=root/f"{method}_B{capacity}";env=native_env(native)
    t=time.perf_counter();txt=run_cmd([native["dynamic"],records,tree_base,str(capacity),str(effective_fill),str(page_bytes),variant,"2"],root/"build.log",env);seconds=time.perf_counter()-t
    lines=[x for x in txt.splitlines() if x.startswith("DYNAMIC_RTREE_BUILD_RESULT")]
    if not lines:raise RuntimeError("missing dynamic build result")
    vals=dict(item.split("=",1) for item in lines[-1].split(",")[1:]);idx=int(vals["index_identifier"])
    m={"method":method,"variant":variant,"tree_base":str(tree_base),"index_identifier":idx,"total_construction_seconds":seconds,"capacity":capacity,"fill_factor":effective_fill};atomic_json(meta,m);return tree_base,idx,m
