from __future__ import annotations

import argparse
import gc
import json
import math
import os
import shutil
import time
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .baselines import build_str_tree, build_tgs
from .config import Config, DEFAULT_CONFIG
from .data import load_all_domains
from .eval import evaluate_knn_suite, evaluate_range_suite, summarize_rows, tree_metrics
from .native import build_dynamic_native, build_platon, compile_native, eval_native_tree
from .teacher import label_state, load_records, sample_states_for_domain, save_records
from .train import load_model, train_one
from .tree import build_neural_tree, load_tree, save_tree
from .utils import Budget, TimeBudgetStop, atomic_json, derive_seed, geometric_mean, read_csv, seed_everything, sha256_file, write_csv
from .workload import KNN_K, create_dataset_workloads, load_query_suite

DOMAIN_NAMES=("twitter","crimes","arizona")


def _load_cfg(path:Path|None)->Config:
    if path is None:return DEFAULT_CONFIG
    raw=json.loads(Path(path).read_text())
    return Config(**raw)


def _state_marker(root:Path,cfg:Config,status:str,**extra):
    atomic_json(root/"STATE.json",{"protocol_id":cfg.protocol_id,"seed":cfg.seed,"capacity":cfg.capacity,"status":status,"updated_unix":time.time(),**extra})


def _evaluate_python_method(tree,suite:dict[str,Any],nobj:int,capacity:int,method:str)->list[dict[str,Any]]:
    rows=[]
    for name,q in suite["ranges"].items():rows+=evaluate_range_suite(tree,q,nobj,capacity,"range",name)
    p=suite["points"];pb=np.column_stack([p[:,0],p[:,1],p[:,0],p[:,1]]).astype(np.float32);rows+=evaluate_range_suite(tree,pb,nobj,capacity,"point","point")
    for k in KNN_K:rows+=evaluate_knn_suite(tree,suite["knn_points"],k,f"knn_k{k}")
    for r in rows:r["method"]=method
    return rows


def _mean_access(rows:list[dict[str,Any]])->float:
    return float(np.mean([float(r["total_node_accesses"]) for r in rows])) if rows else float("inf")


def _make_initial_teacher(domains,workroots,root,cfg):
    paths=[]
    for di,name in enumerate(DOMAIN_NAMES):
        path=root/"teacher"/f"initial_{name}.npz";paths.append(path)
        if path.is_file():continue
        construction=np.load(workroots[name]/"construction_boxes.npy")
        records=sample_states_for_domain(domains[name],construction,di,cfg.initial_states_per_domain,cfg.capacity,cfg.state_min_pages,cfg.state_max_pages,cfg.local_query_cap,cfg.hist_bins,cfg.candidates_per_state,derive_seed(cfg.protocol_id,cfg.seed,f"teacher_initial|{name}"),cfg.teacher_overlap_weight,cfg.teacher_margin_weight,progress_prefix=f"[{name}] ")
        save_records(path,records)
    return paths


def _train_bootstrap(arrays,root,cfg):
    ck=root/"models"/"bootstrap.pt"
    if not ck.is_file():
        train_one(arrays,ck,derive_seed(cfg.protocol_id,cfg.seed,"bootstrap_model"),cfg.bootstrap_epochs,cfg.train_batch_size,cfg.learning_rate,cfg.weight_decay,cfg.early_stop_patience,True)
    return ck


def _make_dagger(domains,workroots,bootstrap_path,root,cfg):
    device="cuda:0" if torch.cuda.is_available() else "cpu";model=load_model(bootstrap_path,device)
    paths=[]
    for di,name in enumerate(DOMAIN_NAMES):
        path=root/"teacher"/f"dagger_{name}.npz";paths.append(path)
        if path.is_file():continue
        rng=np.random.default_rng(derive_seed(cfg.protocol_id,cfg.seed,f"dagger|{name}"));records=[];construction=np.load(workroots[name]/"construction_boxes.npy")
        n=min(cfg.shadow_rows_per_domain,len(domains[name]));idx=rng.choice(len(domains[name]),size=n,replace=False) if n<len(domains[name]) else np.arange(n);sub=domains[name][idx]
        target=cfg.dagger_states_per_domain;expected=max(1,math.ceil(n/cfg.capacity));prob=min(0.85,max(0.10,target/max(expected,1)*0.75))
        def capture(entries,queries,level):
            if len(records)>=target or rng.random()>prob:return
            try:records.append(label_state(entries,queries,di,level,cfg.capacity,cfg.hist_bins,cfg.candidates_per_state,rng,cfg.teacher_overlap_weight,cfg.teacher_margin_weight))
            except Exception:return
            if len(records)%100==0:print(f"[{name}] DAgger labels {len(records)}/{target}",flush=True)
        build_neural_tree(sub,construction,model,device,cfg.capacity,cfg.hist_bins,cfg.local_query_cap,capture)
        # If stochastic capture undershot, fill with fresh real states rather than changing target size.
        if len(records)<target:
            extra=sample_states_for_domain(domains[name],construction,di,target-len(records),cfg.capacity,cfg.state_min_pages,cfg.state_max_pages,cfg.local_query_cap,cfg.hist_bins,cfg.candidates_per_state,derive_seed(cfg.protocol_id,cfg.seed,f"dagger_fill|{name}"),cfg.teacher_overlap_weight,cfg.teacher_margin_weight,progress_prefix=f"[{name}/fill] ")
            records.extend(extra)
        save_records(path,records[:target])
    return paths


def _train_ensemble(arrays,root,cfg):
    paths=[];metas=[]
    for m in range(cfg.ensemble_members):
        p=root/"models"/f"member_{m}.pt";paths.append(p)
        if not p.is_file():
            meta=train_one(arrays,p,derive_seed(cfg.protocol_id,cfg.seed,f"final_member|{m}"),cfg.final_epochs,cfg.train_batch_size,cfg.learning_rate,cfg.weight_decay,cfg.early_stop_patience,True)
        else:meta=json.loads(p.with_suffix(".training.json").read_text())
        metas.append(meta)
    atomic_json(root/"models"/"ENSEMBLE_TRAINING.json",{"members":metas})
    return paths


def _select_member(member_paths,domains,workroots,root,cfg):
    sp=root/"models"/"SHADOW_SELECTION.json"
    if sp.is_file():return int(json.loads(sp.read_text())["selected_member"])
    device="cuda:0" if torch.cuda.is_available() else "cpu";results=[]
    for mi,p in enumerate(member_paths):
        model=load_model(p,device);per={}
        for name in DOMAIN_NAMES:
            rng=np.random.default_rng(derive_seed(cfg.protocol_id,cfg.seed,f"shadow_objects|{name}"));n=min(cfg.shadow_rows_per_domain,len(domains[name]));idx=rng.choice(len(domains[name]),size=n,replace=False) if n<len(domains[name]) else np.arange(n);sub=domains[name][idx]
            construction=np.load(workroots[name]/"construction_boxes.npy");tree,_=build_neural_tree(sub,construction,model,device,cfg.capacity,cfg.hist_bins,cfg.local_query_cap)
            suite=load_query_suite(workroots[name],"validation");rows=_evaluate_python_method(tree,suite,len(sub),cfg.capacity,f"member_{mi}");per[name]=_mean_access(rows)
            print(f"shadow member={mi} dataset={name} mean_access={per[name]:.6f}",flush=True)
        results.append(per)
    best_by_domain={name:min(r[name] for r in results) for name in DOMAIN_NAMES};scores=[]
    for mi,r in enumerate(results):
        ratios=[r[n]/max(best_by_domain[n],1e-12) for n in DOMAIN_NAMES];scores.append({"member":mi,"means":r,"domain_ratios":dict(zip(DOMAIN_NAMES,ratios)),"score":max(ratios)+0.25*float(np.mean(ratios))})
    selected=min(scores,key=lambda x:x["score"])["member"]
    payload={"selected_member":int(selected),"selection_rule":"minimize worst-domain validation access ratio plus 0.25 mean ratio; PLATON/final queries not used","members":scores};atomic_json(sp,payload);return int(selected)


def _bootstrap_ci(neural:np.ndarray,baseline:np.ndarray,seed:int,draws:int)->dict[str,float]:
    neural=np.asarray(neural,dtype=np.float64);baseline=np.asarray(baseline,dtype=np.float64);ratio=float(neural.sum()/baseline.sum());rng=np.random.default_rng(seed);vals=[];n=len(neural)
    for _ in range(draws):
        idx=rng.integers(0,n,size=n);vals.append(float(neural[idx].sum()/baseline[idx].sum()))
    return {"ratio":ratio,"ci_low":float(np.quantile(vals,.025)),"ci_high":float(np.quantile(vals,.975))}


def _compare(neural_rows,all_rows,dataset,cfg):
    nmap={r["query_uid"]:r for r in neural_rows};summ=[]
    for method,rows in all_rows.items():
        if method=="Neural":continue
        bmap={r["query_uid"]:r for r in rows};common=[u for u in nmap if u in bmap]
        na=np.asarray([float(nmap[u]["total_node_accesses"]) for u in common]);ba=np.asarray([float(bmap[u]["total_node_accesses"]) for u in common])
        wins=int(np.sum(na<ba));ties=int(np.sum(na==ba));loss=int(np.sum(na>ba));ci=_bootstrap_ci(na,ba,derive_seed(cfg.protocol_id,cfg.seed,f"bootstrap|{dataset}|{method}"),cfg.bootstrap_draws)
        summ.append({"dataset":dataset,"baseline":method,"queries":len(common),"neural_total_accesses":float(na.sum()),"baseline_total_accesses":float(ba.sum()),"ratio_neural_over_baseline":ci["ratio"],"bootstrap_ci_low":ci["ci_low"],"bootstrap_ci_high":ci["ci_high"],"strict_win_rate":wins/max(len(common),1),"tie_rate":ties/max(len(common),1),"loss_rate":loss/max(len(common),1)})
    return summ


def _correctness(reference_rows,method_rows)->dict[str,Any]:
    ref={r["query_uid"]:r for r in reference_rows};mism=[];knn=[]
    for method,rows in method_rows.items():
        if rows is reference_rows:continue
        for r in rows:
            p=ref.get(r["query_uid"])
            if p is None:continue
            if r["query_type"] in ("range","point"):
                a=(int(r["result_count"]),int(r["id_sum"]),int(r["id_xor"]));b=(int(p["result_count"]),int(p["id_sum"]),int(p["id_xor"]))
                if a!=b and len(mism)<50:mism.append({"method":method,"query_uid":r["query_uid"],"got":a,"reference":b})
            else:
                a=float(r["kth_distance"]);b=float(p["kth_distance"])
                if not math.isclose(a,b,rel_tol=2e-6,abs_tol=2e-6) and len(knn)<50:knn.append({"method":method,"query_uid":r["query_uid"],"got":a,"reference":b})
    return {"passed":len(mism)==0 and len(knn)==0,"range_point_mismatch_examples":mism,"knn_distance_mismatch_examples":knn,"range_point_mismatch_count":len(mism),"knn_distance_mismatch_count":len(knn)}


def _write_report(root,cfg,selection,all_pair,construction):
    lines=["# WAHARP Real-Train v1 Result","",f"Protocol: `{cfg.protocol_id}`",f"Capacity: `{cfg.capacity}`",f"Selected neural member: `{selection}`","","## Pairwise node-access results","","| Dataset | Baseline | Neural/Baseline | 95% CI | Win rate |","|---|---|---:|---:|---:|"]
    for r in all_pair:lines.append(f"| {r['dataset']} | {r['baseline']} | {r['ratio_neural_over_baseline']:.6f} | [{r['bootstrap_ci_low']:.6f}, {r['bootstrap_ci_high']:.6f}] | {100*r['strict_win_rate']:.2f}% |")
    lines += ["","## Construction time","","| Dataset | Method | Seconds |","|---|---|---:|"]
    for r in construction:lines.append(f"| {r['dataset']} | {r['method']} | {r['seconds']:.3f} |")
    lines += ["","The primary scientific metric is logical node access. Native-vs-Python latency is implementation-level only.","PLATON was used only as an evaluation baseline, not as a teacher, action source, fallback, or checkpoint-selection signal."]
    (root/"FINAL_REPORT.md").write_text("\n".join(lines)+"\n",encoding="utf-8")


def _compact_zip(root:Path)->Path:
    out=root/"WAHARP_REALTRAIN_V1_HANDOFF.zip"
    skip_ext={".dat",".idx"}
    with zipfile.ZipFile(out,"w",compression=zipfile.ZIP_DEFLATED,compresslevel=5) as z:
        for p in root.rglob("*"):
            if not p.is_file() or p==out:continue
            if p.suffix in skip_ext or p.name in ("records.txt",):continue
            if p.stat().st_size>200_000_000:continue
            z.write(p,p.relative_to(root))
    return out


def run(input_root:Path,output_root:Path,cfg:Config)->None:
    output_root=Path(output_root);output_root.mkdir(parents=True,exist_ok=True);cfg.dump(output_root/"CONFIG.json");_state_marker(output_root,cfg,"RUNNING")
    budget=Budget(cfg.wall_budget_seconds,cfg.reserve_seconds);seed_everything(cfg.seed)
    try:
        if torch.cuda.is_available():print("GPUs:",torch.cuda.device_count(),[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],flush=True)
        if torch.cuda.device_count()<2:print("WARNING: expected 2 GPUs; training will still run but may be slower",flush=True)
        budget.require(120,"data loading")
        domains,metadata=load_all_domains(input_root,cfg.twitter_max_rows,cfg.crimes_max_rows,cfg.arizona_max_rows);atomic_json(output_root/"DATASETS.json",metadata)
        workroots={}
        for name in DOMAIN_NAMES:
            wr=output_root/"workloads"/name;workroots[name]=wr
            create_dataset_workloads(domains[name],name,wr,cfg.protocol_id,cfg.seed,cfg.construction_query_count,cfg.validation_queries_per_range_condition,cfg.validation_point_queries,cfg.validation_knn_queries_per_k,cfg.final_queries_per_range_condition,cfg.final_point_queries,cfg.final_knn_queries_per_k)
        budget.require(1800,"initial teacher generation")
        initial_paths=_make_initial_teacher(domains,workroots,output_root,cfg);initial=load_records(initial_paths)
        bootstrap=_train_bootstrap(initial,output_root,cfg)
        budget.require(1200,"DAgger state generation")
        dagger_paths=_make_dagger(domains,workroots,bootstrap,output_root,cfg);combined=load_records(initial_paths+dagger_paths)
        member_paths=_train_ensemble(combined,output_root,cfg)
        budget.require(1200,"shadow member selection")
        selected=_select_member(member_paths,domains,workroots,output_root,cfg);device="cuda:0" if torch.cuda.is_available() else "cpu";model=load_model(member_paths[selected],device)
        native=None
        if cfg.run_platon or cfg.run_guttman_native or cfg.run_rstar_native:
            budget.require(300,"native preflight");native=compile_native(input_root,output_root,Path(__file__).resolve().parents[1])
        all_pair=[];construction_rows=[];all_correct=True
        for name in DOMAIN_NAMES:
            budget.require(1800,f"final dataset {name}")
            ds=output_root/"final"/name;ds.mkdir(parents=True,exist_ok=True);suite=load_query_suite(workroots[name],"final");construction=np.load(workroots[name]/"construction_boxes.npy");rects=domains[name]
            methods={};buildmeta={}
            # Neural-only method.
            tpath=ds/"Neural.tree.pkl.gz";dpath=ds/"Neural_CONSTRUCTION.json";ppath=ds/"Neural_PER_QUERY.csv"
            if tpath.is_file() and dpath.is_file():tree,diag=load_tree(tpath);buildmeta["Neural"]=json.loads(dpath.read_text())
            else:
                tree,diag=build_neural_tree(rects,construction,model,device,cfg.capacity,cfg.hist_bins,cfg.local_query_cap);save_tree(tpath,tree,diag);buildmeta["Neural"]={"method":"Neural","seconds":diag["build_seconds"],**diag};atomic_json(dpath,buildmeta["Neural"])
            if ppath.is_file():methods["Neural"]=read_csv(ppath)
            else:methods["Neural"]=_evaluate_python_method(tree,suite,len(rects),cfg.capacity,"Neural");write_csv(ppath,methods["Neural"])
            atomic_json(ds/"Neural_TREE_METRICS.json",tree_metrics(tree,cfg.capacity))
            del tree
            gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            if cfg.run_str:
                p=ds/"STR_PER_QUERY.csv";m=ds/"STR_CONSTRUCTION.json"
                if p.is_file():methods["STR"]=read_csv(p);buildmeta["STR"]=json.loads(m.read_text())
                else:
                    tr,meta=build_str_tree(rects,cfg.capacity);buildmeta["STR"]={"method":"STR","seconds":meta["build_seconds"],**meta};atomic_json(m,buildmeta["STR"]);methods["STR"]=_evaluate_python_method(tr,suite,len(rects),cfg.capacity,"STR");write_csv(p,methods["STR"]);atomic_json(ds/"STR_TREE_METRICS.json",tree_metrics(tr,cfg.capacity));del tr;gc.collect()
            if cfg.run_tgs:
                p=ds/"TGS_PER_QUERY.csv";m=ds/"TGS_CONSTRUCTION.json"
                if p.is_file():methods["TGS"]=read_csv(p);buildmeta["TGS"]=json.loads(m.read_text())
                else:
                    budget.require(900,f"TGS {name}");tr,meta=build_tgs(rects,cfg.capacity);buildmeta["TGS"]={"method":"TGS","seconds":meta["build_seconds"],**meta};atomic_json(m,buildmeta["TGS"]);methods["TGS"]=_evaluate_python_method(tr,suite,len(rects),cfg.capacity,"TGS");write_csv(p,methods["TGS"]);atomic_json(ds/"TGS_TREE_METRICS.json",tree_metrics(tr,cfg.capacity));del tr;gc.collect()
            if cfg.run_platon:
                pr=ds/"PLATON";tb,idx,meta=build_platon(name,rects,construction,cfg.capacity,pr,native,cfg.seed,cfg.platon_rollouts,cfg.platon_simulation_steps,cfg.platon_utilization,cfg.storage_page_bytes);methods["PLATON"]=eval_native_tree(tb,idx,suite,len(rects),cfg.capacity,pr/"eval",native,"PLATON");buildmeta["PLATON"]={"method":"PLATON","seconds":meta["total_construction_seconds"],**meta}
            if cfg.run_guttman_native:
                gr=ds/"Guttman";tb,idx,meta=build_dynamic_native("Guttman",rects,cfg.capacity,gr,native,cfg.native_fill_factor,cfg.storage_page_bytes);methods["Guttman"]=eval_native_tree(tb,idx,suite,len(rects),cfg.capacity,gr/"eval",native,"Guttman");buildmeta["Guttman"]={"method":"Guttman","seconds":meta["total_construction_seconds"],**meta}
            if cfg.run_rstar_native:
                rr=ds/"RStar";tb,idx,meta=build_dynamic_native("RStar",rects,cfg.capacity,rr,native,cfg.native_fill_factor,cfg.storage_page_bytes);methods["RStar"]=eval_native_tree(tb,idx,suite,len(rects),cfg.capacity,rr/"eval",native,"RStar");buildmeta["RStar"]={"method":"RStar","seconds":meta["total_construction_seconds"],**meta}
            reference=methods.get("PLATON",methods["Neural"]);corr=_correctness(reference,methods);atomic_json(ds/"CORRECTNESS.json",corr);all_correct=all_correct and corr["passed"]
            wm=[]
            for meth,rows in methods.items():wm+=summarize_rows(rows,meth,name)
            write_csv(ds/"WORKLOAD_METRICS.csv",wm)
            pair=_compare(methods["Neural"],methods,name,cfg);write_csv(ds/"PAIRWISE.csv",pair);all_pair+=pair
            for meth,meta in buildmeta.items():construction_rows.append({"dataset":name,"method":meth,"seconds":float(meta.get("seconds",meta.get("total_construction_seconds",0.0)))})
            _state_marker(output_root,cfg,"RUNNING",completed_dataset=name,selected_member=selected,elapsed_seconds=budget.elapsed)
        write_csv(output_root/"PAIRWISE_ALL.csv",all_pair);write_csv(output_root/"CONSTRUCTION_ALL.csv",construction_rows)
        # Combined decision versus every baseline present on all three datasets.
        combined=[]
        for baseline in sorted(set(r["baseline"] for r in all_pair)):
            rows=[r for r in all_pair if r["baseline"]==baseline]
            if len(rows)==len(DOMAIN_NAMES):
                n=sum(r["neural_total_accesses"] for r in rows);b=sum(r["baseline_total_accesses"] for r in rows);combined.append({"baseline":baseline,"ratio_neural_over_baseline":n/b,"datasets":len(rows)})
        write_csv(output_root/"COMBINED_BASELINES.csv",combined)
        _write_report(output_root,cfg,selected,all_pair,construction_rows);handoff=_compact_zip(output_root)
        verdict={"decision":"COMPLETE","all_correctness_passed":all_correct,"selected_member":selected,"combined":combined,"handoff":str(handoff),"elapsed_seconds":budget.elapsed,"platon_used_for_training":False,"platon_used_for_selection":False,"final_queries_used_for_training_or_selection":False}
        atomic_json(output_root/"FINAL_DECISION.json",verdict);_state_marker(output_root,cfg,"COMPLETE",**verdict);print(json.dumps(verdict,indent=2),flush=True)
    except TimeBudgetStop as e:
        payload={"decision":"CONTINUE_WAHARP_REALTRAIN_REATTACH_STATE_AND_RERUN","reason":str(e),"elapsed_seconds":budget.elapsed,"remaining_seconds":budget.remaining};atomic_json(output_root/"CONTINUE.json",payload);_state_marker(output_root,cfg,"CONTINUE",**payload);_compact_zip(output_root);print(json.dumps(payload,indent=2),flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--input-root",type=Path,default=Path("/kaggle/input"));ap.add_argument("--output-root",type=Path,default=Path("/kaggle/working/WAHARP_REALTRAIN_V1_STATE"));ap.add_argument("--config",type=Path,default=None);args=ap.parse_args();run(args.input_root,args.output_root,_load_cfg(args.config))

if __name__=="__main__":main()
