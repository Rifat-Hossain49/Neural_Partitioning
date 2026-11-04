"""Preregistered frozen-model workload-conditioning pilot; no training."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import shutil
import struct
import sys
import time
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from realtrain.data import load_arizona, load_point_csv, locate_arizona_npy, locate_crimes_csv
from realtrain.eval import result_signature, tree_metrics
from realtrain.geometry import centers
from realtrain.tree import NeuralPolicy, SimplePolicy, build_policy_tree, load_tree, save_tree
from realtrain.utils import atomic_json, derive_seed, read_csv, sha256_array, sha256_file
from realtrain.workload import KNN_K, estimate_knn_surrogate_boxes, make_points
from scripts.kaggle.run_reviewer_checks import CHECKPOINT, MIX, Experiment, ResumeRequired, atomic_csv, depth_audit

PROTOCOL = 'waharp_workload_adaptation_pilot_v1_20261009'
ARMS = ('Horizontal', 'Vertical')
METHODS = ('Widest-Axis-Quarter', 'WAHARP-Horizontal', 'WAHARP-Vertical')
SEEDS = tuple(range(5))
AREAS = (0.0001, 0.0004, 0.001, 0.004, 0.01)
DATA = {
    'arizona': (1464257, '58eb17e00bbed2a46790f0e373e85198cfc4d2bb92a2355afa5b8db8f83ac5c6'),
    'crimes': (7697816, 'e229e88e15541ed945211487216c013cd14eebd55178f05749071983b290cd4b'),
}


def specification():
    return dict(protocol_id=PROTOCOL, checkpoint=CHECKPOINT, capacity=128,
                datasets=DATA, seeds=SEEDS, arms=ARMS, methods=METHODS,
                area_fractions=AREAS, elongated_fraction=0.8, aspect=16,
                robust_quantiles=(0.01, 0.99), center_jitter=0.001,
                construction_mix=dict(range=1960, point=140, knn_surrogate=700),
                evaluation_mix=dict(MIX), hist_bins=12, local_query_cap=128,
                primary='range accesses', bootstrap_draws=10000,
                practical_improvement=0.05, retrained=False, device='cpu', threads=1,
                query_engine='realtrain.eval', mixed_latency='descriptive single pass',
                original_benchmark_modified=False)


def json_canonical(value):
    return json.loads(json.dumps(value, sort_keys=True))


def seed_for(dataset, seed, label):
    return derive_seed(PROTOCOL, seed, f'{dataset}/{label}')


def range_pair(rects, extent, count, seed):
    if count % len(AREAS):
        raise ValueError('Range count must evenly allocate area tiers')
    rng = np.random.default_rng(seed)
    xy = centers(rects[rng.integers(0, len(rects), count)]).astype(np.float64)
    xy += rng.normal(size=xy.shape) * extent[None, :] * 0.001
    per_area = count // len(AREAS)
    area = np.repeat(np.asarray(AREAS), per_area)
    counts = np.full(len(AREAS), int(per_area * .8), dtype=int)
    counts[:int(round(count*.8))-int(counts.sum())] += 1
    elongated = np.concatenate([np.arange(per_area) < n for n in counts])
    aspect = np.where(elongated, 16., 1.)
    out = {}
    for arm, ratio in zip(ARMS, (aspect, 1 / aspect)):
        wh = np.column_stack((extent[0] * np.sqrt(area * ratio),
                              extent[1] * np.sqrt(area / ratio)))
        out[arm] = np.column_stack((xy - wh / 2, xy + wh / 2)).astype(np.float32)
    return out


def workload_arrays(rects, dataset, seed):
    c = centers(rects)
    extent = np.maximum(np.quantile(c, .99, axis=0) - np.quantile(c, .01, axis=0), 1e-6)
    del c
    construction = range_pair(rects, extent, 1960, seed_for(dataset, seed, 'construction-range'))
    evaluation = range_pair(rects, extent, 4200, seed_for(dataset, seed, 'evaluation-range'))
    cp = make_points(rects, 840, seed_for(dataset, seed, 'construction-points'))
    rng = np.random.default_rng(seed_for(dataset, seed, 'construction-knn'))
    ks = rng.choice(np.asarray(KNN_K), 700)
    kb = estimate_knn_surrogate_boxes(rects, cp[140:], ks, seed_for(dataset, seed, 'knn-radius'))
    pb = np.column_stack((cp[:140], cp[:140]))
    order = rng.permutation(2800)
    arrays = {'points': make_points(rects, 300, seed_for(dataset, seed, 'evaluation-points')),
              'knn_points': make_points(rects, 300, seed_for(dataset, seed, 'evaluation-knn')),
              'robust_extent': extent}
    for arm in ARMS:
        arrays[f'construction_{arm}'] = np.concatenate((construction[arm], pb, kb))[order]
        arrays[f'range_{arm}'] = evaluation[arm]
    return arrays


def suite_for(arrays, arm):
    q = arrays[f'range_{arm}']
    return dict(ranges={f'area_{i}': a for i, a in enumerate(np.split(q, len(AREAS)))},
                points=arrays['points'], knn_points=arrays['knn_points'])


def suite_identity(suite):
    return {**{k: sha256_array(v) for k, v in suite['ranges'].items()},
            'points': sha256_array(suite['points']), 'knn_points': sha256_array(suite['knn_points'])}


def checked_rows(root):
    marker = json.loads((root/'COMPLETE.json').read_text())
    if marker['csv_sha256'] != sha256_file(root/'PER_QUERY.csv'):
        raise RuntimeError('Completed query CSV changed')
    rows = read_csv(root/'PER_QUERY.csv')
    if (len(rows) != 6000 or len({r['query_uid'] for r in rows}) != 6000
            or Counter(r['query_type'] for r in rows) != MIX):
        raise RuntimeError('Incomplete or duplicate evaluation query coverage')
    return rows


def correctness(left, right):
    a = {r['query_uid']: r for r in left}
    b = {r['query_uid']: r for r in right}
    if set(a) != set(b) or len(a) != 6000:
        raise RuntimeError('Correctness comparison has incomplete or mismatched queries')
    checks = Counter()
    for key, x in a.items():
        y = b[key]
        agrees = x['query_type'] == y['query_type']
        if x['query_type'] == 'knn':
            agrees &= (int(x['result_count']) == int(y['result_count']) == int(x['k'])
                       and math.isfinite(float(x['kth_distance']))
                       and math.isclose(float(x['kth_distance']), float(y['kth_distance']),
                                        rel_tol=2e-6, abs_tol=2e-6))
        else:
            agrees &= all(int(x[k]) == int(y[k]) for k in ('result_count', 'id_sum', 'id_xor'))
        checks[x['query_type']] += not agrees
    if sum(checks.values()):
        raise RuntimeError(f'Correctness mismatch: {dict(checks)}')
    return [dict(query_family=f, mismatches=checks[f], passed=True) for f in ('range', 'point', 'knn')]


def brute_probes(rects, suite, rows):
    by_uid = {r['query_uid']: r for r in rows}
    probes = []
    for name, queries in suite['ranges'].items():
        for i in (0, len(queries)//2, len(queries)-1):
            probes.append((f'range|{name}|{i:05d}', queries[i]))
    for i in np.linspace(0, len(suite['points'])-1, 8, dtype=int):
        p = suite['points'][i]
        probes.append((f'point|point|{i:05d}', np.concatenate((p, p))))
    for key, q in probes:
        mask = ((rects[:, 0] <= q[2]) & (rects[:, 2] >= q[0])
                & (rects[:, 1] <= q[3]) & (rects[:, 3] >= q[1]))
        sig = result_signature(np.flatnonzero(mask))
        if sig != tuple(int(by_uid[key][k]) for k in ('result_count', 'id_sum', 'id_xor')):
            raise RuntimeError(f'Independent brute-force range/point mismatch: {key}')
    for i in np.linspace(0, len(suite['knn_points'])-1, 3, dtype=int):
        x, y = map(float, suite['knn_points'][i])
        r = np.asarray(rects, dtype=np.float64)
        dx = np.maximum(np.maximum(r[:, 0]-x, x-r[:, 2]), 0.)
        dy = np.maximum(np.maximum(r[:, 1]-y, y-r[:, 3]), 0.)
        dist = np.hypot(dx, dy)
        del r, dx, dy
        for k in KNN_K:
            key = f'knn|knn_k{k}|{i:05d}'
            expected = float(np.partition(dist, k-1)[k-1])
            if not math.isclose(expected, float(by_uid[key]['kth_distance']), rel_tol=2e-6, abs_tol=2e-6):
                raise RuntimeError(f'Independent brute-force kNN mismatch: {key}')
    return dict(range_point_probes=len(probes), knn_probes=3*len(KNN_K), mismatches=0, passed=True)


def tree_identity(tree):
    digest = hashlib.sha256()
    stack = [tree.root]
    while stack:
        node = stack.pop()
        digest.update(struct.pack('<?I', node.is_leaf, len(node.entries)))
        for entry in node.entries:
            r = entry.rect
            digest.update(struct.pack('<4d', r.min_x, r.min_y, r.max_x, r.max_y))
            if node.is_leaf:
                digest.update(struct.pack('<q', int(entry.data)))
            else:
                stack.append(entry.child)
    return digest.hexdigest()


class AuditedPolicy(NeuralPolicy):
    def __init__(self, *args):
        super().__init__(*args)
        self.first_state = {}
        self.by_level = {}

    def choose(self, entries, level, query_cap=128):
        action, q, features = super().choose(entries, level, query_cap)
        self.by_level.setdefault(str(level), Counter())[action] += 1
        if str(level) not in self.first_state:
            self.first_state[str(level)] = dict(action=action, objects=len(entries),
                                               feature_sha256=sha256_array(features),
                                               queries_sha256=sha256_array(q))
        return action, q, features


def family_summary(rows, meta):
    out = []
    for family in ('all', 'range', 'point', 'knn'):
        sub = [r for r in rows if family == 'all' or r['query_type'] == family]
        accesses = np.array([int(r['total_node_accesses']) for r in sub])
        out.append({**meta, 'query_family': family, 'queries': len(sub),
                    'logical_node_accesses': int(accesses.sum()),
                    'mean_latency_us': float(np.mean([float(r['latency_us']) for r in sub]))})
    return out


def seed_comparison(left, right, meta):
    a = {r['query_uid']: r for r in left if r['query_type'] == 'range'}
    b = {r['query_uid']: r for r in right if r['query_type'] == 'range'}
    if set(a) != set(b) or len(a) != 4200:
        raise RuntimeError('Primary comparison has mismatched query coverage')
    x = np.array([int(a[k]['total_node_accesses']) for k in sorted(a)])
    y = np.array([int(b[k]['total_node_accesses']) for k in sorted(a)])
    if np.any(x < 1) or np.any(y < 1):
        raise RuntimeError('Invalid logical node access count')
    return {**meta, 'queries': len(x), 'matched_accesses': int(x.sum()),
            'baseline_accesses': int(y.sum()), 'ratio': float(x.sum()/y.sum()),
            'strict_wins': int((x < y).sum()), 'ties': int((x == y).sum()),
            'losses': int((x > y).sum())}


def aggregate_comparisons(rows):
    out = []
    keys = sorted({(r['dataset'], r['test_arm'], r['comparator']) for r in rows})
    for dataset, arm, comparator in keys:
        sub = [r for r in rows if (r['dataset'], r['test_arm'], r['comparator']) == (dataset, arm, comparator)]
        if len(sub) != 5 or {int(r['seed']) for r in sub} != set(SEEDS):
            continue
        logs = np.log([float(r['ratio']) for r in sorted(sub, key=lambda r: int(r['seed']))])
        rng = np.random.default_rng(seed_for(dataset, 0, f'statistics/{arm}/{comparator}'))
        draws = np.exp(logs[rng.integers(0, 5, (10000, 5))].mean(axis=1))
        ratio = float(np.exp(logs.mean()))
        low, high = map(float, np.quantile(draws, (.025, .975)))
        adjusted_low, adjusted_high = map(float, np.quantile(draws, (.00625, .99375)))
        out.append(dict(dataset=dataset, test_arm=arm, comparator=comparator, seeds=5,
                        geometric_mean_ratio=ratio, ci95_low=low, ci95_high=high,
                        ci9875_low=adjusted_low, ci9875_high=adjusted_high,
                        reliable_advantage=ratio < 1 and adjusted_high < 1,
                        practical_five_percent=ratio <= .95 and adjusted_high < 1))
    return out


class Pilot(Experiment):
    def __init__(self, args):
        self.args, self.output = args, args.output
        self.started = time.monotonic()
        self.session = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8]
        self.output.mkdir(parents=True, exist_ok=True)
        if args.resume:
            if (self.output/'PROTOCOL.json').exists():
                raise RuntimeError('Resume must restore into an empty output directory')
            shutil.copytree(args.resume, self.output, dirs_exist_ok=True)
        protocol = self.output/'PROTOCOL.json'
        expected = json_canonical(specification())
        if protocol.exists() and json.loads(protocol.read_text()) != expected:
            raise RuntimeError('Resume protocol differs from the preregistration')
        atomic_json(protocol, expected)
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        self.model = None

    def state(self, status, **extra):
        markers = list(self.output.glob('cells/*/seed_*/*/*/COMPLETE.json'))
        atomic_json(self.output/'experiment_state.json', dict(protocol_id=PROTOCOL, status=status,
                    session=self.session, completed_evaluation_cells=len(markers), target_evaluation_cells=60,
                    elapsed_session_seconds=time.monotonic()-self.started, remaining_session_seconds=self.remaining(),
                    retrained=False, datasets_run=self.args.datasets, **extra))

    def context(self, dataset):
        root = self.output/'datasets'/dataset
        root.mkdir(parents=True, exist_ok=True)
        path = root/'rectangles.npy'
        if not path.exists() or not (root/'DATASET.json').exists():
            self.guard(1800)
            if dataset == 'arizona':
                rects, metadata = load_arizona(locate_arizona_npy(self.args.inputs), 0)
            else:
                rects, metadata = load_point_csv(locate_crimes_csv(self.args.inputs), 0)
            if (len(rects), sha256_array(rects)) != DATA[dataset]:
                raise RuntimeError('Dataset differs from the locked paper data')
            temp = path.with_suffix('.tmp.npy')
            np.save(temp, rects)
            temp.replace(path)
            atomic_json(root/'DATASET.json', metadata)
        rects = np.load(path, mmap_mode='r')
        if (len(rects), sha256_array(rects)) != DATA[dataset]:
            raise RuntimeError('Cached dataset identity mismatch')
        return rects

    def workloads(self, rects, dataset, seed):
        root = self.output/'workloads'/dataset/f'seed_{seed}'
        root.mkdir(parents=True, exist_ok=True)
        path, meta = root/'arrays.npz', root/'IDENTITIES.json'
        if not path.exists() or not meta.exists():
            arrays = workload_arrays(rects, dataset, seed)
            temp = path.with_suffix('.tmp.npz')
            np.savez(temp, **arrays)
            temp.replace(path)
            atomic_json(meta, {k: sha256_array(v) for k, v in arrays.items()})
        with np.load(path) as archive:
            arrays = {k: archive[k] for k in archive.files}
        if not meta.exists() or json.loads(meta.read_text()) != {k: sha256_array(v) for k, v in arrays.items()}:
            raise RuntimeError('Saved workload arrays changed')
        return arrays

    def built_tree(self, rects, dataset, seed, method, arrays):
        root = self.output/'trees'/dataset/(f'seed_{seed}' if method.startswith('WAHARP') else 'shared')/method
        root.mkdir(parents=True, exist_ok=True)
        construction = arrays[f'construction_{method.removeprefix("WAHARP-")}'] if method.startswith('WAHARP') else None
        identity = dict(dataset_sha256=DATA[dataset][1], capacity=128, method=method,
                        construction_sha256=sha256_array(construction) if construction is not None else None,
                        checkpoint=CHECKPOINT if construction is not None else None)
        cache, meta = root/'tree.pkl.gz', root/'CONSTRUCTION.json'
        if cache.exists():
            saved = json.loads(meta.read_text())
            if saved['identity'] != identity or saved['cache_sha256'] != sha256_file(cache):
                raise RuntimeError('Saved tree binding changed')
            return load_tree(cache)[0], saved
        self.guard(1800)
        started = time.perf_counter()
        policy = (AuditedPolicy(self.model, 'cpu', 128, 12, construction)
                  if construction is not None else SimplePolicy(method, 128))
        tree, diag = build_policy_tree(rects, policy, 128)
        elapsed = time.perf_counter()-started
        if construction is not None:
            diag.update(first_state_by_level=policy.first_state,
                        actions_by_level={k: dict(v) for k, v in policy.by_level.items()})
        diag.update(tree_signature=tree_identity(tree), structure={**tree_metrics(tree, 128), **depth_audit(tree)})
        temporary = cache.with_suffix('.tmp')
        save_tree(temporary, tree, diag)
        saved = dict(identity=identity, session=self.session, construction_seconds=elapsed,
                     cache_sha256=sha256_file(temporary), diagnostics=diag)
        atomic_json(meta, saved)
        temporary.replace(cache)
        return tree, saved

    def evaluate_cell(self, tree, tree_meta, suite, dataset, seed, method, arm, rects):
        root = self.output/'cells'/dataset/f'seed_{seed}'/method/arm
        root.mkdir(parents=True, exist_ok=True)
        binding = dict(protocol=PROTOCOL, tree_signature=tree_meta['diagnostics']['tree_signature'],
                       queries=suite_identity(suite), dataset=dataset, seed=seed, method=method, test_arm=arm)
        path = root/'BINDING.json'
        if path.exists() and json.loads(path.read_text()) != binding:
            raise RuntimeError('Evaluation checkpoint binding mismatch')
        atomic_json(path, binding)
        if (root/'COMPLETE.json').exists():
            return checked_rows(root)
        (root/'chunks').mkdir(exist_ok=True)
        rows = super().evaluate(tree, suite, DATA[dataset][0], 128, method, root)
        if method == 'Widest-Axis-Quarter':
            probe = self.output/'correctness'/dataset/f'seed_{seed}'/f'{arm}_BRUTE.json'
            if not probe.exists():
                self.guard()
                atomic_json(probe, {**brute_probes(rects, suite, rows), 'binding': binding})
            if json.loads(probe.read_text()).get('binding') != binding:
                raise RuntimeError('Independent probe checkpoint binding changed')
        atomic_json(root/'COMPLETE.json', dict(queries=6000, csv_sha256=sha256_file(root/'PER_QUERY.csv'),
                    session=self.session, tree_signature=binding['tree_signature']))
        self.state('RUNNING', last_completed=f'{dataset}/{seed}/{method}/{arm}')
        return checked_rows(root)

    def outputs(self):
        summaries, pairs, checks, selectivity, builds = [], [], [], [], []
        complete_cells = 0
        for dataset in DATA:
            for path in (self.output/'trees'/dataset).glob('**/CONSTRUCTION.json'):
                m = json.loads(path.read_text())
                builds.append(dict(dataset=dataset, method=m['identity']['method'], seed=path.parents[1].name,
                                   session=m['session'], construction_seconds=m['construction_seconds'],
                                   tree_signature=m['diagnostics']['tree_signature']))
            for seed in SEEDS:
                seed_root = self.output/'cells'/dataset/f'seed_{seed}'
                arrays = None
                if list(seed_root.glob('*/*/COMPLETE.json')):
                    workload = self.output/'workloads'/dataset/f'seed_{seed}'
                    with np.load(workload/'arrays.npz') as archive:
                        arrays = {k: archive[k] for k in archive.files}
                    if json.loads((workload/'IDENTITIES.json').read_text()) != {k: sha256_array(v) for k, v in arrays.items()}:
                        raise RuntimeError('Workload identity changed during final audit')
                for arm in ARMS:
                    collected = {}
                    for method in METHODS:
                        root = self.output/'cells'/dataset/f'seed_{seed}'/method/arm
                        if (root/'COMPLETE.json').exists():
                            tree_root = self.output/'trees'/dataset/(f'seed_{seed}' if method.startswith('WAHARP') else 'shared')/method
                            tree_meta = json.loads((tree_root/'CONSTRUCTION.json').read_text())
                            diag = tree_meta['diagnostics']
                            construction = arrays[f'construction_{method.removeprefix("WAHARP-")}'] if method.startswith('WAHARP') else None
                            expected_tree = dict(dataset_sha256=DATA[dataset][1], capacity=128, method=method,
                                construction_sha256=sha256_array(construction) if construction is not None else None,
                                checkpoint=CHECKPOINT if construction is not None else None)
                            expected_binding = dict(protocol=PROTOCOL, tree_signature=diag['tree_signature'],
                                queries=suite_identity(suite_for(arrays, arm)), dataset=dataset, seed=seed,
                                method=method, test_arm=arm)
                            if (tree_meta['identity'] != expected_tree or not diag['validation']['passed']
                                    or not diag['structure']['equal_leaf_depth']
                                    or json.loads((root/'BINDING.json').read_text()) != expected_binding
                                    or json.loads((root/'COMPLETE.json').read_text())['tree_signature'] != diag['tree_signature']):
                                raise RuntimeError('Completed evaluation/tree provenance mismatch')
                            collected[method] = checked_rows(root)
                            complete_cells += 1
                            summaries.extend(family_summary(collected[method], dict(dataset=dataset, seed=seed,
                                             method=method, test_arm=arm)))
                    baseline = collected.get('Widest-Axis-Quarter')
                    if baseline is None:
                        continue
                    brute = self.output/'correctness'/dataset/f'seed_{seed}'/f'{arm}_BRUTE.json'
                    if not brute.exists():
                        raise RuntimeError('Completed reference cell lacks independent correctness probes')
                    record = json.loads(brute.read_text())
                    binding = self.output/'cells'/dataset/f'seed_{seed}'/'Widest-Axis-Quarter'/arm/'BINDING.json'
                    if (not record.get('passed') or record.get('mismatches') != 0
                            or record.get('binding') != json.loads(binding.read_text())):
                        raise RuntimeError('Independent correctness probes failed')
                    checks.append(dict(dataset=dataset, seed=seed, test_arm=arm, method='Widest-Axis-Quarter',
                                       query_family='independent_probes',
                                       **{k: v for k, v in record.items() if k != 'binding'}))
                    for area in range(5):
                        counts = np.array([int(r['result_count']) for r in baseline
                                           if r['query_type'] == 'range' and r['workload'] == f'area_{area}'])
                        selectivity.append(dict(dataset=dataset, seed=seed, test_arm=arm, area_fraction=AREAS[area],
                                           queries=len(counts), mean_results=float(counts.mean()),
                                           p50_results=float(np.median(counts)), p95_results=float(np.quantile(counts, .95)),
                                           empty_fraction=float((counts == 0).mean()), mean_selectivity=float(counts.mean()/DATA[dataset][0])))
                    for method in METHODS[1:]:
                        if method in collected:
                            checks.extend({**r, 'dataset': dataset, 'seed': seed, 'test_arm': arm, 'method': method}
                                          for r in correctness(collected[method], baseline))
                    matched = collected.get(f'WAHARP-{arm}')
                    opposite = ARMS[1] if arm == ARMS[0] else ARMS[0]
                    if matched is not None:
                        pairs.append(seed_comparison(matched, baseline, dict(dataset=dataset, seed=seed,
                                     test_arm=arm, comparator='Widest-Axis-Quarter')))
                        if f'WAHARP-{opposite}' in collected:
                            pairs.append(seed_comparison(matched, collected[f'WAHARP-{opposite}'], dict(dataset=dataset,
                                         seed=seed, test_arm=arm, comparator='Mismatched-WAHARP')))
        comparisons = aggregate_comparisons(pairs)
        for name, rows in (('RESULTS.csv', [r for r in summaries if r['query_family'] == 'all']),
                           ('QUERY_FAMILY_RESULTS.csv', [r for r in summaries if r['query_family'] != 'all']),
                           ('SEED_COMPARISONS.csv', pairs), ('COMPARISONS.csv', comparisons),
                           ('CORRECTNESS.csv', checks), ('SELECTIVITY.csv', selectivity), ('CONSTRUCTION_TIMES.csv', builds)):
            atomic_csv(self.output/name, rows)
        done = complete_cells == 60 and len(pairs) == 40 and len(comparisons) == 8
        audit = dict(status='COMPLETE_VERIFIED' if done else 'PARTIAL_VERIFIED',
                     completed_evaluation_cells=complete_cells, target_evaluation_cells=60,
                     correctness_mismatches=0, seed_comparisons=len(pairs), retrained=False)
        atomic_json(self.output/'COMPLETION_AUDIT.json', audit)
        lines = ['# Frozen-Model Workload Adaptation Pilot', '',
                 'Separate from the original locked benchmark. Five independent workload seeds; B=128.',
                 'Range accesses are primary. Ratios below one favor matched WAHARP.',
                 'Intervals resample seeds, not individual queries. Five seeds have limited precision.', '',
                 '| Dataset | Test workload | Comparator | Range ratio | 98.75% interval |',
                 '|---|---|---|---:|---|']
        for r in comparisons:
            lines.append(f"| {r['dataset']} | {r['test_arm']} | {r['comparator']} | {r['geometric_mean_ratio']:.6f} | [{r['ci9875_low']:.6f}, {r['ci9875_high']:.6f}] |")
        adapt = [r for r in comparisons if r['comparator'] == 'Mismatched-WAHARP']
        widest = [r for r in comparisons if r['comparator'] == 'Widest-Axis-Quarter']
        signal = done and len(adapt) == 4 and all(r['reliable_advantage'] for r in adapt)
        strong = signal and len(widest) == 4 and all(r['practical_five_percent'] for r in widest)
        lines += ['', f"Audit: {audit['status']}; cells {complete_cells}/60.",
                  f'Predeclared crossover criterion satisfied: {signal}.',
                  f'Predeclared stronger pilot criterion satisfied: {strong}.',
                  'False criteria are not suppressed or redefined after observing results.',
                  'See SELECTIVITY.csv: equal area does not imply equal result selectivity.',
                  'Latency is descriptive Python timing, not a confirmed production break-even.',
                  'No acceptance guarantee; no universal superiority claim follows from a pilot.']
        temporary = self.output/'FINAL_REPORT.md.tmp'
        temporary.write_text('\n'.join(lines)+'\n', encoding='utf-8')
        temporary.replace(self.output/'FINAL_REPORT.md')
        if done:
            self.plot(comparisons)
        return audit

    def plot(self, comparisons):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
        for ax, dataset in zip(axes, DATA):
            for j, comparator in enumerate(('Widest-Axis-Quarter', 'Mismatched-WAHARP')):
                rows = [next(r for r in comparisons if r['dataset'] == dataset and r['test_arm'] == arm
                             and r['comparator'] == comparator) for arm in ARMS]
                y = np.array([r['geometric_mean_ratio'] for r in rows])
                ax.errorbar(np.arange(2)+.12*(j*2-1), y,
                            yerr=[np.maximum(0, y-np.array([r['ci9875_low'] for r in rows])),
                                  np.maximum(0, np.array([r['ci9875_high'] for r in rows])-y)],
                            fmt='o', capsize=4, label=comparator)
            ax.axhline(1, color='gray', linestyle='--', linewidth=1)
            ax.set_xticks(range(2), ARMS)
            ax.set_title(dataset.title())
            ax.set_xlabel('Held-out workload')
        axes[0].set_ylabel('Matched WAHARP / comparator range accesses')
        axes[1].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(self.output/'WORKLOAD_CROSSOVER.png', dpi=180)
        plt.close(fig)

    def run(self):
        self.load_model()
        self.state('RUNNING')
        for dataset in self.args.datasets:
            dataset_cells = list(self.output.glob(f'cells/{dataset}/seed_*/*/*/COMPLETE.json'))
            if len(dataset_cells) == 30:
                continue
            rects = self.context(dataset)
            for seed in SEEDS:
                arrays = self.workloads(rects, dataset, seed)
                for method in METHODS:
                    roots = [self.output/'cells'/dataset/f'seed_{seed}'/method/arm for arm in ARMS]
                    if all((r/'COMPLETE.json').exists() for r in roots):
                        continue
                    tree, tree_meta = self.built_tree(rects, dataset, seed, method, arrays)
                    for arm in ARMS:
                        suite = suite_for(arrays, arm)
                        self.evaluate_cell(tree, tree_meta, suite, dataset, seed, method, arm, rects)
                    del tree
                    gc.collect()
                    self.outputs()
                    if method.startswith('WAHARP'):
                        (self.output/'trees'/dataset/f'seed_{seed}'/method/'tree.pkl.gz').unlink(missing_ok=True)
            del rects
            gc.collect()
            (self.output/'trees'/dataset/'shared'/'Widest-Axis-Quarter'/'tree.pkl.gz').unlink(missing_ok=True)
        verified = self.outputs()
        self.state('COMPLETE_VERIFIED' if verified['status'] == 'COMPLETE_VERIFIED' else 'SHARD_COMPLETE')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputs', type=Path, default=Path('/kaggle/input'))
    parser.add_argument('--output', type=Path, default=Path('/kaggle/working/WAHARP_WORKLOAD_ADAPTATION_B128'))
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--datasets', nargs='+', choices=tuple(DATA), default=list(DATA))
    parser.add_argument('--hours', type=float, default=10.5)
    args = parser.parse_args()
    if not 0 < args.hours <= 10.5 or len(set(args.datasets)) != len(args.datasets):
        parser.error('Use a nonduplicated dataset scope and a budget up to 10.5 hours')
    experiment = Pilot(args)
    try:
        experiment.run()
    except ResumeRequired as exc:
        experiment.outputs()
        experiment.state('PAUSED_RESUME_REQUIRED', reason=str(exc))
        print('Checkpoint saved; resume required:', exc, flush=True)
    except Exception as exc:
        experiment.state('FAILED_REQUIRES_INSPECTION', reason=f'{type(exc).__name__}: {exc}')
        raise


if __name__ == '__main__':
    main()
