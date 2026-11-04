from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from realtrain.eval import evaluate_knn_suite, evaluate_range_suite
from realtrain.tree import SimplePolicy, build_policy_tree
from realtrain.utils import atomic_json
from scripts.kaggle.run_workload_adaptation import (
    ARMS, AREAS, MIX, Pilot, aggregate_comparisons, brute_probes, checked_rows,
    correctness, range_pair, seed_comparison, suite_for, tree_identity, workload_arrays,
)
from scripts.kaggle.run_reviewer_checks import atomic_csv


def rectangles(n=1000):
    xy = np.random.default_rng(10).uniform(.1, .8, (n, 2)).astype(np.float32)
    return np.column_stack((xy, xy + .0001))


def fake_rows():
    out = []
    for family, count in MIX.items():
        for i in range(count):
            out.append(dict(query_uid=f'{family}|{i}', query_type=family, total_node_accesses=10,
                            result_count=1, id_sum=0, id_xor=0, k=1, kth_distance=0.))
    return out


class WorkloadAdaptationTests(unittest.TestCase):
    def test_paired_range_centers_areas_and_orientation(self):
        ext = np.array([.5, .4])
        for n in (1960, 4200):
            pair = range_pair(rectangles(), ext, n, 20261009)
            h, v = (pair[a] for a in ARMS)
            np.testing.assert_allclose((h[:, :2]+h[:, 2:])/2, (v[:, :2]+v[:, 2:])/2, atol=1e-7)
            hw, vw = h[:, 2:]-h[:, :2], v[:, 2:]-v[:, :2]
            np.testing.assert_allclose(hw.prod(axis=1), vw.prod(axis=1), rtol=.0003, atol=1e-10)
            aspect = hw[:, 0]/ext[0]/(hw[:, 1]/ext[1])
            self.assertEqual(int((aspect > 2).sum()), int(n*.8))
            self.assertEqual(int((aspect < 2).sum()), int(n*.2))
            np.testing.assert_allclose(hw[:, 0]/ext[0], vw[:, 1]/ext[1], rtol=.0003)

    def test_construction_evaluation_and_seeds_are_distinct(self):
        a = workload_arrays(rectangles(), 'arizona', 0)
        b = workload_arrays(rectangles(), 'arizona', 1)
        self.assertEqual(a['construction_Horizontal'].shape, (2800, 4))
        self.assertFalse(np.array_equal(a['construction_Horizontal'][:100], a['range_Horizontal'][:100]))
        self.assertFalse(np.array_equal(a['range_Horizontal'], b['range_Horizontal']))
        for arm in ARMS:
            suite = suite_for(a, arm)
            self.assertEqual(sum(len(x) for x in suite['ranges'].values()), 4200)
            self.assertEqual(len(suite['points']), 300)
            self.assertEqual(len(suite['knn_points']), 300)
        self.assertEqual(len(AREAS), 5)

    def test_correctness_and_primary_comparison_include_unfavorable_results(self):
        reference = fake_rows()
        worse = [{**r, 'total_node_accesses': 11} for r in reference]
        self.assertTrue(all(r['passed'] for r in correctness(worse, reference)))
        comparison = seed_comparison(worse, reference, dict(seed=0))
        self.assertAlmostEqual(comparison['ratio'], 1.1)
        self.assertEqual(comparison['losses'], 4200)
        self.assertEqual(comparison['strict_wins'], 0)

    def test_correctness_detects_signatures_distances_and_missing_queries(self):
        a = fake_rows()
        wrong = [dict(r) for r in a]
        wrong[0]['id_xor'] = 1
        with self.assertRaisesRegex(RuntimeError, 'Correctness mismatch'):
            correctness(wrong, a)
        wrong = [dict(r) for r in a]
        wrong[-1]['kth_distance'] = .1
        with self.assertRaisesRegex(RuntimeError, 'Correctness mismatch'):
            correctness(wrong, a)
        with self.assertRaisesRegex(RuntimeError, 'incomplete'):
            correctness(a[:-1], a)

    def test_bootstrap_uses_all_five_seeds_and_does_not_rename_losses(self):
        rows = [dict(dataset='arizona', test_arm='Horizontal', comparator='Widest-Axis-Quarter',
                     seed=s, ratio=1.1) for s in range(5)]
        self.assertEqual(aggregate_comparisons(rows[:-1]), [])
        r = aggregate_comparisons(rows)[0]
        self.assertAlmostEqual(r['geometric_mean_ratio'], 1.1)
        self.assertFalse(r['reliable_advantage'])
        self.assertFalse(r['practical_five_percent'])

    def test_independent_probes_match_small_real_tree_and_detect_corruption(self):
        rects = rectangles()
        tree, _ = build_policy_tree(rects, SimplePolicy('Widest-Axis-Quarter', 128))
        arrays = workload_arrays(rects, 'arizona', 0)
        suite = suite_for(arrays, 'Horizontal')
        # Probe helper uses first, middle, and last queries in each group.
        suite['ranges'] = {k: v[:3] for k, v in suite['ranges'].items()}
        suite['points'] = suite['points'][:8]
        suite['knn_points'] = suite['knn_points'][:3]
        rows = []
        for name, queries in suite['ranges'].items():
            rows += evaluate_range_suite(tree, queries, len(rects), 128, 'range', name)
        points = np.column_stack((suite['points'], suite['points']))
        rows += evaluate_range_suite(tree, points, len(rects), 128, 'point', 'point')
        for k in (1, 5, 25, 125, 625):
            rows += evaluate_knn_suite(tree, suite['knn_points'], k, f'knn_k{k}')
        self.assertTrue(brute_probes(rects, suite, rows)['passed'])
        signature = tree_identity(tree)
        self.assertEqual(signature, tree_identity(tree))
        rows[0]['result_count'] += 1
        with self.assertRaisesRegex(RuntimeError, 'brute-force'):
            brute_probes(rects, suite, rows)

    def test_completed_csv_mutation_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            atomic_csv(root/'PER_QUERY.csv', fake_rows())
            atomic_json(root/'COMPLETE.json', dict(csv_sha256='not-the-file-hash'))
            with self.assertRaisesRegex(RuntimeError, 'CSV changed'):
                checked_rows(root)

    def test_partial_report_cannot_claim_success(self):
        with tempfile.TemporaryDirectory() as directory:
            pilot = Pilot.__new__(Pilot)
            pilot.output = Path(directory)
            result = pilot.outputs()
            self.assertEqual(result['status'], 'PARTIAL_VERIFIED')
            self.assertEqual(result['completed_evaluation_cells'], 0)
            text = (pilot.output/'FINAL_REPORT.md').read_text()
            self.assertIn('criterion satisfied: False', text)

    def test_evaluation_refuses_changed_query_binding_before_using_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            pilot = Pilot.__new__(Pilot)
            pilot.output = Path(directory)
            root = pilot.output/'cells'/'arizona'/'seed_0'/'WAHARP-Horizontal'/'Horizontal'
            atomic_json(root/'BINDING.json', dict(changed=True))
            suite = suite_for(workload_arrays(rectangles(), 'arizona', 0), 'Horizontal')
            meta = dict(diagnostics=dict(tree_signature='tree'))
            with self.assertRaisesRegex(RuntimeError, 'binding mismatch'):
                pilot.evaluate_cell(None, meta, suite, 'arizona', 0, 'WAHARP-Horizontal', 'Horizontal', rectangles())

    def test_evaluation_checkpoints_batches_and_reuses_completed_cell(self):
        with tempfile.TemporaryDirectory() as directory:
            pilot = Pilot.__new__(Pilot)
            pilot.output = Path(directory)
            pilot.session = 'test-session'
            pilot.guard = lambda *args: None
            pilot.state = lambda *args, **kwargs: None
            suite = suite_for(workload_arrays(rectangles(), 'arizona', 0), 'Horizontal')
            meta = dict(diagnostics=dict(tree_signature='tree'))

            def ranges(tree, queries, n, capacity, family, workload):
                return [dict(result_count=0, id_sum=0, id_xor=0, total_node_accesses=1)
                        for _ in queries]

            def knn(tree, queries, k, workload):
                return [dict(result_count=k, id_sum=0, id_xor=0, total_node_accesses=1,
                             k=k, kth_distance=0.) for _ in queries]

            with patch('scripts.kaggle.run_reviewer_checks.evaluate_range_suite', side_effect=ranges), \
                 patch('scripts.kaggle.run_reviewer_checks.evaluate_knn_suite', side_effect=knn):
                rows = pilot.evaluate_cell(None, meta, suite, 'arizona', 0, 'WAHARP-Horizontal', 'Horizontal', rectangles())
            self.assertEqual(len(rows), 6000)
            root = pilot.output/'cells'/'arizona'/'seed_0'/'WAHARP-Horizontal'/'Horizontal'
            self.assertGreater(len(list((root/'chunks').glob('*.csv'))), 1)
            with patch('scripts.kaggle.run_reviewer_checks.evaluate_range_suite', side_effect=AssertionError('must not rerun')):
                cached = pilot.evaluate_cell(None, meta, suite, 'arizona', 0, 'WAHARP-Horizontal', 'Horizontal', rectangles())
            self.assertEqual(rows, cached)


if __name__ == '__main__':
    unittest.main()
