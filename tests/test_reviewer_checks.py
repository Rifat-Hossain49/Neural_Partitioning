from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from scripts.kaggle.run_reviewer_checks import (Experiment, ResumeRequired, compare,
                                              query_batches, timing_tasks, uid)
from realtrain.utils import atomic_json, read_csv, sha256_file


def suite():
    return {"ranges": {f"range_{i}": np.zeros((300, 4), dtype=np.float32) for i in range(14)},
            "points": np.zeros((300, 2), dtype=np.float32),
            "knn_points": np.zeros((300, 2), dtype=np.float32)}


def fake_rows(tree, queries, *args):
    return [dict(result_count=0, id_sum=0, id_xor=0, kth_distance=0,
                 total_node_accesses=1) for _ in queries]


class ReviewerChecksTests(unittest.TestCase):
    def test_default_timing_schedule_is_unchanged(self):
        tasks = timing_tasks()
        self.assertEqual(len(tasks), 27)
        self.assertEqual(tasks[0], dict(dataset="arizona", capacity=128, repetition=0))
        self.assertEqual(tasks[-1], dict(dataset="twitter", capacity=512, repetition=2))

    def test_shard_rejects_duplicate_and_unsupported_tasks(self):
        task = dict(dataset="twitter", capacity=128, repetition=1)
        manifest = dict(protocol="waharp_reviewer_checks_v1_20261003", shard_id="test", tasks=[task])
        self.assertEqual(timing_tasks(manifest), [task])
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            timing_tasks({**manifest, "tasks": [task, task]})
        with self.assertRaises(ValueError):
            timing_tasks({**manifest, "tasks": [{**task, "capacity": 64}]})
        with self.assertRaises(ValueError):
            timing_tasks({**manifest, "tasks": [{**task, "admission_seconds": float("nan")}]})
        with self.assertRaises(ValueError):
            timing_tasks({**manifest, "tasks": []})

    def test_shard_executes_only_owned_task_and_never_claims_global_completion(self):
        with tempfile.TemporaryDirectory() as directory:
            experiment = Experiment.__new__(Experiment)
            experiment.output = Path(directory)
            experiment.shard = dict(shard_id="test")
            experiment.tasks = [dict(dataset="twitter", capacity=256, repetition=2, admission_seconds=3000)]
            experiment.load_model = Mock()
            experiment.state = Mock()
            experiment.outputs = Mock()
            experiment.guard = Mock()
            experiment.context = Mock(return_value=(None, None, None, None))
            experiment.timing = Mock()
            for dataset in ("arizona", "crimes", "twitter"):
                for capacity in (128, 256, 512):
                    for method in ("WAHARP", "Fixed-X-Quarter", "Widest-Axis-Quarter"):
                        atomic_json(experiment.output/"cells"/dataset/f"B{capacity}"/"methods"/method/"COMPLETE.json", {})
            experiment.run()
            experiment.context.assert_called_once_with("twitter")
            experiment.timing.assert_called_once_with("twitter", 256, 2, (None, None, None, None))
            experiment.guard.assert_called_once_with(3900)
            experiment.state.assert_called_with("SHARD_COMPLETE")

    def test_missing_requested_checkpoint_fails_before_repeating_work(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            experiment = Experiment.__new__(Experiment)
            experiment.output = root/"output"
            experiment.args = SimpleNamespace(inputs=root, expected_resume_state_sha256="missing")
            atomic_json(root/"old"/"experiment_state.json", dict(
                protocol_id="waharp_reviewer_checks_v1_20261003", completed_units=3))
            with self.assertRaisesRegex(RuntimeError, "Requested checkpoint is not mounted"):
                experiment.restore()
            self.assertFalse(experiment.output.exists())

    def test_resume_selects_exact_requested_state_not_another_candidate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            experiment = Experiment.__new__(Experiment)
            experiment.output = root/"output"
            atomic_json(root/"old"/"experiment_state.json", dict(
                protocol_id="waharp_reviewer_checks_v1_20261003", completed_units=99))
            expected = root/"requested"/"experiment_state.json"
            atomic_json(expected, dict(protocol_id="waharp_reviewer_checks_v1_20261003",
                                      completed_units=0, completed_quality_cells=0, completed_timing_pairs=0))
            experiment.args = SimpleNamespace(inputs=root, expected_resume_state_sha256=sha256_file(expected))
            experiment.restore()
            self.assertEqual(sha256_file(experiment.output/"experiment_state.json"), sha256_file(expected))

    def test_query_batches_cover_locked_mix_without_duplicate_ids(self):
        batches = list(query_batches(suite()))
        self.assertEqual(len(batches), 240)
        self.assertEqual(sum(len(b[3]) for b in batches if b[0] == "range"), 4200)
        self.assertEqual(sum(len(b[3]) for b in batches if b[0] == "point"), 300)
        self.assertEqual(sum(len(b[3]) for b in batches if b[0] == "knn"), 1500)

    def test_partial_query_evaluation_resumes_only_missing_batches(self):
        with tempfile.TemporaryDirectory() as directory:
            experiment = Experiment.__new__(Experiment)
            experiment.state = Mock()
            experiment.guard = Mock(side_effect=[None, None, ResumeRequired("test")])
            root = Path(directory)
            with (
                patch("scripts.kaggle.run_reviewer_checks.evaluate_range_suite", side_effect=fake_rows) as ranges,
                patch("scripts.kaggle.run_reviewer_checks.evaluate_knn_suite", side_effect=fake_rows) as knn,
            ):
                with self.assertRaises(ResumeRequired):
                    experiment.evaluate(None, suite(), 10, 4, "test", root)
                self.assertEqual(len(list((root/"chunks").glob("*.csv"))), 2)
                experiment.guard = Mock()
                rows = experiment.evaluate(None, suite(), 10, 4, "test", root)
                self.assertEqual(len(rows), 6000)
                self.assertEqual(len({uid(row) for row in rows}), 6000)
                self.assertEqual(ranges.call_count+knn.call_count, 240)
                summaries, checks = compare(rows, rows, "test", 4, "identity")
                self.assertTrue(all(r["passed"] for r in checks))
                self.assertTrue(all(r["waharp_over_baseline"] == 1 for r in summaries))
                altered = [dict(row) for row in rows]
                altered[0]["id_sum"] = 1
                with self.assertRaisesRegex(RuntimeError, "Correctness failure"):
                    compare(rows, altered, "test", 4, "altered")

    def test_completed_timing_pair_resumes_pending_quality_not_the_timer(self):
        with tempfile.TemporaryDirectory() as directory:
            experiment = Experiment.__new__(Experiment)
            experiment.output = Path(directory)
            experiment.platon_quality = Mock()
            experiment.guard = Mock(side_effect=AssertionError("must not start another timer"))
            marker = experiment.output/"cells"/"arizona"/"B128"/"timing"/"rep_0"/"COMPLETE.json"
            atomic_json(marker, dict(session="original", waharp_seconds=1, platon_seconds=2))
            experiment.timing("arizona", 128, 0, (None, None, None, None))
            experiment.platon_quality.assert_called_once()
            self.assertEqual(experiment.platon_quality.call_args.args[3]["session"], "original")

    def test_output_aggregation_handles_build_metadata_with_capacity(self):
        with tempfile.TemporaryDirectory() as directory:
            experiment = Experiment.__new__(Experiment)
            experiment.output = Path(directory)
            root = experiment.output/"cells"/"arizona"/"B128"/"methods"/"WAHARP"
            atomic_json(root/"CONSTRUCTION.json", dict(capacity=128, method="WAHARP", build_seconds=1.5))
            atomic_json(root/"TREE_STRUCTURE.json", dict(capacity=128, method="WAHARP", height=3))
            experiment.outputs()
            rows = read_csv(experiment.output/"SIMPLE_POLICY_BUILD_TIMES.csv")
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["capacity"], "128")
            self.assertEqual(rows[0]["dataset"], "arizona")
            self.assertTrue((experiment.output/"FINAL_REPORT.md").exists())

    def test_resume_skips_input_loading_for_completed_work(self):
        with tempfile.TemporaryDirectory() as directory:
            experiment = Experiment.__new__(Experiment)
            experiment.output = Path(directory)
            experiment.load_model = Mock()
            experiment.state = Mock()
            experiment.outputs = Mock()
            experiment.context = Mock(side_effect=AssertionError("completed input must not be loaded"))
            for dataset in ("arizona", "crimes", "twitter"):
                for capacity in (128, 256, 512):
                    cell = experiment.output/"cells"/dataset/f"B{capacity}"
                    for method in ("WAHARP", "Fixed-X-Quarter", "Widest-Axis-Quarter"):
                        atomic_json(cell/"methods"/method/"COMPLETE.json", {})
                    for rep in range(3):
                        atomic_json(cell/"timing"/f"rep_{rep}"/"COMPLETE.json", {})
                    (cell/"PLATON").mkdir()
                    (cell/"PLATON"/"PAIRWISE.csv").touch()
            experiment.run()
            experiment.context.assert_not_called()
            experiment.state.assert_called_with("COMPLETE")


if __name__ == "__main__":
    unittest.main()
