from __future__ import annotations

import unittest

import numpy as np
import torch

from realtrain.eval import evaluate_knn_query, evaluate_range_query, result_signature
from realtrain.tree import SimplePolicy, build_neural_tree, build_policy_tree


class AlwaysXQuarter(torch.nn.Module):
    def forward(self, x):
        scores = torch.ones((len(x), 80), device=x.device)
        scores[:, 0] = 0
        return scores


def leaves(tree):
    pending = [tree.root]
    result = []
    while pending:
        node = pending.pop()
        if node.is_leaf:
            result.append(tuple(e.data for e in node.entries))
        else:
            pending.extend(e.child for e in reversed(node.entries))
    return result


class SimplePolicyTests(unittest.TestCase):
    def test_widest_axis_and_tie(self):
        policy = SimplePolicy("Widest-Axis-Quarter", 4)
        x = np.array([[0, 0, 0, 0], [2, 1, 2, 1]], dtype=np.float32)
        self.assertEqual(policy.choose(x, 0)[0], 0)
        self.assertEqual(policy.choose(x[:, [1, 0, 3, 2]], 0)[0], 40)
        tied = np.array([[0, 0, 0, 0], [1, 1, 1, 1]], dtype=np.float32)
        self.assertEqual(policy.choose(tied, 0)[0], 0)

    def test_shared_executor_matches_neural_constant_action_at_all_levels(self):
        low = np.random.default_rng(23).uniform(0, 0.9, (385, 2)).astype(np.float32)
        rectangles = np.column_stack([low, low + 0.02])
        query = np.array([[0, 0, 1, 1]], dtype=np.float32)
        neural, nd = build_neural_tree(rectangles, query, AlwaysXQuarter(), "cpu", 4, 12, 128)
        simple, sd = build_policy_tree(rectangles, SimplePolicy("Fixed-X-Quarter", 4))
        self.assertEqual(leaves(neural), leaves(simple))
        self.assertEqual(nd["validation"], sd["validation"])
        self.assertEqual(nd["action_counts"], sd["action_counts"])
        self.assertGreater(nd["validation"]["height"], 2)
        self.assertEqual(sd["neural_decision_count"], 0)

    def test_simple_trees_preserve_range_point_and_knn_answers(self):
        low = np.random.default_rng(31).uniform(0, 0.9, (97, 2)).astype(np.float32)
        rectangles = np.column_stack([low, low + 0.01])
        for name in ("Fixed-X-Quarter", "Widest-Axis-Quarter"):
            tree, diag = build_policy_tree(rectangles, SimplePolicy(name, 4))
            self.assertTrue(diag["validation"]["passed"])
            for q in (np.array([0.1, 0.1, 0.7, 0.7]),
                      np.array([*low[0], *low[0]])):
                hits = np.flatnonzero((rectangles[:, 0] <= q[2]) &
                                      (rectangles[:, 1] <= q[3]) &
                                      (rectangles[:, 2] >= q[0]) &
                                      (rectangles[:, 3] >= q[1]))
                actual = evaluate_range_query(tree, q, len(rectangles), 4)
                self.assertEqual((actual["result_count"], actual["id_sum"], actual["id_xor"]),
                                 result_signature(hits))
            point = np.array([0.5, 0.5])
            delta = np.maximum(np.maximum(rectangles[:, :2] - point,
                                          point - rectangles[:, 2:]), 0)
            expected = np.sort(np.linalg.norm(delta, axis=1))[4]
            self.assertAlmostEqual(evaluate_knn_query(tree, point, 5)["kth_distance"], expected)


if __name__ == "__main__":
    unittest.main()
