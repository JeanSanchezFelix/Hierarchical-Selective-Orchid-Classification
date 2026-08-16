import tempfile
import unittest
from pathlib import Path

import numpy as np

from model_compression.src.orchid.calibration import UnknownPolicy, apply_unknown_policy
from model_compression.src.orchid.phylogeny import build_distance_matrix, build_posterior_distance_summary, mean_phylogenetic_error
from model_compression.src.orchid.routing import HierarchicalCascadeRouter


class TestOrchidDistanceMatrix(unittest.TestCase):
    def test_unknown_policy_keeps_best_candidate(self):
        cascade = HierarchicalCascadeRouter(["A", "B"], {"A": ["A::one"], "B": ["B::one"]})
        result = cascade.route([0.0, 0.0], {"A": [1.0], "B": [0.0]}, top_k=2)
        decision = apply_unknown_policy(result, UnknownPolicy(0.9, 0.9, 0.2))
        self.assertTrue(decision.is_unknown)
        self.assertEqual(decision.best_candidate_species_id, "A::one")
        self.assertIn("best candidate", decision.display_label)

    def test_phylogeny_distances_need_reviewed_coverage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tree = root / "tree.tre"
            mapping = root / "mapping.csv"
            tree.write_text("(A:1,(B:1,C:1):1);", encoding="utf-8")
            mapping.write_text(
                "species_id,source_tip,mapping_status\nA::a,A,matched\nB::b,B,matched\nC::c,C,matched\n",
                encoding="utf-8",
            )
            labels, matrix, coverage = build_distance_matrix(tree, ["A::a", "B::b", "C::c"], mapping)
            self.assertEqual(coverage.ratio, 1.0)
            self.assertEqual(matrix.shape, (3, 3))
            self.assertAlmostEqual(mean_phylogenetic_error(["A::a"], ["B::b"], labels, matrix), matrix[0, 1])
            posterior_labels, posterior_mean, posterior_std, _ = build_posterior_distance_summary(root, ["A::a", "B::b", "C::c"], mapping)
            self.assertEqual(posterior_labels, labels)
            self.assertTrue(np.allclose(posterior_mean, matrix))
            self.assertTrue(np.allclose(posterior_std, 0.0))


if __name__ == "__main__":
    unittest.main()
