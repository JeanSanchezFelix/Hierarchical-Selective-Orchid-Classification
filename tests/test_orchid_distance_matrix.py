import tempfile
import unittest
from pathlib import Path

import numpy as np

from model_compression.src.orchid.calibration import UnknownPolicy, apply_unknown_policy
from model_compression.src.orchid.phylogeny import build_distance_matrix, build_posterior_distance_summary, mean_phylogenetic_error
from model_compression.src.orchid.routing import HierarchicalCascadeRouter
from model_compression.src.orchid.evaluation import EvaluationRecord, summarize_records, write_evaluation_report
from model_compression.src.orchid.deployment_manifest import deployment_entry, make_deployment_manifest
from model_compression.src.orchid.model_packs import create_model_pack, install_model_pack


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

    def test_evaluation_summary_and_report(self):
        records = [
            EvaluationRecord("A::one", "A", "A::one", "A", "A", ("A", "B"), 0.8, False, None),
            EvaluationRecord("B::one", "B", "A::one", "A", "A", ("A", "B"), 0.4, True, "low_genus_confidence"),
        ]
        summary = summarize_records(records)
        self.assertEqual(summary["n_test_images"], 2)
        self.assertEqual(summary["unknown_rate"], 0.5)
        with tempfile.TemporaryDirectory() as directory:
            output = write_evaluation_report(records, summary, directory)
            self.assertTrue((output / "metrics.json").is_file())
            self.assertTrue((output / "predictions.csv").is_file())

    def test_compressed_model_pack_round_trip(self):
        metadata = {"task": "genus", "class_labels": ["A"], "model_name": "mobilenet_v2", "img_size": 224, "normalization": {"mean": [0], "std": [1]}}
        expert_metadata = {**metadata, "task": "genus_species", "target_genus": "A", "class_labels": ["A::one"]}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "router.tflite").write_bytes(b"router")
            (root / "expert_A.tflite").write_bytes(b"expert")
            manifest = make_deployment_manifest([
                deployment_entry(root / "router.tflite", metadata, "router"),
                deployment_entry(root / "expert_A.tflite", expert_metadata, "expert", "A"),
            ])
            pack = create_model_pack(manifest, root, root / "pack.zip")
            cache = install_model_pack(pack, root / "cache")
            self.assertTrue((cache / "models" / "router.tflite").is_file())
            self.assertTrue((cache / "models" / "expert_A.tflite").is_file())


if __name__ == "__main__":
    unittest.main()
