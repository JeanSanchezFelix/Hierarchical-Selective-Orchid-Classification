import csv
import tempfile
import unittest
from pathlib import Path

from model_compression.src.orchid.paper_results import hierarchical_aurc, paired_bootstrap_hauc_difference, summarize_matrix


class PaperResultsTests(unittest.TestCase):
    def write_rows(self, path, rows):
        with Path(path).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)

    def test_paired_bootstrap_requires_identical_test_images(self):
        with tempfile.TemporaryDirectory() as directory:
            rows = [{"image_file": "A/a.jpg", "true_species_id": "A::a", "true_genus_id": "A", "predicted_species_id": "A::a", "predicted_genus_id": "A", "decision_level": "species", "confidence": "0.9"}]
            left, right = Path(directory) / "left.csv", Path(directory) / "right.csv"
            self.write_rows(left, rows)
            changed = [dict(rows[0], image_file="A/b.jpg")]
            self.write_rows(right, changed)
            with self.assertRaises(ValueError):
                paired_bootstrap_hauc_difference(left, right, samples=5)

    def test_hierarchical_aurc_and_matrix_summary(self):
        rows = [
            {"image_file": "A/a.jpg", "true_species_id": "A::a", "true_genus_id": "A", "predicted_species_id": "A::a", "predicted_genus_id": "A", "decision_level": "species", "confidence": "0.9"},
            {"image_file": "B/b.jpg", "true_species_id": "B::b", "true_genus_id": "B", "predicted_species_id": "", "predicted_genus_id": "B", "decision_level": "genus", "confidence": "0.5"},
        ]
        self.assertEqual(hierarchical_aurc(rows), 0.0)
        with tempfile.TemporaryDirectory() as directory:
            candidate, reference = Path(directory) / "candidate.csv", Path(directory) / "reference.csv"
            self.write_rows(candidate, rows); self.write_rows(reference, rows)
            output = summarize_matrix({"ours": {17: candidate}, "flat_hsc": {17: reference}}, "flat_hsc", Path(directory) / "out")
            self.assertTrue((output / "seed_metrics.csv").is_file())


if __name__ == "__main__":
    unittest.main()
