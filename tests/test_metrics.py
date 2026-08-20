import unittest

import numpy as np
from sklearn.metrics import roc_auc_score

from model_compression.src.utils.metrics import calculate_metrics


class CalculateMetricsTests(unittest.TestCase):
    def test_multiclass_auc_ignores_unobserved_output_classes(self):
        # The model retains four output classes, but class 1 has no samples in
        # this split.  This occurs when a reviewed split assigns a class only
        # to validation or test.
        y_true = np.array([0, 2, 3, 0, 2, 3])
        y_proba = np.array(
            [
                [0.70, 0.05, 0.20, 0.05],
                [0.10, 0.10, 0.70, 0.10],
                [0.10, 0.10, 0.10, 0.70],
                [0.60, 0.10, 0.20, 0.10],
                [0.10, 0.10, 0.80, 0.00],
                [0.10, 0.10, 0.10, 0.80],
            ]
        )

        metrics = calculate_metrics(y_true, np.argmax(y_proba, axis=1), y_proba)
        observed = np.array([0, 2, 3])
        expected = roc_auc_score(
            y_true,
            y_proba[:, observed] / y_proba[:, observed].sum(axis=1, keepdims=True),
            labels=observed,
            multi_class="ovr",
            average="macro",
        )

        self.assertAlmostEqual(metrics["AUC-Score"], expected)


if __name__ == "__main__":
    unittest.main()
