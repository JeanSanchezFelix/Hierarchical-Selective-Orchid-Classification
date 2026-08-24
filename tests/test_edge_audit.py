import unittest

import numpy as np
import torch

from model_compression.src.orchid.edge_audit import audit_models, litert_parity


class EdgeAuditTests(unittest.TestCase):
    def test_host_audit_counts_models_and_calls(self):
        report = audit_models([torch.nn.Flatten(), torch.nn.Flatten()], 4, warmup=0, trials=2)
        self.assertEqual(report["model_files"], 2)
        self.assertEqual(report["neural_inference_calls_per_input"], 2)
        self.assertEqual(report["measurement_scope"], "host_cpu_only_not_mobile_device")

    def test_parity_reports_top1_agreement(self):
        report = litert_parity(np.asarray([[3.0, 1.0], [0.0, 2.0]]), np.asarray([[2.9, 1.1], [1.0, 2.0]]))
        self.assertEqual(report["top1_agreement"], 1.0)


if __name__ == "__main__":
    unittest.main()
