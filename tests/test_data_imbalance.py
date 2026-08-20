import unittest

from model_compression.src.utils.data_imbalance import calculate_model_weights


class _DatasetWithAbsentClass:
    classes = ["Bletia::Bti. patula", "Cattleya::C. trianae", "Empty::Unobserved"]
    targets = [0, 0, 1]


class DataImbalanceTests(unittest.TestCase):
    def test_class_weights_allow_declared_class_without_training_images(self):
        weights = calculate_model_weights(_DatasetWithAbsentClass())

        self.assertEqual(weights.shape[0], 3)
        self.assertGreater(weights[0].item(), 0.0)
        self.assertGreater(weights[1].item(), 0.0)
        self.assertEqual(weights[2].item(), 0.0)


if __name__ == "__main__":
    unittest.main()
