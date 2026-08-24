import unittest

import torch

from model_compression.src.orchid.models import (
    BalancedSoftmaxLoss,
    METHOD_DUAL_HEAD_HSC,
    METHOD_FLAT_CE,
    OrchidMultiTaskLoss,
    OrchidTaxonomyIndex,
    aggregate_species_probabilities,
    build_orchid_model,
)


class OrchidModelTests(unittest.TestCase):
    def setUp(self):
        self.taxonomy = OrchidTaxonomyIndex.from_species_ids(("Alpha::one", "Alpha::two", "Beta::three"))

    def test_taxonomy_aggregation_is_a_probability_distribution(self):
        logits = torch.tensor([[0.0, 0.0, 0.0]])
        aggregated = aggregate_species_probabilities(logits, self.taxonomy)
        self.assertTrue(torch.allclose(aggregated, torch.tensor([[2 / 3, 1 / 3]]), atol=1e-6))
        self.assertTrue(torch.allclose(aggregated.sum(dim=1), torch.ones(1)))

    def test_balanced_softmax_uses_declared_class_counts(self):
        loss = BalancedSoftmaxLoss((10, 2, 1))
        values = loss(torch.zeros((2, 3)), torch.tensor([0, 2]))
        self.assertTrue(torch.isfinite(values))

    def test_dual_head_loss_has_finite_taxonomy_components(self):
        criterion = OrchidMultiTaskLoss(
            self.taxonomy, BalancedSoftmaxLoss((10, 5, 2)), genus_weight=1.0, consistency_weight=0.5
        )
        loss, components = criterion(torch.randn(2, 3), torch.tensor([0, 2]), torch.randn(2, 2))
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual({"species_loss", "genus_loss", "consistency_loss", "total_loss"}, set(components))

    def test_model_factory_preserves_flat_and_dual_head_output_contracts(self):
        images = torch.randn(1, 3, 32, 32)
        flat = build_orchid_model(METHOD_FLAT_CE, self.taxonomy, use_imagenet_weights=False).eval()
        dual = build_orchid_model(METHOD_DUAL_HEAD_HSC, self.taxonomy, use_imagenet_weights=False).eval()
        with torch.no_grad():
            self.assertEqual((1, 3), tuple(flat(images).shape))
            species, genus = dual(images)
        self.assertEqual((1, 3), tuple(species.shape))
        self.assertEqual((1, 2), tuple(genus.shape))


if __name__ == "__main__":
    unittest.main()
