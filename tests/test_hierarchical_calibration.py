import unittest

import numpy as np

from model_compression.src.orchid.calibration import (
    fit_hierarchical_selective_policy,
    hierarchical_decisions,
)
from model_compression.src.orchid.models import OrchidTaxonomyIndex


class HierarchicalCalibrationTests(unittest.TestCase):
    def setUp(self):
        self.taxonomy = OrchidTaxonomyIndex.from_species_ids(("Alpha::one", "Alpha::two", "Beta::three"))
        self.logits = np.asarray([[5.0, 1.0, 0.0], [0.0, 4.0, 1.0], [0.0, 0.0, 5.0], [4.0, 1.0, 0.0]])
        self.targets = np.asarray([0, 1, 2, 0])

    def test_flat_policy_uses_implied_genus_probabilities(self):
        policy = fit_hierarchical_selective_policy(self.logits, self.targets, self.taxonomy, target_known_coverage=0.75)
        decisions = hierarchical_decisions(self.logits, self.taxonomy, policy)
        self.assertFalse(policy.uses_genus_head)
        self.assertEqual(4, len(decisions))
        self.assertTrue(all(decision["decision_level"] in {"species", "genus", "unknown"} for decision in decisions))

    def test_dual_head_policy_requires_and_uses_genus_logits(self):
        genus_logits = np.asarray([[4.0, 0.0], [4.0, 0.0], [0.0, 4.0], [4.0, 0.0]])
        policy = fit_hierarchical_selective_policy(self.logits, self.targets, self.taxonomy, genus_logits, target_known_coverage=0.75)
        decisions = hierarchical_decisions(self.logits, self.taxonomy, policy, genus_logits)
        self.assertTrue(policy.uses_genus_head)
        self.assertEqual("species", decisions[0]["decision_level"])


if __name__ == "__main__":
    unittest.main()
