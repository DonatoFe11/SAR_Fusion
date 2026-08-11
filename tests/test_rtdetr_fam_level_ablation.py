import unittest

import torch
from torch import nn

from scripts.run_rtdetr_fam_level_ablation import (
    CONDITIONS,
    active_fam_levels,
    factorial_effects,
    find_fam_modules,
)


class FeatureAlignmentModule(nn.Module):
    def __init__(self, increment):
        super().__init__()
        self.increment = increment

    def forward(self, rgb, ir):
        return ir + self.increment


class BoundedFeatureAlignmentModule(FeatureAlignmentModule):
    pass


class ThreeLevelModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fam_modules = nn.ModuleList(
            [FeatureAlignmentModule(10), FeatureAlignmentModule(20), FeatureAlignmentModule(30)]
        )

    def forward(self, rgb, ir):
        return [module(rgb, ir) for module in self.fam_modules]


class TestRTDetrFAMLevelAblation(unittest.TestCase):
    def test_conditions_are_the_full_three_level_power_set(self):
        actual = {frozenset(levels) for levels in CONDITIONS.values()}
        expected = {
            frozenset(levels)
            for levels in [
                (), (0,), (1,), (2,), (0, 1), (0, 2), (1, 2), (0, 1, 2)
            ]
        }
        self.assertEqual(actual, expected)

    def test_inactive_levels_return_raw_ir_and_hooks_are_removed(self):
        model = ThreeLevelModel()
        rgb = torch.tensor(0.0)
        ir = torch.tensor(1.0)

        with active_fam_levels(model, {0, 2}):
            outputs = model(rgb, ir)
        self.assertEqual([output.item() for output in outputs], [11.0, 1.0, 31.0])

        outputs_after = model(rgb, ir)
        self.assertEqual(
            [output.item() for output in outputs_after],
            [11.0, 21.0, 31.0],
        )

    def test_wrong_number_of_fam_modules_fails_closed(self):
        model = nn.Sequential(FeatureAlignmentModule(1))
        with self.assertRaisesRegex(RuntimeError, "exactly three"):
            find_fam_modules(model)

    def test_bounded_variant_is_discovered_at_all_three_levels(self):
        model = nn.Sequential(
            BoundedFeatureAlignmentModule(1),
            BoundedFeatureAlignmentModule(2),
            BoundedFeatureAlignmentModule(3),
        )
        self.assertEqual(len(find_fam_modules(model)), 3)

    def test_invalid_level_fails_before_forward(self):
        model = ThreeLevelModel()
        with self.assertRaisesRegex(ValueError, "Invalid FAM level"):
            with active_fam_levels(model, {3}):
                pass

    def test_factorial_effects_recover_additive_level_contributions(self):
        responses = {}
        for condition, active in CONDITIONS.items():
            responses[condition] = (
                10.0
                + (2.0 if 0 in active else 0.0)
                + (3.0 if 1 in active else 0.0)
                + (5.0 if 2 in active else 0.0)
            )
        effects = factorial_effects(responses)
        self.assertAlmostEqual(effects["P3"], 2.0)
        self.assertAlmostEqual(effects["P4"], 3.0)
        self.assertAlmostEqual(effects["P5"], 5.0)
        self.assertAlmostEqual(effects["P3:P4"], 0.0)
        self.assertAlmostEqual(effects["P3:P5"], 0.0)
        self.assertAlmostEqual(effects["P4:P5"], 0.0)
        self.assertAlmostEqual(effects["P3:P4:P5"], 0.0)


if __name__ == "__main__":
    unittest.main()
