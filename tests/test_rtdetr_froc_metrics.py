import unittest

import numpy as np

from scripts.rtdetr_froc_metrics import detection_events, empirical_curve, recall_at_budgets


EMPTY_BOXES = np.empty((0, 4), dtype=float)


class DetectionEventsTests(unittest.TestCase):
    def test_confidence_order_duplicate_is_false_positive(self):
        scores, tp = detection_events(
            [[0, 0, 10, 10]], [[0, 0, 10, 10], [0, 0, 8, 10]], [0.2, 0.9]
        )
        np.testing.assert_array_equal(scores, [0.9, 0.2])
        np.testing.assert_array_equal(tp, [True, False])

    def test_prediction_ties_preserve_original_order(self):
        # Prediction 0 can match either GT, but prefers GT 0. Prediction 1 can
        # only match GT 0; reversing their order would yield two matches.
        _, tp = detection_events(
            [[0, 0, 10, 10], [4, 0, 14, 10]],
            [[1, 0, 11, 10], [0, 0, 8, 10]],
            [0.8, 0.8],
        )
        np.testing.assert_array_equal(tp, [True, False])
        _, reversed_tp = detection_events(
            [[0, 0, 10, 10], [4, 0, 14, 10]],
            [[0, 0, 8, 10], [1, 0, 11, 10]],
            [0.8, 0.8],
        )
        np.testing.assert_array_equal(reversed_tp, [True, True])

    def test_ground_truth_ties_choose_lowest_original_index(self):
        _, tp = detection_events(
            [[0, 0, 10, 10], [4, 0, 14, 10]],
            [[2, 0, 12, 10], [0, 0, 8, 10]],
            [0.9, 0.8],
        )
        np.testing.assert_array_equal(tp, [True, False])

    def test_chooses_best_unused_ground_truth(self):
        _, tp = detection_events(
            [[0, 0, 10, 10], [4, 0, 14, 10]],
            [[0, 0, 10, 10], [1, 0, 11, 10]],
            [0.9, 0.8],
        )
        np.testing.assert_array_equal(tp, [True, True])

    def test_exact_iou_boundary_is_inclusive(self):
        _, tp = detection_events([[0, 0, 10, 10]], [[0, 0, 5, 10]], [0.7])
        np.testing.assert_array_equal(tp, [True])

    def test_empty_images_and_absent_predictions(self):
        _, tp = detection_events(EMPTY_BOXES, [[1, 1, 3, 3]], [0.2])
        np.testing.assert_array_equal(tp, [False])
        scores, tp = detection_events([[0, 0, 10, 10]], EMPTY_BOXES, [])
        self.assertEqual(scores.size, 0)
        self.assertEqual(tp.size, 0)
        self.assertEqual(tp.dtype, np.dtype(bool))

    def test_input_arrays_are_not_mutated(self):
        ground_truth = np.array([[0.0, 0.0, 10.0, 10.0]])
        prediction = ground_truth.copy()
        scores = np.array([0.4])
        originals = [value.copy() for value in (ground_truth, prediction, scores)]
        detection_events(ground_truth, prediction, scores)
        for before, after in zip(originals, (ground_truth, prediction, scores)):
            np.testing.assert_array_equal(before, after)

    def test_invalid_inputs(self):
        valid = [[0, 0, 10, 10]]
        cases = [
            ([], valid, [0.5], 0.5),
            ([[0, 0, 1]], valid, [0.5], 0.5),
            ([[0, 0, 0, 10]], valid, [0.5], 0.5),
            ([[0, 0, np.nan, 10]], valid, [0.5], 0.5),
            (valid, [[0, 0, np.inf, 10]], [0.5], 0.5),
            (valid, valid, [], 0.5),
            (valid, valid, [[0.5]], 0.5),
            (valid, valid, [np.nan], 0.5),
            (valid, valid, [1.1], 0.5),
            (valid, valid, [-0.1], 0.5),
            (valid, valid, [0.5], np.nan),
            (valid, valid, [0.5], -0.1),
            (valid, valid, [0.5], 1.1),
            (valid, valid, [0.5], True),
        ]
        for args in cases:
            with self.subTest(args=args), self.assertRaises(ValueError):
                detection_events(*args)


class EmpiricalCurveTests(unittest.TestCase):
    def test_cross_image_matching_and_empty_frame_denominator(self):
        # Coincident coordinates in different images cannot share matches.
        events = [
            detection_events([[0, 0, 2, 2]], EMPTY_BOXES, []),
            detection_events(EMPTY_BOXES, [[0, 0, 2, 2]], [0.9]),
            detection_events([[0, 0, 2, 2]], [[0, 0, 2, 2]], [0.8]),
        ]
        curve = empirical_curve(events, num_images=4, total_gt=2)
        np.testing.assert_array_equal(curve["tp"], [0, 0, 1])
        np.testing.assert_array_equal(curve["fp"], [0, 1, 1])
        np.testing.assert_array_equal(curve["recall"], [0, 0, 0.5])
        np.testing.assert_array_equal(curve["fppi"], [0, 0.25, 0.25])

    def test_ties_are_atomic_across_images(self):
        curve = empirical_curve(
            [(np.array([0.9, 0.5]), np.array([True, False])),
             (np.array([0.5]), np.array([True]))],
            num_images=2, total_gt=2,
        )
        np.testing.assert_array_equal(curve["threshold"], [np.inf, 0.9, 0.5])
        np.testing.assert_array_equal(curve["tp"], [0, 1, 2])
        np.testing.assert_array_equal(curve["fp"], [0, 0, 1])
        np.testing.assert_array_equal(recall_at_budgets(curve, [0, 0.49, 0.5]), [0.5, 0.5, 1])

    def test_all_empty_predictions_have_reject_all_point(self):
        curve = empirical_curve([], num_images=5, total_gt=3)
        np.testing.assert_array_equal(curve["threshold"], [np.inf])
        for name in ("recall", "fppi", "tp", "fp"):
            np.testing.assert_array_equal(curve[name], [0])
        np.testing.assert_array_equal(recall_at_budgets(curve, [0, 1]), [0, 0])

    def test_monotonicity_and_exact_counts_at_every_threshold(self):
        events = [
            (np.array([1.0, 0.8, 0.3, 0.0]), np.array([False, True, False, True])),
            (np.array([0.8, 0.3]), np.array([True, False])),
        ]
        curve = empirical_curve(events, num_images=3, total_gt=4)
        self.assertTrue(np.all(np.diff(curve["threshold"]) < 0))
        for name in ("recall", "fppi", "tp", "fp"):
            self.assertTrue(np.all(np.diff(curve[name]) >= 0))
        for i, threshold in enumerate(curve["threshold"]):
            expected_tp = sum(int(flags[scores >= threshold].sum()) for scores, flags in events)
            expected_fp = sum(int((~flags[scores >= threshold]).sum()) for scores, flags in events)
            self.assertEqual(curve["tp"][i], expected_tp)
            self.assertEqual(curve["fp"][i], expected_fp)
        self.assertTrue(np.issubdtype(curve["tp"].dtype, np.integer))
        self.assertTrue(np.issubdtype(curve["fp"].dtype, np.integer))

    def test_invalid_counts_and_events(self):
        valid_event = (np.array([0.9]), np.array([True]))
        cases = [
            ([valid_event], 0, 1),
            ([valid_event], 1, 0),
            ([valid_event], 1.5, 1),
            ([valid_event], True, 1),
            ([valid_event], 1, -1),
            ([valid_event, valid_event], 1, 2),
            ([valid_event, valid_event], 2, 1),
            ([(np.array([0.2, 0.9]), np.array([True, False]))], 1, 1),
            ([(np.array([0.2]), np.array([1]))], 1, 1),
            ([(np.array([0.2]), np.array([True, False]))], 1, 1),
            ([(np.array([np.inf]), np.array([False]))], 1, 1),
            ([(np.array([0.2]),)], 1, 1),
        ]
        for args in cases:
            with self.subTest(args=args), self.assertRaises(ValueError):
                empirical_curve(*args)

    def test_budgets_use_empirical_points_without_interpolation(self):
        curve = empirical_curve(
            [(np.array([0.9, 0.8, 0.7, 0.6]), np.array([True, False, True, False]))],
            num_images=2, total_gt=4,
        )
        np.testing.assert_array_equal(
            recall_at_budgets(curve, [1.0, 0.25, 0.5, 0.0, 20.0]),
            [0.5, 0.25, 0.5, 0.25, 0.5],
        )
        self.assertEqual(recall_at_budgets(curve, []).size, 0)

    def test_invalid_budgets_and_curves(self):
        curve = empirical_curve([(np.array([0.8]), np.array([True]))], 1, 1)
        for budget in ([np.nan], [np.inf], [-1], [[1]]):
            with self.subTest(budget=budget), self.assertRaises(ValueError):
                recall_at_budgets(curve, budget)
        changes = [
            ("recall", [0, 1.1]), ("recall", [0.2, 1]),
            ("recall", [0, np.nan]), ("fppi", [0, -1]),
            ("threshold", [1, 0.8]), ("threshold", [np.inf, np.inf]),
            ("tp", [0, 0.5]), ("tp", [0]),
        ]
        for name, value in changes:
            malformed = {key: array.copy() for key, array in curve.items()}
            malformed[name] = np.array(value)
            with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                recall_at_budgets(malformed, [0.5])
        with self.assertRaises(ValueError):
            recall_at_budgets({}, [0.5])


if __name__ == "__main__":
    unittest.main()
