import unittest

from sarfusion.experiment.run import Run


class _FakeTracker:
    def __init__(self):
        self.end_calls = 0

    def end(self):
        self.end_calls += 1


class _FakeAccelerator:
    def __init__(self):
        self.free_memory_calls = 0
        self.released_objects = None

    def free_memory(self, *objects):
        self.free_memory_calls += 1
        self.released_objects = objects
        return [None] * len(objects)


class TestRunCleanup(unittest.TestCase):
    def test_end_releases_training_objects_and_is_idempotent(self):
        run = Run()
        tracker = _FakeTracker()
        accelerator = _FakeAccelerator()
        run.tracker = tracker
        run.accelerator = accelerator
        run.model = object()
        run.optimizer = object()
        run.scheduler = object()
        run.train_loader = object()
        run.val_loader = object()
        run.test_loader = object()
        run.criterion = object()
        run.train_evaluator = object()
        run.val_evaluator = object()
        run.denormalize = object()
        run.validation_json = object()
        run.compute_val_metrics = lambda: run.val_evaluator

        run.end()

        self.assertEqual(tracker.end_calls, 1)
        self.assertEqual(accelerator.free_memory_calls, 1)
        self.assertEqual(len(accelerator.released_objects), 6)
        self.assertIsNone(run.model)
        self.assertIsNone(run.optimizer)
        self.assertIsNone(run.scheduler)
        self.assertIsNone(run.train_loader)
        self.assertIsNone(run.val_loader)
        self.assertIsNone(run.test_loader)
        self.assertIsNone(run.criterion)
        self.assertIsNone(run.train_evaluator)
        self.assertIsNone(run.val_evaluator)
        self.assertIsNone(run.compute_val_metrics)
        self.assertIsNone(run.tracker)
        self.assertIsNone(run.accelerator)

        run.end()
        self.assertEqual(tracker.end_calls, 1)
        self.assertEqual(accelerator.free_memory_calls, 1)


if __name__ == "__main__":
    unittest.main()
