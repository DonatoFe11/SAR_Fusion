import subprocess
import unittest
from unittest.mock import patch

from sarfusion.experiment.experiment import Experimenter
from sarfusion.utils.utils import load_yaml


class TestExperimentIsolation(unittest.TestCase):
    def test_expanded_run_is_executed_in_a_fresh_python_process(self):
        params = {
            "experiment": {"name": "isolated-test", "isolate_runs": True},
            "seed": 40,
            "task": "detection",
        }
        captured = {}

        def fake_run(command, check):
            captured["command"] = command
            captured["check"] = check
            captured["params"] = load_yaml(command[-1])
            return subprocess.CompletedProcess(command, 0)

        with patch(
            "sarfusion.experiment.experiment.subprocess.run",
            side_effect=fake_run,
        ):
            return_code = Experimenter._execute_isolated_run(params)

        self.assertEqual(return_code, 0)
        self.assertTrue(captured["check"])
        self.assertEqual(captured["command"][2], "run")
        self.assertEqual(captured["command"][3], "--parameters")
        self.assertEqual(captured["params"], params)


if __name__ == "__main__":
    unittest.main()
