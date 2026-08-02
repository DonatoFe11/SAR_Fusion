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

        def fake_run(command, check, env):
            captured["command"] = command
            captured["check"] = check
            captured["env"] = env
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
        self.assertEqual(captured["env"]["PYTHONHASHSEED"], "40")

    def test_deterministic_run_sets_cublas_environment_before_python_starts(self):
        params = {
            "experiment": {"isolate_runs": True},
            "seed": 42,
            "reproducibility": {"deterministic": True},
        }
        captured = {}

        def fake_run(command, check, env):
            captured.update(command=command, check=check, env=env)
            return subprocess.CompletedProcess(command, 0)

        with patch(
            "sarfusion.experiment.experiment.subprocess.run",
            side_effect=fake_run,
        ):
            Experimenter._execute_isolated_run(params)

        self.assertEqual(captured["env"]["PYTHONHASHSEED"], "42")
        self.assertEqual(captured["env"]["CUBLAS_WORKSPACE_CONFIG"], ":4096:8")


if __name__ == "__main__":
    unittest.main()
