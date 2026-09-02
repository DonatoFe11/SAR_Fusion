import unittest
from unittest.mock import patch

from click.testing import CliRunner

from main import main


class TestExperimentCLI(unittest.TestCase):
    def test_start_from_run_is_forwarded_without_editing_yaml(self):
        runner = CliRunner()

        with patch("sarfusion.experiment.experiment.experiment") as run_experiment:
            result = runner.invoke(
                main,
                [
                    "experiment",
                    "--parameters",
                    "protocol.yaml",
                    "--yolo",
                    "--start-from-run",
                    "1",
                ],
            )

        self.assertEqual(result.exit_code, 0, result.output)
        run_experiment.assert_called_once_with(
            param_path="protocol.yaml",
            parallel=False,
            only_create=False,
            yolo=True,
            start_from_run=1,
            max_runs=None,
        )

    def test_max_runs_is_forwarded_for_single_seed_pilots(self):
        runner = CliRunner()

        with patch("sarfusion.experiment.experiment.experiment") as run_experiment:
            result = runner.invoke(
                main,
                [
                    "experiment",
                    "--parameters",
                    "protocol.yaml",
                    "--start-from-run",
                    "0",
                    "--max-runs",
                    "1",
                ],
            )

        self.assertEqual(result.exit_code, 0, result.output)
        run_experiment.assert_called_once_with(
            param_path="protocol.yaml",
            parallel=False,
            only_create=False,
            yolo=False,
            start_from_run=0,
            max_runs=1,
        )


if __name__ == "__main__":
    unittest.main()
