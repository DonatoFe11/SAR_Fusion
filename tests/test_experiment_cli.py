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
        )


if __name__ == "__main__":
    unittest.main()
