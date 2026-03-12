from pathlib import Path
import sys
import types
import unittest

from click.testing import CliRunner
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

if "planetary_computer" not in sys.modules:
    sys.modules["planetary_computer"] = types.SimpleNamespace(sign=lambda asset: asset)

if "ml4floods" not in sys.modules:
    ml4floods = types.ModuleType("ml4floods")
    ml4floods_scripts = types.ModuleType("ml4floods.scripts")
    ml4floods_inference = types.ModuleType("ml4floods.scripts.inference")
    ml4floods_inference.load_inference_function = lambda *args, **kwargs: None
    sys.modules["ml4floods"] = ml4floods
    sys.modules["ml4floods.scripts"] = ml4floods_scripts
    sys.modules["ml4floods.scripts.inference"] = ml4floods_inference

if "torch" not in sys.modules:
    sys.modules["torch"] = types.SimpleNamespace(tensor=lambda value: value)

from app_ml4floods.cli import main


class CliTests(unittest.TestCase):
    def test_cli_invokes_pipeline_with_explicit_arguments(self):
        runner = CliRunner()

        with patch("app_ml4floods.cli.run_pipeline") as run_pipeline:
            result = runner.invoke(
                main,
                [
                    "--product-uri",
                    "https://example.com/item.json",
                    "--water-threshold",
                    "0.9",
                    "--brightness-threshold",
                    "4200",
                ],
            )

        self.assertEqual(result.exit_code, 0)
        run_pipeline.assert_called_once_with(
            product_uri="https://example.com/item.json",
            water_threshold=0.9,
            brightness_threshold=4200.0,
            collection_uri=None,
        )

    def test_cli_uses_default_threshold_values(self):
        runner = CliRunner()

        with patch("app_ml4floods.cli.run_pipeline") as run_pipeline:
            result = runner.invoke(
                main,
                [
                    "--product-uri",
                    "https://example.com/item.json",
                ],
            )

        self.assertEqual(result.exit_code, 0)
        run_pipeline.assert_called_once_with(
            product_uri="https://example.com/item.json",
            water_threshold=0.7,
            brightness_threshold=3500,
            collection_uri=None,
        )

    def test_cli_forwards_collection_uri(self):
        runner = CliRunner()

        with patch("app_ml4floods.cli.run_pipeline") as run_pipeline:
            result = runner.invoke(
                main,
                [
                    "--product-uri",
                    "https://example.com/item.json",
                    "--collection-uri",
                    "https://example.com/collection.json",
                ],
            )

        self.assertEqual(result.exit_code, 0)
        run_pipeline.assert_called_once_with(
            product_uri="https://example.com/item.json",
            water_threshold=0.7,
            brightness_threshold=3500,
            collection_uri="https://example.com/collection.json",
        )

    def test_cli_requires_product_uri(self):
        runner = CliRunner()

        result = runner.invoke(main, [])

        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("Missing option '--product-uri'", result.output)
