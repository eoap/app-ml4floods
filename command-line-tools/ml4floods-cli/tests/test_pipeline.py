import os
from pathlib import Path
import sys
import types
import unittest

from types import SimpleNamespace
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

from app_ml4floods import pipeline


class FakeTensor:
    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self

    def astype(self, _dtype):
        return [[1]]


class FakeSource:
    def __init__(self):
        self.meta = {"width": 1, "height": 1}
        self.closed = False

    def block_windows(self, _band_index):
        return [((0, 0), "window-1")]

    def close(self):
        self.closed = True


class FakeWriter:
    def __init__(self):
        self.colormap = None
        self.writes = []
        self.overviews = None
        self.tags = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def write_colormap(self, band, colormap):
        self.colormap = (band, colormap)

    def write(self, data, band, window=None):
        self.writes.append((data, band, window))

    def build_overviews(self, levels, resampling):
        self.overviews = (levels, resampling)

    def update_tags(self, **kwargs):
        self.tags = kwargs


class PipelineTests(unittest.TestCase):
    def test_run_pipeline_uses_product_uri_and_writes_output(self):
        product_uri = "https://example.com/item.json"
        common_assets = ["blue", "green", "red", "nir", "swir16", "swir22"]
        item = SimpleNamespace(
            id="scene-1",
            assets={
                asset_key: SimpleNamespace(href=f"/tmp/{asset_key}.tif")
                for asset_key in common_assets
            },
        )
        opened_sources = {}
        writer = FakeWriter()

        def fake_rasterio_open(path, mode=None, **kwargs):
            if mode == "w":
                return writer
            source = FakeSource()
            opened_sources[path] = source
            return source

        with (
            patch.object(pipeline, "WORKDIR", "/tmp/ml4flood-test"),
            patch("app_ml4floods.pipeline.os.makedirs") as makedirs,
            patch(
                "app_ml4floods.pipeline.read_stac_item", return_value=item
            ) as read_stac_item,
            patch(
                "app_ml4floods.pipeline.item_filter_assets",
                return_value=(item, common_assets),
            ) as item_filter_assets,
            patch(
                "app_ml4floods.pipeline.model_configuration",
                return_value=("inference-function", {"name": "config"}),
            ) as model_configuration,
            patch(
                "app_ml4floods.pipeline.update_item_assets",
                return_value=["/tmp/local-asset.tif"],
            ) as update_item_assets,
            patch(
                "app_ml4floods.pipeline.stack_separated_bands",
                return_value="stacked-bands",
            ) as stack_separated_bands,
            patch(
                "app_ml4floods.pipeline.predict",
                return_value=(FakeTensor(), None),
            ) as predict,
            patch("app_ml4floods.pipeline.create_stac_catalog") as create_stac_catalog,
            patch("app_ml4floods.pipeline.clean_up") as clean_up,
            patch(
                "app_ml4floods.pipeline.rasterio.open",
                side_effect=fake_rasterio_open,
            ),
        ):
            pipeline.run_pipeline(
                product_uri=product_uri,
                water_threshold=0.8,
                brightness_threshold=4100,
            )

        makedirs.assert_called_once_with("/tmp/ml4flood-test", exist_ok=True)
        read_stac_item.assert_called_once_with(product_uri)
        item_filter_assets.assert_called_once_with(item)
        model_configuration.assert_called_once_with(
            num_of_available_bands=6,
            th_water=0.8,
            th_brightness=4100,
        )
        self.assertEqual(update_item_assets.call_count, 2)
        stack_separated_bands.assert_called_once()
        predict.assert_called_once_with(
            "inference-function",
            "stacked-bands",
            channels=[0, 1, 2, 3, 4, 5],
        )
        create_stac_catalog.assert_called_once_with(
            item=item,
            geotiff_path="/tmp/ml4flood-test/flood-delineation.tif",
            output_root=os.getcwd(),
            collection_uri=None,
        )
        clean_up.assert_called_once_with(["/tmp/local-asset.tif"])

        self.assertIsNotNone(writer.colormap)
        self.assertEqual(writer.writes, [([[1]], 1, "window-1")])
        self.assertIsNotNone(writer.overviews)
        self.assertEqual(writer.tags, {"ns": "rio_overview", "resampling": "nearest"})
        self.assertTrue(all(source.closed for source in opened_sources.values()))
