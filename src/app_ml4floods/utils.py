from loguru import logger
import os
from shutil import move
import numpy as np
import rasterio
import pystac
from rasterio.enums import Resampling
from rio_stac.stac import create_stac_item
import torch
from rasterio.warp import Resampling
from ml4floods.scripts.inference import load_inference_function
from yarl import URL
import math
from typing import Dict, List
import planetary_computer as pc



def get_asset(item, common_name) -> pystac.Asset | None:
    """Returns the asset of a STAC Item defined with its common band name"""
    for key, asset in item.get_assets().items():
        if not "data" in asset.to_dict()["roles"]:
            continue

        eo_asset = pystac.extensions.eo.AssetEOExtension(asset)
        if not eo_asset.bands:
            continue
        for b in eo_asset.bands:
            if (
                "common_name" in b.properties.keys()
                and common_name in b.properties["common_name"]
            ):
                return asset


def item_filter_assets(item: pystac.Item) -> pystac.Item:
    """Filter STAC Item assets to keep only those relevant for ML4Floods processing."""

    new_item = pystac.Item(
        id=item.id,
        geometry=item.geometry,
        bbox=item.bbox,
        datetime=item.datetime,
        properties=item.properties,
    )

    new_item.assets = {}

    if get_mission(item) == "sentinel-2":
        common_names = [
            "blue",
            "green",
            "red",
            "nir",
            "swir16",
            "swir22",
        ]
    elif get_mission(item) == "landsat":
        common_names = ["blue", "green", "red", "nir08", "swir16", "swir22"]

    for key in common_names:
        new_item.add_asset(key=key, asset=get_asset(item, key))

    return new_item, common_names


def resize_and_convert_to_cog(asset: pystac.Asset, target_resolution=10) -> str:

    logger.info(f"Resizing and converting asset {asset.get_absolute_href()} to COG format...")

    eo_asset = pystac.extensions.eo.AssetEOExtension(asset)
    common_band_name = eo_asset.bands[0].properties.get("common_name")

    url = URL(asset.get_absolute_href())
    if url.host is not None and url.host.endswith(".blob.core.windows.net"):
        logger.info(
            f"Asset {common_band_name} is on Azure Blob Storage. Sign with ms planetary computer."
        )
        raw_asset = asset
        asset = pc.sign(raw_asset)

    with rasterio.open(asset.get_absolute_href()) as src:
        output_file = f"./{URL(asset.get_absolute_href()).name}"
        if src.transform.a > target_resolution and (not (os.path.isfile(output_file))):

            # Calculate the new shape based on the target resolution
            scale_x = int(src.width * (src.res[0] / target_resolution))
            scale_y = int(src.height * (src.res[1] / target_resolution))

            logger.info(
                f"Resampling asset {common_band_name} from {src.transform.a}m to {target_resolution}m resolution ({scale_x}x{scale_y})..."
            )
            # Read and resample data
            data = src.read(
                out_shape=(src.count, int(scale_y), int(scale_x)),
                resampling=Resampling.bilinear,
            )

            # Update the metadata for the transformed dataset
            transform = src.transform * src.transform.scale(
                (src.width / data.shape[-1]), (src.height / data.shape[-2])
            )
            profile = src.profile
            profile.update(
                {
                    "driver": "COG",
                    "dtype": data.dtype,
                    "height": data.shape[1],
                    "width": data.shape[2],
                    "transform": transform,
                    "compress": "LZW",
                    "interleave": "pixel",
                }
            )

            # Write data directly to a COG file
            with rasterio.open(output_file, "w", **profile) as dst:
                dst.write(data)

            logger.info(f"Resampled asset saved to {output_file}")
            return output_file

        else:
            logger.info(
                f"Asset {common_band_name} already in {target_resolution}m resolution."
            )

    return asset.get_absolute_href()


def update_and_resample_asset(asset: pystac.Asset, target_resolution=10) -> str:
    """Update asset href by resampling if needed."""
    if "data" in asset.to_dict()["roles"]:
        return resize_and_convert_to_cog(asset, target_resolution)
    else:
        return asset.get_absolute_href()    

def stack_separated_bands(window, srcs, common_assets):
    """Stack bands from separate assets into a single array for the given window."""
    block = np.empty((len(common_assets), window.height, window.width), dtype=np.uint16)
    for i, (band, src) in enumerate(srcs.items()):
        block[i, :, :] = src.read(1, window=window)
    return block


def predict(inference_function, input_tensor, channels=[1, 2, 3, 7, 11, 12]):
    """Make prediction using the inference function and input tensor."""
    input_tensor = input_tensor.astype(np.float32)
    input_tensor = input_tensor[channels]

    torch_inputs = torch.tensor(np.nan_to_num(input_tensor))
    return inference_function(torch_inputs)


def _get_stats(arr: np.ma.MaskedArray) -> Dict:
    """Calculate array statistics."""
    # Avoid non masked nan/inf values
    np.ma.fix_invalid(arr, copy=False)
    sample, edges = np.histogram(arr[~arr.mask], bins=np.arange(6))
    return {
        "statistics": {
            "mean": arr.mean().item(),
            "minimum": arr.min().item(),
            "maximum": arr.max().item(),
            "stddev": arr.std().item(),
            "valid_percent": np.count_nonzero(~arr.mask) / float(arr.data.size) * 100,
        },
        "histogram": {
            "count": len(edges),
            "min": float(edges.min()),
            "max": float(edges.max()),
            "buckets": sample.tolist(),
        },
    }


def get_raster_info(
    src_dst,
    max_size: int = 1024,
):
    """Get raster metadata.
    see: https://github.com/stac-extensions/raster#raster-band-object
    """
    height = src_dst.height
    width = src_dst.width
    if max_size:
        if max(width, height) > max_size:
            ratio = height / width
            if ratio > 1:
                height = max_size
                width = math.ceil(height / ratio)
            else:
                width = max_size
                height = math.ceil(width * ratio)

    meta: List[Dict] = []

    area_or_point = src_dst.tags().get("AREA_OR_POINT", "").lower()

    # Missing `bits_per_sample` and `spatial_resolution`
    print(src_dst.indexes)
    for band in src_dst.indexes:
        value = {
            "data_type": src_dst.dtypes[band - 1],
            "scale": src_dst.scales[band - 1],
            "offset": src_dst.offsets[band - 1],
        }
        if area_or_point:
            value["sampling"] = area_or_point

        # If the Nodata is not set we don't forward it.
        if src_dst.nodata is not None:
            if np.isnan(src_dst.nodata):
                value["nodata"] = "nan"
            elif np.isposinf(src_dst.nodata):
                value["nodata"] = "inf"
            elif np.isneginf(src_dst.nodata):
                value["nodata"] = "-inf"
            else:
                value["nodata"] = src_dst.nodata

        if src_dst.units[band - 1] is not None:
            value["unit"] = src_dst.units[band - 1]

        value.update(
            _get_stats(
                src_dst.read(indexes=band, out_shape=(height, width), masked=True)
            )
        )
        meta.append(value)

    return meta


def generate_asset_overview(asset_in_key, target_dir):
    """Generate a STAC Asset for the overview of a raster file."""
    asset_out_key = f"{asset_in_key}"
    raster_info = {
        "raster:bands": get_raster_info(
            rasterio.open(target_dir),
            max_size=1024,
        )
    }

    return asset_in_key, pystac.asset.Asset(
        href=f"{asset_out_key}.tif",
        media_type=pystac.media_type.MediaType.COG,
        title=f"{asset_out_key} overview",
        roles=["visual", "overview"],
        extra_fields=raster_info,
    )


def to_stac(geotiff_path, item):
    """Convert a GeoTIFF file to a STAC Item."""
    asset_key, asset = generate_asset_overview(
        asset_in_key="flood-delineation", target_dir=geotiff_path
    )
    result_item = create_stac_item(
        id=f"{item.id}-flood-delineation",
        source=geotiff_path,
        assets={asset_key: asset},
        with_proj=True,
        with_raster=False,
        properties={},
    )

    return result_item


def save_prediction(data, output_href, meta):
    """
    Save the prediction result to a GeoTIFF file.
    """
    meta.update(
        {
            "driver": "COG",
            "dtype": "uint8",
            "blockxsize": 256,
            "blockysize": 256,
            "count": 1,
            "tiled": True,
            "compress": "deflate",
            "interleave": "band",
        }
    )
    with rasterio.open(output_href, "w", **meta) as dst:

        dst.write(data, indexes=1)
        dst.write_colormap(
            1,
            {
                0: (0, 0, 0),
                1: (0, 128, 0),
                2: (0, 0, 255),
                3: (255, 255, 255),
                5: (255, 0, 0),
            },
        )
        cmap = dst.colormap(1)
        dst.build_overviews([2, 4, 8, 16], Resampling.nearest)
        dst.update_tags(ns="rio_overview", resampling="nearest")


def save_overview(data, output_href, meta):
    """
    Save the overview of a raster file to a GeoTIFF file.
    """
    meta.update(
        {
            "driver": "COG",
            "dtype": "uint8",
            "blockxsize": 256,
            "blockysize": 256,
            "count": 1,
            "tiled": True,
            "compress": "deflate",
            "interleave": "band",
        }
    )
    with rasterio.open(output_href, "w", **meta) as dst:

        dst.write(data, indexes=1)
        dst.write_colormap(
            1,
            {
                0: (0, 0, 0, 255),
                1: (0, 128, 0, 255),
                2: (0, 0, 255, 255),
                3: (255, 255, 255, 255),
                5: (255, 0, 0, 255),
            },
        )
        cmap = dst.colormap(1)

        dst.build_overviews([2, 4, 8, 16], Resampling.nearest)
        dst.update_tags(ns="rio_overview", resampling="nearest")


def model_configuration(
    num_of_available_bands: int, th_water: float = 0.7, th_brightness: float = 3500
):
    """Configure the model and load the inference function based on available bands."""
    distinguish_flood_traces = True if num_of_available_bands > 4 else False
    experiment_name = (
        "WF2_unetv2_bgriswirs" if num_of_available_bands > 4 else "WF2_unetv2_rgbi"
    )

    return load_inference_function(
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "models",
            experiment_name,
        ),
        device_name="cpu",
        max_tile_size=1024,
        th_water=th_water,
        th_brightness=th_brightness,
        distinguish_flood_traces=distinguish_flood_traces,
    )


def create_stac_catalog(item: pystac.Item, result_prefix: str):
    """Create a STAC Catalog for the flood delineation result."""
    logger.info(f"Creating STAC Item for the flood delineation result")
    out_item = to_stac(f"{result_prefix}.tif", item)
    logger.info(f"Creating a STAC Catalog for the flood delineation result")
    cat = pystac.Catalog(
        id="catalog",
        description="flood delineation result",
        title="flood delineation result",
    )
    cat.add_items([out_item])
    cat.normalize_and_save(
        root_href=f"./{out_item.id}", catalog_type=pystac.CatalogType.SELF_CONTAINED
    )
    move(
        f"{result_prefix}.tif",
        os.path.join(out_item.id, out_item.id, f"{result_prefix}.tif"),
    )
    move(
        f"{result_prefix}-overview.tif",
        os.path.join(out_item.id, out_item.id, f"{result_prefix}-overview.tif"),
    )


def get_target_resolution(item: pystac.Item) -> int:
    """Get the target resolution (in meters) for resampling based on item properties."""
    gsd = item.properties.get("gsd", None)
    if gsd is not None:
        return int(gsd)
    elif "sentinel-2" in item.properties.get("constellation", ""):
        return 10
    else:
        logger.warning(
            "Item does not have 'gsd' property. Defaulting to 10m resolution."
        )
        return 10


def get_mission(item: pystac.Item) -> str:
    """Get the mission name from the item properties."""
    if "sentinel-2" in item.properties.get("constellation", ""):
        return "sentinel-2"
    elif "landsat-c2-l2" in item.collection_id:
        return "landsat"
    elif "sentinel-2-l2a" in item.collection_id:
        return "sentinel-2"
    elif "landsat-8" in item.properties.get("constellation", ""):
        return "landsat"
    return item.properties.get("mission", "unknown")


def clean_up(temp_files: List[str]) -> None:
    """Remove temporary files."""
    for href in temp_files:
        if URL(href).scheme not in ["http", "https"]:
            os.remove(href)