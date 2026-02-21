from __future__ import annotations
from loguru import logger
import os
import rasterio
from pystac import Item, Asset
from pystac.extensions.eo import AssetEOExtension
from rasterio.enums import Resampling
from yarl import URL
import planetary_computer as pc
from typing import List
from ..utils.misc import get_target_resolution


def resize_and_convert_to_cog(
    asset: Asset,
    target_resolution: int = 10,
) -> str:
    """
    Ensure asset is available locally in /tmp/ml4flood.
    If resolution is higher than target_resolution, resample to target_resolution.
    Otherwise copy locally as-is.

    Returns
    -------
    str
        Local file path to the prepared COG.
    """

    original_href = asset.get_absolute_href()
    logger.info(f"Preparing asset locally: {original_href}")

    eo_asset = AssetEOExtension(asset)
    common_band_name = eo_asset.bands[0].properties.get("common_name")

    # Sign Azure Blob URLs if needed
    url = URL(original_href)
    if url.host is not None and url.host.endswith(".blob.core.windows.net"):
        logger.info(
            f"Asset {common_band_name} is on Azure Blob Storage. Signing with Planetary Computer."
        )
        asset = pc.sign(asset)
        original_href = asset.get_absolute_href()

    # Prepare local working directory
    local_dir = "/tmp/ml4flood"
    os.makedirs(local_dir, exist_ok=True)

    local_path = os.path.join(local_dir, URL(original_href).name)

    # If already processed locally, reuse
    if os.path.isfile(local_path):
        logger.info(f"Reusing local asset: {local_path}")
        return local_path

    with rasterio.open(original_href) as src:
        # --------------------------------------------------
        # Case 1: Resample required
        # --------------------------------------------------
        if src.transform.a > target_resolution:
            scale_x = int(src.width * (src.res[0] / target_resolution))
            scale_y = int(src.height * (src.res[1] / target_resolution))

            logger.info(
                f"Resampling {common_band_name} from "
                f"{src.transform.a}m to {target_resolution}m "
                f"({scale_x}x{scale_y})"
            )

            data = src.read(
                out_shape=(src.count, scale_y, scale_x),
                resampling=Resampling.bilinear,
            )

            transform = src.transform * src.transform.scale(
                (src.width / data.shape[-1]),
                (src.height / data.shape[-2]),
            )

            profile = src.profile.copy()
            profile.update(
                {
                    "driver": "COG",
                    "height": data.shape[1],
                    "width": data.shape[2],
                    "transform": transform,
                    "compress": "LZW",
                    "interleave": "pixel",
                }
            )

            with rasterio.open(local_path, "w", **profile) as dst:
                dst.write(data)

            logger.info(f"Resampled asset saved locally: {local_path}")
            return local_path

        # --------------------------------------------------
        # Case 2: Already correct resolution → copy locally
        # --------------------------------------------------
        logger.info(
            f"{common_band_name} already at {target_resolution}m. Copying locally."
        )

        profile = src.profile.copy()
        profile.update(
            {
                "driver": "COG",
                "compress": "LZW",
                "interleave": "pixel",
            }
        )

        with rasterio.open(local_path, "w", **profile) as dst:
            for band in range(1, src.count + 1):
                dst.write(src.read(band), band)

        logger.info(f"Copied asset locally: {local_path}")
        return local_path


def update_item_assets(item: Item) -> List[str]:
    """
    Update each asset href (sign/resample if needed) and return the list of updated hrefs.
    Mutates `item.assets` in place.
    """
    local_hrefs: List[str] = []

    target_resolution: int = get_target_resolution(item)

    for key, asset in item.get_assets().items():
        logger.info(f"Processing asset {key}")

        updated_asset_href: str = update_and_resample_asset(
            asset=asset,
            target_resolution=target_resolution,
        )

        asset.href = updated_asset_href
        logger.info(f"Updated asset href for {key}: {updated_asset_href}")

        item.assets[key] = asset
        local_hrefs.append(updated_asset_href)

    return local_hrefs


def update_and_resample_asset(asset: Asset, target_resolution=10) -> str:
    """Update asset href by resampling if needed."""
    if "data" in asset.to_dict()["roles"]:
        return resize_and_convert_to_cog(asset, target_resolution)
    else:
        return asset.get_absolute_href()


def get_asset(item, common_name) -> Asset | None:
    """Returns the asset of a STAC Item defined with its common band name"""
    for key, asset in item.get_assets().items():
        if "data" not in asset.to_dict()["roles"]:
            continue

        eo_asset = AssetEOExtension(asset)
        if not eo_asset.bands:
            continue
        for b in eo_asset.bands:
            if (
                "common_name" in b.properties.keys()
                and common_name in b.properties["common_name"]
            ):
                return asset
