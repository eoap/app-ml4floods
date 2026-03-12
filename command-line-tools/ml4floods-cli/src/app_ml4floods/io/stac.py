from __future__ import annotations
from loguru import logger
import os

from pystac import Item, Catalog, Asset, read_file, CatalogType
from rio_stac.stac import create_stac_item
from pystac.extensions.render import RenderExtension, Render
from pystac.media_type import MediaType
from pathlib import Path
from typing import List
from ..utils.misc import get_mission
from .assets import get_asset
from shutil import move


def read_stac_item(input_item: str) -> Item:
    """
    Read a STAC Item from either:
    - a local STAC Catalog directory (expects catalog.json inside), or
    - a direct STAC Item file/URL.
    """

    if os.path.isdir(input_item):
        logger.info(
            "Reading STAC catalog from a local STAC Catalog at %s",
            input_item,
        )
        catalog: Catalog = read_file(os.path.join(input_item, "catalog.json"))
        item: Item = next(catalog.get_items())
    else:
        logger.info(
            "Reading STAC Item from %s",
            input_item,
        )
        item: Item = read_file(input_item)

    return item


def to_stac(geotiff_path: str, item: Item) -> Item:
    asset_key = "flood-delineation"

    asset = Asset(
        href=os.path.basename(geotiff_path),
        media_type=MediaType.COG,
        roles=["data", "visual"],
    )

    result_item = create_stac_item(
        id=f"{item.id}-flood-delineation",
        source=geotiff_path,
        assets={asset_key: asset},
        with_proj=True,
        with_raster=True,
        properties={},
    )

    # Create empty Render object
    render = Render({})

    # Apply properties to it
    render.apply(
        assets=[asset_key],
        title="Flood delineation",
        colormap={
            "0": [0, 0, 0, 255],
            "1": [0, 128, 0, 255],
            "2": [0, 0, 255, 255],
            "3": [255, 255, 255, 255],
            "5": [255, 0, 0, 255],
        },
        nodata="0",
    )

    # Attach to item
    render_ext = RenderExtension.ext(result_item, add_if_missing=True)
    render_ext.apply({"default": render})

    return result_item


def create_stac_catalog(
    item: Item,
    geotiff_path: str,
    output_root: str,
        collection_uri: str = None,
) -> None:
    """
    Create STAC catalog in output_root with layout:

    output_root/
        catalog.json
        <item-id>/
            <item-id>.json
            flood-delineation.tif
    """

    output_root: Path = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    # Create derived STAC item
    result_item: Item = to_stac(geotiff_path, item)

    if collection_uri:
        collection = read_file(collection_uri)
        result_item.collection = collection.id

    catalog_id = result_item.id

    # Create catalog
    catalog = Catalog(
        id="catalog",
        description="Flood delineation result",
        title="Flood delineation result",
    )

    catalog.add_item(result_item)

    # Normalize directly into final directory
    catalog.normalize_and_save(
        root_href=str(output_root),
        catalog_type=CatalogType.SELF_CONTAINED,
    )

    # Move TIFF into item directory
    item_dir = output_root / catalog_id
    item_dir.mkdir(exist_ok=True)

    target_tif = item_dir / Path(geotiff_path).name

    move(str(geotiff_path), str(target_tif))

    logger.info(f"STAC written to {output_root}")


def item_filter_assets(item: Item) -> tuple[Item, List[str]]:
    """Filter STAC Item assets to keep only those relevant for ML4Floods processing."""

    new_item = Item(
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
