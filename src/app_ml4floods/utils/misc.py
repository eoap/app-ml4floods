from __future__ import annotations
from loguru import logger
import os
import pystac
from yarl import URL
from typing import List


def clean_up(temp_files: List[str]) -> None:
    """Remove temporary files."""
    for href in temp_files:
        logger.info(f"Removing temporary file: {href}")
        if URL(href).scheme not in ["http", "https"]:
            os.remove(href)


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
