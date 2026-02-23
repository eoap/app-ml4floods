# pipeline.py

import os
from loguru import logger
import rasterio
from rasterio.enums import Resampling
from .io.stac import read_stac_item, create_stac_catalog, item_filter_assets
from .io.assets import update_item_assets
from .inference.model import model_configuration, predict
from .inference.processing import stack_separated_bands
from .utils.misc import clean_up

base_tmp = os.environ.get("TMPDIR", "/tmp")
WORKDIR = os.path.join(base_tmp, "ml4flood")


def run_pipeline(input_item: str, water_threshold: float, brightness_threshold: float):

    os.makedirs(WORKDIR, exist_ok=True)

    # -----------------------------------------
    # Read STAC
    # -----------------------------------------
    item = read_stac_item(input_item)
    item, common_assets = item_filter_assets(item)

    logger.info(f"Read {item.id}")

    # -----------------------------------------
    # Model
    # -----------------------------------------
    inference_function, config = model_configuration(
        num_of_available_bands=len(common_assets),
        th_water=water_threshold,
        th_brightness=brightness_threshold,
    )

    # -----------------------------------------
    # Assets preparation
    # -----------------------------------------
    local_hrefs = update_item_assets(item)

    # -----------------------------------------
    # Streaming prediction
    # -----------------------------------------
    # (COG writing block stays here)
    if len(common_assets) > 4:
        channels = [1, 2, 3, 7, 11, 12]
    else:
        channels = [1, 2, 3, 7]

    logger.info(f"Using channels: {channels}")

    # --------------------------------------------------
    # Prepare local assets (/tmp)
    # --------------------------------------------------
    local_hrefs = update_item_assets(item)

    srcs = {
        asset_key: rasterio.open(asset.href)
        for asset_key, asset in item.assets.items()
        if asset_key in common_assets
    }

    try:
        referenced_src = srcs[common_assets[4]]
    except (IndexError, KeyError):
        referenced_src = srcs[common_assets[0]]

    meta = referenced_src.meta.copy()

    # --------------------------------------------------
    # Prepare streaming COG output
    # --------------------------------------------------
    result_prefix = "flood-delineation"
    tmp_output = os.path.join(WORKDIR, f"{result_prefix}.tif")

    meta.update(
        {
            "driver": "COG",
            "dtype": "uint8",
            "count": 1,
            "blockxsize": 256,
            "blockysize": 256,
            "tiled": True,
            "compress": "deflate",
            "interleave": "band",
        }
    )

    logger.info(f"Writing output to {tmp_output}")

    with rasterio.open(tmp_output, "w", **meta) as dst:
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

        logger.info("Calculating block windows for streaming processing")
        windows = list(referenced_src.block_windows(1))
        total_windows = len(windows)
        logger.info(f"Total number of blocks to process: {total_windows}")
        logger.info("Starting prediction loop")

        log_every = max(1, total_windows // 20)  # 5% increments

        for i, (_, window) in enumerate(windows):
            arr_block = stack_separated_bands(window, srcs, common_assets)

            prediction_block, _ = predict(
                inference_function,
                arr_block,
                channels=list(range(len(channels))),
            )

            prediction_block_np = (
                prediction_block.detach().cpu().numpy().astype("uint8")
            )

            if i % log_every == 0:
                percent = 100 * i / total_windows
                logger.info(
                    f"Prediction progress: {i}/{total_windows} ({percent:.1f}%)"
                )

            dst.write(prediction_block_np, 1, window=window)

        logger.info("Finished prediction loop")

        logger.info("Building overviews")
        dst.build_overviews([2, 4, 8, 16], Resampling.nearest)
        logger.info("Finished building overviews")
        dst.update_tags(ns="rio_overview", resampling="nearest")

    # Close sources
    for src in srcs.values():
        src.close()

    # --------------------------------------------------
    # Create STAC catalog directly in final output dir
    # --------------------------------------------------

    logger.info("Creating STAC catalog for output")

    # final_output_dir = os.getcwd()
    # -----------------------------------------
    # STAC generation
    # -----------------------------------------

    create_stac_catalog(
        item=item,
        geotiff_path=tmp_output,
        output_root=os.getcwd(),
    )

    clean_up(local_hrefs)

    logger.info("Done!")
