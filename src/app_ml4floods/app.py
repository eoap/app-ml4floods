from loguru import logger
import rasterio
import click
from app_ml4floods.utils import (
    read_stac_item,
    update_item_assets,
    stack_separated_bands,
    create_stac_catalog,
    item_filter_assets,
    model_configuration,
    predict,
    clean_up,
)


# Run:
# ml4flood --input-item https://earth-search.aws.element84.com/v1/collections/sentinel-2-l2a/items/S2A_10SFG_20230618_0_L2A
# ml4flood --input-item https://earth-search.aws.element84.com/v1/collections/sentinel-2-l2a/items/S2B_10SFH_20230613_0_L2A
@click.command(
    short_help="ML4Floods inference for flood extent estimation",
    help="ML4Floods inference for flood extent estimation using pre-trained model on Sentinel-2 or Landsat-9 data",
)
@click.option(
    "--input-item",
    "input_item",
    help="STAC Item URL or staged STAC catalog",
    required=True,
    type=click.Path(),
)
@click.option(
    "--water-threshold",
    "water_threshold",
    help="Water threshold (default: 0.7)",
    required=True,
    default=0.7,
    type=float,
)
@click.option(
    "--brightness-threshold",
    "brightness_threshold",
    help="Brightness threshold (default: 3500)",
    required=True,
    default=3500,
    type=float,
)
def main(input_item, water_threshold, brightness_threshold):

    import os
    from rasterio.enums import Resampling

    WORKDIR = "/tmp/ml4flood"
    os.makedirs(WORKDIR, exist_ok=True)

    # --------------------------------------------------
    # Read and filter item
    # --------------------------------------------------
    item = read_stac_item(input_item)
    item, common_assets = item_filter_assets(item)

    logger.info(f"Read {item.id}, Available common bands: {common_assets}")

    # --------------------------------------------------
    # Model configuration
    # --------------------------------------------------
    inference_function, config = model_configuration(
        num_of_available_bands=len(common_assets),
        th_water=water_threshold,
        th_brightness=brightness_threshold,
    )

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

    final_output_dir = os.getcwd()

    create_stac_catalog(
        item=item,
        geotiff_path=tmp_output,
        output_root=final_output_dir,
    )

    # --------------------------------------------------
    # Cleanup temporary local assets
    # --------------------------------------------------

    clean_up(local_hrefs)

    logger.info("Done!")


if __name__ == "__main__":
    main()
