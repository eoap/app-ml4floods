from loguru import logger
import os
import numpy as np
import rasterio
import pystac
import click
from loguru import logger
from georeader.geotensor import GeoTensor
from app_ml4floods.utils import (
    get_target_resolution,
    stack_separated_bands,
    save_prediction,
    save_overview,
    create_stac_catalog,
    update_and_resample_asset,
    item_filter_assets,
    model_configuration,
    predict,
    clean_up,
)
from tqdm import tqdm


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

    # Read Item and its assets
    if os.path.isdir(input_item):
        logger.info(f"Reading STAC catalog from a local STAC Catalog at {input_item}")
        catalog = pystac.read_file(os.path.join(input_item, "catalog.json"))
        item = next(catalog.get_items())
        item, common_assets = item_filter_assets(item)
    else:
        logger.info(f"Reading STAC Item from {input_item}")
        item = pystac.read_file(input_item)
        item, common_assets = item_filter_assets(item)

    logger.info(f"Read {item.id}, Available common bands: {common_assets}")

    ### Configure model and load inference function
    inference_function, config = model_configuration(
        num_of_available_bands=len(common_assets),
        th_water=water_threshold,
        th_brightness=brightness_threshold,
    )
    channel_configuration = config["data_params"]["channel_configuration"]
    if len(common_assets) > 4:
        channels = [
            1,
            2,
            3,
            7,
            11,
            12,
        ]  # ['blue', 'green', 'red', 'nir', 'swir16', 'swir22']
    else:
        channels = [1, 2, 3, 7]  # ['blue', 'green', 'red', 'nir']

    local_hrefs = []
    ### Sign and resample assets if needed
    for key, asset in item.get_assets().items():
        logger.info(f"Processing asset {key}: {type(asset)}")
        updated_asset_href = update_and_resample_asset(
            asset=asset, target_resolution=get_target_resolution(item)
        )
        asset.href = updated_asset_href
        logger.info(f"Updated asset href for {key}: {updated_asset_href}")
        item.assets[key] = asset
        local_hrefs.append(updated_asset_href)

    
    ### Open the tif file
    srcs = {
        asset_key: rasterio.open(asset.href)
        for asset_key, asset in item.assets.items()
        if asset_key in common_assets
    }
    try:
        # if we have RGB+NIR+SWIRS
        referenced_src, meta = (
            srcs[common_assets[4]],
            srcs[common_assets[4]].meta.copy(),
        )
    except:
        # if we have only RGB+NIR
        referenced_src, meta = (
            srcs[common_assets[0]],
            srcs[common_assets[0]].meta.copy(),
        )
    prediction = np.empty(
        (
            referenced_src.height,
            referenced_src.width,
        ),
        dtype=np.uint8,
    )  # create the empty array

    ### read bands window by window and make prediction using pre-trained model
    logger.info(
        f"Using channels: {channels} for inference {list(range(len(channels)))}"
    )

    tqdm_loop = tqdm(
        referenced_src.block_windows(1),
        total=sum(1 for _ in referenced_src.block_windows(1)),
        desc=f"Predicting",
    )

    for ji, window in tqdm_loop:
        arr_block = stack_separated_bands(window, srcs, common_assets)
        tqdm_loop.set_postfix(
            ordered_dict={
                "col_off": window.col_off,
                "row_off": window.row_off,
                "block shape": arr_block.shape,
            }
        )
        prediction_block, _ = predict(
            inference_function, arr_block, channels=list(range(len(channels)))
        )
        prediction[
            window.row_off : window.row_off + window.height,
            window.col_off : window.col_off + window.width,
        ] = prediction_block

    # Save prediction as a COG tif image and provide STAC objects for that
    result_prefix = "flood-delineation"
    logger.info(f"Saving classification result to {result_prefix}.tif")
    prediction_block_raster = GeoTensor(
        prediction, transform=meta["transform"], fill_value_default=0, crs=meta["crs"]
    )
    save_prediction(prediction_block_raster.values, f"{result_prefix}.tif", meta)
    save_overview(prediction_block_raster.values, f"{result_prefix}-overview.tif", meta)
    del arr_block, prediction
    create_stac_catalog(item=item, result_prefix=result_prefix)

    clean_up(local_hrefs)

    logger.info("Done!")


if __name__ == "__main__":
    main()
