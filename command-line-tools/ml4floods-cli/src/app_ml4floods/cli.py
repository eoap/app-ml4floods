# cli.py

import click
from .pipeline import run_pipeline


@click.command(
    short_help="ML4Floods inference for flood extent estimation",
)
@click.option(
    "--product-uri",
    required=True,
    type=click.STRING,
)
@click.option(
    "--water-threshold",
    default=0.7,
    type=float,
)
@click.option(
    "--brightness-threshold",
    default=3500,
    type=float,
)
@click.option(
    "--collection-uri",
    default=None,
    type=click.STRING,
    help="Collection for publishing the results (optional)",
    required=False,
)
def main(product_uri, water_threshold, brightness_threshold, collection_uri):
    run_pipeline(
        product_uri=product_uri,
        water_threshold=water_threshold,
        brightness_threshold=brightness_threshold,
        collection_uri=collection_uri,
    )


if __name__ == "__main__":
    main()
