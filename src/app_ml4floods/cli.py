# cli.py

import click
from .pipeline import run_pipeline


@click.command(
    short_help="ML4Floods inference for flood extent estimation",
)
@click.option(
    "--input-item",
    required=True,
    type=click.Path(),
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
def main(input_item, water_threshold, brightness_threshold):
    run_pipeline(
        input_item=input_item,
        water_threshold=water_threshold,
        brightness_threshold=brightness_threshold,
    )


if __name__ == "__main__":
    main()
