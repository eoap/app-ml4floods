from __future__ import annotations
import os
import numpy as np
import torch
from ml4floods.scripts.inference import load_inference_function


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


def predict(inference_function, input_tensor, channels=[1, 2, 3, 7, 11, 12]):
    """Make prediction using the inference function and input tensor."""
    input_tensor = input_tensor.astype(np.float32)
    input_tensor = input_tensor[channels]

    torch_inputs = torch.tensor(np.nan_to_num(input_tensor))
    return inference_function(torch_inputs)
