import numpy as np


def stack_separated_bands(window, srcs, common_assets):
    """Stack bands from separate assets into a single array for the given window."""
    block = np.empty((len(common_assets), window.height, window.width), dtype=np.uint16)
    for i, (_, src) in enumerate(srcs.items()):
        block[i, :, :] = src.read(1, window=window)
    return block
