import numpy as np
from numpy import ndarray


def adjust_disparity_scale(epoch: int, alpha: float = 0.03,
                           beta: float = 0.15, min_scale: float = 0.3,
                           max_scale: float = 1.0) -> float:

    scale = (epoch * alpha) + beta
    return np.clip(scale, min_scale, max_scale)


def post_process_disparity(disparity: ndarray, alpha: float = 20,
                           beta: float = 0.05) -> ndarray:

    left_disparity = disparity[0]
    right_disparity = np.fliplr(disparity[1])

    mean_disparity = (left_disparity + right_disparity) / 2

    _, height, width = disparity.shape

    x = np.linspace(0, 1, width)
    y = np.linspace(0, 1, height)
    xv, _ = np.meshgrid(x, y)

    left_mask = 1 - np.clip(alpha * (xv - beta), 0, 1)
    right_mask = np.fliplr(left_mask)

    mean_mask = 1 - (left_mask + right_mask)

    return (right_mask * left_disparity) + (left_mask * right_disparity) \
        + (mean_mask * mean_disparity)
