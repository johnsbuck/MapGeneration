"""Math

This script contains functions commonly used for mathematical functions between vectors.
"""

import numpy as np


def lerp(t, a, b):
    """Linear Interpolation function

    Args:
        t (float): Alpha Value
        a (np.ndarray): Vector 1
        b (np.ndarray): Vector 2

    Returns:
        (np.ndarray) Vector from linear interpolation
    """
    return (1 - t) * a + t * b


def dot(g, x, y, z=None):
    """ Dot Product for 2 or 3-Dimensional vectors

    Args:
        g (np.ndarray): First Vector
        x (float): X Coordinate
        y (float): Y Coordinate
        z (float): Z Coordinate (Default: None)

    Returns:
        (float) Dot Product
    """
    if z is None:
        return g[0] * x + g[1] * y
    return g[0] * x + g[1] * y + g[2] * z


def fract(value):
    return value - np.floor(value)
