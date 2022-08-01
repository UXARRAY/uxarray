import numpy as np


# helper function to project node on the unit sphere
def normalize_in_place(node):
    """Helper function to project an arbitrary node in 3D coordinates [x, y, z]
    on the unit sphere.

    Parameters
    ----------
    node: float array [x, y, z]

    Returns: float array, the result vector [x, y, z]
    """
    magnitude = np.sqrt(node[0] * node[0] + node[1] * node[1] +
                        node[2] * node[2])
    return [node[0] / magnitude, node[1] / magnitude, node[2] / magnitude]
