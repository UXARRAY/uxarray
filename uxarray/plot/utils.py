import holoviews as hv
import matplotlib as mpl

import numpy as np


class HoloviewsBackend:
    """Utility class to compare and set a HoloViews plotting backend for
    visualization."""

    def __init__(self):
        self.matplotlib_backend = mpl.get_backend()

    def assign(self, backend: str):
        """Assigns a backend for use with HoloViews visualization.

        Parameters
        ----------
        backend : str
            Plotting backend to use, one of 'matplotlib', 'bokeh'
        """

        if backend not in ["bokeh", "matplotlib", None]:
            raise ValueError(
                f"Unsupported backend. Expected one of ['bokeh', 'matplotlib'], but received {backend}"
            )
        if backend is not None and backend != hv.Store.current_backend:
            # only call hv.extension if it needs to be changed
            hv.extension(backend)

    def reset_mpl_backend(self):
        """Resets the default backend for the ``matplotlib`` module."""
        mpl.use(self.matplotlib_backend)


# global reference to holoviews backend utility class
backend = HoloviewsBackend()


def great_circle(p1, p2, num_points=100):
    """
    Generate points along the great circle between two points.
    p1 and p2 are arrays/lists of Cartesian coordinates.
    """
    # Normalize the input points
    p1 = p1 / np.linalg.norm(p1)
    p2 = p2 / np.linalg.norm(p2)

    # Compute the angle between the points
    omega = np.arccos(np.clip(np.dot(p1, p2), -1.0, 1.0))

    if omega == 0:
        return np.array([p1 for _ in range(num_points)])

    # Generate interpolation factors
    t = np.linspace(0, 1, num_points)

    # Compute points along the great circle
    sin_omega = np.sin(omega)
    points = (
        np.sin((1 - t) * omega)[:, np.newaxis] * p1
        + np.sin(t * omega)[:, np.newaxis] * p2
    ) / sin_omega
    return points
