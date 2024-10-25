from matplotlib.colors import LinearSegmentedColormap


class _Colormaps:
    """Singleton class for lazy-loaded colormaps."""

    def __init__(self):
        self._diverging = None
        self._sequential = None
        self._sequential_blue = None
        self._sequential_green = None
        self._sequential_green_blue = None

    @property
    def diverging(self):
        if self._diverging is None:
            self._diverging = LinearSegmentedColormap.from_list(
                "diverging",
                [
                    (0.000, (0.016, 0.576, 0.565)),
                    (0.500, (1.000, 1.000, 1.000)),
                    (1.000, (0.004, 0.400, 0.569)),
                ],
            )
        return self._diverging

    @property
    def sequential(self):
        if self._sequential is None:
            self._sequential = LinearSegmentedColormap.from_list(
                "sequential",
                [
                    (0.000, (0.004, 0.400, 0.569)),
                    (1.000, (0.016, 0.576, 0.565)),
                ],
            )
        return self._sequential

    @property
    def sequential_blue(self):
        if self._sequential_blue is None:
            self._sequential_blue = LinearSegmentedColormap.from_list(
                "sequential_blue",
                [
                    (0.000, (1.000, 1.000, 1.000)),
                    (1.000, (0.004, 0.400, 0.569)),
                ],
            )
        return self._sequential_blue

    @property
    def sequential_green(self):
        if self._sequential_green is None:
            self._sequential_green = LinearSegmentedColormap.from_list(
                "sequential_green",
                [
                    (0.000, (1.000, 1.000, 1.000)),
                    (1.000, (0.016, 0.576, 0.565)),
                ],
            )
        return self._sequential_green

    @property
    def sequential_green_blue(self):
        if self._sequential_green_blue is None:
            self._sequential_green_blue = LinearSegmentedColormap.from_list(
                "sequential_green_blue",
                [
                    (0.000, (1.000, 1.000, 1.000)),
                    (0.500, (0.016, 0.576, 0.565)),
                    (1.000, (0.004, 0.400, 0.569)),
                ],
            )
        return self._sequential_green_blue


# Create a singleton instance for the colormaps
colormaps = _Colormaps()
