from matplotlib.colors import LinearSegmentedColormap


diverging = LinearSegmentedColormap.from_list(
    "diverging",
    (
        (0.000, (0.698, 0.416, 0.137)),
        (0.400, (0.867, 0.678, 0.498)),
        (0.500, (1.000, 1.000, 1.000)),
        (0.600, (0.757, 0.859, 0.541)),
        (1.000, (0.498, 0.584, 0.322)),
    ),
)

# UXarray themed sequential color map
sequential = LinearSegmentedColormap.from_list(
    "sequential", ((0.000, (0.698, 0.463, 0.231)), (1.000, (0.498, 0.584, 0.322)))
)

sequential_orange = LinearSegmentedColormap.from_list(
    "sequential_orange",
    ((0.000, (1.000, 1.000, 1.000)), (1.000, (0.698, 0.416, 0.137))),
)

sequential_green = LinearSegmentedColormap.from_list(
    "my_gradient", ((0.000, (1.000, 1.000, 1.000)), (1.000, (0.498, 0.584, 0.322)))
)
