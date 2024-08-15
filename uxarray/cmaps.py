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
    "sequential", ((0.000, (0.004, 0.400, 0.569)), (1.000, (0.016, 0.576, 0.565)))
)

sequential_blue = LinearSegmentedColormap.from_list(
    "blue", ((0.000, (0.004, 0.400, 0.569)), (1.000, (1.000, 1.000, 1.000)))
)

sequential_green = LinearSegmentedColormap.from_list(
    "green", ((0.000, (0.016, 0.576, 0.565)), (1.000, (1.000, 1.000, 1.000)))
)
