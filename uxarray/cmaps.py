from matplotlib.colors import LinearSegmentedColormap


diverging = LinearSegmentedColormap.from_list(
    "diverging",
    (
        (0.000, (0.016, 0.576, 0.565)),
        (0.500, (1.000, 1.000, 1.000)),
        (1.000, (0.004, 0.400, 0.569)),
    ),
)

# UXarray themed sequential color map
sequential = LinearSegmentedColormap.from_list(
    "sequential", ((0.000, (0.004, 0.400, 0.569)), (1.000, (0.016, 0.576, 0.565)))
)

sequential_blue = LinearSegmentedColormap.from_list(
    "sequential_blue", ((0.000, (1.000, 1.000, 1.000)), (1.000, (0.004, 0.400, 0.569)))
)

sequential_green = LinearSegmentedColormap.from_list(
    "sequential_green", ((0.000, (1.000, 1.000, 1.000)), (1.000, (0.016, 0.576, 0.565)))
)

sequential_green_blue = LinearSegmentedColormap.from_list(
    "sequential_green_blue",
    (
        (0.000, (1.000, 1.000, 1.000)),
        (0.500, (0.016, 0.576, 0.565)),
        (1.000, (0.004, 0.400, 0.569)),
    ),
)
