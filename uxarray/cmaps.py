from matplotlib.colors import LinearSegmentedColormap


# UXarray themed sequential color map
sequential = LinearSegmentedColormap.from_list(
    "sequential", ((0.000, (0.698, 0.463, 0.231)), (1.000, (0.200, 0.235, 0.125)))
)

# UXarray themed diverging color map
diverging = LinearSegmentedColormap.from_list(
    "diverging",
    (
        (0.000, (0.698, 0.463, 0.231)),
        (0.400, (0.867, 0.678, 0.498)),
        (0.500, (1.000, 1.000, 1.000)),
        (0.600, (0.757, 0.859, 0.541)),
        (1.000, (0.200, 0.235, 0.125)),
    ),
)
