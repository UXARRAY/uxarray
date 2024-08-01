from matplotlib.colors import LinearSegmentedColormap


# UXarray themed sequential color map
def sequential():
    sequential_map = LinearSegmentedColormap.from_list(
        "sequential", ((0.000, (0.698, 0.463, 0.231)), (1.000, (0.200, 0.235, 0.125)))
    )

    return sequential_map


# UXarray themed diverging color map
def diverging():
    diverging_map = LinearSegmentedColormap.from_list(
        "diverging",
        (
            (0.000, (0.698, 0.463, 0.231)),
            (0.400, (0.867, 0.678, 0.498)),
            (0.500, (1.000, 1.000, 1.000)),
            (0.600, (0.757, 0.859, 0.541)),
            (1.000, (0.200, 0.235, 0.125)),
        ),
    )

    return diverging_map
