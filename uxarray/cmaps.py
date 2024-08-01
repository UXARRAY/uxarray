from matplotlib.colors import LinearSegmentedColormap


# UXarray themed sequential color map
def sequential():
    sequential_map = LinearSegmentedColormap.from_list(
        "sequential_map",
        ((0.000, (0.698, 0.463, 0.231)), (1.000, (0.200, 0.235, 0.125))),
    )

    return sequential_map


# UXarray themed diverging color map
def diverging():
    diverging_map = LinearSegmentedColormap.from_list(
        "diverging_map",
        (
            (0.000, (0.612, 0.361, 0.114)),
            (0.400, (0.973, 0.792, 0.616)),
            (0.500, (1.000, 1.000, 1.000)),
            (0.600, (0.741, 0.820, 0.580)),
            (1.000, (0.145, 0.192, 0.051)),
        ),
    )

    return diverging_map
