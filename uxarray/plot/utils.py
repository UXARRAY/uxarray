import holoviews as hv


class HoloviewsBackend:
    """Utility class to compare and set holoviews backends for
    visualization."""

    def __init__(self):
        self.backend = None

    def assign(self, backend: str):
        """Assigns a backend for use with HoloViz visualization."""

        if backend not in ["bokeh", "matplotlib"]:
            raise ValueError(
                f"Unsupported backend. Expected one of ['bokeh', 'matplotlib'], but received {backend}"
            )

        if backend != self.backend:
            # only call hv.extension if it needs to be changed
            hv.extension(backend)
            self.backend = backend


# global reference to class to comparing and setting backend
backend = HoloviewsBackend()
