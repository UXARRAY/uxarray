import holoviews as hv


class HoloviewsBackend:
    """Utility class to compare and set holoviews backends for
    visualization."""

    def __init__(self):
        self.backend = None

    def compare_and_set(self, backend: str):
        """Compares the currently set holoviews backend and sets it to the
        desired one, if different."""

        if backend not in ["bokeh", "matplotlib"]:
            raise ValueError(
                f"Unsupported backend. Expected one of ['bokeh', 'matplotlib'], but received {backend}"
            )

        if backend != self.backend:
            hv.extension(backend)
            self.backend = backend


# global reference to class to comparing and setting backend
hv_backend_ref = HoloviewsBackend()
