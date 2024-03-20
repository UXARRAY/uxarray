import holoviews as hv


class HoloviewsBackend:
    def __init__(self):
        self.backend = None

    def compare_and_set(self, backend: str):
        """TODO:"""

        if backend != self.backend:
            hv.extension(backend)
            self.backend = backend


# global reference to class to comparing and setting backend
hv_backend_ref = HoloviewsBackend()
