class HoloviewsBackend:
    """Utility class to compare and set a HoloViews plotting backend for
    visualization."""

    def __init__(self):
        import matplotlib as mpl

        self.matplotlib_backend = mpl.get_backend()

    def assign(self, backend: str):
        """Assigns a backend for use with HoloViews visualization.

        Parameters
        ----------
        backend : str
            Plotting backend to use, one of 'matplotlib', 'bokeh'
        """

        from holoviews import Store, extension

        if backend not in ["bokeh", "matplotlib", None]:
            raise ValueError(
                f"Unsupported backend. Expected one of ['bokeh', 'matplotlib'], but received {backend}"
            )
        if backend is not None:
            from holoviews import Store, extension

            if backend != Store.current_backend:
                # only call hv.extension if it needs to be changed
                extension(backend)

    def reset_mpl_backend(self):
        """Resets the default backend for the ``matplotlib`` module."""
        import matplotlib as mpl

        mpl.use(self.matplotlib_backend)


# global reference to holoviews backend utility class
backend = HoloviewsBackend()
