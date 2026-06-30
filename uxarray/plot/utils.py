import holoviews as hv


class HoloviewsBackend:
    """Compare and set the HoloViews plotting backend."""

    def __init__(self):
        self.matplotlib_backend = None

    def assign(self, backend: str):
        """Assign a HoloViews backend, one of 'matplotlib', 'bokeh'."""
        if backend not in ["bokeh", "matplotlib", None]:
            raise ValueError(
                f"Unsupported backend. Expected one of ['bokeh', 'matplotlib'], but received {backend}"
            )
        if backend is not None and backend != hv.Store.current_backend:
            import matplotlib as mpl

            # Capture the live backend now (not once at init) so a backend the
            # user set later is what we restore.
            self.matplotlib_backend = mpl.get_backend()
            hv.extension(backend)

            if backend == "matplotlib":
                # hv.extension("matplotlib") switches the active Matplotlib
                # backend (e.g. to agg in Jupyter), breaking subsequent native
                # matplotlib/xarray .plot() calls. HoloViews renders through
                # hv.Store.current_backend, so restoring is safe. See #1537.
                self.reset_mpl_backend()

    def reset_mpl_backend(self):
        """Switch Matplotlib back to the backend captured before the last switch."""
        if self.matplotlib_backend is None:
            return

        import matplotlib as mpl

        mpl.use(self.matplotlib_backend)


backend = HoloviewsBackend()
