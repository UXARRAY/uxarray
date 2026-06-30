import holoviews as hv


class HoloviewsBackend:
    """Utility class to compare and set a HoloViews plotting backend for
    visualization."""

    def __init__(self):
        self.matplotlib_backend = None

    def assign(self, backend: str):
        """Assigns a backend for use with HoloViews visualization.

        Parameters
        ----------
        backend : str
            Plotting backend to use, one of 'matplotlib', 'bokeh'
        """

        if self.matplotlib_backend is None:
            import matplotlib as mpl

            self.matplotlib_backend = mpl.get_backend()

        if backend not in ["bokeh", "matplotlib", None]:
            raise ValueError(
                f"Unsupported backend. Expected one of ['bokeh', 'matplotlib'], but received {backend}"
            )
        if backend is not None and backend != hv.Store.current_backend:
            # only call hv.extension if it needs to be changed
            hv.extension(backend)

            if backend == "matplotlib":
                # ``hv.extension("matplotlib")`` switches the active Matplotlib
                # backend (e.g. to ``agg`` in Jupyter), which clobbers the
                # IPython inline display hook. This silently breaks any
                # subsequent native ``matplotlib``/``xarray`` ``.plot()`` calls,
                # which render nothing. HoloViews objects still display because
                # they render through ``hv.Store.current_backend`` rather than
                # the active Matplotlib backend, so restoring the original
                # backend here is safe and fixes downstream plotting. See
                # https://github.com/UXARRAY/uxarray/issues/1537
                self.reset_mpl_backend()

    def reset_mpl_backend(self):
        """Restores the original ``matplotlib`` backend (and its IPython inline
        display hook) that was active before a HoloViews backend switch."""
        if self.matplotlib_backend is None:
            return

        import matplotlib as mpl

        mpl.use(self.matplotlib_backend)


backend = HoloviewsBackend()
