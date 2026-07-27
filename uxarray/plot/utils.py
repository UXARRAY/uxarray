class HoloviewsBackend:
    """Compare and set the HoloViews plotting backend."""

    def __init__(self):
        self.matplotlib_backend = None

    def assign(self, backend: str):
        """Assign a HoloViews backend, one of 'matplotlib', 'bokeh'."""
        import holoviews as hv

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
        """Restore the Matplotlib backend captured before the last switch.

        ``hv.extension("matplotlib")`` does not just change the active backend;
        in an IPython/Jupyter kernel it also tears down the display integration
        that auto-renders figures at the end of a cell. Simply calling
        ``mpl.use`` puts the backend name back but leaves that integration
        broken, so subsequent native ``matplotlib``/``xarray`` ``.plot()`` calls
        silently produce no output unless ``plt.show()`` is called explicitly.

        Inside IPython we therefore re-run the shell's own backend activation
        (the public equivalent of the ``%matplotlib`` magic), which rebuilds the
        full integration in one step. Outside IPython we fall back to
        ``mpl.use``.
        """
        if self.matplotlib_backend is None:
            return

        try:
            from IPython import get_ipython

            shell = get_ipython()
        except ImportError:
            shell = None

        if shell is not None:
            # Map the stored backend to the gui name enable_matplotlib expects.
            gui = self.matplotlib_backend
            if gui.startswith("module://") and "inline" in gui:
                gui = "inline"
            try:
                shell.enable_matplotlib(gui)
                return
            except Exception:
                pass

        import matplotlib as mpl

        mpl.use(self.matplotlib_backend)


backend = HoloviewsBackend()
