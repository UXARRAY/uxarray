class Imports:
    """Benchmark importing uxarray."""

    def time_import_uxarray(self):
        import uxarray as ux
        ux.__version__
