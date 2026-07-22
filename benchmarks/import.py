class Imports:
    """Benchmark importing uxarray."""

    def timeraw_import_uxarray(self):
        return "import uxarray"

    def peakmem_import_uxarray(self):
        """Peak memory of a process that has imported uxarray."""
        import uxarray  # noqa: F401
