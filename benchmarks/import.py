from .helpers._peakmem import subprocess_peak_rss


class Imports:
    """Benchmark importing uxarray."""

    def timeraw_import_uxarray(self):
        return "import uxarray"

    def track_peakmem_import_uxarray(self):
        """Peak resident memory of a process that has imported uxarray."""
        return subprocess_peak_rss("import uxarray")

    track_peakmem_import_uxarray.unit = "bytes"
