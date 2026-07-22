class Imports:
    """Benchmark importing uxarray."""

    def timeraw_import_uxarray(self):
        return "import uxarray"

    def peakmem_import_uxarray(self):
        """Peak memory of a process that has imported uxarray.

        This is the floor under every other ``peakmem_*`` result, so it is worth
        tracking on its own: a heavy new top-level import shows up here rather
        than silently inflating everything else.
        """
        import uxarray  # noqa: F401
