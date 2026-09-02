import inspect
import os
from functools import lru_cache


@lru_cache(maxsize=None)
def _package_dir():
    import uxarray

    return os.path.dirname(os.path.abspath(uxarray.__file__))


def find_stack_level():
    """``stacklevel`` that attributes a warning to the first frame outside uxarray.

    A hardcoded ``stacklevel`` names whichever frame happens to sit that many
    levels up, so a warning raised deep in a reader either blames another
    internal module or silently starts pointing somewhere else the moment a
    helper is added or removed. Counting frames until the stack leaves the
    package keeps the warning pinned to the caller's own line regardless of how
    many internal calls separate them.
    """
    package_dir = _package_dir()

    frame = inspect.currentframe()
    n = 0
    try:
        while frame is not None:
            if os.path.abspath(frame.f_code.co_filename).startswith(package_dir):
                frame = frame.f_back
                n += 1
            else:
                break
    finally:
        # Break the reference cycle a live frame object creates.
        del frame

    # ``n`` counts this function's own frame, which is exactly the off-by-one
    # between "frames inside the package" and the ``stacklevel`` the caller of
    # ``warnings.warn`` needs.
    return n
