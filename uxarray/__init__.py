from .grid import *
from .helpers import *


def __version__():
    """Returns the version of uxarray currently installed."""
    # Attempt to import the needed modules
    try:
        from importlib.metadata import version as version
    except ImportError:
        from importlib_metadata import version as version

    try:
        __version = version("uxarray")
    except NameError:
        # Placeholder version incase an error occurs, such as the library isn't installed
        __version = "0000"

    return __version
