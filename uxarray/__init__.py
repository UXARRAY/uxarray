from .grid import *
from .helpers import *


def __version__():
    # Attempt to import the needed modules
    try:
        from importlib.metadata import version as version
    except ImportError:
        from importlib_metadata import version as version

    try:
        __version = version("uxarray")
        print(__version)
    except AttributeError:
        # Incase something happens, such as the library isn't installed
        __version = "0000"
