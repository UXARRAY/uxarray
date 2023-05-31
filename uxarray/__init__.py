from .grid import *
from .helpers import *
# Sets the version of uxarray currently installed
# Attempt to import the needed modules
try:
    from importlib.metadata import version as version
except ImportError:
    from importlib_metadata import version as version

try:
    __version = version("uxarray")
except Exception:
    # Placeholder version incase an error occurs, such as the library isn't installed
    __version = "999"
