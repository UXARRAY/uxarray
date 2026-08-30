"""
File Purpose: all custom error types in uxarray

Defining custom error types helps with:
    - quickly presenting relevant info
    - cleaner error handling within uxarray
    - cleaner error handling for packages using uxarray

Defining all such custom error types in this file, instead of throughout the codebase,
    helps with maintainability, discoverability, and convenience,
    as the import syntax will always be "import uxarray.errors.CustomError",
    and help(uxarray.errors) will list all custom error types in one place.

For more details, see: https://github.com/UXARRAY/uxarray/issues/1556
"""


class DataCenteringError(ValueError):
    """data centering issue, such as expecting node-centered data but got edge-centered."""


class DimensionError(ValueError):
    """issue with dimension(s), such as wrong size(s), name(s), or number of dimensions."""


class GridInvalidError(ValueError):
    """data does not correspond to a valid uxarray.Grid.
    E.g., unrecognized format, duplicate nodes, or some faces with area < 0.
    """


class GridsMismatchError(ValueError):
    """attempted to perform an operation involving two incompatible uxarray.Grid objects"""


# # # ----- Io-type-specific Errors ----- # # #


class YacNotAvailableError(RuntimeError):
    """Raised when the YAC backend is requested but unavailable."""


# # # ----- Miscellaneous Errors ----- # # #


class OptionalDependencyNotFoundError(ModuleNotFoundError):
    """indicates functionality relies on a not-yet-installed optional dependency."""
