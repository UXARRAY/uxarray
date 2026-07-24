"""
File Purpose: custom error types in uxarray

Defining custom error types helps with:
    - quickly presenting relevant info
    - cleaner error handling within uxarray
    - cleaner error handling for packages using uxarray

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
