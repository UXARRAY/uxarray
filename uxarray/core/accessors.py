"""
Delegation-based accessor classes for UxArray groupby operations.

These classes wrap xarray's groupby/resample/etc objects and ensure that
operations return UxDataArray/UxDataset objects with preserved uxgrid.
"""

from typing import Any

import xarray as xr

# Mapping of method names to their accessor classes
DATASET_ACCESSOR_METHODS = {
    "groupby": "UxDatasetGroupByAccessor",
    "groupby_bins": "UxDatasetGroupByAccessor",  # Uses same accessor as groupby
    "resample": "UxDatasetResampleAccessor",
    "rolling": "UxDatasetRollingAccessor",
    "coarsen": "UxDatasetCoarsenAccessor",
    "weighted": "UxDatasetWeightedAccessor",
    "rolling_exp": "UxDatasetRollingExpAccessor",
    "cumulative": "UxDatasetRollingAccessor",  # Uses same accessor as rolling
}

DATAARRAY_ACCESSOR_METHODS = {
    "groupby": "UxDataArrayGroupByAccessor",
    "groupby_bins": "UxDataArrayGroupByAccessor",  # Uses same accessor as groupby
    "resample": "UxDataArrayResampleAccessor",
    "rolling": "UxDataArrayRollingAccessor",
    "coarsen": "UxDataArrayCoarsenAccessor",
    "weighted": "UxDataArrayWeightedAccessor",
    "rolling_exp": "UxDataArrayRollingExpAccessor",
    "cumulative": "UxDataArrayRollingAccessor",  # Uses same accessor as rolling
}


class BaseAccessor:
    """Base class for all UxArray accessor classes."""

    # Default methods known to return DataArrays/Datasets - optimized for performance
    _DEFAULT_PRESERVE_METHODS = {
        "mean",
        "sum",
        "std",
        "var",
        "min",
        "max",
        "count",
        "median",
        "quantile",
        "first",
        "last",
        "prod",
        "all",
        "any",
        "argmax",
        "argmin",
        "ffill",
        "bfill",
        "fillna",
        "where",
        "interpolate_na",
        "reduce",
        "map",
        "apply",
        "assign",
        "assign_coords",
    }

    # Class-level configuration
    _preserve_methods = _DEFAULT_PRESERVE_METHODS.copy()
    _auto_wrap = False

    def __init__(self, xr_obj, uxgrid, source_datasets=None):
        """
        Parameters
        ----------
        xr_obj : xarray accessor object
            The xarray accessor object to wrap
        uxgrid : uxarray.Grid
            The grid to preserve in results
        source_datasets : str, optional
            Source dataset information to preserve (Dataset only)
        """
        self._xr_obj = xr_obj
        self._uxgrid = uxgrid
        self._source_datasets = source_datasets

    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to the wrapped xarray object."""
        attr = getattr(self._xr_obj, name)

        if not callable(attr):
            return attr

        # Check if this method should be wrapped
        if self._should_wrap_method(name):
            return self._wrap_method(attr)

        return attr

    def _should_wrap_method(self, name: str) -> bool:
        """Determine if a method should be wrapped."""
        if self._auto_wrap:
            # In auto-wrap mode, check if it's callable
            return hasattr(getattr(self._xr_obj, name, None), "__call__")
        return name in self._preserve_methods

    def _wrap_method(self, method):
        """Wrap a method to preserve uxgrid in results."""

        def wrapped(*args, **kwargs):
            result = method(*args, **kwargs)
            return self._process_result(result)

        # Preserve method metadata
        wrapped.__name__ = getattr(method, "__name__", "wrapped")
        wrapped.__doc__ = getattr(method, "__doc__", "")
        return wrapped

    def _process_result(self, result):
        """Process method results to preserve uxgrid. To be overridden by subclasses."""
        raise NotImplementedError("Subclasses must implement _process_result")

    # Delegation for common dunder methods
    def __iter__(self):
        return iter(self._xr_obj)

    def __len__(self):
        return len(self._xr_obj)

    def __repr__(self):
        # Subclasses should override this for specific names
        return f"{self.__class__.__name__}({repr(self._xr_obj)})"

    def __dir__(self):
        # Combine our methods with xarray's for better IDE support
        return sorted(set(dir(self.__class__)) | set(dir(self._xr_obj)))

    # Class methods for configuration
    @classmethod
    def add_preserve_method(cls, method_name: str):
        """Add a method to the list of methods that preserve uxgrid."""
        cls._preserve_methods.add(method_name)

    @classmethod
    def remove_preserve_method(cls, method_name: str):
        """Remove a method from the list of methods that preserve uxgrid."""
        cls._preserve_methods.discard(method_name)

    @classmethod
    def set_preserve_methods(cls, methods: set[str]):
        """Set the complete list of methods that preserve uxgrid."""
        cls._preserve_methods = set(methods)

    @classmethod
    def reset_preserve_methods(cls):
        """Reset to the default list of methods."""
        cls._preserve_methods = cls._DEFAULT_PRESERVE_METHODS.copy()

    @classmethod
    def enable_auto_wrap(cls, enabled: bool = True):
        """Enable or disable automatic wrapping of all callable methods."""
        cls._auto_wrap = enabled


class UxDataArrayGroupByAccessor(BaseAccessor):
    """Wraps xarray DataArrayGroupBy to preserve uxgrid in results."""

    def __init__(self, xr_obj, uxgrid):
        """
        Parameters
        ----------
        xr_obj : xarray.DataArrayGroupBy
            The xarray groupby object to wrap
        uxgrid : uxarray.Grid
            The grid to preserve in results
        """
        super().__init__(xr_obj, uxgrid)

    def _process_result(self, result):
        """Process method results to preserve uxgrid."""
        if isinstance(result, xr.DataArray):
            from uxarray.core.dataarray import UxDataArray

            return UxDataArray(result, uxgrid=self._uxgrid)
        elif isinstance(result, xr.Dataset):
            from uxarray.core.dataset import UxDataset

            return UxDataset(result, uxgrid=self._uxgrid)
        return result

    def __repr__(self):
        return f"UxDataArrayGroupBy({repr(self._xr_obj)})"


class UxDatasetGroupByAccessor(BaseAccessor):
    """Wraps xarray DatasetGroupBy to preserve uxgrid in results."""

    def _process_result(self, result):
        """Process method results to preserve uxgrid."""
        if isinstance(result, xr.Dataset):
            from uxarray.core.dataset import UxDataset

            return UxDataset(
                result, uxgrid=self._uxgrid, source_datasets=self._source_datasets
            )
        elif isinstance(result, xr.DataArray):
            from uxarray.core.dataarray import UxDataArray

            return UxDataArray(result, uxgrid=self._uxgrid)
        return result

    def __repr__(self):
        return f"UxDatasetGroupBy({repr(self._xr_obj)})"


# Similar accessors for other operations
class UxDataArrayResampleAccessor(UxDataArrayGroupByAccessor):
    """Wraps xarray DataArrayResample to preserve uxgrid in results."""

    def __repr__(self):
        return f"UxDataArrayResample({repr(self._xr_obj)})"


class UxDatasetResampleAccessor(UxDatasetGroupByAccessor):
    """Wraps xarray DatasetResample to preserve uxgrid in results."""

    def __repr__(self):
        return f"UxDatasetResample({repr(self._xr_obj)})"


class UxDataArrayRollingAccessor(UxDataArrayGroupByAccessor):
    """Wraps xarray DataArrayRolling to preserve uxgrid in results."""

    def __repr__(self):
        return f"UxDataArrayRolling({repr(self._xr_obj)})"


class UxDatasetRollingAccessor(UxDatasetGroupByAccessor):
    """Wraps xarray DatasetRolling to preserve uxgrid in results."""

    def __repr__(self):
        return f"UxDatasetRolling({repr(self._xr_obj)})"


class UxDataArrayCoarsenAccessor(UxDataArrayGroupByAccessor):
    """Wraps xarray DataArrayCoarsen to preserve uxgrid in results."""

    def __repr__(self):
        return f"UxDataArrayCoarsen({repr(self._xr_obj)})"


class UxDatasetCoarsenAccessor(UxDatasetGroupByAccessor):
    """Wraps xarray DatasetCoarsen to preserve uxgrid in results."""

    def __repr__(self):
        return f"UxDatasetCoarsen({repr(self._xr_obj)})"


class UxDataArrayWeightedAccessor(UxDataArrayGroupByAccessor):
    """Wraps xarray DataArrayWeighted to preserve uxgrid in results."""

    def __repr__(self):
        return f"UxDataArrayWeighted({repr(self._xr_obj)})"


class UxDatasetWeightedAccessor(UxDatasetGroupByAccessor):
    """Wraps xarray DatasetWeighted to preserve uxgrid in results."""

    def __repr__(self):
        return f"UxDatasetWeighted({repr(self._xr_obj)})"


class UxDataArrayRollingExpAccessor(UxDataArrayGroupByAccessor):
    """Wraps xarray DataArrayRollingExp to preserve uxgrid in results."""

    def __repr__(self):
        return f"UxDataArrayRollingExp({repr(self._xr_obj)})"


class UxDatasetRollingExpAccessor(UxDatasetGroupByAccessor):
    """Wraps xarray DatasetRollingExp to preserve uxgrid in results."""

    def __repr__(self):
        return f"UxDatasetRollingExp({repr(self._xr_obj)})"
