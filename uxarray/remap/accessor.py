from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from uxarray.core.dataarray import UxDataArray
    from uxarray.core.dataset import UxDataset
    from uxarray.grid.grid import Grid

from uxarray.remap.bilinear import _bilinear
from uxarray.remap.inverse_distance_weighted import _inverse_distance_weighted_remap
from uxarray.remap.nearest_neighbor import _nearest_neighbor_remap

_VALID_BACKENDS = ("uxarray", "yac")


def _validate_backend(backend: str) -> None:
    if backend not in _VALID_BACKENDS:
        raise ValueError(
            f"Invalid backend '{backend}'. Expected one of {_VALID_BACKENDS}."
        )


class RemapAccessor:
    """Expose remapping methods on UxDataArray and UxDataset objects."""

    def __init__(self, ux_obj: UxDataArray | UxDataset):
        self.ux_obj = ux_obj

    def __repr__(self) -> str:
        prefix = f"<{type(self.ux_obj).__name__}.remap>\n"
        return (
            prefix
            + "Supported methods:\n"
            + "  • nearest_neighbor(destination_grid, remap_to='faces')\n"
            + "  • inverse_distance_weighted(destination_grid, remap_to='faces', power=2, k=8)\n"
            + "  • to_rectilinear(lon, lat, backend='yac')\n"
        )

    def __call__(
        self,
        *args,
        backend: str = "uxarray",
        yac_method: str | None = None,
        yac_options: dict | None = None,
        **kwargs,
    ) -> UxDataArray | UxDataset:
        """
        Shortcut for nearest-neighbor remapping.

        Calling `.remap(...)` with no explicit method will invoke
        `nearest_neighbor(...)`.

        When ``backend="yac"``, this generic entrypoint can also be used to
        select a YAC-specific interpolation method through ``yac_method``.
        """
        nn_kwargs: dict = {"backend": backend, "yac_options": yac_options}
        if yac_method is not None:
            nn_kwargs["yac_method"] = yac_method
        return self.nearest_neighbor(*args, **nn_kwargs, **kwargs)

    def nearest_neighbor(
        self,
        destination_grid: Grid,
        remap_to: str = "faces",
        backend: str = "uxarray",
        yac_method: str | None = "nnn",
        yac_options: dict | None = None,
        **kwargs,
    ) -> UxDataArray | UxDataset:
        """
        Perform nearest-neighbor remapping.

        Each destination point takes the value of its closest source point.

        Parameters
        ----------
        destination_grid : Grid
            The UXarray grid to which data will be interpolated.
        remap_to : {'nodes', 'edges', 'faces'}, default='faces'
            Which grid element receives the remapped values.

        backend : {'uxarray', 'yac'}, default='uxarray'
            Remapping backend to use. When set to 'yac', requires YAC to be
            available on PYTHONPATH.
        yac_method : {'nnn', 'average', 'conservative'}, optional
            YAC interpolation method. Defaults to 'nnn' when backend='yac'.
        yac_options : dict, optional
            YAC interpolation configuration options.

        Returns
        -------
        UxDataArray or UxDataset
            A new object with data mapped onto `destination_grid`.

        Notes
        -----
        When ``backend="yac"``, remapping uses YAC's low-level ``yac.core``
        Python bindings. See the YAC documentation and installation guide:

        - https://dkrz-sw.gitlab-pages.dkrz.de/yac/
        - https://dkrz-sw.gitlab-pages.dkrz.de/yac/d1/d9f/installing_yac.html
        """

        _validate_backend(backend)
        if backend == "yac":
            from uxarray.remap.yac import _yac_remap

            yac_kwargs = yac_options or {}
            return _yac_remap(
                self.ux_obj, destination_grid, remap_to, yac_method, yac_kwargs
            )
        return _nearest_neighbor_remap(self.ux_obj, destination_grid, remap_to)

    def inverse_distance_weighted(
        self,
        destination_grid: Grid,
        remap_to: str = "faces",
        power=2,
        k=8,
        backend: str = "uxarray",
        yac_method: str | None = None,
        yac_options: dict | None = None,
        **kwargs,
    ) -> UxDataArray | UxDataset:
        """
        Perform inverse-distance-weighted (IDW) remapping.

        Each destination point is a weighted average of nearby source points,
        with weights proportional to 1/(distance**power).

        Parameters
        ----------
        destination_grid : Grid
            The UXarray grid to which data will be interpolated.
        remap_to : {'nodes', 'edges', 'faces'}, default='faces'
            Which grid element receives the remapped values.
        power : int, default=2
            Exponent controlling distance decay. Larger values make the
            interpolation more local.
        k : int, default=8
            Number of nearest source points to include in the weighted average.

        backend : {'uxarray', 'yac'}, default='uxarray'
            Remapping backend to use. When set to 'yac', requires YAC to be
            available on PYTHONPATH.
        yac_method : {'nnn', 'conservative'}, optional
            YAC interpolation method. Required when backend='yac'.
        yac_options : dict, optional
            YAC interpolation configuration options.

        Returns
        -------
        UxDataArray or UxDataset
            A new object with data mapped onto `destination_grid`.

        Notes
        -----
        When ``backend="yac"``, this method delegates to YAC's ``average``
        interpolation method through the low-level ``yac.core`` Python
        bindings. See the YAC documentation and installation guide:

        - https://dkrz-sw.gitlab-pages.dkrz.de/yac/
        - https://dkrz-sw.gitlab-pages.dkrz.de/yac/d1/d9f/installing_yac.html
        """

        _validate_backend(backend)
        if backend == "yac":
            raise NotImplementedError(
                "inverse_distance_weighted with backend='yac' is not currently "
                "exposed through the UXarray YAC accessor. "
                "Use backend='uxarray' for IDW, or use the YAC backend through "
                ".remap(..., backend='yac', yac_method=..., yac_options=...)."
            )
        return _inverse_distance_weighted_remap(
            self.ux_obj, destination_grid, remap_to, power, k
        )

    def to_rectilinear(
        self,
        lon,
        lat,
        backend: str = "yac",
        yac_method: str | None = "nnn",
        yac_options: dict | None = None,
        **kwargs,
    ):
        """
        Remap onto a rectilinear longitude/latitude grid.

        This convenience method targets 1-D longitude and latitude coordinate
        arrays and returns a plain xarray object with ``lat`` and ``lon`` axes,
        making the output suitable for downstream structured-grid workflows.

        Parameters
        ----------
        lon : array-like or xarray.DataArray
            1-D target longitude cell-center coordinate in degrees.
        lat : array-like or xarray.DataArray
            1-D target latitude cell-center coordinate in degrees.
        backend : {'uxarray', 'yac'}, default='yac'
            Remapping backend to use. The YAC backend uses YAC's rectilinear
            grid support directly.
        yac_method : {'nnn', 'average', 'conservative'}, optional
            YAC interpolation method. Defaults to ``'nnn'``.
        yac_options : dict, optional
            YAC interpolation configuration options.

        Returns
        -------
        xarray.DataArray or xarray.Dataset
            Remapped data with the source spatial dimension replaced by the
            provided latitude and longitude dimensions.
        """

        _validate_backend(backend)
        if backend == "yac":
            from uxarray.remap.yac import _yac_remap_to_rectilinear

            return _yac_remap_to_rectilinear(
                self.ux_obj,
                lon,
                lat,
                yac_method or "nnn",
                yac_options or {},
            )

        from uxarray.remap.structured import _native_remap_to_rectilinear

        return _native_remap_to_rectilinear(self.ux_obj, lon, lat)

    def bilinear(
        self,
        destination_grid: Grid,
        remap_to: str = "faces",
        backend: str = "uxarray",
        yac_method: str | None = "average",
        yac_options: dict | None = None,
        **kwargs,
    ) -> UxDataArray | UxDataset:
        """
        Perform bilinear remapping.

        Parameters
        ---------
        destination_grid : Grid
            Destination Grid for remapping
        remap_to : {'nodes', 'edges', 'faces'}, default='faces'
            Which grid element receives the remapped values.

        backend : {'uxarray', 'yac'}, default='uxarray'
            Remapping backend to use. When set to 'yac', bilinear remapping is
            routed through YAC's average interpolation.
        yac_method : {'average'}, optional
            YAC interpolation method for the bilinear convenience wrapper.
            Only ``'average'`` is supported here.
        yac_options : dict, optional
            YAC interpolation configuration options for the average method.

        Returns
        -------
        UxDataArray or UxDataset
            A new object with data mapped onto `destination_grid`.
        """

        _validate_backend(backend)
        if backend == "yac":
            from uxarray.remap.yac import _yac_remap

            if yac_method not in (None, "average"):
                raise ValueError(
                    "bilinear with backend='yac' only supports yac_method='average'. "
                    "Use .remap(..., backend='yac', yac_method=...) for other YAC methods."
                )
            yac_kwargs = yac_options or {}
            return _yac_remap(
                self.ux_obj,
                destination_grid,
                remap_to,
                yac_method or "average",
                yac_kwargs,
            )
        return _bilinear(self.ux_obj, destination_grid, remap_to)
