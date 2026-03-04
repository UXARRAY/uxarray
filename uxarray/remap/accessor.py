from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from uxarray.core.dataarray import UxDataArray
    from uxarray.core.dataset import UxDataset
    from uxarray.grid.grid import Grid

from uxarray.remap.bilinear import _bilinear
from uxarray.remap.inverse_distance_weighted import _inverse_distance_weighted_remap
from uxarray.remap.nearest_neighbor import _nearest_neighbor_remap


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
        """
        return self.nearest_neighbor(
            *args,
            backend=backend,
            yac_method=yac_method,
            yac_options=yac_options,
            **kwargs,
        )

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
        yac_method : {'nnn', 'conservative'}, optional
            YAC interpolation method. Defaults to 'nnn' when backend='yac'.
        yac_options : dict, optional
            YAC interpolation configuration options.

        Returns
        -------
        UxDataArray or UxDataset
            A new object with data mapped onto `destination_grid`.
        """

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
        """

        if backend == "yac":
            from uxarray.remap.yac import _yac_remap

            yac_kwargs = yac_options or {}
            return _yac_remap(
                self.ux_obj, destination_grid, remap_to, yac_method, yac_kwargs
            )
        return _inverse_distance_weighted_remap(
            self.ux_obj, destination_grid, remap_to, power, k
        )

    def bilinear(
        self,
        destination_grid: Grid,
        remap_to: str = "faces",
        backend: str = "uxarray",
        yac_method: str | None = None,
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
        """

        if backend == "yac":
            from uxarray.remap.yac import _yac_remap

            yac_kwargs = yac_options or {}
            return _yac_remap(
                self.ux_obj, destination_grid, remap_to, yac_method, yac_kwargs
            )
        return _bilinear(self.ux_obj, destination_grid, remap_to)
