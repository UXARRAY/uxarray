from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import xarray as xr


@dataclass(frozen=True)
class RectilinearGridSpec:
    """Normalized 1-D longitude/latitude target grid metadata."""

    lon: xr.DataArray
    lat: xr.DataArray
    lon_edges: np.ndarray
    lat_edges: np.ndarray
    cyclic_lon: bool

    @property
    def shape(self) -> tuple[int, int]:
        return self.lat.size, self.lon.size

    @property
    def size(self) -> int:
        return self.lat.size * self.lon.size

    @property
    def lon_dim(self) -> str:
        return self.lon.dims[0]

    @property
    def lat_dim(self) -> str:
        return self.lat.dims[0]

    @property
    def lon_name(self) -> str:
        return self.lon.name or self.lon_dim

    @property
    def lat_name(self) -> str:
        return self.lat.name or self.lat_dim

    @property
    def lon_rad(self) -> np.ndarray:
        return np.deg2rad(np.asarray(self.lon.values, dtype=np.float64))

    @property
    def lat_rad(self) -> np.ndarray:
        return np.deg2rad(np.asarray(self.lat.values, dtype=np.float64))

    @property
    def lon_edges_rad(self) -> np.ndarray:
        return np.deg2rad(self.lon_edges)

    @property
    def lat_edges_rad(self) -> np.ndarray:
        return np.deg2rad(self.lat_edges)

    def flattened_centers_rad(self) -> tuple[np.ndarray, np.ndarray]:
        lon_2d, lat_2d = np.meshgrid(self.lon_rad, self.lat_rad)
        return lon_2d.ravel(), lat_2d.ravel()


def _as_1d_coord(coord, default_name: str) -> xr.DataArray:
    if isinstance(coord, xr.DataArray):
        values = np.asarray(coord.values, dtype=np.float64)
        if coord.ndim != 1:
            raise ValueError(
                f"Rectilinear {default_name!r} coordinate must be 1-D, "
                f"got {coord.ndim}-D."
            )
        dim = coord.dims[0]
        name = coord.name or default_name
        attrs = coord.attrs.copy()
    else:
        values = np.asarray(coord, dtype=np.float64)
        if values.ndim != 1:
            raise ValueError(
                f"Rectilinear {default_name!r} coordinate must be 1-D, "
                f"got {values.ndim}-D."
            )
        dim = default_name
        name = default_name
        attrs = {}

    if values.size < 2:
        raise ValueError(
            f"Rectilinear {default_name!r} coordinate must contain at least two values."
        )
    return xr.DataArray(values, dims=(dim,), name=name, attrs=attrs)


def _centers_to_edges(values: np.ndarray, coord_name: str) -> np.ndarray:
    diffs = np.diff(values)
    if np.any(diffs == 0):
        raise ValueError(f"Rectilinear {coord_name!r} coordinate contains duplicates.")
    if not (np.all(diffs > 0) or np.all(diffs < 0)):
        raise ValueError(
            f"Rectilinear {coord_name!r} coordinate must be strictly monotonic."
        )

    edges = np.empty(values.size + 1, dtype=np.float64)
    edges[1:-1] = values[:-1] + 0.5 * diffs
    edges[0] = values[0] - 0.5 * diffs[0]
    edges[-1] = values[-1] + 0.5 * diffs[-1]
    return edges


def _looks_global_lon(edges: np.ndarray) -> bool:
    return np.isclose(abs(edges[-1] - edges[0]), 360.0, rtol=0.0, atol=1.0e-8)


def _normalize_rectilinear_target(lon, lat) -> RectilinearGridSpec:
    lon_coord = _as_1d_coord(lon, "lon")
    lat_coord = _as_1d_coord(lat, "lat")
    lon_edges = _centers_to_edges(np.asarray(lon_coord.values), lon_coord.name or "lon")
    lat_edges = _centers_to_edges(np.asarray(lat_coord.values), lat_coord.name or "lat")

    return RectilinearGridSpec(
        lon=lon_coord,
        lat=lat_coord,
        lon_edges=lon_edges,
        lat_edges=lat_edges,
        cyclic_lon=_looks_global_lon(lon_edges),
    )


def _reshape_array_to_rectilinear(
    da: xr.DataArray, spec: RectilinearGridSpec
) -> xr.DataArray:
    if "n_face" not in da.dims:
        return xr.DataArray(da)

    axis = da.get_axis_num("n_face")
    if da.sizes["n_face"] != spec.size:
        raise ValueError(
            "Cannot reshape remapped data to the requested rectilinear grid. "
            f"Expected {spec.size} face values, got {da.sizes['n_face']}."
        )

    shape = da.shape[:axis] + spec.shape + da.shape[axis + 1 :]
    dims = da.dims[:axis] + (spec.lat_dim, spec.lon_dim) + da.dims[axis + 1 :]
    coords = {
        name: coord
        for name, coord in da.coords.items()
        if "n_face" not in coord.dims
        and name not in {spec.lat_name, spec.lon_name, spec.lat_dim, spec.lon_dim}
    }
    coords[spec.lat_name] = spec.lat
    coords[spec.lon_name] = spec.lon

    return xr.DataArray(
        np.asarray(da.values).reshape(shape),
        dims=dims,
        coords=coords,
        name=da.name,
        attrs=da.attrs,
    )


def _reshape_to_rectilinear(obj, spec: RectilinearGridSpec):
    """Convert flattened ``n_face`` remap output to lat/lon-shaped xarray."""

    if isinstance(obj, xr.DataArray):
        return _reshape_array_to_rectilinear(obj, spec)

    if isinstance(obj, xr.Dataset):
        xr_obj = obj
    elif hasattr(obj, "to_xarray"):
        xr_obj = obj.to_xarray()
    else:
        xr_obj = xr.Dataset(obj)
    data_vars = {
        name: _reshape_array_to_rectilinear(da, spec)
        for name, da in xr_obj.data_vars.items()
    }
    coords = {
        name: coord
        for name, coord in xr_obj.coords.items()
        if "n_face" not in coord.dims
        and name not in {spec.lat_name, spec.lon_name, spec.lat_dim, spec.lon_dim}
    }
    coords[spec.lat_name] = spec.lat
    coords[spec.lon_name] = spec.lon
    return xr.Dataset(data_vars=data_vars, coords=coords, attrs=xr_obj.attrs)


def _native_remap_to_rectilinear(source, lon, lat):
    """Remap to a rectilinear target through UXarray's native NN backend."""

    from uxarray.grid import Grid
    from uxarray.remap.nearest_neighbor import _nearest_neighbor_remap

    spec = _normalize_rectilinear_target(lon, lat)
    destination_grid = Grid.from_structured(lon=spec.lon.values, lat=spec.lat.values)
    remapped = _nearest_neighbor_remap(
        source,
        destination_grid=destination_grid,
        remap_to="faces",
    )
    return _reshape_to_rectilinear(remapped, spec)
