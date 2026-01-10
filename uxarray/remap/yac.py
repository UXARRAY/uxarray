from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from uuid import uuid4

import numpy as np

import uxarray.core.dataarray
from uxarray.remap.utils import (
    LABEL_TO_COORD,
    _assert_dimension,
    _construct_remapped_ds,
    _get_remap_dims,
    _to_dataset,
)


class YacNotAvailableError(RuntimeError):
    """Raised when the YAC backend is requested but unavailable."""


@dataclass
class _YacOptions:
    method: str
    kwargs: dict[str, Any]


def _import_yac():
    try:
        import yac  # type: ignore
    except Exception as exc:  # pragma: no cover - import failure handled in tests
        raise YacNotAvailableError(
            "YAC backend requested but 'yac' is not available. "
            "Build YAC with Python bindings and ensure it is on PYTHONPATH."
        ) from exc
    return yac


def _get_lon_lat(grid, dim_kind: str) -> tuple[np.ndarray, np.ndarray]:
    if dim_kind == "node":
        for prefix in ("node", "vertex"):
            lon = getattr(grid, f"{prefix}_lon", None)
            lat = getattr(grid, f"{prefix}_lat", None)
            if lon is not None and lat is not None:
                return np.asarray(lon, dtype=np.float64), np.asarray(
                    lat, dtype=np.float64
                )
        raise AttributeError(
            "Grid has neither node_lon/node_lat nor vertex_lon/vertex_lat"
        )
    if dim_kind == "edge":
        lon = getattr(grid, "edge_lon", None)
        lat = getattr(grid, "edge_lat", None)
        if lon is None or lat is None:
            raise AttributeError("Grid does not provide edge_lon/edge_lat")
        return np.asarray(lon, dtype=np.float64), np.asarray(lat, dtype=np.float64)
    if dim_kind == "face":
        lon = getattr(grid, "face_lon", None)
        lat = getattr(grid, "face_lat", None)
        if lon is None or lat is None:
            raise AttributeError("Grid does not provide face_lon/face_lat")
        return np.asarray(lon, dtype=np.float64), np.asarray(lat, dtype=np.float64)
    raise ValueError(f"Unsupported grid dimension kind: {dim_kind!r}")


def _get_connectivity(grid) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    try:
        from uxarray import INT_FILL_VALUE

        fill_value = INT_FILL_VALUE
    except Exception:
        fill_value = -1
    connectivity = np.asarray(getattr(grid, "face_node_connectivity")).astype(np.int64)
    num_vertices = np.sum(connectivity != fill_value, axis=1).astype(np.intc)
    cell_to_vertex = connectivity[connectivity != fill_value].astype(np.intc)
    return connectivity, num_vertices, cell_to_vertex


def _build_unstructured_grid(yac, grid, grid_name: str):
    node_lon, node_lat = _get_lon_lat(grid, "node")
    _, num_vertices, cell_to_vertex = _get_connectivity(grid)
    return yac.UnstructuredGrid(
        grid_name,
        num_vertices,
        np.deg2rad(node_lon),
        np.deg2rad(node_lat),
        cell_to_vertex,
        use_ll_edges=False,
    )


def _normalize_yac_method(yac_method: str | None) -> _YacOptions:
    if not yac_method:
        raise ValueError(
            "backend='yac' requires yac_method to be set to 'nnn' or 'conservative'."
        )
    method = yac_method.lower()
    if method not in {"nnn", "conservative"}:
        raise ValueError(f"Unsupported YAC method: {yac_method!r}")
    return _YacOptions(method=method, kwargs={})


class _YacRemapper:
    def __init__(
        self,
        src_grid,
        tgt_grid,
        src_dim: str,
        tgt_dim: str,
        yac_method: str,
        yac_kwargs: dict[str, Any],
    ):
        yac = _import_yac()
        self._yac = yac
        yac.def_calendar(yac.Calendar.PROLEPTIC_GREGORIAN)
        self._yac_inst = yac.YAC(default_instance=True)
        self._yac_inst.def_datetime("2000-01-01T00:00:00", "2000-01-01T00:01:00")

        unique = uuid4().hex
        self._comp_name = f"uxarray_yac_{unique}"
        self._comp = self._yac_inst.def_comp(self._comp_name)
        self._src_grid_name = f"src_{unique}"
        self._tgt_grid_name = f"tgt_{unique}"

        self._src_points, self._tgt_points = self._build_points(
            src_grid, tgt_grid, src_dim, tgt_dim, yac_method
        )

        self._src_field = yac.Field.create(
            "src_field",
            self._comp,
            self._src_points,
            1,
            "1",
            yac.TimeUnit.SECOND,
        )
        self._tgt_field = yac.Field.create(
            "tgt_field",
            self._comp,
            self._tgt_points,
            1,
            "1",
            yac.TimeUnit.SECOND,
        )

        stack = yac.InterpolationStack()
        if yac_method == "nnn":
            reduction = yac_kwargs.get("reduction_type", yac.NNNReductionType.AVG)
            if isinstance(reduction, str):
                reduction = yac.NNNReductionType[reduction.upper()]
            stack.add_nnn(
                reduction_type=reduction,
                n=yac_kwargs.get("n", 1),
                max_search_distance=yac_kwargs.get("max_search_distance", 0.0),
                scale=yac_kwargs.get("scale", 1.0),
            )
        elif yac_method == "conservative":
            normalisation = yac_kwargs.get(
                "normalisation", yac.ConservNormalizationType.DESTAREA
            )
            if isinstance(normalisation, str):
                normalisation = yac.ConservNormalizationType[normalisation.upper()]
            stack.add_conservative(
                order=yac_kwargs.get("order", 1),
                enforced_conserv=yac_kwargs.get("enforced_conserv", False),
                partial_coverage=yac_kwargs.get("partial_coverage", False),
                normalisation=normalisation,
            )

        self._yac_inst.def_couple(
            self._comp_name,
            self._src_grid_name,
            "src_field",
            self._comp_name,
            self._tgt_grid_name,
            "tgt_field",
            "1",
            yac.TimeUnit.SECOND,
            yac.Reduction.TIME_NONE,
            stack,
        )
        self._yac_inst.enddef()

    def _build_points(self, src_grid, tgt_grid, src_dim, tgt_dim, yac_method):
        yac = self._yac
        if yac_method == "conservative":
            if src_dim != "n_face" or tgt_dim != "n_face":
                raise ValueError(
                    "YAC conservative remapping only supports face-centered data."
                )
            self._src_grid = _build_unstructured_grid(
                yac, src_grid, self._src_grid_name
            )
            self._tgt_grid = _build_unstructured_grid(
                yac, tgt_grid, self._tgt_grid_name
            )
            src_lon, src_lat = _get_lon_lat(src_grid, "face")
            tgt_lon, tgt_lat = _get_lon_lat(tgt_grid, "face")
            src_points = self._src_grid.def_points(
                yac.Location.CELL, np.deg2rad(src_lon), np.deg2rad(src_lat)
            )
            tgt_points = self._tgt_grid.def_points(
                yac.Location.CELL, np.deg2rad(tgt_lon), np.deg2rad(tgt_lat)
            )
            return src_points, tgt_points

        src_kind = src_dim.replace("n_", "")
        tgt_kind = tgt_dim.replace("n_", "")
        src_lon, src_lat = _get_lon_lat(src_grid, src_kind)
        tgt_lon, tgt_lat = _get_lon_lat(tgt_grid, tgt_kind)
        self._src_grid = yac.CloudGrid(
            self._src_grid_name, np.deg2rad(src_lon), np.deg2rad(src_lat)
        )
        self._tgt_grid = yac.CloudGrid(
            self._tgt_grid_name, np.deg2rad(tgt_lon), np.deg2rad(tgt_lat)
        )
        src_points = self._src_grid.def_points(np.deg2rad(src_lon), np.deg2rad(src_lat))
        tgt_points = self._tgt_grid.def_points(np.deg2rad(tgt_lon), np.deg2rad(tgt_lat))
        return src_points, tgt_points

    def remap(self, values: np.ndarray) -> np.ndarray:
        values = np.ascontiguousarray(values, dtype=np.float64).reshape(-1)
        if values.size != self._src_field.size:
            raise ValueError(
                f"YAC remap expects {self._src_field.size} values, got {values.size}."
            )
        self._src_field.put(values)
        out, _ = self._tgt_field.get()
        return np.asarray(out, dtype=np.float64).reshape(-1)

    def close(self) -> None:
        self._yac_inst.cleanup()


def _yac_remap(source, destination_grid, remap_to: str, yac_method: str, yac_kwargs):
    _assert_dimension(remap_to)
    destination_dim = LABEL_TO_COORD[remap_to]
    options = _normalize_yac_method(yac_method)
    options.kwargs.update(yac_kwargs or {})
    ds, is_da, name = _to_dataset(source)
    dims_to_remap = _get_remap_dims(ds)
    remappers: dict[str, _YacRemapper] = {}
    remapped_vars = {}

    for src_dim in dims_to_remap:
        remappers[src_dim] = _YacRemapper(
            ds.uxgrid,
            destination_grid,
            src_dim,
            destination_dim,
            options.method,
            options.kwargs,
        )

    try:
        for var_name, da in ds.data_vars.items():
            src_dim = next((d for d in da.dims if d in dims_to_remap), None)
            if src_dim is None:
                remapped_vars[var_name] = da
                continue

            other_dims = [d for d in da.dims if d != src_dim]
            da_t = da.transpose(*other_dims, src_dim)
            src_values = np.asarray(da_t.values)
            flat_src = src_values.reshape(-1, src_values.shape[-1])
            remapper = remappers[src_dim]
            out_flat = np.empty(
                (flat_src.shape[0], remapper._tgt_field.size), dtype=np.float64
            )
            for idx in range(flat_src.shape[0]):
                out_flat[idx] = remapper.remap(flat_src[idx])

            out_shape = src_values.shape[:-1] + (remapper._tgt_field.size,)
            out_values = out_flat.reshape(out_shape)
            coords = {dim: da.coords[dim] for dim in other_dims if dim in da.coords}
            da_out = uxarray.core.dataarray.UxDataArray(
                out_values,
                dims=other_dims + [destination_dim],
                coords=coords,
                name=da.name,
                attrs=da.attrs,
                uxgrid=destination_grid,
            )
            remapped_vars[var_name] = da_out
    finally:
        for remapper in remappers.values():
            remapper.close()

    ds_remapped = _construct_remapped_ds(
        source, remapped_vars, destination_grid, destination_dim
    )
    return ds_remapped[name] if is_da else ds_remapped
