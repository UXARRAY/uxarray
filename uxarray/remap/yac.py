from __future__ import annotations

import importlib
import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
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


def _load_yac_core_from_file() -> ModuleType | None:
    if "yac.core" in sys.modules:
        return sys.modules["yac.core"]

    for path_entry in sys.path:
        pkg_dir = Path(path_entry) / "yac"
        if not pkg_dir.is_dir():
            continue

        matches = sorted(pkg_dir.glob("core*.so"))
        if not matches:
            matches = sorted(pkg_dir.glob("core*.pyd"))
        if not matches:
            continue

        pkg = sys.modules.get("yac")
        if pkg is None:
            pkg = ModuleType("yac")
            sys.modules["yac"] = pkg
        pkg.__path__ = [str(pkg_dir)]

        spec = importlib.util.spec_from_file_location("yac.core", matches[0])
        if spec is None or spec.loader is None:
            continue

        module = importlib.util.module_from_spec(spec)
        sys.modules["yac.core"] = module
        spec.loader.exec_module(module)
        setattr(pkg, "core", module)
        return module

    return None


def _import_yac():
    module = _load_yac_core_from_file()
    if module is not None:
        return module

    try:
        return importlib.import_module("yac.core")
    except Exception as exc:  # pragma: no cover - fallback depends on local install
        raise YacNotAvailableError(
            "YAC backend requested but 'yac.core' is not available. "
            "Build YAC with Python bindings and ensure its site-packages and "
            "shared libraries are discoverable."
        ) from exc


def _normalize_yac_method(yac_method: str | None) -> _YacOptions:
    if not yac_method:
        raise ValueError(
            "backend='yac' requires yac_method to be set to 'nnn' or 'conservative'."
        )
    method = yac_method.lower()
    if method not in {"nnn", "conservative"}:
        raise ValueError(f"Unsupported YAC method: {yac_method!r}")
    return _YacOptions(method=method, kwargs={})


def _get_location(yac_core, dim: str):
    mapping = {
        "n_face": yac_core.yac_location.YAC_LOC_CELL,
        "n_node": yac_core.yac_location.YAC_LOC_CORNER,
        "n_edge": yac_core.yac_location.YAC_LOC_EDGE,
    }
    try:
        return mapping[dim]
    except KeyError as exc:
        raise ValueError(f"Unsupported remap dimension for YAC: {dim!r}") from exc


def _coerce_enum(enum_type, value: Any):
    if not isinstance(value, str):
        return value

    normalized = value.upper()
    for member in enum_type:
        if member.name == normalized or member.name.endswith(f"_{normalized}"):
            return member

    raise ValueError(f"Unsupported value {value!r} for enum {enum_type.__name__}.")


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
        yac_core = _import_yac()
        self._src_location = _get_location(yac_core, src_dim)
        self._tgt_location = _get_location(yac_core, tgt_dim)

        define_edges = "n_edge" in (src_dim, tgt_dim)
        unique = uuid4().hex
        self._src_grid = yac_core.BasicGrid.from_uxgrid(
            f"uxarray_src_{unique}",
            src_grid,
            def_edges=define_edges,
        )
        self._tgt_grid = yac_core.BasicGrid.from_uxgrid(
            f"uxarray_tgt_{unique}",
            tgt_grid,
            def_edges=define_edges,
        )

        self._src_field = yac_core.InterpField(
            self._src_grid.add_coordinates(self._src_location)
        )
        self._tgt_field = yac_core.InterpField(
            self._tgt_grid.add_coordinates(self._tgt_location)
        )

        stack = yac_core.InterpolationStack()
        if yac_method == "nnn":
            weight_type = _coerce_enum(
                yac_core.yac_interp_nnn_weight_type,
                yac_kwargs.get("reduction_type", yac_kwargs.get("nnn_type")),
            )
            if weight_type is None:
                weight_type = yac_core.yac_interp_nnn_weight_type.YAC_INTERP_NNN_AVG
            stack.add_nnn(
                nnn_type=weight_type,
                n=yac_kwargs.get("n", 1),
                max_search_distance=yac_kwargs.get("max_search_distance", 0.0),
                scale=yac_kwargs.get("scale", 1.0),
            )
        elif yac_method == "conservative":
            normalisation = _coerce_enum(
                yac_core.yac_interp_method_conserv_normalisation,
                yac_kwargs.get("normalisation"),
            )
            if normalisation is None:
                normalisation = yac_core.yac_interp_method_conserv_normalisation.YAC_INTERP_CONSERV_DESTAREA
            stack.add_conservative(
                order=yac_kwargs.get("order", 1),
                enforced_conserv=yac_kwargs.get("enforced_conserv", False),
                partial_coverage=yac_kwargs.get("partial_coverage", False),
                normalisation=normalisation,
            )

        self._weights = yac_core.compute_weights(
            stack,
            self._src_field,
            self._tgt_field,
        )
        self._interpolation = self._weights.get_interpolation(collection_size=1)
        self._src_size = self._src_grid.get_data_size(self._src_location)
        self._tgt_size = self._tgt_grid.get_data_size(self._tgt_location)

    def remap(self, values: np.ndarray) -> np.ndarray:
        """Apply the pre-computed interpolation weights to *values*.

        The interpolation method (NNN or conservative) is determined by
        *yac_method* passed to the constructor and is fixed for the lifetime of
        this remapper instance.  This method simply executes the weight
        application; it does not select or alter the interpolation algorithm.

        Parameters
        ----------
        values : np.ndarray
            1-D array of source-grid values with length equal to the number of
            source points registered with YAC (``self._src_size``).

        Returns
        -------
        np.ndarray
            1-D array of remapped values on the destination grid.
        """
        values = np.ascontiguousarray(values, dtype=np.float64).reshape(1, -1)
        if values.shape[1] != self._src_size:
            raise ValueError(
                f"YAC remap expects {self._src_size} values, got {values.shape[1]}."
            )
        out = self._interpolation(values)
        return np.asarray(out, dtype=np.float64).reshape(-1)


def _yac_remap(source, destination_grid, remap_to: str, yac_method: str, yac_kwargs):
    _assert_dimension(remap_to)
    destination_dim = LABEL_TO_COORD[remap_to]
    options = _normalize_yac_method(yac_method)
    options.kwargs.update(yac_kwargs or {})
    ds, is_da, name = _to_dataset(source)
    dims_to_remap = _get_remap_dims(ds)

    if options.method == "conservative":
        if destination_dim != "n_face":
            raise ValueError(
                "YAC conservative remapping requires the destination to be "
                "face-centered (remap_to='faces'). "
                f"Got remap_to={remap_to!r} which maps to dimension {destination_dim!r}."
            )
        non_face_src = dims_to_remap - {"n_face"}
        if non_face_src:
            raise ValueError(
                "YAC conservative remapping requires all source data to be "
                f"face-centered (dimension 'n_face'). "
                f"Found non-face source dimension(s): {non_face_src}. "
                "Use yac_method='nnn' for node- or edge-centered data."
            )
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
        out_flat = np.empty((flat_src.shape[0], remapper._tgt_size), dtype=np.float64)
        for idx in range(flat_src.shape[0]):
            out_flat[idx] = remapper.remap(flat_src[idx])

        out_shape = src_values.shape[:-1] + (remapper._tgt_size,)
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

    ds_remapped = _construct_remapped_ds(
        source, remapped_vars, destination_grid, destination_dim
    )
    return ds_remapped[name] if is_da else ds_remapped
