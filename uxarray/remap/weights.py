from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import xarray as xr
from numpy.random.mtrand import f

from uxarray.core.utils import _open_dataset_with_fallback

if TYPE_CHECKING:
    from scipy import sparse

# LRU-bounded cache for loaded remap operators.
_WEIGHTS_CACHE_MAXSIZE = 32
_WEIGHTS_CACHE: "OrderedDict[tuple[str, int, int], RemapWeights]" = OrderedDict()


def clear_remap_weights_cache() -> None:
    """Drop all cached remap weight operators."""
    _WEIGHTS_CACHE.clear()


def _first_present(mapping, names: tuple[str, ...], kind: str):
    for name in names:
        if name in mapping:
            return mapping[name]
    raise ValueError(f"Could not find {kind}. Expected one of: {', '.join(names)}.")


def _normalize_indices(indices, size: int, label: str) -> np.ndarray:
    """Convert SCRIP/ESMF-style index arrays to 0-based ``np.int64``.

    Prefers an explicit ``start_index`` attribute (SCRIP/ESMF convention).
    Without one, treats indices as 1-based only when ``max == size`` (the
    only unambiguous evidence); otherwise treats them as 0-based when in
    range, and raises on anything else.
    """
    start_index = None
    if hasattr(indices, "attrs"):
        start_index = indices.attrs.get("start_index")

    arr = np.asarray(indices, dtype=np.int64).ravel()
    if arr.size == 0:
        return arr

    arr_min = int(arr.min())
    arr_max = int(arr.max())

    if start_index is None:
        if arr_max == size:
            start_index = 1
        elif arr_min >= 0 and arr_max < size:
            start_index = 0
        else:
            raise ValueError(
                f"{label} indices are out of bounds for size {size}. "
                f"Found min={arr_min}, max={arr_max}."
            )
    else:
        start_index = int(start_index)

    arr = arr - start_index
    if arr.min() < 0 or arr.max() >= size:
        raise ValueError(
            f"{label} indices are out of bounds for size {size} "
            f"with start_index={start_index}. "
            f"Found min={int(arr.min())}, max={int(arr.max())}."
        )

    return arr


@dataclass(frozen=True)
class RemapWeights:
    """Reusable sparse remapping operator loaded from a standard weight file."""

    matrix: sparse.csr_matrix
    source_size: int
    destination_size: int
    path: str | None = None

    @classmethod
    def from_file(cls, filename_or_obj: str | PathLike[str] | xr.Dataset):
        """Load a standard sparse remap-weight file into memory once."""
        from scipy import sparse

        if isinstance(filename_or_obj, xr.Dataset):
            ds = filename_or_obj
            close_ds = False
            path = None
        else:
            ds = _open_dataset_with_fallback(filename_or_obj)
            close_ds = True
            path = str(filename_or_obj)

        try:
            source_size = int(
                _first_present(ds.sizes, ("n_a", "src_grid_size"), "source dimension")
            )
            destination_size = int(
                _first_present(
                    ds.sizes, ("n_b", "dst_grid_size"), "destination dimension"
                )
            )

            row = _normalize_indices(
                _first_present(ds.variables, ("row", "dst_address"), "row indices"),
                destination_size,
                "Row",
            )
            col = _normalize_indices(
                _first_present(ds.variables, ("col", "src_address"), "column indices"),
                source_size,
                "Column",
            )
            values = np.asarray(
                _first_present(ds.variables, ("S", "weights"), "weight values"),
                dtype=np.float64,
            ).ravel()

            if not (row.size == col.size == values.size):
                raise ValueError(
                    "Remap weights require row, col, and weight arrays of equal length."
                )

            matrix = sparse.coo_matrix(
                (values, (row, col)),
                shape=(destination_size, source_size),
            ).tocsr()
        finally:
            if close_ds:
                ds.close()

        return cls(
            matrix=matrix,
            source_size=source_size,
            destination_size=destination_size,
            path=path,
        )

    def _apply(self, values: np.ndarray) -> np.ndarray:
        """Apply the sparse remap operator along the trailing dimension.

        Internal kernel used by :func:`uxarray.remap.apply_weights._apply_weights`.
        The public entrypoint for applying weights is the ``.remap.apply_weights``
        accessor method.
        """
        values = np.asarray(values)

        if values.ndim == 0:
            raise ValueError("Remap weights require at least a 1-D input array.")

        if values.shape[-1] != self.source_size:
            raise ValueError(
                f"Expected trailing dimension of size {self.source_size}, "
                f"got {values.shape[-1]}."
            )

        flat_values = values.reshape(-1, values.shape[-1])
        remapped = (self.matrix @ flat_values.T).T
        return remapped.reshape(values.shape[:-1] + (self.destination_size,))


def _cache_key(filename_or_obj: str | PathLike[str]) -> tuple[str, int, int]:
    path = Path(filename_or_obj).expanduser().resolve()
    stat = path.stat()
    return str(path), stat.st_mtime_ns, stat.st_size


def load_remap_weights(
    filename_or_obj: str | PathLike[str] | xr.Dataset | RemapWeights,
) -> RemapWeights:
    """Load or normalize reusable remap weights.

    Path-based inputs are cached by resolved path, mtime, and file size so
    repeated loads avoid rebuilding the sparse matrix. The cache is
    LRU-bounded; call :func:`clear_remap_weights_cache` to drop all entries.
    """
    if isinstance(filename_or_obj, RemapWeights):
        return filename_or_obj

    if isinstance(filename_or_obj, xr.Dataset):
        return RemapWeights.from_file(filename_or_obj)

    cache_key = _cache_key(filename_or_obj)
    weights = _WEIGHTS_CACHE.get(cache_key)
    if weights is None:
        weights = RemapWeights.from_file(filename_or_obj)
        _WEIGHTS_CACHE[cache_key] = weights
        while len(_WEIGHTS_CACHE) > _WEIGHTS_CACHE_MAXSIZE:
            _WEIGHTS_CACHE.popitem(last=False)
    else:
        _WEIGHTS_CACHE.move_to_end(cache_key)

    return weights
