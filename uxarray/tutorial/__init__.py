from __future__ import annotations

import os
import urllib.request
import warnings
from pathlib import Path

from uxarray.tutorial.registry import DATASETS, Component, TutorialDataset

_BASE_DATA_URL = "https://raw.githubusercontent.com/UXARRAY/uxarray/main/test/meshfiles"
_CACHE_DIR = Path(
    os.environ.get(
        "UXARRAY_TUTORIAL_CACHE_DIR",
        Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
        / "uxarray"
        / "tutorial",
    )
)


def available_datasets() -> tuple[str, ...]:
    """Return the names of available tutorial datasets."""
    return tuple(sorted(DATASETS))


def describe_dataset(name: str) -> str:
    """Return a short description for a tutorial dataset."""
    return _get_dataset_entry(name).description


def file_path(name: str, component: Component | None = None) -> Path:
    """Return the local path for a tutorial dataset file."""
    dataset = _get_dataset_entry(name)

    if component is None:
        components = _available_components(dataset)
        if len(components) != 1:
            raise ValueError(
                f"Dataset {name!r} has multiple files. Please specify one of: "
                f"{', '.join(components)}."
            )
        component = components[0]

    path_parts = _get_component_path(dataset, name, component)

    local_root = _local_meshfiles_path()
    if local_root is not None:
        path = local_root.joinpath(*path_parts)
        if path.exists():
            return path

    return _ensure_cached(path_parts)


def file_paths(name: str) -> tuple[Path, ...]:
    """Return the local paths for a multi-file tutorial dataset."""
    dataset = _get_dataset_entry(name)

    if dataset.data_files is None:
        raise ValueError(
            f"Dataset {name!r} does not define multiple data files. "
            "Use uxarray.tutorial.file_path(...) instead."
        )

    local_root = _local_meshfiles_path()
    if local_root is not None:
        paths = tuple(local_root.joinpath(*parts) for parts in dataset.data_files)
        if all(path.exists() for path in paths):
            return paths

    return tuple(_ensure_cached(parts) for parts in dataset.data_files)


def open_grid(name: str, **kwargs):
    """Open the grid file for a tutorial dataset."""
    from uxarray import open_grid as ux_open_grid

    return ux_open_grid(file_path(name, component="grid"), **kwargs)


def open_dataset(name: str, **kwargs):
    """Open a tutorial dataset with its associated grid."""
    dataset = _get_dataset_entry(name)

    if dataset.data_files is not None:
        raise ValueError(
            f"Dataset {name!r} defines multiple data files. "
            "Use uxarray.tutorial.open_mfdataset(...) instead."
        )

    if dataset.data is None:
        raise ValueError(
            f"Dataset {name!r} does not define a data file. "
            "Use uxarray.tutorial.open_grid(...) instead."
        )

    from uxarray import open_dataset as ux_open_dataset

    return ux_open_dataset(
        file_path(name, component="grid"),
        file_path(name, component="data"),
        **kwargs,
    )


def open_mfdataset(name: str, **kwargs):
    """Open a multi-file tutorial dataset with its associated grid."""
    dataset = _get_dataset_entry(name)

    if dataset.data_files is None:
        raise ValueError(
            f"Dataset {name!r} does not define multiple data files. "
            "Use uxarray.tutorial.open_dataset(...) instead."
        )

    from uxarray import open_mfdataset as ux_open_mfdataset

    return ux_open_mfdataset(
        file_path(name, component="grid"),
        file_paths(name),
        **kwargs,
    )


def _get_dataset_entry(name: str) -> TutorialDataset:
    try:
        dataset = DATASETS[name]
    except KeyError as err:
        available = ", ".join(available_datasets())
        raise ValueError(
            f"Unknown tutorial dataset {name!r}. Available datasets: {available}."
        ) from err

    _validate_dataset_entry(name, dataset)
    return dataset


def _local_meshfiles_path() -> Path | None:
    """Return the local ``test/meshfiles`` directory if it is available."""
    candidates = []

    data_dir = os.environ.get("UXARRAY_DATA_DIR")
    if data_dir:
        candidates.append(Path(data_dir).expanduser().resolve())

    candidates.append(Path(__file__).resolve().parents[2] / "test" / "meshfiles")

    for path in candidates:
        if path.exists():
            return path

    return None


def _ensure_cached(path_parts: tuple[str, ...]) -> Path:
    """Ensure a tutorial dataset file is available locally and return its path."""
    destination = _CACHE_DIR.joinpath(*path_parts)

    if destination.exists():
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    url = "/".join([_BASE_DATA_URL, *path_parts])
    label = "/".join(path_parts)

    warnings.warn(
        f"Downloading tutorial dataset file {label!r} to {destination}.",
        stacklevel=2,
    )
    tmp_path = destination.with_suffix(destination.suffix + ".tmp")

    try:
        urllib.request.urlretrieve(url, str(tmp_path))
        tmp_path.replace(destination)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise

    return destination


def _available_components(dataset: TutorialDataset) -> tuple[Component, ...]:
    if dataset.data is None:
        return ("grid",)
    return ("grid", "data")


def _get_component_path(
    dataset: TutorialDataset,
    name: str,
    component: Component,
) -> tuple[str, ...]:
    if component == "grid":
        return dataset.grid

    if component == "data":
        if dataset.data is None:
            raise ValueError(f"Dataset {name!r} does not define a data file.")
        return dataset.data

    raise ValueError("Invalid component. Expected one of: grid, data.")


def _validate_dataset_entry(name: str, dataset: TutorialDataset) -> None:
    if dataset.data is not None and dataset.data_files is not None:
        raise ValueError(
            f"Tutorial dataset {name!r} cannot define both 'data' and 'data_files'."
        )
