from __future__ import annotations

from pathlib import Path

from uxarray.tutorial.registry import DATASETS, Component, TutorialDataset


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
    path = _meshfiles_path().joinpath(*path_parts)

    if not path.exists():
        raise FileNotFoundError(
            f"Tutorial dataset file for {name!r} was not found at {path}."
        )

    return path


def file_paths(name: str) -> tuple[Path, ...]:
    """Return the local paths for a multi-file tutorial dataset."""
    dataset = _get_dataset_entry(name)

    if dataset.data_files is None:
        raise ValueError(
            f"Dataset {name!r} does not define multiple data files. "
            "Use uxarray.tutorial.file_path(...) instead."
        )

    meshfiles_path = _meshfiles_path()
    paths = tuple(meshfiles_path.joinpath(*parts) for parts in dataset.data_files)

    for path in paths:
        if not path.exists():
            raise FileNotFoundError(
                f"Tutorial dataset file for {name!r} was not found at {path}."
            )

    return paths


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


def _meshfiles_path() -> Path:
    """Support both local development from a cloned
    repo and cwd fallback supports docs builds that run from the root repo"""
    candidates = (
        Path(__file__).resolve().parents[2] / "test" / "meshfiles",
        Path.cwd() / "test" / "meshfiles",
    )

    for path in candidates:
        if path.exists():
            return path

    checked = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        f"Could not locate the tutorial dataset directory. Checked: {checked}"
    )


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
