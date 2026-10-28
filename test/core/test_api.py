from contextlib import ExitStack
from functools import partial

import dask.array as da
import numpy.testing as nt
import uxarray as ux
import numpy as np
import pytest
import tempfile
import xarray as xr
from pathlib import Path
from unittest.mock import patch
from uxarray.core.utils import _open_dataset_with_fallback
import os

TEST_MESHFILES = Path(__file__).resolve().parent.parent / "meshfiles"

def test_open_geoflow_dataset(gridpath, datasetpath):
    """Loads a single dataset with its grid topology file using uxarray's
    open_dataset call."""

    # Paths to Data Variable files
    data_paths = [
        datasetpath("ugrid", "geoflow-small", "v1.nc"),
        datasetpath("ugrid", "geoflow-small", "v2.nc"),
        datasetpath("ugrid", "geoflow-small", "v3.nc")
    ]

    uxds_v1 = ux.open_dataset(gridpath("ugrid", "geoflow-small", "grid.nc"), data_paths[0])

    # Ideally uxds_v1.uxgrid should NOT be None
    nt.assert_equal(uxds_v1.uxgrid is not None, True)

def test_open_dataset(gridpath, datasetpath, mesh_constants):
    """Loads a single dataset with its grid topology file using uxarray's
    open_dataset call."""

    grid_path = gridpath("ugrid", "outCSne30", "outCSne30.ug")
    data_path = datasetpath("ugrid", "outCSne30", "outCSne30_var2.nc")
    uxds_var2_ne30 = ux.open_dataset(grid_path, data_path)

    nt.assert_equal(uxds_var2_ne30.uxgrid.node_lon.size, mesh_constants['NNODES_outCSne30'])
    nt.assert_equal(len(uxds_var2_ne30.uxgrid._ds.data_vars), mesh_constants['DATAVARS_outCSne30'])
    nt.assert_equal(uxds_var2_ne30.source_datasets, str(data_path))


def test_open_dataset_single_combined_mpas_file(gridpath):
    """Loads a combined MPAS grid-and-data file with a single argument."""

    # Use a known combined grid-and-data MPAS file.
    file_path = gridpath("mpas", "QU", "oQU480.231010.nc")

    uxds_single = ux.open_dataset(file_path)
    uxds_pair = ux.open_dataset(file_path, file_path)

    # Ensure that the single-argument path actually loads data variables
    assert len(uxds_single.data_vars) > 0
    nt.assert_equal(uxds_single.uxgrid.source_grid_spec, "MPAS")
    nt.assert_equal(uxds_single.source_datasets, str(file_path))
    nt.assert_equal(uxds_single.sizes["n_face"], uxds_pair.sizes["n_face"])
    nt.assert_equal(set(uxds_single.data_vars), set(uxds_pair.data_vars))
    assert "ssh" in uxds_single.data_vars


def test_open_dataset_single_combined_xarray_dataset(gridpath):
    """Loads a combined MPAS grid-and-data xarray.Dataset with a single argument."""

    file_path = gridpath("mpas", "QU", "oQU480.231010.nc")

    with xr.open_dataset(file_path) as ds:
        uxds = ux.open_dataset(ds)

    nt.assert_equal(uxds.uxgrid.source_grid_spec, "MPAS")
    nt.assert_equal(uxds.source_datasets, None)
    assert "ssh" in uxds.data_vars


def test_open_dataset_single_argument_rejects_directory_grid(tmp_path):
    """Requires a separate data file for directory-based grids."""

    with pytest.raises(
        ValueError, match="single directory is not supported"
    ):
        ux.open_dataset(tmp_path)


def test_open_dataset_single_argument_rejects_invalid_combined_file(datasetpath):
    """Rejects one-file inputs that do not contain recognizable grid metadata."""

    data_path = datasetpath("ugrid", "outCSne30", "outCSne30_var2.nc")

    with pytest.raises(ux.errors.GridInvalidError, match="Failed to parse uxgrid information from xarray.Dataset."):
        ux.open_dataset(data_path)


@pytest.fixture
def reuse_owners(tmp_path, monkeypatch):
    """Close original backend owners even if Grid/UxDataset loses callbacks."""
    with ExitStack() as owners:

        def tracked_open(opener, *args, **kwargs):
            """Track the actual backend owner, not just its UxDataset wrapper."""
            source = opener(*args, **kwargs)
            owners.callback(source.close)
            return source

        for name in ("open_dataset", "open_mfdataset"):
            monkeypatch.setattr(xr, name, partial(tracked_open, getattr(xr, name)))
        yield owners


@pytest.fixture
def reuse_grid(reuse_owners):
    """Use the suite's small unstructured quad-hexagon grid for ordinary reuse."""
    grid = ux.open_grid(TEST_MESHFILES / "ugrid" / "quad-hexagon" / "grid.nc")
    grid.get_ball_tree(coordinates="nodes")
    return grid


@pytest.fixture
def reuse_open(tmp_path, reuse_owners):
    """Exercise Dataset/file/multifile inputs with identical preservation checks."""
    counter = 0

    def open_members(grid, members, kind, **kwargs):
        """Open members and verify preservation before returning expected data."""
        nonlocal counter
        snapshots = [member.copy(deep=True) for member in members]
        arrays = [{name: member[name].data for name in member} for member in members]
        grid_snapshot, mapping = grid._ds.copy(deep=True), grid._source_dims_dict.copy()
        grid_dataset = grid._ds
        # Preserve storage, not materialized .data arrays from lazy backends.
        grid_arrays = {name: grid._ds[name].variable._data for name in grid._ds}
        tree = grid._ball_tree
        chunks = kwargs.get("chunks")
        saved_chunks = chunks.copy() if isinstance(chunks, dict) else chunks
        grid_kwargs = kwargs.get("grid_kwargs")
        saved_grid_kwargs = grid_kwargs.copy() if grid_kwargs is not None else None
        paths = []
        if kind != "dataset":
            for member in members:
                path = tmp_path / f"member_{counter}.nc"
                counter += 1
                member.to_netcdf(path)
                paths.append(path)
        # Chunk ownership for True is a separate open review question.
        if chunks is not None:
            kwargs.setdefault("chunk_grid", False)
        with patch("uxarray.core.api.open_grid") as construct:
            try:
                if kind == "multiple":
                    results = [
                        ux.open_mfdataset(
                            grid, paths, combine="nested", concat_dim="member", **kwargs
                        )
                    ]
                    expected = [xr.concat(members, dim="member")]
                    sources = [str(paths)]
                elif kind == "dataset":
                    with patch("xarray.open_dataset") as backend:
                        results = [
                            ux.open_dataset(grid, member, **kwargs)
                            for member in members
                        ]
                    backend.assert_not_called()
                    expected, sources = members, [None] * len(members)
                else:
                    results = [ux.open_dataset(grid, path, **kwargs) for path in paths]
                    expected, sources = members, list(map(str, paths))
                for result, source in zip(results, sources):
                    reuse_owners.callback(result.close)
                    assert result.uxgrid is grid
                    assert result.source_datasets == source
                return list(zip(results, expected))
            finally:
                construct.assert_not_called()
                assert grid._ds is grid_dataset
                assert grid._source_dims_dict == mapping
                assert grid._ball_tree is tree
                xr.testing.assert_identical(grid._ds, grid_snapshot)
                for name, data in grid_arrays.items():
                    assert grid._ds[name].variable._data is data
                assert chunks == saved_chunks
                assert grid_kwargs == saved_grid_kwargs
                for member, snapshot, data in zip(members, snapshots, arrays):
                    xr.testing.assert_identical(member, snapshot)
                    for name in member:
                        assert member[name].chunks == snapshot[name].chunks
                        assert member[name].data is data[name]

    return open_members


def _reuse_members(face_dim="n_face"):
    """Generate two deterministic members, independently of grid fixture data."""
    return [
        xr.Dataset(
            {"temperature": (("time", face_dim), np.arange(12).reshape(3, 4) + i)}
        )
        for i in range(2)
    ]


def _equal_size_grid(source_dims):
    """Equal node/face counts make size inference and alias collisions observable."""
    return ux.Grid.from_topology(
        node_lon=np.array([0.0, 120.0, -120.0, 0.0]),
        node_lat=np.array([0.0, 0.0, 0.0, 90.0]),
        face_node_connectivity=np.array([[0, 1, 2], [0, 3, 1], [1, 3, 2], [2, 3, 0]]),
        dims_dict=source_dims.copy(),
    )


@pytest.mark.parametrize("kind", ["dataset", "file", "multiple"])
@pytest.mark.parametrize("chunks", [None, -1, "auto", {}, {"n_face": 2, "time": 1}])
def test_open_dataset_reuse_members(reuse_grid, reuse_open, kind, chunks):
    """Reuse topology/cached tree without grid I/O, preserving member values."""
    for result, expected in reuse_open(
        reuse_grid, _reuse_members(), kind, chunks=chunks, grid_kwargs={}
    ):
        assert result.temperature.dims == expected.temperature.dims
        nt.assert_array_equal(result.temperature, expected.temperature)
        assert isinstance(result.temperature.data, da.Array) == (
            kind == "multiple" or chunks is not None
        )
        if chunks == {"n_face": 2, "time": 1}:
            assert result.temperature.chunks[-2:] == ((1, 1, 1), (2, 2))


@pytest.mark.parametrize("kind", ["dataset", "file", "multiple"])
@pytest.mark.parametrize(
    "chunks",
    [
        {"n_face": 2, "n_node": 1, "time": 1},
        {"cells": 2, "vertices": 1, "time": 1},
        {"cells": 4, "vertices": 4, "n_face": 2, "n_node": 1, "time": 1},
    ],
)
def test_open_dataset_reuse_source_chunks(reuse_open, kind, chunks):
    """Translate source/canonical chunks without mutating caller options/data."""
    grid = _equal_size_grid({"vertices": "n_node", "cells": "n_face"})
    members = [
        ds.assign(velocity=("vertices", np.arange(4) + i))
        for i, ds in enumerate(_reuse_members("cells"))
    ]
    for result, expected in reuse_open(grid, members, kind, chunks=chunks):
        assert (
            result.temperature.dims == expected.temperature.rename(cells="n_face").dims
        )
        assert result.velocity.dims == expected.velocity.rename(vertices="n_node").dims
        assert result.temperature.chunks[-2:] == ((1, 1, 1), (2, 2))
        assert result.velocity.chunks[-1] == (1, 1, 1, 1)
        nt.assert_array_equal(result.temperature, expected.temperature)
        nt.assert_array_equal(result.velocity, expected.velocity)


@pytest.mark.parametrize("kind", ["dataset", "file", "multiple"])
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("face_chunks", [False, True])
def test_open_dataset_reuse_overlapping_aliases(reuse_open, kind, reverse, face_chunks):
    """Read requested values, not newly translated keys, in either mapping order."""
    items = [("n_face", "n_node"), ("cells", "n_face")]
    grid = _equal_size_grid(dict(items[::-1] if reverse else items))
    chunks = {"n_node": 1, "time": 1}
    if face_chunks:
        chunks["n_face"] = 2
    members = [
        ds.assign(velocity=("n_face", np.arange(4) + i))
        for i, ds in enumerate(_reuse_members("cells"))
    ]
    for result, expected in reuse_open(grid, members, kind, chunks=chunks):
        prefix = ("member",) if kind == "multiple" else ()
        assert result.temperature.dims == prefix + ("time", "n_face")
        assert result.velocity.dims == prefix + ("n_node",)
        assert result.temperature.chunks[-2:] == (
            (1, 1, 1),
            (2, 2) if face_chunks else (4,),
        )
        assert result.velocity.chunks[-1] == (1, 1, 1, 1)
        nt.assert_array_equal(result.temperature, expected.temperature)
        nt.assert_array_equal(result.velocity, expected.velocity)


@pytest.mark.parametrize("kind", ["dataset", "file", "multiple"])
@pytest.mark.parametrize("source_dims", [{}, {"vertices": "n_node", "cells": "n_face"}])
def test_open_dataset_reuse_canonical_equal_counts(reuse_open, kind, source_dims):
    """Canonical dimensions remain distinct, including topology without a map."""
    grid = _equal_size_grid(source_dims)
    members = [
        ds.assign(velocity=("n_node", np.arange(4) + i))
        for i, ds in enumerate(_reuse_members())
    ]
    for result, expected in reuse_open(
        grid, members, kind, chunks={"n_face": 2, "n_node": 1}
    ):
        assert result.temperature.dims == expected.temperature.dims
        assert result.velocity.dims == expected.velocity.dims
        assert result.temperature.chunks[-1] == (2, 2)
        assert result.velocity.chunks[-1] == (1, 1, 1, 1)
        nt.assert_array_equal(result.temperature, expected.temperature)
        nt.assert_array_equal(result.velocity, expected.velocity)
    if source_dims:
        result, _ = reuse_open(grid, _reuse_members("cells")[:1], "dataset")[0]
        assert result.temperature.dims == ("time", "n_face")


def test_open_dataset_reuse_face_node_sequence(reuse_open):
    """Face-only members must not discard mappings needed by later node data."""
    grid = _equal_size_grid({"vertices": "n_node", "cells": "n_face"})
    members = [
        _reuse_members("cells")[0],
        xr.Dataset({"velocity": ("vertices", np.arange(4))}),
        _reuse_members("cells")[1],
    ]
    for result, expected in reuse_open(grid, members, "dataset"):
        xr.testing.assert_equal(
            result.to_xarray(),
            expected.rename(
                {
                    dim: grid._source_dims_dict[dim]
                    for dim in expected.dims
                    if dim in grid._source_dims_dict
                }
            ),
        )


@pytest.mark.parametrize("prechunked", [False, True])
def test_open_dataset_reuse_input_dataset_chunks(reuse_grid, reuse_open, prechunked):
    """Rechunk the returned Dataset without replacing the caller's arrays."""
    members = _reuse_members()
    if prechunked:
        members = [member.chunk({"n_face": 4, "time": 3}) for member in members]
    for result, _ in reuse_open(
        reuse_grid, members, "dataset", chunks={"n_face": 2, "time": 1}
    ):
        assert result.temperature.chunks == ((1, 1, 1), (2, 2))


@pytest.mark.parametrize("kind", ["dataset", "file", "multiple"])
def test_open_dataset_reuse_prechunked_grid(reuse_grid, reuse_open, kind):
    """With chunk_grid=False, preserve an already-chunked Grid and its caches."""
    reuse_grid.chunk(n_node=2, n_face=2)
    for result, _ in reuse_open(
        reuse_grid, _reuse_members(), kind, chunks={"n_face": 1}, chunk_grid=False
    ):
        assert result.temperature.chunks[-1] == (1, 1, 1, 1)
    assert reuse_grid.face_node_connectivity.chunks[0] == (2, 2)


@pytest.mark.parametrize("kind", ["dataset", "file", "multiple"])
@pytest.mark.parametrize(
    "options, message",
    [
        ({"use_dual": True}, "use_dual"),
        ({"grid_kwargs": {"drop_variables": "face_nodes"}}, "grid_kwargs"),
    ],
)
def test_open_dataset_reuse_rejects_construction_options(
    reuse_grid, reuse_open, kind, options, message
):
    """Reject options that only apply when constructing a Grid."""
    with pytest.raises(ValueError, match=message):
        reuse_open(reuse_grid, _reuse_members(), kind, **options)


def test_open_dataset_reuse_requires_data(reuse_grid):
    """An existing Grid alone is not a member Dataset."""
    with pytest.raises(TypeError, match="filename_or_obj"):
        ux.open_dataset(reuse_grid)


@pytest.mark.parametrize("kind", ["file", "dataset"])
def test_open_dataset_reuse_existing_construction(
    reuse_grid, reuse_owners, tmp_path, kind
):
    """Existing grid-file and grid-Dataset inputs still construct equivalent grids."""
    path = TEST_MESHFILES / "ugrid" / "quad-hexagon" / "grid.nc"
    grid_source = path if kind == "file" else xr.open_dataset(path)
    member = _reuse_members()[0]
    data_path = tmp_path / "construction.nc"
    member.to_netcdf(data_path)
    result = ux.open_dataset(grid_source, data_path, chunks={"n_face": 2})
    reuse_owners.callback(result.close)
    assert result.uxgrid is not reuse_grid
    assert result.uxgrid == reuse_grid
    assert result.temperature.dims == ("time", "n_face")
    assert result.temperature.chunks[-1] == (2, 2)
    nt.assert_array_equal(result.temperature, member.temperature)
    xr.testing.assert_identical(result.uxgrid._ds.compute(), reuse_grid._ds)


def test_open_mf_dataset(gridpath, datasetpath, mesh_constants):
    """Loads multiple datasets with their grid topology file using
    uxarray's open_dataset call."""

    grid_path = gridpath("ugrid", "outCSne30", "outCSne30.ug")
    dsfiles_mf_ne30 = datasetpath(
        "ugrid",
        "outCSne30",
        ["outCSne30_var2.nc", "outCSne30_vortex.nc"],
    )
    uxds_mf_ne30 = ux.open_mfdataset(grid_path, dsfiles_mf_ne30)

    nt.assert_equal(uxds_mf_ne30.uxgrid.node_lon.size, mesh_constants['NNODES_outCSne30'])
    nt.assert_equal(len(uxds_mf_ne30.uxgrid._ds.data_vars), mesh_constants['DATAVARS_outCSne30'])
    nt.assert_equal(uxds_mf_ne30.source_datasets, str(dsfiles_mf_ne30))

def test_open_grid(gridpath, mesh_constants):
    """Loads only a grid topology file using uxarray's open_grid call."""
    uxgrid = ux.open_grid(gridpath("ugrid", "geoflow-small", "grid.nc"))

    nt.assert_almost_equal(uxgrid.calculate_total_face_area(), mesh_constants['MESH30_AREA'], decimal=3)

def test_copy_dataset(gridpath, datasetpath):
    """Loads a single dataset with its grid topology file using uxarray's
    open_dataset call and make a copy of the object."""

    uxds_var2_ne30 = ux.open_dataset(
        gridpath("ugrid", "outCSne30", "outCSne30.ug"),
        datasetpath("ugrid", "outCSne30", "outCSne30_var2.nc")
    )

    # make a shallow and deep copy of the dataset object
    uxds_var2_ne30_copy_deep = uxds_var2_ne30.copy(deep=True)
    uxds_var2_ne30_copy = uxds_var2_ne30.copy(deep=False)

    # Ideally uxds_var2_ne30_copy.uxgrid should NOT be None
    nt.assert_equal(uxds_var2_ne30_copy.uxgrid is not None, True)

    # Check that the copy is a shallow copy
    assert uxds_var2_ne30_copy.uxgrid is uxds_var2_ne30.uxgrid
    assert uxds_var2_ne30_copy.uxgrid == uxds_var2_ne30.uxgrid

    # Check that the deep copy is a deep copy
    assert uxds_var2_ne30_copy_deep.uxgrid == uxds_var2_ne30.uxgrid
    assert uxds_var2_ne30_copy_deep.uxgrid is not uxds_var2_ne30.uxgrid

def test_copy_dataarray(gridpath, datasetpath):
    """Loads an unstructured grid and data using uxarray's open_dataset
    call and make a copy of the dataarray object."""

    # Paths to Data Variable files
    data_paths = [
        datasetpath("ugrid", "geoflow-small", "v1.nc"),
        datasetpath("ugrid", "geoflow-small", "v2.nc"),
        datasetpath("ugrid", "geoflow-small", "v3.nc")
    ]

    uxds_v1 = ux.open_dataset(gridpath("ugrid", "geoflow-small", "grid.nc"), data_paths[0])

    # get the uxdataarray object
    v1_uxdata_array = uxds_v1['v1']

    # make a shallow and deep copy of the dataarray object
    v1_uxdata_array_copy_deep = v1_uxdata_array.copy(deep=True)
    v1_uxdata_array_copy = v1_uxdata_array.copy(deep=False)

    # Check that the copy is a shallow copy
    assert v1_uxdata_array_copy.uxgrid is v1_uxdata_array.uxgrid
    assert v1_uxdata_array_copy.uxgrid == v1_uxdata_array.uxgrid

    # Check that the deep copy is a deep copy
    assert v1_uxdata_array_copy_deep.uxgrid == v1_uxdata_array.uxgrid
    assert v1_uxdata_array_copy_deep.uxgrid is not v1_uxdata_array.uxgrid

def test_open_dataset_grid_kwargs(gridpath, datasetpath):
    """Drops ``Mesh2_face_nodes`` from the inputted grid file using
    ``grid_kwargs``"""

    with pytest.raises(ValueError):
        # attempt to open a dataset after dropping face nodes should raise a KeyError
        uxds = ux.open_dataset(
            gridpath("ugrid", "outCSne30", "outCSne30.ug"),
            datasetpath("ugrid", "outCSne30", "outCSne30_var2.nc"),
            grid_kwargs={"drop_variables": "Mesh2_face_nodes"}
                )


def test_open_dataset_with_fallback():
    """Test that the fallback mechanism works when the default engine fails."""

    tmp_path = ""
    ds = None
    ds_fallback = None
    try:
        # Create a simple test dataset
        with tempfile.NamedTemporaryFile(suffix='.nc', delete=False) as tmp:
            data = xr.Dataset({'temp': (['x', 'y'], np.random.rand(5, 5))})
            data.to_netcdf(tmp.name)
            tmp_path = tmp.name

        # Test normal case
        ds = _open_dataset_with_fallback(tmp_path)
        assert isinstance(ds, xr.Dataset)
        assert 'temp' in ds.data_vars

        # Test fallback mechanism with mocked failure
        original_open = xr.open_dataset
        call_count = 0
        def mock_open_dataset(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1 and 'engine' not in kwargs:
                raise Exception("Simulated engine failure")
            return original_open(*args, **kwargs)

        with patch('uxarray.core.utils.xr.open_dataset', side_effect=mock_open_dataset):
            ds_fallback = _open_dataset_with_fallback(tmp_path)
            assert isinstance(ds_fallback, xr.Dataset)
            assert call_count == 2  # First failed, second succeeded

    finally:
        if ds is not None:
            ds.close()
        if ds_fallback is not None:
            ds_fallback.close()
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def test_open_dataset_with_fallback_chains_both_engine_errors(tmp_path):
    """When both engines fail, the fallback error must be chained onto the
    default engine's error rather than replacing it."""

    not_netcdf = tmp_path / "not_netcdf.nc"
    not_netcdf.write_text("this is not a netcdf file")

    with pytest.raises(Exception) as excinfo:
        _open_dataset_with_fallback(str(not_netcdf))

    assert excinfo.value.__cause__ is not None, (
        "the default engine's error was discarded"
    )


def test_list_grid_names_multigrid(gridpath):
    """List grids from an OASIS-style multi-grid file."""
    grid_file = gridpath("scrip", "oasis", "grids.nc")
    grid_names = ux.list_grid_names(grid_file)

    assert isinstance(grid_names, list)
    assert set(grid_names) == {"ocn", "atm"}


def test_list_grid_names_single_scrip():
    """List grids from a standard single-grid SCRIP file."""
    grid_path = TEST_MESHFILES / "scrip" / "outCSne8" / "outCSne8.nc"
    grid_names = ux.list_grid_names(grid_path)

    assert isinstance(grid_names, list)
    assert grid_names == ["grid"]


def test_open_multigrid_all_grids(gridpath):
    """Open all grids from a multi-grid file."""
    grid_file = gridpath("scrip", "oasis", "grids.nc")
    grids = ux.open_multigrid(grid_file)

    assert isinstance(grids, dict)
    assert set(grids.keys()) == {"ocn", "atm"}
    assert grids["ocn"].n_face == 12
    assert grids["atm"].n_face == 20


def test_open_multigrid_specific_grids(gridpath):
    """Open a subset of grids from a multi-grid file."""
    grid_file = gridpath("scrip", "oasis", "grids.nc")
    grids = ux.open_multigrid(grid_file, gridnames=["ocn"])

    assert set(grids.keys()) == {"ocn"}
    assert grids["ocn"].n_face == 12


def test_open_multigrid_with_masks(gridpath):
    """Open grids with a companion mask file."""
    grid_file = gridpath("scrip", "oasis", "grids.nc")
    mask_file = gridpath("scrip", "oasis", "masks.nc")

    grids = ux.open_multigrid(grid_file, mask_filename=mask_file)

    assert grids["ocn"].n_face == 8
    assert grids["atm"].n_face == 20


def test_open_multigrid_mask_zero_faces(gridpath):
    """Applying masks that deactivate an entire grid should not fail."""
    grid_file = gridpath("scrip", "oasis", "grids.nc")
    mask_file = gridpath("scrip", "oasis", "masks_no_atm.nc")

    grids = ux.open_multigrid(grid_file, mask_filename=mask_file)

    assert grids["ocn"].n_face == 8
    assert grids["atm"].n_face == 0


def test_open_multigrid_missing_grid_error(gridpath):
    """Requesting a missing grid should raise."""
    grid_file = gridpath("scrip", "oasis", "grids.nc")

    with pytest.raises(ValueError, match="grid 'land' not found in the provided file"):
        ux.open_multigrid(grid_file, gridnames=["land"])


def test_concat_various_inputs():
    """Ensure concat() requires uxarray objs and raises clear TypeError otherwise,
    and that concat() works in basic cases (exactly two UxDataArrays or two UxDatasets)
    Includes regression test for first and third bugs mentioned in #1642.
    """
    # ensure reasonable crash when providing no inputs
    with pytest.raises(ValueError, match="requires at least one object"):
        ux.concat((), dim='anything')
    # ensure crash when providing no uxarray objects
    with pytest.raises(TypeError, match="expected either all UxDataArray or all UxDataset"):
        ux.concat((7, "not a uxarray object"), dim='anything')
    # ensure crash when providing some non-uxarray objects
    uxds0 = ux.tutorial.open_dataset('quad-hexagon')
    uxds1 = uxds0 + 10
    with pytest.raises(TypeError, match="expected either all UxDataArray or all UxDataset"):
        ux.concat((uxds0, "not a uxarray object", uxds1), dim='new_dim')
    # ensure crash when providing a mix of UxDataArray and UxDataset objects
    uxarr0 = uxds0['t2m']
    with pytest.raises(TypeError, match="expected either all UxDataArray or all UxDataset"):
        ux.concat((uxds0, uxds1, uxarr0), dim='new_dim')
    with pytest.raises(TypeError, match="expected either all UxDataArray or all UxDataset"):
        ux.concat((uxarr0, uxds0), dim='new_dim')
    # ensure concat works in basic cases: two UxDatasets or two UxDataArrays
    ux.concat((uxds0, uxds1), 'new_dim')  # also ensures can provide dim as positional arg.
    uxarr1 = uxds1['t2m']
    assert uxarr0.uxgrid == uxarr1.uxgrid
    result = ux.concat((uxarr0, uxarr1), 'new_dim')
    # ^^ also serves as regression test for third bug in #1642,
    # i.e.: ensures can actually concat UxDataArray objects.
    # include a few quick tests that the result looks reasonable:
    assert isinstance(result, ux.UxDataArray)
    assert result.uxgrid == uxarr0.uxgrid
    assert result.sizes == {**uxarr0.sizes, 'new_dim': 2}

    # regression test for first bug in #1642: using non-uxarray objects with equal uxgrid
    #    should raise error message mentioning uxarray objects, not xarray objects.
    class Foo():
        def __init__(self, uxgrid):
            self.uxgrid = uxgrid
    foo0 = Foo(7)
    foo1 = Foo(7)  # also 7 because: want to use the same uxgrid value in both cases.
    with pytest.raises(TypeError, match="expected either all UxDataArray or all UxDataset"):
        ux.concat([foo0, foo1], dim='anything')


def test_concat_checks_uxgrid():
    """Ensure concat() requires all objects' uxgrids to be equal,
    but not necessarily the same exact object.
    Includes regression test for second bug mentioned in #1642.
    """
    arrA = ux.tutorial.open_dataset("outCSne30-vortex")['psi']
    arrB = arrA.copy()
    assert arrA.uxgrid == arrB.uxgrid
    assert arrA.uxgrid is not arrB.uxgrid
    ux.concat([arrA, arrB], 'new_dim')

    arrC = ux.tutorial.open_dataset("quad-hexagon")['t2m']
    assert arrA.uxgrid != arrC.uxgrid
    with pytest.raises(ux.errors.GridsMismatchError, match=r"got objs\[1\]\.uxgrid != objs\[0\]\.uxgrid"):
        ux.concat([arrA, arrC], dim='new_dim')
    with pytest.raises(ux.errors.GridsMismatchError, match=r"got objs\[2\]\.uxgrid != objs\[0\]\.uxgrid"):
        ux.concat([arrA, arrB, arrC], dim='new_dim')
