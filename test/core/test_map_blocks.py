import dask.array as da
import numpy as np
import numpy.testing as nt
import pytest
import xarray as xr

import uxarray as ux
from uxarray.errors import DimensionError


@pytest.fixture
def uxds(gridpath, datasetpath):
    # multi_dim_data has dims (time=8, lev=8, n_face=4)
    return ux.open_dataset(
        gridpath("ugrid", "quad-hexagon", "grid.nc"),
        datasetpath("ugrid", "quad-hexagon", "multi_dim_data.nc"),
    )


def _times_face_areas(block):
    return block * block.uxgrid.face_areas


def test_blocks_split_off_grid_get_whole_grid(uxds):
    """Chunked only along time, every block carries the original grid object."""
    uxda = uxds["multi_dim_data"]
    seen = []

    def func(block):
        if block.size:  # skip the zero-size call xarray uses to infer the output
            seen.append((type(block), block.uxgrid))
        return _times_face_areas(block)

    result = uxda.chunk(time=2).map_blocks(func)

    assert isinstance(result, ux.UxDataArray)
    assert result.uxgrid is uxda.uxgrid
    assert isinstance(result.data, da.Array)
    nt.assert_allclose(result.values, (uxda * uxda.uxgrid.face_areas).values)
    assert len(seen) == 4
    assert all(cls is ux.UxDataArray and grid is uxda.uxgrid for cls, grid in seen)


def test_blocks_split_along_n_face_get_matching_subgrid(uxds):
    """Chunked along n_face, each block carries the sub-grid of exactly its faces."""
    uxda = uxds["multi_dim_data"]
    face_lon = uxda.uxgrid.face_lon.values

    def stamp_face_lon(block):
        assert block.uxgrid.n_face == block.sizes["n_face"]
        return block.copy(data=np.broadcast_to(block.uxgrid.face_lon.values, block.shape).copy())

    for n_face in (1, 3):
        result = ux.map_blocks(stamp_face_lon, uxda.chunk(n_face=n_face))
        nt.assert_array_equal(result.isel(time=0, lev=0).values, face_lon)

    result = ux.map_blocks(_times_face_areas, uxda.chunk(time=4, n_face=2))
    assert result.uxgrid is uxda.uxgrid
    nt.assert_allclose(result.values, (uxda * uxda.uxgrid.face_areas).values)
    # nothing used to locate each block leaks into the result
    assert set(result.coords) == set(uxda.coords)


def test_dataset_map_blocks(uxds):
    result = uxds.chunk(n_face=2).map_blocks(_times_face_areas)

    assert isinstance(result, ux.UxDataset)
    assert result.uxgrid is uxds.uxgrid
    nt.assert_allclose(
        result["multi_dim_data"].values,
        (uxds["multi_dim_data"] * uxds.uxgrid.face_areas).values,
    )


def test_array_args_share_block_grid(uxds):
    """DataArray args are split alongside obj and carry the same sub-grid as its block."""
    uxda = uxds["multi_dim_data"].chunk(n_face=2)

    def add(block, other):
        assert isinstance(other, ux.UxDataArray)
        assert other.uxgrid is block.uxgrid
        return block + other

    result = ux.map_blocks(add, uxda, args=[uxda * 2])
    nt.assert_allclose(result.values, 3 * uxds["multi_dim_data"].values)


def test_template(uxds):
    uxda = uxds["multi_dim_data"].chunk(n_face=2)

    result = ux.map_blocks(_times_face_areas, uxda, template=uxda)

    assert isinstance(result, ux.UxDataArray)
    nt.assert_allclose(result.values, (uxda * uxda.uxgrid.face_areas).values)


def test_not_dask_calls_func_directly(uxds):
    uxda = uxds["multi_dim_data"]
    result = uxda.map_blocks(lambda block: block + 1)
    assert isinstance(result.data, np.ndarray)
    nt.assert_allclose(result.values, uxda.values + 1)


def test_rejects_plain_xarray():
    with pytest.raises(TypeError, match="UxDataArray or UxDataset"):
        ux.map_blocks(lambda block: block, xr.DataArray(da.ones(3)))


@pytest.fixture
def mpas_grid(gridpath):
    return ux.open_grid(gridpath("mpas", "QU", "oQU480.231010.nc"))


def _random_on(uxgrid, grid_dim, n_time=3):
    rng = np.random.default_rng(0)
    data = rng.random((n_time, uxgrid.sizes[grid_dim]))
    return ux.UxDataArray(xr.DataArray(data, dims=["time", grid_dim]), uxgrid=uxgrid)


def test_node_split_coriolis(mpas_grid):
    """Split along n_node, blocks keep the whole grid and get node_lat for their own nodes."""
    vorticity = _random_on(mpas_grid, "n_node")
    omega = 7.292e-5
    seen = []

    def absolute_vorticity(block, node_lat):
        if block.size:  # skip the zero-size call xarray uses to infer the output
            seen.append((block.uxgrid, block.sizes["n_node"], node_lat.sizes["n_node"]))
        return block + 2 * omega * np.sin(np.deg2rad(node_lat))

    result = ux.map_blocks(
        absolute_vorticity, vorticity.chunk(n_node=1000), args=[mpas_grid.node_lat]
    )

    assert isinstance(result, ux.UxDataArray)
    assert result.uxgrid is mpas_grid
    expected = vorticity + 2 * omega * np.sin(np.deg2rad(mpas_grid.node_lat))
    nt.assert_allclose(result.values, expected.values)
    assert len(seen) == 4
    assert all(grid is mpas_grid and n == n_lat for grid, n, n_lat in seen)


def test_edge_split_transport(mpas_grid):
    """Split along n_edge, blocks get the edge lengths for their own edges."""
    normal_velocity = _random_on(mpas_grid, "n_edge")
    edge_length = mpas_grid.edge_node_distances

    result = normal_velocity.chunk(n_edge=2000, time=1).map_blocks(
        lambda block, length: block * length, args=[edge_length]
    )

    assert result.uxgrid is mpas_grid
    nt.assert_allclose(result.values, (normal_velocity * edge_length).values)


def test_face_split_gathers_node_args(mpas_grid):
    """With n_face split, node data in args is gathered onto each block's sub-grid,
    so a node-to-face aggregation per block matches the eager result."""
    node_data = _random_on(mpas_grid, "n_node")
    face_template = _random_on(mpas_grid, "n_face").chunk(n_face=500, time=1)
    seen = []

    def node_to_face(block, nodes):
        if block.size:  # skip the zero-size call xarray uses to infer the output
            seen.append((nodes.uxgrid is block.uxgrid, nodes.sizes["n_node"], block.uxgrid.n_node))
        return nodes.topological_mean(destination="face")

    # args must match obj's chunks along shared non-grid dims, as in xarray.map_blocks
    result = ux.map_blocks(node_to_face, face_template, args=[node_data.chunk(time=1)])

    expected = node_data.topological_mean(destination="face")
    nt.assert_allclose(result.values, expected.values)
    assert len(seen) == 12
    assert all(same_grid and n == n_grid for same_grid, n, n_grid in seen)


def test_face_split_gathers_chunked_node_args(mpas_grid):
    """A node arg chunked along n_node is still gathered whole onto each block's sub-grid."""
    node_data = _random_on(mpas_grid, "n_node")
    face_template = _random_on(mpas_grid, "n_face").chunk(n_face=500)

    result = ux.map_blocks(
        lambda block, nodes: nodes.topological_mean(destination="face"),
        face_template,
        args=[node_data.chunk(n_node=1000)],
    )

    expected = node_data.topological_mean(destination="face")
    nt.assert_allclose(result.values, expected.values)


def _edge_to_face_mean(edges, uxgrid):
    """Mean of each face's edge values, read through the grid's face_edge_connectivity."""
    conn = uxgrid.face_edge_connectivity.values
    valid = conn != ux.INT_FILL_VALUE
    values = edges.values[..., np.where(valid, conn, 0)]
    return (values * valid).sum(axis=-1) / valid.sum(axis=-1)


def test_face_split_gathers_edge_args(gridpath):
    """With n_face split, edge data in args is gathered onto each block's sub-grid,
    even when the grid had not built face_edge_connectivity yet."""
    uxgrid = ux.open_grid(gridpath("mpas", "QU", "oQU480.231010.nc"))
    edge_data = _random_on(uxgrid, "n_edge")
    face_template = _random_on(uxgrid, "n_face").chunk(n_face=500)

    def edge_to_face(block, edges):
        return block.copy(data=_edge_to_face_mean(edges, block.uxgrid))

    result = ux.map_blocks(edge_to_face, face_template, args=[edge_data])

    nt.assert_allclose(result.values, _edge_to_face_mean(edge_data, uxgrid))


def test_face_split_rejects_args_on_face_and_node(mpas_grid):
    face_data = _random_on(mpas_grid, "n_face")
    both = ux.UxDataset({"face": face_data, "node": _random_on(mpas_grid, "n_node")}, uxgrid=mpas_grid)

    with pytest.raises(DimensionError, match="not both"):
        ux.map_blocks(lambda block, other: block, face_data.chunk(n_face=500), args=[both])


def test_args_rechunked_to_match(mpas_grid):
    """Grid variables chunked differently from obj are rechunked to match it."""
    face_data = _random_on(mpas_grid, "n_face").chunk(n_face=500)
    areas = mpas_grid.face_areas.chunk(n_face=7)

    result = ux.map_blocks(lambda block, area: block * area, face_data, args=[areas])

    nt.assert_allclose(result.values, (face_data * mpas_grid.face_areas).values)


def test_rejects_mixed_grid_dims(mpas_grid):
    face_data = _random_on(mpas_grid, "n_face")
    node_data = _random_on(mpas_grid, "n_node")
    edge_data = _random_on(mpas_grid, "n_edge")

    # n_face split while node data is also present: the blocks cover unrelated parts of the grid
    mixed = ux.UxDataset({"face": face_data, "node": node_data}, uxgrid=mpas_grid)
    with pytest.raises(DimensionError, match="do not line up"):
        ux.map_blocks(lambda block: block, mixed.chunk(n_face=500))

    # two grid dimensions split at once
    mixed = ux.UxDataset({"node": node_data, "edge": edge_data}, uxgrid=mpas_grid)
    with pytest.raises(DimensionError, match="do not line up"):
        ux.map_blocks(lambda block: block, mixed.chunk(n_node=1000, n_edge=2000))

    # mixed grid dims split only off-grid are fine, and get the whole grid
    result = mixed.chunk(time=1).map_blocks(lambda block: block + block.uxgrid.node_lon)
    nt.assert_allclose(result["node"].values, (node_data + mpas_grid.node_lon).values)
