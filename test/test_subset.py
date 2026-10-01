import dask.array as da
import numpy as np
import uxarray as ux
import uxarray.grid.slice as slice_module
from uxarray.grid.slice import _remap_dense, _remap_kernel, _remap_searchsorted

import pytest


def test_repr(gridpath, datasetpath):
    uxds = ux.open_dataset(gridpath("ugrid", "quad-hexagon", "grid.nc"), datasetpath("ugrid", "quad-hexagon", "data.nc"))

    # grid repr
    grid_repr = uxds.uxgrid.subset.__repr__()
    assert "bounding_box" in grid_repr
    assert "bounding_circle" in grid_repr
    assert "nearest_neighbor" in grid_repr

    # data array repr
    da_repr = uxds['t2m'].subset.__repr__()
    assert "bounding_box" in da_repr
    assert "bounding_circle" in da_repr
    assert "nearest_neighbor" in da_repr


def test_grid_face_isel(gridpath):
    GRID_PATHS = [
        gridpath("mpas", "QU", "oQU480.231010.nc"),
        gridpath("ugrid", "geoflow-small", "grid.nc"),
        gridpath("ugrid", "outCSne30", "outCSne30.ug")
    ]
    for grid_path in GRID_PATHS:
        grid = ux.open_grid(grid_path)

        grid_contains_edge_node_conn = "edge_node_connectivity" in grid._ds

        face_indices = [0, 1, 2, 3, 4]
        for n_max_faces in range(1, len(face_indices)):
            grid_subset = grid.isel(n_face=face_indices[:n_max_faces])
            assert grid_subset.n_face == n_max_faces
            if not grid_contains_edge_node_conn:
                assert "edge_node_connectivity" not in grid_subset._ds

        face_indices = [0, 1, 2, grid.n_face]
        with pytest.raises(IndexError):
            grid_subset = grid.isel(n_face=face_indices)
            if not grid_contains_edge_node_conn:
                assert "edge_node_connectivity" not in grid_subset._ds


def test_grid_node_isel(gridpath):
    GRID_PATHS = [
        gridpath("mpas", "QU", "oQU480.231010.nc"),
        gridpath("ugrid", "geoflow-small", "grid.nc"),
        gridpath("ugrid", "outCSne30", "outCSne30.ug")
    ]
    for grid_path in GRID_PATHS:
        grid = ux.open_grid(grid_path)

        node_indices = [0, 1, 2, 3, 4]
        for n_max_nodes in range(1, len(node_indices)):
            grid_subset = grid.isel(n_node=node_indices[:n_max_nodes])
            assert grid_subset.n_node >= n_max_nodes

        face_indices = [0, 1, 2, grid.n_node]
        with pytest.raises(IndexError):
            grid_subset = grid.isel(n_face=face_indices)


def test_grid_nn_subset(gridpath):
    GRID_PATHS = [
        gridpath("mpas", "QU", "oQU480.231010.nc"),
        gridpath("ugrid", "geoflow-small", "grid.nc"),
        gridpath("ugrid", "outCSne30", "outCSne30.ug")
    ]
    coord_locs = [[0, 0], [-180, 0], [180, 0], [0, 90], [0, -90]]

    for grid_path in GRID_PATHS:
        grid = ux.open_grid(grid_path)

        # corner-nodes
        ks = [1, 2, grid.n_node - 1]
        for coord in coord_locs:
            for k in ks:
                grid_subset = grid.subset.nearest_neighbor(coord,
                                                           k,
                                                           element="nodes")
                assert grid_subset.n_node >= k

        # face-centers
        ks = [1, 2, grid.n_face - 1]
        for coord in coord_locs:
            for k in ks:
                grid_subset = grid.subset.nearest_neighbor(
                    coord, k, "face centers")

                assert grid_subset.n_face == k
                assert isinstance(grid_subset, ux.Grid)


def test_grid_bounding_circle_subset(gridpath):
    GRID_PATHS = [
        gridpath("mpas", "QU", "oQU480.231010.nc"),
        gridpath("ugrid", "geoflow-small", "grid.nc"),
        gridpath("ugrid", "outCSne30", "outCSne30.ug")
    ]
    center_locs = [[0, 0], [-180, 0], [180, 0], [0, 90], [0, -90]]
    coord_locs = center_locs  # Use the same locations
    rs = [45, 90, 180]  # Define radii

    for grid_path in GRID_PATHS:
        grid = ux.open_grid(grid_path)
        for element in ["nodes", "face centers"]:
            for coord in coord_locs:
                for r in rs:
                    grid_subset = grid.subset.bounding_circle(coord, r, element)

                    assert isinstance(grid_subset, ux.Grid)


def test_grid_bounding_box_subset(gridpath):
    GRID_PATHS = [
        gridpath("mpas", "QU", "oQU480.231010.nc"),
        gridpath("ugrid", "geoflow-small", "grid.nc"),
        gridpath("ugrid", "outCSne30", "outCSne30.ug")
    ]
    bbox = [(-10, 10), (-10, 10)]
    bbox_antimeridian = [(-170, 170), (-45, 45)]

    for element in ["nodes", "face centers"]:
        for grid_path in GRID_PATHS:
            grid = ux.open_grid(grid_path)

            grid_subset = grid.subset.bounding_box(bbox[0],
                                                   bbox[1],)

            grid_subset_antimeridian = grid.subset.bounding_box(
                bbox_antimeridian[0], bbox_antimeridian[1])


def test_uxda_isel(gridpath, datasetpath):
    uxds = ux.open_dataset(gridpath("mpas", "QU", "oQU480.231010.nc"), gridpath("mpas", "QU", "oQU480.231010.nc"))

    sub = uxds['bottomDepth'].isel(n_face=[1, 2, 3])

    assert len(sub) == 3


def test_uxda_isel_with_coords(gridpath, datasetpath):
    uxds = ux.open_dataset(gridpath("mpas", "QU", "oQU480.231010.nc"), gridpath("mpas", "QU", "oQU480.231010.nc"))
    uxds = uxds.assign_coords({"lon_face": uxds.uxgrid.face_lon})
    sub = uxds['bottomDepth'].isel(n_face=[1, 2, 3])

    assert "lon_face" in sub.coords
    assert len(sub.coords['lon_face']) == 3


def test_inverse_indices(gridpath):
    grid = ux.open_grid(gridpath("mpas", "QU", "oQU480.231010.nc"))

    # Test nearest neighbor subsetting
    coord = [0, 0]
    subset = grid.subset.nearest_neighbor(coord, k=1, element="face centers", inverse_indices=True)

    assert subset.inverse_indices is not None

    # Test bounding box subsetting
    box = [(-10, 10), (-10, 10)]
    subset = grid.subset.bounding_box(box[0], box[1], inverse_indices=True)

    assert subset.inverse_indices is not None

    # Test bounding circle subsetting
    center_coord = [0, 0]
    subset = grid.subset.bounding_circle(center_coord, r=10, element="face centers", inverse_indices=True)

    assert subset.inverse_indices is not None

    # Ensure code raises exceptions when the element is edges or nodes or inverse_indices is incorrect
    assert pytest.raises(Exception, grid.subset.bounding_circle, center_coord, r=10, element="edge centers", inverse_indices=True)
    assert pytest.raises(Exception, grid.subset.bounding_circle, center_coord, r=10, element="nodes", inverse_indices=True)
    assert pytest.raises(ValueError, grid.subset.bounding_circle, center_coord, r=10, element="face center", inverse_indices=(['not right'], True))

    # Test isel directly
    subset = grid.isel(n_face=[1], inverse_indices=True)
    assert subset.inverse_indices.face.values == 1


def test_da_subset(gridpath, datasetpath):
    uxds = ux.open_dataset(gridpath("ugrid", "quad-hexagon", "grid.nc"), datasetpath("ugrid", "quad-hexagon", "data.nc"))

    res1 = uxds['t2m'].subset.bounding_box(lon_bounds=(-10, 10), lat_bounds=(-10, 10))
    res2 = uxds['t2m'].subset.bounding_circle(center_coord=(0,0), r=10)
    res3 = uxds['t2m'].subset.nearest_neighbor(center_coord=(0, 0), k=4)

    assert len(res1) == len(res2) == len(res3) == 4


def test_empty_subset(gridpath, datasetpath):
    """ensure that subsetting methods still return a valid UXarray object even if subset is empty.
    (The resulting object should have size=0.)
    This test ensures issue #1285 has been fixed.
    """
    uxds = ux.open_dataset(gridpath("ugrid", "quad-hexagon", "grid.nc"), datasetpath("ugrid", "quad-hexagon", "data.nc"))
    arr = uxds['t2m']
    min_lon = arr.uxgrid.face_lon.min().item()
    definitely_out_of_bounds = (min_lon - 10, min_lon - 5)
    # mostly just want to ensure this doesn't crash:
    res = arr.subset.bounding_box(lon_bounds=definitely_out_of_bounds, lat_bounds=(-10, 10))
    assert isinstance(res, ux.UxDataArray)
    assert res.size == 0
    # should still have all the same dim names even if resulting subset is empty:
    assert set(res.dims) == set(arr.dims)


def test_remap_kernel_selection():
    """The dense lookup table is only built when it is cheap in absolute terms,
    or small relative to the slice itself."""

    tiny_selection = np.arange(1000, dtype=ux.constants.INT_DTYPE)

    # small grid, dense is affordable regardless of how little is selected
    assert _remap_kernel(tiny_selection, 10_000)[0] is _remap_dense

    # large grid, tiny slice: the lookup must stay proportional to the slice
    func, kwargs = _remap_kernel(tiny_selection, 100_000_000)
    assert func is _remap_searchsorted
    assert kwargs["orig_indices"].size == tiny_selection.size

    # large grid, most of it selected: dense is back within budget
    big_selection = np.arange(50_000_000, dtype=ux.constants.INT_DTYPE)
    assert _remap_kernel(big_selection, 100_000_000)[0] is _remap_dense


def test_remap_kernels_agree():
    """Both remapping kernels must produce identical results, including for
    fill values and for indices that fall outside of the slice."""

    fill = ux.constants.INT_FILL_VALUE
    dtype = ux.constants.INT_DTYPE

    n_node = 20
    selected = np.array([2, 3, 7, 11, 19], dtype=dtype)
    conn = np.array(
        [
            [2, 3, 7, fill],  # all within the slice
            [11, 19, 2, 3],
            [0, 5, 7, 18],  # 0, 5 and 18 are not part of the slice
            [fill, fill, fill, fill],
        ],
        dtype=dtype,
    )

    dense = np.full(n_node, fill, dtype=dtype)
    dense[selected] = np.arange(selected.size, dtype=dtype)

    expected = np.array(
        [
            [0, 1, 2, fill],
            [3, 4, 0, 1],
            [fill, fill, 2, fill],
            [fill, fill, fill, fill],
        ],
        dtype=dtype,
    )

    for result in (_remap_dense(conn, dense), _remap_searchsorted(conn, selected)):
        assert result.dtype == dtype
        np.testing.assert_array_equal(result, expected)

    # an empty slice maps everything to the fill value
    empty = np.array([], dtype=dtype)
    np.testing.assert_array_equal(
        _remap_searchsorted(conn, empty), np.full(conn.shape, fill, dtype=dtype)
    )


def test_isel_dask_connectivity(gridpath, monkeypatch):
    """Chunked connectivity must stay lazy through a slice and still match the
    eager result.

    The remapping kernels run per block, so a bug that only shows up on a partial
    chunk is invisible when the connectivity is a single eager array. Both kernels
    are exercised, since only one of them is selected for any given grid.
    """
    grid_path = gridpath("mpas", "QU", "oQU480.231010.nc")

    def open_with_edges(**kwargs):
        grid = ux.open_grid(grid_path, **kwargs)
        # build the edge connectivity so the edge remap is exercised as well
        _ = grid.face_edge_connectivity
        return grid

    eager_grid = open_with_edges()
    face_indices = np.arange(0, eager_grid.n_face, 2)
    eager_subset = eager_grid.isel(n_face=face_indices)

    for force_sparse in (False, True):
        grid = open_with_edges(chunks=-1)

        # split each connectivity into several blocks; `chunks=-1` alone is
        # dask-backed but single-block, which would not cover the blockwise path
        for name in list(grid._ds.data_vars):
            if "_connectivity" in name:
                dim = grid._ds[name].dims[0]
                grid._ds[name] = grid._ds[name].chunk(
                    {dim: max(grid._ds.sizes[dim] // 5, 1)}
                )

        if force_sparse:
            monkeypatch.setattr(slice_module, "_DENSE_REMAP_MAX_SIZE", 0)
            monkeypatch.setattr(slice_module, "_DENSE_REMAP_MAX_RATIO", 0)
        subset = grid.isel(n_face=face_indices)
        monkeypatch.undo()

        remapped = {
            name: var
            for name, var in subset._ds.data_vars.items()
            if "_connectivity" in name
        }
        assert remapped
        assert set(remapped) == {
            name for name in eager_subset._ds.data_vars if "_connectivity" in name
        }

        for name, var in remapped.items():
            assert isinstance(var.data, da.Array), f"{name} was computed eagerly"
            assert len(var.data.chunks[0]) > 1, f"{name} did not span multiple blocks"
            np.testing.assert_array_equal(
                var.values,
                eager_subset._ds[name].values,
                err_msg=f"force_sparse={force_sparse}: {name}",
            )


class TestBoundingBoxSubset:
    """``subset.bounding_box`` computes bounds for candidate faces only (#1778).

    Its result must equal the whole-mesh path: ``get_faces_between_longitudes``
    intersected with ``get_faces_between_latitudes`` on ``Grid.bounds``.
    """

    # A face crossing the antimeridian, one on each side of it, a face with a
    # pole inside it, and a face whose nodes sit exactly on round coordinates.
    VERTICES = [
        [[170, 10], [-170, 10], [-170, 20], [170, 20]],
        [[172, -5], [179, -5], [179, 5], [172, 5]],
        [[-179, -5], [-172, -5], [-172, 5], [-179, 5]],
        [[5, 80], [95, 80], [-175, 80], [-85, 80]],
        [[-30, -20], [30, -20], [30, 20], [-30, 20]],
    ]
    CROSSING, EAST, WEST, POLE, ROUND = range(5)

    GRIDS = {
        "healpix-z3": lambda gridpath: ux.Grid.from_healpix(3),
        "healpix-z4": lambda gridpath: ux.Grid.from_healpix(4),
        "mpas": lambda gridpath: ux.open_grid(gridpath("mpas", "QU", "oQU480.231010.nc")),
        "geoflow": lambda gridpath: ux.open_grid(gridpath("ugrid", "geoflow-small", "grid.nc")),
        "outCSne30": lambda gridpath: ux.open_grid(gridpath("ugrid", "outCSne30", "outCSne30.ug")),
        "scrip": lambda gridpath: ux.open_grid(gridpath("scrip", "outCSne8", "outCSne8.nc")),
        "vertices": lambda gridpath: ux.Grid.from_face_vertices(
            TestBoundingBoxSubset.VERTICES, latlon=True
        ),
    }

    BOXES = [
        ([-30, 30], [-20, 20]),
        ([-1, 1], [-1, 1]),
        ([-180, 180], [-90, 90]),
        ([0, 10], [80, 90]),
        ([-180, 180], [80, 90]),
        ([0, 10], [-90, -80]),
        ([170, -170], [-10, 10]),
        ([170, -170], [-30, 30]),
        ([170, 180], [-10, 10]),
        ([-180, -170], [-10, 10]),
        ([-106.6, -93.5], [25.8, 36.5]),
        ([100, 140], [0, 45]),
    ]

    @staticmethod
    def whole_mesh(grid, lon_bounds, lat_bounds):
        return np.intersect1d(
            grid.get_faces_between_longitudes(lon_bounds),
            grid.get_faces_between_latitudes(lat_bounds),
        )

    @staticmethod
    def screened(grid, lon_bounds, lat_bounds):
        from uxarray.subset.grid_accessor import _faces_in_bounding_box

        assert "bounds" not in grid._ds
        faces = _faces_in_bounding_box(grid, lon_bounds, lat_bounds)
        assert "bounds" not in grid._ds, "whole-mesh bounds were computed"
        return np.sort(faces)

    @pytest.mark.parametrize("lon_bounds,lat_bounds", BOXES)
    @pytest.mark.parametrize("name", list(GRIDS))
    def test_matches_whole_mesh_path(self, gridpath, name, lon_bounds, lat_bounds):
        make = self.GRIDS[name]
        expected = self.whole_mesh(make(gridpath), lon_bounds, lat_bounds)
        actual = self.screened(make(gridpath), lon_bounds, lat_bounds)
        np.testing.assert_array_equal(actual, np.sort(expected))

    def test_accessor_matches_whole_mesh_path(self, gridpath):
        grid = self.GRIDS["mpas"](gridpath)
        expected = self.whole_mesh(self.GRIDS["mpas"](gridpath), [-30, 30], [-20, 20])
        assert expected.size > 0

        sub = grid.subset.bounding_box([-30, 30], [-20, 20], inverse_indices=True)

        assert "bounds" not in grid._ds
        np.testing.assert_array_equal(
            np.sort(sub.inverse_indices.face.values), np.sort(expected)
        )

    def test_bounds_are_computed_for_candidates_only(self, gridpath, monkeypatch):
        import uxarray.grid.bounds as bounds_module
        from uxarray.constants import INT_FILL_VALUE

        grid = self.GRIDS["geoflow"](gridpath)
        lon_bounds, lat_bounds = [-30, 30], [-20, 20]

        conn = grid.face_node_connectivity.values
        valid = conn != INT_FILL_VALUE
        lon = np.where(valid, grid.node_lon.values[np.where(valid, conn, 0)], 0)
        lat = np.where(valid, grid.node_lat.values[np.where(valid, conn, 0)], 0)
        nodes_inside = (
            (lon >= lon_bounds[0]) & (lon <= lon_bounds[1])
            & (lat >= lat_bounds[0]) & (lat <= lat_bounds[1])
        ) | ~valid
        n_candidates = int(np.all(nodes_inside, axis=1).sum())
        assert 0 < n_candidates < grid.n_face

        sizes = []
        kernel = bounds_module._construct_face_bounds_array

        def counting_kernel(face_node_connectivity, *args):
            sizes.append(face_node_connectivity.shape[0])
            return kernel(face_node_connectivity, *args)

        monkeypatch.setattr(bounds_module, "_construct_face_bounds_array", counting_kernel)

        sub = grid.subset.bounding_box(lon_bounds, lat_bounds)

        assert sizes == [n_candidates]
        assert 0 < sub.n_face <= n_candidates
        assert "bounds" not in grid._ds

    def test_cached_bounds_are_used(self, gridpath, monkeypatch):
        import uxarray.grid.bounds as bounds_module
        from uxarray.subset.grid_accessor import _faces_in_bounding_box

        grid = self.GRIDS["mpas"](gridpath)
        grid.bounds
        monkeypatch.setattr(
            bounds_module,
            "_faces_with_nodes_within_box",
            lambda *a: pytest.fail("cached bounds were ignored"),
        )

        actual = _faces_in_bounding_box(grid, [-30, 30], [-20, 20])
        np.testing.assert_array_equal(
            np.sort(actual), np.sort(self.whole_mesh(grid, [-30, 30], [-20, 20]))
        )

    def test_partially_contained_faces_are_excluded(self, gridpath):
        reference = self.GRIDS["geoflow"](gridpath)
        bounds_lat = reference.face_bounds_lat.values
        bounds_lon = reference.face_bounds_lon.values

        # The box's top edge runs through the middle of face ``f``.
        lon_bounds = [-60, 60]
        inside_lon = (bounds_lon[:, 0] >= lon_bounds[0]) & (bounds_lon[:, 1] <= lon_bounds[1])
        f = int(np.argmin(np.where(inside_lon, np.abs(bounds_lat[:, 0] - 10), np.inf)))
        lat_hi = bounds_lat[f].mean()
        lat_bounds = [-60, lat_hi]

        faces = self.screened(self.GRIDS["geoflow"](gridpath), lon_bounds, lat_bounds)

        assert f not in faces
        straddling = np.flatnonzero((bounds_lat[:, 0] < lat_hi) & (bounds_lat[:, 1] > lat_hi))
        assert straddling.size > 0
        assert np.intersect1d(faces, straddling).size == 0
        assert np.all(bounds_lat[faces, 0] >= lat_bounds[0])
        assert np.all(bounds_lat[faces, 1] <= lat_hi)
        assert np.all(bounds_lon[faces, 0] >= lon_bounds[0])
        assert np.all(bounds_lon[faces, 1] <= lon_bounds[1])

        # Moving the edge to the face's own top bound brings exactly it in.
        lat_bounds = [-60, bounds_lat[f, 1]]
        with_face = self.screened(self.GRIDS["geoflow"](gridpath), lon_bounds, lat_bounds)
        assert f in with_face
        np.testing.assert_array_equal(
            with_face, np.sort(self.whole_mesh(reference, lon_bounds, lat_bounds))
        )

    def test_nodes_on_the_box_edge(self, gridpath):
        make = self.GRIDS["vertices"]
        reference = make(gridpath)
        lat_lo, lat_hi = reference.face_bounds_lat.values[self.ROUND]

        # The nodes sit on the box's meridians; the arcs between them bulge
        # past +-20 latitude, so the face fits its own bounds and not [-20, 20].
        assert lat_hi > 20
        exact_fit = self.screened(make(gridpath), [-30, 30], [lat_lo, lat_hi])
        assert self.ROUND in exact_fit
        np.testing.assert_array_equal(
            exact_fit, np.sort(self.whole_mesh(reference, [-30, 30], [lat_lo, lat_hi]))
        )
        assert self.ROUND not in self.screened(make(gridpath), [-30, 30], [-20, 20])

    def test_antimeridian(self, gridpath):
        make = self.GRIDS["vertices"]
        crossing = self.screened(make(gridpath), [170, -170], [-30, 30])
        east = self.screened(make(gridpath), [170, 180], [-30, 30])
        west = self.screened(make(gridpath), [-180, -170], [-30, 30])

        assert set(crossing) == {self.CROSSING, self.EAST, self.WEST}
        assert set(east) == {self.EAST}
        assert set(west) == {self.WEST}

    def test_pole_face(self, gridpath):
        make = self.GRIDS["vertices"]
        assert self.POLE not in self.screened(make(gridpath), [0, 10], [70, 90])
        assert self.POLE not in self.screened(make(gridpath), [170, -170], [70, 90])
        assert self.POLE in self.screened(make(gridpath), [-180, 180], [70, 90])

    def test_region_outside_a_partial_grid_is_empty(self, gridpath, datasetpath):
        grid = ux.open_grid(gridpath("ugrid", "quad-hexagon", "grid.nc"))
        assert grid.face_lon.values.max() < 90
        assert self.screened(grid, [100, 110], [-10, 10]).size == 0

        sub = grid.subset.bounding_box([100, 110], [-10, 10])
        assert sub.n_face == 0
