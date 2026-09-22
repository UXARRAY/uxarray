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


class TestBoundingBoxPrefilter:
    """The centre pre-filter must be an optimization, not a behaviour change.

    ``bounding_box`` used to compute ``Grid.bounds`` -- a per-face box for
    every face in the mesh -- before it could test which faces fall inside
    the region asked for. That cost is proportional to the mesh rather than
    the crop, so a few-hundred-face region out of a 300M-face grid needed
    more memory than the machine had (#1778). Face centres are already
    resident and are used first to discard faces that cannot reach the box;
    exact bounds are then computed for the survivors only.

    A faster subset that quietly drops faces near the edge of the box would
    be worse than a slow one, so what is pinned here is that the answer does
    not move.
    """

    @staticmethod
    def _exact(grid, lon_bounds, lat_bounds):
        """The pre-filter-free path, kept here as the reference answer."""
        return np.intersect1d(
            grid.get_faces_between_longitudes(lon_bounds),
            grid.get_faces_between_latitudes(lat_bounds),
        )

    # Boxes chosen for the cases a centre-based filter could plausibly get
    # wrong: tiny (most faces excluded), whole-globe (none excluded), polar
    # (faces whose bounds span far more longitude than their centre suggests)
    # and antimeridian-spanning (where the interval is a union, not a range).
    BOXES = [
        ([-30, 30], [-20, 20]),
        ([-1, 1], [-1, 1]),
        ([-180, 180], [-90, 90]),
        ([0, 10], [80, 89]),
        ([0, 10], [-89, -80]),
        ([170, -170], [-10, 10]),
        ([-106.6, -93.5], [25.8, 36.5]),
        ([100, 140], [0, 45]),
    ]

    @pytest.mark.parametrize("zoom", [3, 4, 5])
    @pytest.mark.parametrize("lon_bounds,lat_bounds", BOXES)
    def test_prefilter_matches_the_exact_path(self, zoom, lon_bounds, lat_bounds):
        from uxarray.subset.grid_accessor import _faces_in_bounding_box

        grid = ux.Grid.from_healpix(zoom)
        expected = self._exact(ux.Grid.from_healpix(zoom), lon_bounds, lat_bounds)
        actual = _faces_in_bounding_box(grid, lon_bounds, lat_bounds)

        np.testing.assert_array_equal(np.sort(expected), np.sort(actual))

    def test_a_precomputed_bounds_array_is_used_as_is(self):
        """A grid that already has ``bounds`` should not pay for the filter.

        The saving comes from *not* building the whole-mesh index. Once it
        exists the exact path is free, and taking the filter anyway would add
        work for nothing.
        """
        from uxarray.subset.grid_accessor import _faces_in_bounding_box

        grid = ux.Grid.from_healpix(4)
        _ = grid.bounds  # pay for it up front
        assert "bounds" in grid._ds

        expected = self._exact(grid, [-30, 30], [-20, 20])
        actual = _faces_in_bounding_box(grid, [-30, 30], [-20, 20])
        np.testing.assert_array_equal(np.sort(expected), np.sort(actual))

    def test_an_empty_region_returns_no_faces(self):
        """A box over open ocean far from any face must come back empty, not
        raise -- the candidate array is empty before the exact stage runs."""
        from uxarray.subset.grid_accessor import _faces_in_bounding_box

        grid = ux.Grid.from_healpix(3)
        faces = _faces_in_bounding_box(grid, [0.0, 0.001], [0.0, 0.001])
        assert len(faces) == 0

    def test_subsetting_still_returns_a_usable_grid(self):
        """The accessor, not just the helper: the crop must still be a Grid."""
        grid = ux.Grid.from_healpix(4)
        sub = grid.subset.bounding_box(lon_bounds=[-30, 30], lat_bounds=[-20, 20])
        assert int(sub.n_face) > 0
        assert int(sub.n_face) < int(grid.n_face)
        assert sub.face_lon.size == sub.n_face

    def test_a_polar_face_is_not_excluded_by_its_centre(self):
        """Longitude must not be filtered on face centres.

        A face containing a pole is recorded as spanning every longitude, so
        its bounds sit up to 354 degrees from its own centre on a HEALPix z3
        grid. An early version of the pre-filter tested longitude centres
        with a generous margin and still would have dropped those faces for
        a narrow box -- no finite margin can cover a 354 degree gap. The
        equivalence tests above did not catch it, because the boxes they use
        happen not to need any face whose centre lies outside them.

        This pins the measurement directly rather than through a subset, so
        the reason survives even if the filter is rewritten.
        """
        grid = ux.Grid.from_healpix(3)
        bounds_lon = np.asarray(grid.face_bounds_lon.values)
        bounds_lat = np.asarray(grid.face_bounds_lat.values)
        face_lon = np.asarray(grid.face_lon.values)
        face_lat = np.asarray(grid.face_lat.values)

        lon_reach = np.maximum(
            np.abs(bounds_lon[:, 0] - face_lon), np.abs(bounds_lon[:, 1] - face_lon)
        ).max()
        lat_reach = np.maximum(
            np.abs(bounds_lat[:, 0] - face_lat), np.abs(bounds_lat[:, 1] - face_lat)
        ).max()

        assert lon_reach > 180.0, (
            "longitude bounds no longer reach far beyond the face centre; if "
            "that is genuinely true the longitude centre-filter could be "
            "reinstated, but check why before doing so"
        )
        assert lat_reach < 45.0, (
            "latitude bounds now reach further from the centre than the "
            f"filter's margin assumes ({lat_reach:.1f} deg); the latitude "
            "pre-filter is no longer safe"
        )

    def test_containment_is_what_makes_a_centre_filter_safe(self):
        """The assumption the pre-filter rests on, checked rather than trusted.

        ``faces_within_lat_bounds`` reads as an overlap test -- its docstring
        says "overlap" -- but the implementation keeps a face only when its
        bounds are *fully contained* in the query interval. That distinction
        is the whole justification for filtering on centres: a fully
        contained face must have a contained centre, so nothing the exact
        path keeps can be discarded by the centre test, and no safety margin
        is required.

        If that ever becomes a genuine overlap test, this fails and the
        pre-filter needs a margin before it is correct again.
        """
        grid = ux.Grid.from_healpix(5)
        face_lat = np.asarray(grid.face_lat.values)

        straddlers = 0
        rng = np.random.default_rng(0)
        for _ in range(40):
            la = rng.uniform(-70, 60)
            lat_bounds = [la, la + 4]
            kept = grid.get_faces_between_latitudes(lat_bounds)
            centre_inside = np.flatnonzero(
                (face_lat >= lat_bounds[0]) & (face_lat <= lat_bounds[1])
            )
            straddlers += len(np.setdiff1d(kept, centre_inside))

        assert straddlers == 0, (
            f"{straddlers} faces were kept whose centre lies outside the "
            "query interval, so the latitude filter is now an overlap test "
            "and the centre pre-filter would silently drop them"
        )
