import warnings
import numpy as np
import uxarray as ux
from uxarray.errors import DataCenteringError, DimensionError, GridInvalidError
from uxarray.grid.geometry import _build_polygon_shells, _build_corrected_polygon_shells
from uxarray.core.dataset import UxDataset, UxDataArray
import pytest


def test_to_dataset(gridpath, datasetpath):
    """Tests the conversion of UxDataArrays to a UXDataset."""
    uxds = ux.open_dataset(
        gridpath("ugrid", "outCSne30", "outCSne30.ug"),
        datasetpath("ugrid", "outCSne30", "outCSne30_var2.nc")
    )
    uxds_converted = uxds['var2'].to_dataset()

    assert isinstance(uxds_converted, UxDataset)
    assert uxds_converted.uxgrid == uxds.uxgrid


def test_get_dual(gridpath, datasetpath):
    """Tests the creation of the dual mesh on a data array."""
    uxds = ux.open_dataset(
        gridpath("ugrid", "outCSne30", "outCSne30.ug"),
        datasetpath("ugrid", "outCSne30", "outCSne30_var2.nc")
    )
    dual = uxds['var2'].get_dual()

    assert isinstance(dual, UxDataArray)
    assert dual._node_centered()


def test_to_geodataframe(gridpath, datasetpath):
    """Tests the conversion to ``GeoDataFrame``"""
    # GeoFlow
    uxds_geoflow = ux.open_dataset(
        gridpath("ugrid", "geoflow-small", "grid.nc"),
        datasetpath("ugrid", "geoflow-small", "v1.nc")
    )

    # v1 is mapped to nodes, should raise a value error
    with pytest.raises(ValueError):
        uxds_geoflow['v1'].to_geodataframe()

    # grid conversion
    gdf_geoflow_grid = uxds_geoflow.uxgrid.to_geodataframe(periodic_elements='split')

    # number of elements
    assert gdf_geoflow_grid.shape == (uxds_geoflow.uxgrid.n_face, 1)

    # NE30
    uxds_ne30 = ux.open_dataset(
        gridpath("ugrid", "outCSne30", "outCSne30.ug"),
        datasetpath("ugrid", "outCSne30", "outCSne30_var2.nc")
    )

    gdf_geoflow_data = uxds_ne30['var2'].to_geodataframe(periodic_elements='split')

    assert gdf_geoflow_data.shape == (uxds_ne30.uxgrid.n_face, 2)


def test_to_polycollection(gridpath, datasetpath):
    """Tests the conversion to ``PolyCollection``"""
    # GeoFlow
    uxds_geoflow = ux.open_dataset(
        gridpath("ugrid", "geoflow-small", "grid.nc"),
        datasetpath("ugrid", "geoflow-small", "v1.nc")
    )

    # v1 is mapped to nodes, should raise a value error
    with pytest.raises(ValueError):
        uxds_geoflow['v1'].to_polycollection()

    # grid conversion
    pc_geoflow_grid = uxds_geoflow.uxgrid.to_polycollection(periodic_elements="ignore")

    # number of elements
    assert len(pc_geoflow_grid._paths) == uxds_geoflow.uxgrid.n_face


def test_to_geodataframe_preserves_antimeridian_faces(gridpath, datasetpath):
    uxds = ux.open_dataset(
        gridpath("scrip", "ne30pg2", "grid.nc"),
        datasetpath("scrip", "ne30pg2", "data.nc"),
    )
    uxda = uxds["RELHUM"]
    for dim in uxda.dims[:-1]:
        uxda = uxda.isel({dim: 0})

    gdf = uxda.to_geodataframe(periodic_elements="split")
    polygons = uxda.plot.polygons(rasterize=False)

    assert gdf.shape == (uxds.uxgrid.n_face, 2)
    assert len(polygons.data) == uxds.uxgrid.n_face
    assert len(uxds.uxgrid.antimeridian_face_indices) == 120


def test_geodataframe_caching(gridpath, datasetpath):
    uxds = ux.open_dataset(
        gridpath("ugrid", "outCSne30", "outCSne30.ug"),
        datasetpath("ugrid", "outCSne30", "outCSne30_var2.nc")
    )

    gdf_start = uxds['var2'].to_geodataframe()
    gdf_next = uxds['var2'].to_geodataframe()

    # with caching, they point to the same area in memory
    assert gdf_start is gdf_next

    gdf_end = uxds['var2'].to_geodataframe(override=True)

    # override will recompute the grid
    assert gdf_start is not gdf_end

def test_isel_invalid_dim(gridpath, datasetpath):
    """Tests that isel raises a ValueError with a helpful message when an
    invalid dimension is provided."""
    uxds = ux.open_dataset(
        gridpath("ugrid", "outCSne30", "outCSne30.ug"),
        datasetpath("ugrid", "outCSne30", "outCSne30_var2.nc"),
    )

    # create a UxDataArray with an extra dimension
    data = np.random.rand(2, uxds.uxgrid.n_face)
    uxda = UxDataArray(data, dims=["time", "n_face"], uxgrid=uxds.uxgrid)

    with pytest.raises(
        DimensionError,
        match=r"Dimensions \{'invalid_dim'\} do not exist\..*Available dimensions: \('time', 'n_face'\)",
    ):
        uxda.isel(invalid_dim=0)

    with pytest.raises(
        ValueError,
        match=r"Dimensions \{'level'\} do not exist\..*Available dimensions: \('time', 'n_face'\)",
    ):
        uxda.isel(level=0)


def test_data_location():
    """Tests data_location for face/node/edge centered data and non-grid data."""
    uxgrid = ux.Grid.from_healpix(zoom=1)

    face_da = UxDataArray(
        np.ones(uxgrid.n_face), dims=["n_face"], uxgrid=uxgrid
    )
    node_da = UxDataArray(
        np.ones(uxgrid.n_node), dims=["n_node"], uxgrid=uxgrid
    )
    edge_da = UxDataArray(
        np.ones(uxgrid.n_edge), dims=["n_edge"], uxgrid=uxgrid
    )
    other_da = UxDataArray(
        np.ones(5), dims=["other_dim"], uxgrid=uxgrid
    )

    assert face_da.data_location == "face_centered"
    assert node_da.data_location == "node_centered"
    assert edge_da.data_location == "edge_centered"
    assert other_da.data_location is None

    # Works when an extra (non-grid) dimension is present
    face_time = UxDataArray(
        np.ones((3, uxgrid.n_face)), dims=["time", "n_face"], uxgrid=uxgrid
    )
    assert face_time.data_location == "face_centered"


class TestNeighborhoodFilter:
    """Tests for ``UxDataArray.neighborhood_filter``."""

    def test_face_centered(self, gridpath, datasetpath):
        """A large enough radius should average every face together."""
        uxds = ux.open_dataset(
            gridpath("ugrid", "outCSne30", "outCSne30.ug"),
            datasetpath("ugrid", "outCSne30", "outCSne30_vortex.nc"),
        )
        uxda = uxds["psi"]

        # radius of 0 should select each face's own coordinate, leaving the
        # data unchanged
        filtered = uxda.neighborhood_filter(func=np.mean, r=0.0)
        np.testing.assert_allclose(filtered.values, uxda.values)

        # a large enough radius should include the entire grid in the
        # neighborhood of every face, so every filtered value should match
        # the global mean of the field
        filtered_all = uxda.neighborhood_filter(func=np.mean, r=360.0)
        np.testing.assert_allclose(filtered_all.values, uxda.values.mean())

        assert isinstance(filtered, UxDataArray)
        assert filtered.uxgrid == uxda.uxgrid
        assert filtered.dims == uxda.dims
        assert filtered.shape == uxda.shape

    def test_node_centered(self):
        """Neighborhood filter should work for node-centered data."""
        uxgrid = ux.Grid.from_healpix(zoom=1)
        data = np.arange(uxgrid.n_node, dtype=float)
        uxda = UxDataArray(data, dims=["n_node"], uxgrid=uxgrid, name="node_var")

        filtered = uxda.neighborhood_filter(func=np.mean, r=0.0)
        np.testing.assert_allclose(filtered.values, data)

        filtered_all = uxda.neighborhood_filter(func=np.mean, r=360.0)
        np.testing.assert_allclose(filtered_all.values, data.mean())

    def test_edge_centered(self):
        """Neighborhood filter should work for edge-centered data."""
        uxgrid = ux.Grid.from_healpix(zoom=1)
        data = np.arange(uxgrid.n_edge, dtype=float)
        uxda = UxDataArray(data, dims=["n_edge"], uxgrid=uxgrid, name="edge_var")

        filtered = uxda.neighborhood_filter(func=np.mean, r=0.0)
        np.testing.assert_allclose(filtered.values, data)

    def test_custom_func_with_partial(self, gridpath, datasetpath):
        """A user-defined function (i.e. ``functools.partial``) should work."""
        from functools import partial

        uxds = ux.open_dataset(
            gridpath("ugrid", "outCSne30", "outCSne30.ug"),
            datasetpath("ugrid", "outCSne30", "outCSne30_vortex.nc"),
        )
        uxda = uxds["psi"]

        filtered_max = uxda.neighborhood_filter(func=np.max, r=5.0)
        filtered_percentile = uxda.neighborhood_filter(
            func=partial(np.percentile, q=100), r=5.0
        )

        np.testing.assert_allclose(filtered_max.values, filtered_percentile.values)

    def test_extra_dimension_preserved(self, gridpath, datasetpath):
        """An extra leading (i.e. time) dimension should be preserved."""
        uxds = ux.open_dataset(
            gridpath("ugrid", "outCSne30", "outCSne30.ug"),
            datasetpath("ugrid", "outCSne30", "outCSne30_vortex.nc"),
        )
        uxda = uxds["psi"]

        data = np.stack([uxda.values, uxda.values * 2.0])
        uxda_time = UxDataArray(
            data, dims=["time", "n_face"], uxgrid=uxda.uxgrid, name="psi_time"
        )

        filtered = uxda_time.neighborhood_filter(func=np.mean, r=0.0)

        assert filtered.dims == uxda_time.dims
        assert filtered.shape == uxda_time.shape
        np.testing.assert_allclose(filtered.values, data)

    def test_invalid_data_location(self):
        """Data that is not mapped to a grid element should raise an error."""
        uxgrid = ux.Grid.from_healpix(zoom=1)
        uxda = UxDataArray(np.ones(5), dims=["other_dim"], uxgrid=uxgrid)

        with pytest.raises(DataCenteringError):
            uxda.neighborhood_filter(func=np.mean, r=1.0)

    def test_radius_edge_cases_never_produce_nan(self):
        """Every element is its own neighbor at distance 0, so no neighborhood
        is ever empty and the output never contains NaN."""
        uxgrid = ux.Grid.from_healpix(zoom=1)
        data = np.arange(uxgrid.n_face, dtype=float)
        uxda = UxDataArray(data, dims=["n_face"], uxgrid=uxgrid, name="face_var")

        # r=0 catches only the element itself, so the data is returned unchanged
        filtered_zero = uxda.neighborhood_filter(func=np.mean, r=0.0)
        assert not np.any(np.isnan(filtered_zero.values))
        np.testing.assert_allclose(filtered_zero.values, data)

        # a radius spanning the sphere catches every element
        filtered_all = uxda.neighborhood_filter(func=np.mean, r=360.0)
        assert not np.any(np.isnan(filtered_all.values))
        np.testing.assert_allclose(filtered_all.values, data.mean())

        # a negative radius is rejected by BallTree.query_radius
        with pytest.raises(AssertionError):
            uxda.neighborhood_filter(func=np.mean, r=-1.0)

    def test_func_without_axis_raises_helpful_error(self):
        """A ``func`` that does not accept ``axis`` should raise a TypeError
        that explains the requirement rather than a raw NumPy message."""
        uxgrid = ux.Grid.from_healpix(zoom=1)
        uxda = UxDataArray(
            np.arange(uxgrid.n_face, dtype=float), dims=["n_face"], uxgrid=uxgrid
        )

        with pytest.raises(TypeError, match="must accept an `axis` keyword"):
            uxda.neighborhood_filter(func=sum, r=5.0)

    def test_uses_spherical_tree_regardless_of_cached_tree(self):
        """``r`` is documented in great-circle degrees, so the filter must build
        a spherical/haversine tree even if a cartesian one was cached first."""
        uxgrid = ux.Grid.from_healpix(zoom=1)
        data = np.arange(uxgrid.n_face, dtype=float)
        uxda = UxDataArray(data, dims=["n_face"], uxgrid=uxgrid, name="face_var")

        expected = uxda.neighborhood_filter(func=np.mean, r=20.0).values

        # Prime the cache with a cartesian tree, then filter again
        uxgrid.get_ball_tree(
            coordinates="face centers",
            coordinate_system="cartesian",
            distance_metric="euclidean",
        )
        np.testing.assert_allclose(
            uxda.neighborhood_filter(func=np.mean, r=20.0).values, expected
        )

    def test_auto_transpose_direct_on_uxdataarray(self, gridpath, datasetpath):
        """Calling neighborhood_filter directly on a (time, n_face) UxDataArray
        (without going through UxDataset) should preserve the original dim order."""
        uxds = ux.open_dataset(
            gridpath("ugrid", "outCSne30", "outCSne30.ug"),
            datasetpath("ugrid", "outCSne30", "outCSne30_vortex.nc"),
        )
        uxda = uxds["psi"]

        # Build a multi-dim UxDataArray with time as the FIRST (non-grid) dim
        data = np.stack([uxda.values, uxda.values * 2.0])  # shape (2, n_face)
        uxda_time = UxDataArray(
            data, dims=["time", "n_face"], uxgrid=uxda.uxgrid, name="psi_time"
        )

        # n_face is already last: no transpose needed internally
        filtered = uxda_time.neighborhood_filter(func=np.mean, r=0.0)
        assert filtered.dims == ("time", "n_face")
        assert filtered.shape == (2, uxda.shape[0])
        np.testing.assert_allclose(filtered.values, data)

        # Also test with a UxDataArray that has grid dim NOT last (n_face, time)
        uxda_face_first = uxda_time.transpose("n_face", "time")
        filtered2 = uxda_face_first.neighborhood_filter(func=np.mean, r=0.0)
        # Dim order must be restored to (n_face, time)
        assert filtered2.dims == ("n_face", "time")
        assert filtered2.shape == (uxda.shape[0], 2)
def test_uxgrid_None_is_invalid_in_uxdataarray():
    """Ensures GridInvalidError gets raised if uxgrid=None when getting UxDataArray.uxgrid.
    Regression test for #1620.
    """
    # construct array without uxgrid. Ideally this line would crash, but allowing uxgrid=None
    # is an important workaround for subclassing from xarray. see #1620 for more details.
    arr = ux.UxDataArray([1,2,3], dims=['n_face'])
    # ensure getting arr.uxgrid crashes with GridInvalidError (it is None...)
    with pytest.raises(GridInvalidError):
        arr.uxgrid

    # trying to set uxgrid to a non-Grid should raise TypeError:
    with pytest.raises(TypeError):
        arr.uxgrid = "not a grid"
    with pytest.raises(TypeError):
        arr.uxgrid = 123
    # this remains true even for None, outside of __init__:
    with pytest.raises(TypeError):
        arr.uxgrid = None
    # it also applies (for non-None non-Grid objects) during __init__:
    with pytest.raises(TypeError):
        ux.UxDataArray([4,5], dims=['n_face'], uxgrid="not a grid")

    """Tests for ``UxDataArray.neighborhood_filter``."""

    @pytest.fixture
    def vortex(self, gridpath, datasetpath):
        """The ``psi`` field on outCSne30, which most of these tests filter."""
        uxds = ux.open_dataset(
            gridpath("ugrid", "outCSne30", "outCSne30.ug"),
            datasetpath("ugrid", "outCSne30", "outCSne30_vortex.nc"),
        )
        return uxds["psi"]

    # Every named reduction, with the NumPy expression it must equal and the
    # parameter it takes. The reference runs through the generic callable path,
    # so this pins each compiled kernel against the loop it bypasses.
    NAMED_REDUCTIONS = [
        ("mean", {}, lambda a, axis: np.mean(a, axis=axis)),
        ("sum", {}, lambda a, axis: np.sum(a, axis=axis)),
        ("min", {}, lambda a, axis: np.min(a, axis=axis)),
        ("max", {}, lambda a, axis: np.max(a, axis=axis)),
        ("median", {}, lambda a, axis: np.median(a, axis=axis)),
        ("ptp", {}, lambda a, axis: np.ptp(a, axis=axis)),
        ("std", {"ddof": 1}, lambda a, axis: np.std(a, axis=axis, ddof=1)),
        ("var", {"ddof": 1}, lambda a, axis: np.var(a, axis=axis, ddof=1)),
        ("quantile", {"q": 0.9}, lambda a, axis: np.quantile(a, 0.9, axis=axis)),
        ("percentile", {"q": 90}, lambda a, axis: np.percentile(a, 90, axis=axis)),
    ]

    @pytest.mark.parametrize("name,kwargs,reference", NAMED_REDUCTIONS)
    def test_named_reduction_matches_numpy(self, name, kwargs, reference):
        """Each compiled reduction must equal its NumPy expression, including
        where NaN lands.

        The field is partly masked on purpose. NaN handling is the easy thing
        to get wrong in a kernel: a hand-written ``if value > best`` loop skips
        NaN where ``np.max`` propagates it, and numba's ``np.median``
        propagates only depending on where the NaN falls in its partition. The
        extra leading dimension exercises the gufunc's broadcast loop.
        """
        uxgrid = ux.Grid.from_healpix(zoom=2)
        rng = np.random.default_rng(0)
        values = rng.random((3, uxgrid.n_face))
        # mask a tenth of the faces, as a land/ocean mask would
        values[:, rng.choice(uxgrid.n_face, uxgrid.n_face // 10, replace=False)] = np.nan
        uxda = UxDataArray(values, dims=["time", "n_face"], uxgrid=uxgrid, name="masked")

        nb = uxgrid.neighborhoods(r=20.0)
        got = nb.reduce(uxda, name, **kwargs).values
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # numpy all-NaN slices
            expected = nb.reduce(uxda, reference).values

        assert np.isnan(got).any(), "expected a neighborhood to hit a masked value"
        assert not np.isnan(got).all(), "expected some neighborhood to be clean"
        np.testing.assert_array_equal(np.isnan(got), np.isnan(expected))
        finite = ~np.isnan(expected)
        np.testing.assert_allclose(got[finite], expected[finite], rtol=1e-12)

    def test_callable_still_accepted(self, vortex):
        """The original ``func=np.mean`` signature keeps working, and reaches
        the same kernel the name does rather than dropping to the generic
        loop."""
        from uxarray.grid.neighbors import _CALLABLE_ALIASES, _resolve_reduction

        for callable_func, name in _CALLABLE_ALIASES.items():
            assert (
                _resolve_reduction(callable_func, {})[0]
                is _resolve_reduction(name, {})[0]
            ), f"{callable_func} should reach the {name!r} kernel"

        np.testing.assert_allclose(
            vortex.neighborhood_filter(np.mean, r=3.0).values,
            vortex.neighborhood_filter("mean", r=3.0).values,
        )

    def test_callable_escape_hatch(self, vortex):
        """A user's own function, with no compiled equivalent, still works on
        the ``axis=-1`` contract.

        ``functools.partial`` is covered by test_custom_func_with_partial.
        """

        # a user's own function, with no NumPy equivalent at all
        def rms(values, axis):
            return np.sqrt(np.mean(values**2, axis=axis))

        filtered = vortex.neighborhood_filter(rms, r=3.0)
        assert filtered.shape == vortex.shape
        assert np.all(filtered.values >= 0)

    @pytest.mark.parametrize(
        "func,kwargs,error,match",
        [
            ("meen", {}, ValueError, "Unknown reduction 'meen'"),
            ("mean", {"q": 90}, TypeError, "unexpected keyword argument"),
            ("quantile", {}, TypeError, "requires the 'q' keyword"),
            ("quantile", {"q": 90}, ValueError, "between 0 and 1"),
            (42, {}, TypeError, "name of a reduction or a callable"),
        ],
    )
    def test_invalid_reduction(self, func, kwargs, error, match):
        """Naming a reduction makes bad input catchable up front, rather than
        as a TypeError from inside the loop."""
        uxgrid = ux.Grid.from_healpix(zoom=1)
        uxda = UxDataArray(
            np.arange(uxgrid.n_face, dtype=float), dims=["n_face"], uxgrid=uxgrid
        )
        with pytest.raises(error, match=match):
            uxda.neighborhood_filter(func, r=1.0, **kwargs)

    def test_neighborhoods_reuse(self, vortex):
        """A reused Neighborhoods must give the same answer as the one-shot
        filter -- the point of holding onto it is that it costs one query."""
        nb = vortex.uxgrid.neighborhoods(r=4.0)

        assert (nb.r, nb.on, nb.grid_dim) == (4.0, "face centers", "n_face")
        counts = nb.n_neighbors
        assert counts.dims == ("n_face",)
        # every element is its own neighbor, so no neighborhood is ever empty
        assert counts.min() >= 1

        for name, kwargs in [("mean", {}), ("percentile", {"q": 90})]:
            np.testing.assert_allclose(
                nb.reduce(vortex, name, **kwargs).values,
                vortex.neighborhood_filter(name, r=4.0, **kwargs).values,
                rtol=1e-12,
            )

    def test_neighborhoods_reject_wrong_data(self):
        """Reducing data mapped elsewhere must fail loudly rather than index
        into the wrong element set."""
        uxgrid = ux.Grid.from_healpix(zoom=1)
        _ = uxgrid.n_node  # populate node coords before the tree is built
        nb = uxgrid.neighborhoods(r=30.0, on="face centers")

        node_data = UxDataArray(
            np.arange(uxgrid.n_node, dtype=float), dims=["n_node"], uxgrid=uxgrid
        )
        with pytest.raises(DataCenteringError, match="reduce over 'n_face'"):
            nb.reduce(node_data, "mean")

        other = ux.Grid.from_healpix(zoom=2)
        wrong_size = UxDataArray(
            np.arange(other.n_face, dtype=float), dims=["n_face"], uxgrid=other
        )
        with pytest.raises(DataCenteringError, match="different grid"):
            nb.reduce(wrong_size, "mean")

        with pytest.raises(ValueError, match="Invalid `on`"):
            uxgrid.neighborhoods(r=1.0, on="face_centers")

    def test_dask_input_stays_lazy(self, vortex):
        """Lazy input stays lazy: the grid dimension is a core dimension, but
        the others stay chunked and unevaluated."""
        eager = vortex.neighborhood_filter("mean", r=2.0)

        stacked = UxDataArray(
            np.tile(vortex.values, (6, 1)),
            dims=["time", "n_face"],
            uxgrid=vortex.uxgrid,
            name="psi",
        ).chunk({"time": 2, "n_face": -1})

        # chunking a non-grid dimension is the supported case: untouched, silent
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            filtered = stacked.neighborhood_filter("mean", r=2.0)

        assert filtered.chunks is not None, "the filter should not force a compute"
        assert filtered.chunksizes["time"] == (2, 2, 2)
        assert isinstance(filtered, UxDataArray)
        np.testing.assert_allclose(
            filtered.compute().values, np.tile(eager.values, (6, 1))
        )

    def test_grid_dim_chunks_are_collapsed_with_warning(self, vortex):
        """A neighborhood may span the whole grid, so the grid dimension cannot
        stay chunked. Collapsing it undoes a memory decision the user made, so
        it is not done silently."""
        expected = vortex.neighborhood_filter("mean", r=2.0).values

        uxda = vortex.chunk({"n_face": 1000})
        assert len(uxda.chunksizes["n_face"]) > 1

        with pytest.warns(UserWarning, match="Rechunking 'n_face'"):
            filtered = uxda.neighborhood_filter("mean", r=2.0)

        assert filtered.chunksizes["n_face"] == (uxda.sizes["n_face"],)
        np.testing.assert_allclose(filtered.compute().values, expected)

    def test_output_is_always_float64(self, vortex):
        """float32 hits the kernel's float32 signature and integers have no
        signature at all; both must come back as float64, as the generic path
        does by writing into a float64 output."""
        uxgrid = ux.Grid.from_healpix(zoom=1)
        integers = UxDataArray(
            np.arange(uxgrid.n_face), dims=["n_face"], uxgrid=uxgrid, name="int_var"
        )
        filtered = integers.neighborhood_filter("mean", r=0.0)
        assert filtered.dtype == np.float64
        np.testing.assert_allclose(filtered.values, integers.values)

        as_float32 = UxDataArray(
            vortex.values.astype(np.float32), dims=vortex.dims, uxgrid=vortex.uxgrid
        )
        filtered32 = as_float32.neighborhood_filter("mean", r=2.0)
        assert filtered32.dtype == np.float64
        np.testing.assert_allclose(
            filtered32.values, vortex.neighborhood_filter("mean", r=2.0).values,
            rtol=1e-6,
        )
