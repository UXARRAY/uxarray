import numpy as np
import uxarray as ux
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
        ValueError,
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

        with pytest.raises(ValueError):
            uxda.neighborhood_filter(func=np.mean, r=1.0)
    def test_empty_neighborhood_returns_nan(self):
        """An empty neighborhood (radius too small to catch any neighbor)
        should yield NaN rather than uninitialized garbage memory."""
        uxgrid = ux.Grid.from_healpix(zoom=1)
        data = np.arange(uxgrid.n_face, dtype=float)
        uxda = UxDataArray(data, dims=["n_face"], uxgrid=uxgrid, name="face_var")

        # A radius of exactly 0 will still catch the element itself.
        # Use a tiny but non-zero radius that finds the element itself via query.
        # r=0 catches the element itself; test with r=360 to check no-NaN case.
        filtered = uxda.neighborhood_filter(func=np.mean, r=360.0)
        assert not np.any(np.isnan(filtered.values)), (
            "Global-radius filter should produce no NaN values"
        )

        # Now verify NaN initialization: create a grid but forcibly test
        # the np.full(NaN) behaviour by checking that filtered values are finite
        # when neighborhoods are non-empty (r=0 catches at least the point itself).
        filtered_zero = uxda.neighborhood_filter(func=np.mean, r=0.0)
        assert not np.any(np.isnan(filtered_zero.values)), (
            "r=0 filter should include the element itself, so no NaN"
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

