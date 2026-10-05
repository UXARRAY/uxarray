
import dask.array as da
import numpy as np
import pytest
import warnings

import numpy.testing as nt

import uxarray as ux
from uxarray.constants import ERROR_TOLERANCE
from uxarray.errors import DataCenteringError

class TestZonalCSne30:

    gridfile_ne30 = None  # Will be set in test methods
    datafile_vortex_ne30 = None  # Will be set in test methods
    dsfile_var2_ne30 = None  # Will be set in test methods
    test_file_2 = None  # Will be set in test methods
    test_file_3 = None  # Will be set in test methods

    def test_non_conservative_zonal_mean_equator(self, gridpath, datasetpath):
        """Tests the zonal mean at the equator. This grid contains points that are exactly """
        grid_path = gridpath("ugrid", "outCSne30", "outCSne30.ug")
        data_path = datasetpath("ugrid", "outCSne30", "outCSne30_vortex.nc")
        uxds = ux.open_dataset(grid_path, data_path)

        res = uxds['psi'].zonal_mean(0)

        assert res.values[0] == pytest.approx(1, abs=ERROR_TOLERANCE)

    def test_non_conservative_zonal_mean(self, gridpath, datasetpath):
        """Tests if the correct number of queries are returned."""
        grid_path = gridpath("ugrid", "outCSne30", "outCSne30.ug")
        data_path = datasetpath("ugrid", "outCSne30", "outCSne30_vortex.nc")
        uxds = ux.open_dataset(grid_path, data_path)

        res = uxds['psi'].zonal_mean((-90.0, 90.0, 1))

        assert len(res) == 181

    def test_non_conservative_zonal_mean_at_pole(self, gridpath, datasetpath):
        """Tests the zonal average execution at both poles."""
        grid_path = gridpath("ugrid", "outCSne30", "outCSne30.ug")
        data_path = datasetpath("ugrid", "outCSne30", "outCSne30_vortex.nc")
        uxds = ux.open_dataset(grid_path, data_path)

        # Test at the poles
        res_n90 = uxds['psi'].zonal_mean(90)
        res_p90 = uxds['psi'].zonal_mean(-90)

        # result should be a scalar
        assert len(res_n90.values) == 1
        assert len(res_p90.values) == 1

    def test_zonal_mean_dask(self, gridpath, datasetpath):
        """Tests if zonal average returns Dask arrays when appropriate."""
        grid_path = gridpath("ugrid", "outCSne30", "outCSne30.ug")
        data_path = datasetpath("ugrid", "outCSne30", "outCSne30_vortex.nc")
        uxds = ux.open_dataset(grid_path, data_path)

        uxds['psi'] = uxds['psi'].chunk()

        res = uxds['psi'].zonal_mean((-90, 90, 1))

        assert isinstance(res.data, da.Array)

        res_computed = res.compute()

        assert isinstance(res_computed.data, np.ndarray)

    def test_zonal_weights(self, gridpath, datasetpath):
        grid_path = gridpath("ugrid", "outCSne30", "outCSne30.ug")
        data_path = datasetpath("ugrid", "outCSne30", "outCSne30_vortex.nc")
        uxds = ux.open_dataset(grid_path, data_path)

        za_1 = uxds['psi'].zonal_mean((-90, 90, 30), use_robust_weights=True)
        za_2 = uxds['psi'].zonal_mean((-90, 90, 30), use_robust_weights=False)

        nt.assert_almost_equal(za_1.data, za_2.data)

    def test_lat_inputs(self, gridpath, datasetpath):
        grid_path = gridpath("ugrid", "outCSne30", "outCSne30.ug")
        data_path = datasetpath("ugrid", "outCSne30", "outCSne30_vortex.nc")
        uxds = ux.open_dataset(grid_path, data_path)

        assert len(uxds['psi'].zonal_mean(lat=1)) == 1
        assert len(uxds['psi'].zonal_mean(lat=(-90, 90, 1))) == 181

    def test_zonal_mean_missing_latitudes_nan(self, gridpath, datasetpath):
        """Zonal mean should return NaN (not zeros) when no faces intersect a latitude."""
        grid_path = gridpath("ugrid", "outCSne30", "outCSne30.ug")
        data_path = datasetpath("ugrid", "outCSne30", "outCSne30_vortex.nc")
        uxds = ux.open_dataset(grid_path, data_path)

        # Restrict to a narrow band so most requested latitudes have no coverage
        narrow = uxds["psi"].subset.bounding_box(lon_bounds=(-20, 20), lat_bounds=(0, 10))

        with warnings.catch_warnings():
            warnings.filterwarnings("error", category=RuntimeWarning)
            res = narrow.zonal_mean(lat=(-90, 90, 10))

        below_band = res.sel(latitudes=res.latitudes < 0)
        assert np.all(np.isnan(below_band))
        assert np.isfinite(res.sel(latitudes=0).item())

        with warnings.catch_warnings():
            warnings.filterwarnings("error", category=RuntimeWarning)
            res_cons = narrow.zonal_mean(lat=(-90, 90, 10), conservative=True)

        below_band_cons = res_cons.sel(latitudes=res_cons.latitudes < 0)
        assert np.all(np.isnan(below_band_cons))
        assert np.isfinite(res_cons.sel(latitudes=5).item())

    def test_zonal_mean_int_data_promotes_dtype(self):
        """Integer inputs should be promoted so NaNs can be stored."""
        grid = ux.Grid.from_healpix(zoom=0)
        faces = np.where(grid.face_lat > 0)[0]  # only northern hemisphere
        uxda = ux.UxDataArray(
            np.ones(grid.n_face, dtype=np.int32), dims=["n_face"], uxgrid=grid
        ).isel(n_face=faces)

        res = uxda.zonal_mean(lat=(-90, 90, 30))

        assert np.issubdtype(res.dtype, np.floating)
        assert np.isnan(res.sel(latitudes=-90)).item()

def test_mismatched_dims():
    uxgrid = ux.Grid.from_healpix(zoom=0)
    uxda = ux.UxDataArray(np.ones((10, uxgrid.n_face, 5)), dims=['a', 'n_face', 'b'], uxgrid=uxgrid)

    za = uxda.zonal_average()

    assert za.shape == (10, 19, 5)
    assert za.dims[1] == "latitudes"


class TestConservativeZonalMean:
    """Test conservative zonal mean functionality."""

    def test_conservative_zonal_mean_basic(self, gridpath, datasetpath):
        """Test basic conservative zonal mean with bands."""
        grid_path = gridpath("ugrid", "outCSne30", "outCSne30.ug")
        data_path = datasetpath("ugrid", "outCSne30", "outCSne30_vortex.nc")
        uxds = ux.open_dataset(grid_path, data_path)

        # Test with explicit bands
        bands = np.array([-90, -30, 0, 30, 90])
        result = uxds["psi"].zonal_mean(lat=bands, conservative=True)

        # Should have one less value than bands (4 bands from 5 edges)
        assert result.shape == (len(bands) - 1,)
        assert np.all(np.isfinite(result.values))

    def test_conservative_float_step_size(self, gridpath, datasetpath):
        """Test conservative zonal mean with float step sizes."""
        grid_path = gridpath("ugrid", "outCSne30", "outCSne30.ug")
        data_path = datasetpath("ugrid", "outCSne30", "outCSne30_vortex.nc")
        uxds = ux.open_dataset(grid_path, data_path)

        # Test with float step size (5.5 degrees)
        result = uxds["psi"].zonal_mean(lat=(-90, 90, 0.05), conservative=True)

        # Should get valid results
        assert len(result) > 0
        assert np.all(np.isfinite(result.values))

        # Test with reasonable float step size (no warning)
        result = uxds["psi"].zonal_mean(lat=(-90, 90, 5.5), conservative=True)
        expected_n_bands = int(np.ceil(180 / 5.5))
        assert result.shape[0] == expected_n_bands
        assert np.all(np.isfinite(result.values))

    def test_conservative_near_pole(self, gridpath, datasetpath):
        """Test conservative zonal mean with bands near the poles."""
        grid_path = gridpath("ugrid", "outCSne30", "outCSne30.ug")
        data_path = datasetpath("ugrid", "outCSne30", "outCSne30_vortex.nc")
        uxds = ux.open_dataset(grid_path, data_path)

        # Test near north pole with float step
        bands_north = np.array([85.0, 87.5, 90.0])
        result_north = uxds["psi"].zonal_mean(lat=bands_north, conservative=True)
        assert result_north.shape == (2,)
        assert np.all(np.isfinite(result_north.values))

        # Test near south pole with float step
        bands_south = np.array([-90.0, -87.5, -85.0])
        result_south = uxds["psi"].zonal_mean(lat=bands_south, conservative=True)
        assert result_south.shape == (2,)
        assert np.all(np.isfinite(result_south.values))

        # Test spanning pole with non-integer step
        bands_span = np.array([88.5, 89.25, 90.0])
        result_span = uxds["psi"].zonal_mean(lat=bands_span, conservative=True)
        assert result_span.shape == (2,)
        assert np.all(np.isfinite(result_span.values))

    def test_conservative_step_size_validation(self, gridpath, datasetpath):
        """Test that step size validation works correctly."""
        grid_path = gridpath("ugrid", "outCSne30", "outCSne30.ug")
        data_path = datasetpath("ugrid", "outCSne30", "outCSne30_vortex.nc")
        uxds = ux.open_dataset(grid_path, data_path)

        # Test negative step size
        with pytest.raises(ValueError, match="Expected step>0"):
            uxds["psi"].zonal_mean(lat=(-90, 90, -10), conservative=True)

        # Test zero step size
        with pytest.raises(ValueError, match="Expected step>0"):
            uxds["psi"].zonal_mean(lat=(-90, 90, 0), conservative=True)

    def test_conservative_full_sphere_conservation(self, gridpath, datasetpath):
        """Test that single band covering entire sphere conserves global mean."""
        grid_path = gridpath("ugrid", "outCSne30", "outCSne30.ug")
        data_path = datasetpath("ugrid", "outCSne30", "outCSne30_vortex.nc")
        uxds = ux.open_dataset(grid_path, data_path)

        # Single band covering entire sphere
        bands = np.array([-90, 90])
        result = uxds["psi"].zonal_mean(lat=bands, conservative=True)

        # Compare with global mean
        global_mean = uxds["psi"].mean()

        assert result.shape == (1,)
        assert result.values[0] == pytest.approx(global_mean.values, rel=0.01)

    def test_conservative_vs_nonconservative_comparison(self, gridpath, datasetpath):
        """Compare conservative and non-conservative methods."""
        grid_path = gridpath("ugrid", "outCSne30", "outCSne30.ug")
        data_path = datasetpath("ugrid", "outCSne30", "outCSne30_vortex.nc")
        uxds = ux.open_dataset(grid_path, data_path)

        # Non-conservative at band centers
        lat_centers = np.array([-60, 0, 60])
        non_conservative = uxds["psi"].zonal_mean(lat=lat_centers)

        # Conservative with bands
        bands = np.array([-90, -30, 30, 90])
        conservative = uxds["psi"].zonal_mean(lat=bands, conservative=True)

        # Results should be similar but not identical
        assert non_conservative.shape == conservative.shape
        # Check they are in the same ballpark
        assert np.all(np.abs(conservative.values - non_conservative.values) <
                     np.abs(non_conservative.values) * 0.5)


class TestZonalAnomaly:
    """Tests for UxDataArray.zonal_anomaly."""

    def _open(self, gridpath, datasetpath):
        grid_path = gridpath("ugrid", "outCSne30", "outCSne30.ug")
        data_path = datasetpath("ugrid", "outCSne30", "outCSne30_vortex.nc")
        return ux.open_dataset(grid_path, data_path)

    def test_output_dims_match_input(self, gridpath, datasetpath):
        """Output shape and dims must equal input (face axis preserved)."""
        uxds = self._open(gridpath, datasetpath)
        psi = uxds["psi"]
        res = psi.zonal_anomaly(lat=(-90, 90, 30))
        assert res.shape == psi.shape
        assert res.dims == psi.dims
        assert "n_face" in res.dims

    def test_conservative_anomaly_band_mean_small(self, gridpath, datasetpath):
        """Conservative anomaly: per-band area-weighted mean is small.

        Faces that straddle a band boundary are intentionally blended across
        neighbouring band means (sharing the same weight kernel as
        zonal_mean), so the per-band mean is not exactly zero — but it must
        be small relative to the raw signal magnitude.
        """
        uxds = self._open(gridpath, datasetpath)
        bands = np.array([-90.0, -30.0, 30.0, 90.0])
        anom = uxds["psi"].zonal_anomaly(lat=bands, conservative=True)

        raw_std = float(uxds["psi"].values.std())
        per_band = _compute_face_band_weights(uxds["psi"].uxgrid, bands)
        for overlapping, w in per_band:
            if overlapping.size == 0:
                continue
            vals = anom.isel(n_face=overlapping, ignore_grid=True).values
            weighted = abs((w * vals).sum() / w.sum())
            assert weighted < raw_std * 0.05

    def test_band_anomaly_centroid_sums_to_zero(self, gridpath, datasetpath):
        """Non-conservative anomaly: simple mean within each band ≈ 0."""
        uxds = self._open(gridpath, datasetpath)
        bands = np.array([-90.0, -30.0, 30.0, 90.0])
        psi = uxds["psi"]
        anom = psi.zonal_anomaly(lat=bands, conservative=False)

        face_lats = psi.uxgrid.face_lat.values
        for bi in range(len(bands) - 1):
            mask = (face_lats >= bands[bi]) & (face_lats < bands[bi + 1])
            if bi == len(bands) - 2:
                mask |= face_lats == bands[bi + 1]
            if not mask.any():
                continue
            assert anom.values[mask].mean() == pytest.approx(0.0, abs=1e-12)

    def test_multidim_face_not_last_axis(self):
        """Works when n_face is not the last axis and preserves other dims."""
        uxgrid = ux.Grid.from_healpix(zoom=1)
        # shape (T, n_face, L); face is axis=1
        T, L = 3, 4
        rng = np.random.default_rng(0)
        data = rng.standard_normal((T, uxgrid.n_face, L))
        uxda = ux.UxDataArray(
            data, dims=["time", "n_face", "level"], uxgrid=uxgrid
        )

        anom = uxda.zonal_anomaly(lat=(-90, 90, 30))
        assert anom.shape == uxda.shape
        assert anom.dims == uxda.dims

        # Per band, per (t, l), the anomaly mean should be ~0.
        face_lats = uxgrid.face_lat.values
        bands = np.linspace(-90, 90, int(round(180 / 30)) + 1)
        for bi in range(len(bands) - 1):
            mask = (face_lats >= bands[bi]) & (face_lats < bands[bi + 1])
            if bi == len(bands) - 2:
                mask |= face_lats == bands[bi + 1]
            if not mask.any():
                continue
            band_vals = anom.values[:, mask, :]
            # Mean across face dim per (t, l) should be ~0
            nt.assert_allclose(band_vals.mean(axis=1), 0.0, atol=1e-12)

    def test_dask_input_stays_lazy(self, gridpath, datasetpath):
        """Centroid path keeps dask laziness when input is chunked."""
        uxds = self._open(gridpath, datasetpath)
        uxds["psi"] = uxds["psi"].chunk()
        res = uxds["psi"].zonal_anomaly(lat=(-90, 90, 30))
        assert isinstance(res.data, da.Array)
        # Verify computation still produces finite numbers
        computed = res.compute()
        assert np.all(np.isfinite(computed.values))

    def test_dask_input_conservative_lazy(self, gridpath, datasetpath):
        """Conservative path keeps dask laziness for the subtract step."""
        uxds = self._open(gridpath, datasetpath)
        uxds["psi"] = uxds["psi"].chunk()
        res = uxds["psi"].zonal_anomaly(lat=(-90, 90, 30), conservative=True)
        assert isinstance(res.data, da.Array)
        computed = res.compute()
        assert np.all(np.isfinite(computed.values))

    def test_conservative_vs_centroid_close(self, gridpath, datasetpath):
        """Conservative and centroid anomalies should be comparable in magnitude."""
        uxds = self._open(gridpath, datasetpath)
        bands = np.array([-90.0, -30.0, 30.0, 90.0])
        a_cons = uxds["psi"].zonal_anomaly(lat=bands, conservative=True)
        a_cent = uxds["psi"].zonal_anomaly(lat=bands, conservative=False)
        # Same shape
        assert a_cons.shape == a_cent.shape
        # Same order of magnitude (allow generous tolerance — methods differ)
        std_cons = float(np.nanstd(a_cons.values))
        std_cent = float(np.nanstd(a_cent.values))
        assert std_cons > 0 and std_cent > 0
        assert 0.25 < std_cons / std_cent < 4.0

    def test_int_input_promotes_dtype(self):
        """Integer inputs are promoted so NaN-bearing anomalies fit."""
        uxgrid = ux.Grid.from_healpix(zoom=1)
        uxda = ux.UxDataArray(
            np.ones(uxgrid.n_face, dtype=np.int32),
            dims=["n_face"],
            uxgrid=uxgrid,
        )
        res = uxda.zonal_anomaly(lat=(-90, 90, 30))
        assert np.issubdtype(res.dtype, np.floating)
        # All-ones input → all-zero anomalies wherever defined
        finite = res.values[np.isfinite(res.values)]
        assert finite.size > 0
        nt.assert_allclose(finite, 0.0, atol=1e-12)

    def test_non_face_centered_raises(self, gridpath, datasetpath):
        """Only face-centered data is supported."""
        uxgrid = ux.Grid.from_healpix(zoom=1)
        uxda = ux.UxDataArray(
            np.zeros(uxgrid.n_node), dims=["n_node"], uxgrid=uxgrid
        )
        with pytest.raises(DataCenteringError, match="non-face-centered data is not currently supported"):
            uxda.zonal_anomaly()

    def test_invalid_lat_input_raises(self):
        """Invalid lat specs raise ValueError."""
        uxgrid = ux.Grid.from_healpix(zoom=1)
        uxda = ux.UxDataArray(
            np.zeros(uxgrid.n_face), dims=["n_face"], uxgrid=uxgrid
        )
        with pytest.raises(ValueError, match="Expected step>0"):
            uxda.zonal_anomaly(lat=(-90, 90, 0))
        with pytest.raises(ValueError, match="Expected step>0"):
            uxda.zonal_anomaly(lat=(-90, 90, -1))
        with pytest.raises(ValueError):
            uxda.zonal_anomaly(lat=[42.0])  # too few edges
        with pytest.raises(ValueError, match="monotonic"):
            uxda.zonal_anomaly(lat=[10.0, -10.0, 30.0])


from uxarray.core.zonal import (  # noqa: E402
    _compute_band_overlap_area,
    _compute_face_band_weights,
)
from uxarray.grid.area import calculate_face_area  # noqa: E402
from uxarray.grid.utils import _get_cartesian_face_edge_nodes_array_subset  # noqa: E402


def _face_edges(lat, lon):
    """Edge array (n_edges, 2, 3) for a face given its vertices in degrees."""
    lat, lon = np.deg2rad(lat), np.deg2rad(lon)
    v = np.stack(
        [np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)], axis=1
    )
    return np.stack([v, np.roll(v, -1, axis=0)], axis=1)


def _band_area(face_edges, lat0, lat1):
    return _compute_band_overlap_area(
        face_edges, np.sin(np.deg2rad(lat0)), np.sin(np.deg2rad(lat1))
    )


def _zone_area(lat0, lat1):
    return 2 * np.pi * (np.sin(np.deg2rad(lat1)) - np.sin(np.deg2rad(lat0)))


class TestBandOverlapArea:
    """Face/latitude-band overlap areas used as conservative zonal weights."""

    @pytest.mark.parametrize(
        "lat0, lat1",
        [(20.0, 60.0), (-10.0, 10.0), (0.0, 30.0), (60.0, 90.0), (85.0, 90.0),
         (-90.0, -85.0), (-37.3, 81.2), (12.0, 12.05)],
    )
    def test_zone_area_identity(self, gridpath, lat0, lat1):
        """The weights of one band on a global grid sum to the zone's area."""
        grid = ux.open_grid(gridpath("ugrid", "outCSne30", "outCSne30.ug"))
        ((_, w),) = _compute_face_band_weights(grid, np.array([lat0, lat1]))
        nt.assert_allclose(w.sum(), _zone_area(lat0, lat1), rtol=1e-9)

    def test_overlaps_partition_face_area(self, gridpath):
        """A face's overlaps with a set of bands tiling the sphere sum to its area."""
        grid = ux.open_grid(gridpath("ugrid", "outCSne30", "outCSne30.ug"))
        totals = np.zeros(grid.n_face)
        for idx, w in _compute_face_band_weights(grid, np.arange(-90.0, 91.0, 7.5)):
            totals[idx] += w
        nt.assert_allclose(totals, grid.face_areas.values, rtol=1e-9)

    def test_contained_face_matches_face_area(self, gridpath):
        """For a face inside the band, the overlap area is the face area."""
        grid = ux.open_grid(gridpath("ugrid", "outCSne30", "outCSne30.ug"))
        faces = grid.get_faces_between_latitudes((20.0, 60.0))
        assert faces.size > 0
        n_nodes = grid.n_nodes_per_face.values
        edges = _get_cartesian_face_edge_nodes_array_subset(
            faces,
            grid.face_node_connectivity.values,
            n_nodes,
            grid.n_max_face_nodes,
            grid.node_x.values,
            grid.node_y.values,
            grid.node_z.values,
        )
        areas = [
            _band_area(edges[k, : n_nodes[f]], 20.0, 60.0) for k, f in enumerate(faces)
        ]
        nt.assert_allclose(areas, grid.face_areas.values[faces], rtol=1e-9)

    @pytest.mark.parametrize("lat0, lat1", [(-90.0, 90.0), (12.0, 33.0), (25.0, 28.0)])
    def test_invariant_under_rotation_and_orientation(self, lat0, lat1):
        """Rotating a face across the antimeridian or reversing it changes nothing."""
        lat = np.array([10.0, 15.0, 35.0, 30.0])
        lon = np.array([-8.0, 9.0, 7.0, -6.0])
        expected = _band_area(_face_edges(lat, lon), lat0, lat1)
        assert expected > 0.0
        for shift in (90.0, 180.0, 183.0, 260.0):
            fe = _face_edges(lat, lon + shift)
            assert _band_area(fe, lat0, lat1) == pytest.approx(expected, rel=1e-12)
            fe_cw = _face_edges(lat[::-1], lon[::-1] + shift)
            assert _band_area(fe_cw, lat0, lat1) == pytest.approx(expected, rel=1e-12)

    @pytest.mark.parametrize("hemisphere", [1.0, -1.0])
    @pytest.mark.parametrize("reverse", [False, True])
    def test_face_enclosing_pole(self, hemisphere, reverse):
        """A face around a pole (no vertex on it) contains the polar cap."""
        lat = hemisphere * np.array([80.0, 80.0, 80.0, 80.0])
        lon = np.array([45.0, 135.0, 225.0, 315.0])
        if reverse:
            lat, lon = lat[::-1], lon[::-1]
        fe = _face_edges(lat, lon)
        x, y, z = (fe[:, 0, i] for i in range(3))
        face_area, _ = calculate_face_area(x, y, z, order=6)

        def band(a, b):
            return _band_area(fe, *sorted((hemisphere * a, hemisphere * b)))

        # Edges peak at ~82.9 deg, so the 85-90 cap lies entirely inside the face.
        assert band(85.0, 90.0) == pytest.approx(_zone_area(85.0, 90.0), rel=1e-12)
        assert band(60.0, 75.0) == 0.0
        assert band(-90.0, 90.0) == pytest.approx(face_area, rel=1e-9)
        parts = band(70.0, 81.0) + band(81.0, 84.0) + band(84.0, 90.0)
        assert parts == pytest.approx(face_area, rel=1e-9)

    def test_face_with_vertex_at_pole(self):
        """A face with a vertex on the pole turns there between meridians."""
        fe = _face_edges(np.array([90.0, 80.0, 80.0]), np.array([0.0, 0.0, 90.0]))
        x, y, z = (fe[:, 0, i] for i in range(3))
        face_area, _ = calculate_face_area(x, y, z, order=6)

        assert _band_area(fe, -90.0, 90.0) == pytest.approx(face_area, rel=1e-9)
        # The edge opposite the pole peaks at ~82.9 deg, so the face covers a
        # quarter of the 85-90 cap.
        quarter_cap = _zone_area(85.0, 90.0) / 4
        assert _band_area(fe, 85.0, 90.0) == pytest.approx(quarter_cap, rel=1e-12)
        parts = _band_area(fe, 70.0, 82.0) + _band_area(fe, 82.0, 90.0)
        assert parts == pytest.approx(face_area, rel=1e-9)

    def test_face_touching_band_has_zero_overlap(self):
        """A face sharing only an edge with the band boundary has zero overlap."""
        fe = _face_edges(np.array([0.0, 0.0, 3.0, 3.0]), np.array([0.0, 3.0, 3.0, 0.0]))
        assert _band_area(fe, -10.0, 0.0) == 0.0
        assert _band_area(fe, 0.0, 10.0) > 0.0
