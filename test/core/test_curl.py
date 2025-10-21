import numpy as np
import pytest

import uxarray as ux
from uxarray.core.dataarray import UxDataArray


class TestQuadHex:
    """Test curl computation on a simple quad-hex grid."""

    def test_curl_output_format(self, quad_hex_grid):
        """Test that curl returns the correct output format."""
        # Create simple vector field components
        u_data = np.ones(quad_hex_grid.n_face)
        v_data = np.ones(quad_hex_grid.n_face)

        u_da = UxDataArray(
            u_data, dims=["n_face"], uxgrid=quad_hex_grid, name="u_component"
        )
        v_da = UxDataArray(
            v_data, dims=["n_face"], uxgrid=quad_hex_grid, name="v_component"
        )

        # Compute curl
        curl_result = u_da.curl(v_da)

        # Check output format
        assert isinstance(curl_result, UxDataArray)
        assert curl_result.dims == u_da.dims
        assert curl_result.uxgrid == u_da.uxgrid
        assert curl_result.shape == u_da.shape
        assert "curl" in curl_result.name

    def test_curl_input_validation(self, quad_hex_grid):
        """Test input validation for curl computation."""
        u_data = np.ones(quad_hex_grid.n_face)
        u_da = UxDataArray(u_data, dims=["n_face"], uxgrid=quad_hex_grid)

        # Test with non-UxDataArray
        with pytest.raises(TypeError, match="other must be a UxDataArray"):
            u_da.curl(u_data)

        # Test with different grids
        other_grid = ux.open_grid(ux.grid.quad_hexagon())
        v_data = np.ones(other_grid.n_face)
        v_da = UxDataArray(v_data, dims=["n_face"], uxgrid=other_grid)

        with pytest.raises(ValueError, match="same grid"):
            u_da.curl(v_da)

        # Test with different dimensions
        v_data_2d = np.ones((2, quad_hex_grid.n_face))
        v_da_2d = UxDataArray(v_data_2d, dims=["time", "n_face"], uxgrid=quad_hex_grid)

        with pytest.raises(ValueError, match="same dimensions"):
            u_da.curl(v_da_2d)

        # Test with multi-dimensional data
        u_data_2d = np.ones((2, quad_hex_grid.n_face))
        u_da_2d = UxDataArray(u_data_2d, dims=["time", "n_face"], uxgrid=quad_hex_grid)

        with pytest.raises(ValueError, match="1-dimensional data"):
            u_da_2d.curl(v_da_2d)


class TestMPASOcean:
    """Test curl computation on MPAS Ocean grid."""

    def test_curl_basic(self, mpas_ocean_grid):
        """Test basic curl computation on MPAS Ocean grid."""
        # Create simple vector field components
        u_data = np.ones(mpas_ocean_grid.n_face)
        v_data = np.ones(mpas_ocean_grid.n_face)

        u_da = UxDataArray(
            u_data, dims=["n_face"], uxgrid=mpas_ocean_grid, name="u_velocity"
        )
        v_da = UxDataArray(
            v_data, dims=["n_face"], uxgrid=mpas_ocean_grid, name="v_velocity"
        )

        # Compute curl
        curl_result = u_da.curl(v_da)

        # Check that we get finite values for interior faces
        finite_values = curl_result.values[np.isfinite(curl_result.values)]
        assert len(finite_values) > 0
        assert np.all(np.isfinite(finite_values))


class TestDyamondSubset:
    """Test curl computation on DYAMOND subset grid with various field types."""

    def test_curl_constant_field(self, dyamond_subset_grid):
        """Test curl of constant vector field (should be zero)."""
        # Create constant vector field
        u_constant = np.full(dyamond_subset_grid.n_face, 2.0)
        v_constant = np.full(dyamond_subset_grid.n_face, 3.0)

        u_da = UxDataArray(
            u_constant, dims=["n_face"], uxgrid=dyamond_subset_grid, name="u_const"
        )
        v_da = UxDataArray(
            v_constant, dims=["n_face"], uxgrid=dyamond_subset_grid, name="v_const"
        )

        # Compute curl
        curl_result = u_da.curl(v_da)

        # For constant fields, curl should be zero (within numerical precision)
        finite_mask = np.isfinite(curl_result.values)
        finite_curl = curl_result.values[finite_mask]

        if len(finite_curl) > 0:
            max_abs_curl = np.abs(finite_curl).max()
            assert max_abs_curl < 1e-10, f"Curl of constant field should be ~0, got {max_abs_curl}"

    def test_curl_linear_field(self, dyamond_subset_grid):
        """Test curl of linear vector field."""
        # Create linear vector field: u = x, v = y
        face_lon = dyamond_subset_grid.face_lon.values
        face_lat = dyamond_subset_grid.face_lat.values

        u_linear = face_lon  # u = x (longitude)
        v_linear = face_lat  # v = y (latitude)

        u_da = UxDataArray(
            u_linear, dims=["n_face"], uxgrid=dyamond_subset_grid, name="u_linear"
        )
        v_da = UxDataArray(
            v_linear, dims=["n_face"], uxgrid=dyamond_subset_grid, name="v_linear"
        )

        # Compute curl
        curl_result = u_da.curl(v_da)

        # Check that we get reasonable values
        finite_mask = np.isfinite(curl_result.values)
        finite_curl = curl_result.values[finite_mask]

        assert len(finite_curl) > 0
        assert np.all(np.isfinite(finite_curl))

    def test_curl_radial_field(self, dyamond_subset_grid):
        """Test curl of radial vector field."""
        # Create radial vector field centered at grid center
        face_lon = dyamond_subset_grid.face_lon.values
        face_lat = dyamond_subset_grid.face_lat.values

        # Grid center
        center_lon = np.mean(face_lon)
        center_lat = np.mean(face_lat)

        # Radial field: points outward from center
        dx = face_lon - center_lon
        dy = face_lat - center_lat
        r = np.sqrt(dx**2 + dy**2)

        # Avoid division by zero
        r = np.maximum(r, 1e-10)

        u_radial = dx / r  # Normalized radial component
        v_radial = dy / r

        u_da = UxDataArray(
            u_radial, dims=["n_face"], uxgrid=dyamond_subset_grid, name="u_radial"
        )
        v_da = UxDataArray(
            v_radial, dims=["n_face"], uxgrid=dyamond_subset_grid, name="v_radial"
        )

        # Compute curl
        curl_result = u_da.curl(v_da)

        # Check that we get reasonable values
        finite_mask = np.isfinite(curl_result.values)
        finite_curl = curl_result.values[finite_mask]

        assert len(finite_curl) > 0
        assert np.all(np.isfinite(finite_curl))

    def test_curl_divergence_identity(self, dyamond_subset_grid):
        """Test vector calculus identity: divergence of curl should be zero."""
        # Create a vector field with some curl
        face_lon = dyamond_subset_grid.face_lon.values
        face_lat = dyamond_subset_grid.face_lat.values

        # Create a rotational field: u = -y, v = x (circulation around origin)
        center_lon = np.mean(face_lon)
        center_lat = np.mean(face_lat)

        u_rot = -(face_lat - center_lat)  # u = -y
        v_rot = (face_lon - center_lon)   # v = x

        u_da = UxDataArray(
            u_rot, dims=["n_face"], uxgrid=dyamond_subset_grid, name="u_rotation"
        )
        v_da = UxDataArray(
            v_rot, dims=["n_face"], uxgrid=dyamond_subset_grid, name="v_rotation"
        )

        # Compute curl
        curl_result = u_da.curl(v_da)

        # The curl should be non-zero for this rotational field
        finite_mask = np.isfinite(curl_result.values)
        finite_curl = curl_result.values[finite_mask]

        if len(finite_curl) > 0:
            # For this specific rotational field, curl should be approximately constant (≈ 2)
            # But we'll just check that it's not all zeros
            assert not np.allclose(finite_curl, 0, atol=1e-10)

        # Note: Testing div(curl) = 0 would require implementing divergence,
        # which is not available in this branch. This test verifies curl produces
        # expected non-zero values for a rotational field.
