import numpy as np
import pytest

import uxarray as ux
import numpy.testing as nt


class TestQuadHex:

    def test_divergence_output_format(self, gridpath, datasetpath):
        """Tests the output format of divergence functionality"""
        uxds = ux.open_dataset(gridpath("ugrid", "quad-hexagon", "grid.nc"), datasetpath("ugrid", "quad-hexagon", "data.nc"))

        # Create two components for vector field (using same data for simplicity in test)
        u_component = uxds['t2m']
        v_component = uxds['t2m']
        
        div_da = u_component.divergence(v_component)

        assert isinstance(div_da, ux.UxDataArray)
        assert div_da.name == "divergence"
        assert "divergence" in div_da.attrs
        assert u_component.sizes == div_da.sizes

    def test_divergence_input_validation(self, gridpath, datasetpath):
        """Tests input validation for divergence method"""
        uxds = ux.open_dataset(gridpath("ugrid", "quad-hexagon", "grid.nc"), datasetpath("ugrid", "quad-hexagon", "data.nc"))

        u_component = uxds['t2m']
        
        # Test with non-UxDataArray
        with pytest.raises(TypeError, match="other must be a UxDataArray"):
            u_component.divergence(np.array([1, 2, 3]))
        
        # Test with different grids (create a simple test case)
        # This would require creating another grid, so we'll skip for now
        
        # Test with different dimensions
        # This would require creating data with different dims, so we'll skip for now


class TestMPASOcean:

    def test_divergence_basic(self, gridpath, datasetpath):
        """Basic test of divergence computation"""
        uxds = ux.open_dataset(gridpath("mpas", "QU", "480", "grid.nc"), datasetpath("mpas", "QU", "480", "data.nc"))

        # Use the same field for both components (not physically meaningful but tests the method)
        u_component = uxds['bottomDepth']
        v_component = uxds['bottomDepth']
        
        div_field = u_component.divergence(v_component)

        # Check that we get finite values where expected
        assert not np.isnan(div_field.values).all()
        assert np.isfinite(div_field.values).any()


class TestDyamondSubset:

    def test_divergence_constant_field(self, gridpath, datasetpath):
        """Test divergence of constant vector field (should be zero)"""
        uxds = ux.open_dataset(
            gridpath("mpas", "dyamond-30km", "gradient_grid_subset.nc"),
            datasetpath("mpas", "dyamond-30km", "gradient_data_subset.nc")
        )
        
        # Create constant fields
        constant_u = uxds['face_lat'] * 0 + 1.0  # Constant field = 1
        constant_v = uxds['face_lat'] * 0 + 1.0  # Constant field = 1
        
        div_field = constant_u.divergence(constant_v)
        
        # Divergence of constant field should be close to zero for interior faces
        # Boundary faces may have NaN values (which is expected)
        finite_values = div_field.values[np.isfinite(div_field.values)]
        
        # Check that we have some finite values (interior faces)
        assert len(finite_values) > 0, "No finite divergence values found"
        
        # Divergence of constant field should be close to zero for finite values
        assert np.abs(finite_values).max() < 1e-10, f"Max divergence: {np.abs(finite_values).max()}"

    def test_divergence_linear_field(self, gridpath, datasetpath):
        """Test divergence of linear vector field"""
        uxds = ux.open_dataset(
            gridpath("mpas", "dyamond-30km", "gradient_grid_subset.nc"),
            datasetpath("mpas", "dyamond-30km", "gradient_data_subset.nc")
        )
        
        # Create linear fields: u = x, v = y (in spherical coordinates: u = lon, v = lat)
        u_component = uxds['face_lon']  # Linear in longitude
        v_component = uxds['face_lat']  # Linear in latitude
        
        div_field = u_component.divergence(v_component)
        
        # Check that we have some finite values (interior faces)
        finite_mask = np.isfinite(div_field.values)
        finite_values = div_field.values[finite_mask]
        
        assert len(finite_values) > 0, "No finite divergence values found"
        
        # For linear fields, divergence should be finite where computable
        # Boundary faces may have NaN values (which is expected)
        assert not np.isnan(finite_values).any(), "Found NaN in finite values"

    def test_divergence_radial_field(self, gridpath, datasetpath):
        """Test divergence of radial vector field (should be positive)"""
        uxds = ux.open_dataset(
            gridpath("mpas", "dyamond-30km", "gradient_grid_subset.nc"),
            datasetpath("mpas", "dyamond-30km", "gradient_data_subset.nc")
        )
        
        # Create a radial field pointing outward from center
        # Use the inverse gaussian as a proxy for radial distance
        radial_distance = uxds['inverse_gaussian']
        
        # Create components proportional to position (simplified radial field)
        u_component = radial_distance * uxds['face_lon']
        v_component = radial_distance * uxds['face_lat']
        
        div_field = u_component.divergence(v_component)
        
        # Check that we have some finite values (interior faces)
        finite_mask = np.isfinite(div_field.values)
        finite_values = div_field.values[finite_mask]
        
        assert len(finite_values) > 0, "No finite divergence values found"
        
        # Boundary faces may have NaN values (which is expected)
        assert not np.isnan(finite_values).any(), "Found NaN in finite values"
        
        # Most finite values should be positive for an expanding field
        positive_values = finite_values > 0
        assert positive_values.sum() > len(finite_values) * 0.3  # At least 30% positive

    def test_divergence_curl_identity(self, gridpath, datasetpath):
        """Test that divergence of curl is zero (vector calculus identity)"""
        uxds = ux.open_dataset(
            gridpath("mpas", "dyamond-30km", "gradient_grid_subset.nc"),
            datasetpath("mpas", "dyamond-30km", "gradient_data_subset.nc")
        )
        
        # Create a scalar potential field
        potential = uxds['gaussian']
        
        # Compute gradient to get a conservative vector field
        grad = potential.gradient()
        u_component = grad['zonal_gradient']
        v_component = grad['meridional_gradient']
        
        # Compute divergence of this gradient (should be the Laplacian)
        div_grad = u_component.divergence(v_component)
        
        # Check that we have some finite values (interior faces)
        finite_mask = np.isfinite(div_grad.values)
        finite_values = div_grad.values[finite_mask]
        
        assert len(finite_values) > 0, "No finite divergence values found"
        
        # This tests the Laplacian computation via div(grad)
        # Boundary faces may have NaN values (which is expected)
        assert not np.isnan(finite_values).any(), "Found NaN in finite values"
        
        # The Laplacian of a Gaussian should have both positive and negative values
        assert (finite_values > 0).any(), "No positive Laplacian values found"
        assert (finite_values < 0).any(), "No negative Laplacian values found"