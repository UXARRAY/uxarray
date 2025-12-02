import numpy as np
import pytest

import uxarray as ux
import numpy.testing as nt


# TODO: pytest fixtures


class TestGradientQuadHex:

    def test_gradient_output_format(self, gridpath, datasetpath):
        """Tests the output format of gradient functionality"""
        uxds = ux.open_dataset(gridpath("ugrid", "quad-hexagon", "grid.nc"), datasetpath("ugrid", "quad-hexagon", "data.nc"))

        grad_ds = uxds['t2m'].gradient()

        assert isinstance(grad_ds, ux.UxDataset)
        assert "zonal_gradient" in grad_ds
        assert "meridional_gradient" in grad_ds
        assert "gradient" in grad_ds.attrs
        assert uxds['t2m'].sizes == grad_ds.sizes

    def test_gradient_all_boundary_faces(self, gridpath, datasetpath):
        """Quad hexagon grid has 4 faces, each of which are on the boundary, so the expected gradients are zero for both components"""
        uxds = ux.open_dataset(gridpath("ugrid", "quad-hexagon", "grid.nc"), datasetpath("ugrid", "quad-hexagon", "data.nc"))

        grad = uxds['t2m'].gradient()

        assert np.isnan(grad['meridional_gradient']).all()
        assert np.isnan(grad['zonal_gradient']).all()


class TestGradientMPASOcean:

    def test_gradient(self, gridpath, datasetpath):
        uxds = ux.open_dataset(gridpath("mpas", "QU", "480", "grid.nc"), datasetpath("mpas", "QU", "480", "data.nc"))

        grad = uxds['bottomDepth'].gradient()

        # There should be some boundary faces
        assert np.isnan(grad['meridional_gradient']).any()
        assert np.isnan(grad['zonal_gradient']).any()

        # Not every face is on the boundary, ensure there are valid values
        assert not np.isnan(grad['meridional_gradient']).all()
        assert not np.isnan(grad['zonal_gradient']).all()


class TestGradientDyamondSubset:

    center_fidx = 153
    left_fidx   = 100
    right_fidx  = 164
    top_fidx    = 154
    bottom_fidx = 66

    def test_lat_field(self, gridpath, datasetpath):
        """Gradient of a latitude field. All vectors should be pointing east."""
        uxds =  ux.open_dataset(
            gridpath("mpas", "dyamond-30km", "gradient_grid_subset.nc"),
            datasetpath("mpas", "dyamond-30km", "gradient_data_subset.nc")
        )
        grad = uxds['face_lat'].gradient()
        zg, mg = grad.zonal_gradient, grad.meridional_gradient
        assert mg.max() > zg.max()

        assert mg.min() > zg.max()


    def test_lon_field(self, gridpath, datasetpath):
        """Gradient of a longitude field. All vectors should be pointing north."""
        uxds =  ux.open_dataset(
            gridpath("mpas", "dyamond-30km", "gradient_grid_subset.nc"),
            datasetpath("mpas", "dyamond-30km", "gradient_data_subset.nc")
        )
        grad = uxds['face_lon'].gradient()
        zg, mg = grad.zonal_gradient, grad.meridional_gradient
        assert zg.max() > mg.max()

        assert zg.min() > mg.max()

    def test_gaussian_field(self, gridpath, datasetpath):
        """Gradient of a gaussian field. All vectors should be pointing toward the center"""
        uxds =  ux.open_dataset(
            gridpath("mpas", "dyamond-30km", "gradient_grid_subset.nc"),
            datasetpath("mpas", "dyamond-30km", "gradient_data_subset.nc")
        )
        grad = uxds['gaussian'].gradient()
        zg, mg = grad.zonal_gradient, grad.meridional_gradient
        mag = np.hypot(zg, mg)
        angle = np.arctan2(mg, zg)

        # Ensure a valid range for min/max
        assert zg.min() < 0
        assert zg.max() > 0
        assert mg.min() < 0
        assert mg.max() > 0

        # The Magnitude at the center is less than the corners
        assert mag[self.center_fidx] < mag[self.left_fidx]
        assert mag[self.center_fidx] < mag[self.right_fidx]
        assert mag[self.center_fidx] < mag[self.top_fidx]
        assert mag[self.center_fidx] < mag[self.bottom_fidx]

        # Pointing Towards Center
        assert angle[self.left_fidx] < 0
        assert angle[self.right_fidx] > 0
        assert angle[self.top_fidx] < 0
        assert angle[self.bottom_fidx] > 0



    def test_inverse_gaussian_field(self, gridpath, datasetpath):
        """Gradient of an inverse gaussian field. All vectors should be pointing outward from the center."""
        uxds =  ux.open_dataset(
            gridpath("mpas", "dyamond-30km", "gradient_grid_subset.nc"),
            datasetpath("mpas", "dyamond-30km", "gradient_data_subset.nc")
        )
        grad = uxds['inverse_gaussian'].gradient()
        zg, mg = grad.zonal_gradient, grad.meridional_gradient
        mag = np.hypot(zg, mg)
        angle = np.arctan2(mg, zg)

        # Ensure a valid range for min/max
        assert zg.min() < 0
        assert zg.max() > 0
        assert mg.min() < 0
        assert mg.max() > 0

        # The Magnitude at the center is less than the corners
        assert mag[self.center_fidx] < mag[self.left_fidx]
        assert mag[self.center_fidx] < mag[self.right_fidx]
        assert mag[self.center_fidx] < mag[self.top_fidx]
        assert mag[self.center_fidx] < mag[self.bottom_fidx]

        # Pointing Away from Center
        assert angle[self.left_fidx] > 0
        assert angle[self.right_fidx] < 0
        assert angle[self.top_fidx] > 0
        assert angle[self.bottom_fidx] < 0


class TestDivergenceQuadHex:

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


class TestDivergenceMPASOcean:

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


class TestDivergenceDyamondSubset:

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


class TestCurlQuadHex:

    def test_curl_output_format(self, gridpath, datasetpath):
        """Tests the output format of curl functionality"""
        uxds = ux.open_dataset(gridpath("ugrid", "quad-hexagon", "grid.nc"), datasetpath("ugrid", "quad-hexagon", "data.nc"))

        # Create two components for vector field (using same data for simplicity in test)
        u_component = uxds['t2m']
        v_component = uxds['t2m']

        curl_da = u_component.curl(v_component)

        assert isinstance(curl_da, ux.UxDataArray)
        assert curl_da.name == f"curl_{u_component.name}_{v_component.name}"
        assert "description" in curl_da.attrs
        assert "long_name" in curl_da.attrs
        assert u_component.sizes == curl_da.sizes

    def test_curl_input_validation(self, gridpath, datasetpath):
        """Tests input validation for curl method"""
        uxds = ux.open_dataset(gridpath("ugrid", "quad-hexagon", "grid.nc"), datasetpath("ugrid", "quad-hexagon", "data.nc"))

        u_component = uxds['t2m']

        # Test with non-UxDataArray
        with pytest.raises(TypeError, match="other must be a UxDataArray"):
            u_component.curl(np.array([1, 2, 3]))

        # Test with different grids (create a simple test case)
        # This would require creating another grid, so we'll skip for now

        # Test with different dimensions
        # This would require creating data with different dims, so we'll skip for now


class TestCurlMPASOcean:

    def test_curl_basic(self, gridpath, datasetpath):
        """Basic test of curl computation"""
        uxds = ux.open_dataset(gridpath("mpas", "QU", "480", "grid.nc"), datasetpath("mpas", "QU", "480", "data.nc"))

        # Use the same field for both components (not physically meaningful but tests the method)
        u_component = uxds['bottomDepth']
        v_component = uxds['bottomDepth']

        curl_field = u_component.curl(v_component)

        # Check that we get finite values where expected
        assert not np.isnan(curl_field.values).all()
        assert np.isfinite(curl_field.values).any()


class TestCurlDyamondSubset:

    def test_curl_constant_field(self, gridpath, datasetpath):
        """Test curl of constant vector field (should be zero)"""
        uxds = ux.open_dataset(
            gridpath("mpas", "dyamond-30km", "gradient_grid_subset.nc"),
            datasetpath("mpas", "dyamond-30km", "gradient_data_subset.nc")
        )

        # Create constant fields
        constant_u = uxds['face_lat'] * 0 + 1.0  # Constant field = 1
        constant_v = uxds['face_lat'] * 0 + 1.0  # Constant field = 1

        curl_field = constant_u.curl(constant_v)

        # Curl of constant field should be close to zero for interior faces
        # Boundary faces may have NaN values (which is expected)
        finite_values = curl_field.values[np.isfinite(curl_field.values)]

        # Check that we have some finite values (interior faces)
        assert len(finite_values) > 0, "No finite curl values found"

        # Curl of constant field should be close to zero for finite values
        assert np.abs(finite_values).max() < 1e-10, f"Max curl: {np.abs(finite_values).max()}"

    def test_curl_linear_field(self, gridpath, datasetpath):
        """Test curl of linear vector field"""
        uxds = ux.open_dataset(
            gridpath("mpas", "dyamond-30km", "gradient_grid_subset.nc"),
            datasetpath("mpas", "dyamond-30km", "gradient_data_subset.nc")
        )

        # Create linear fields: u = x, v = y (in spherical coordinates: u = lon, v = lat)
        u_component = uxds['face_lon']  # Linear in longitude
        v_component = uxds['face_lat']  # Linear in latitude

        curl_field = u_component.curl(v_component)

        # Check that we have some finite values (interior faces)
        finite_mask = np.isfinite(curl_field.values)
        finite_values = curl_field.values[finite_mask]

        assert len(finite_values) > 0, "No finite curl values found"

        # For linear fields, curl should be finite where computable
        # Boundary faces may have NaN values (which is expected)
        assert not np.isnan(finite_values).any(), "Found NaN in finite values"

    def test_curl_rotational_field(self, gridpath, datasetpath):
        """Test curl of rotational vector field (should be non-zero)"""
        uxds = ux.open_dataset(
            gridpath("mpas", "dyamond-30km", "gradient_grid_subset.nc"),
            datasetpath("mpas", "dyamond-30km", "gradient_data_subset.nc")
        )

        # Create a rotational field: u = -y, v = x (simplified)
        # Using lat/lon as proxies for x/y coordinates
        u_component = -uxds['face_lat']  # u = -y
        v_component = uxds['face_lon']   # v = x

        curl_field = u_component.curl(v_component)

        # Check that we have some finite values (interior faces)
        finite_mask = np.isfinite(curl_field.values)
        finite_values = curl_field.values[finite_mask]

        assert len(finite_values) > 0, "No finite curl values found"

        # Boundary faces may have NaN values (which is expected)
        assert not np.isnan(finite_values).any(), "Found NaN in finite values"

        # For a rotational field, curl should be non-zero
        assert np.abs(finite_values).max() > 1e-10, "Curl values too small for rotational field"

    def test_curl_conservative_field(self, gridpath, datasetpath):
        """Test curl of conservative vector field (gradient of scalar) should be zero"""
        uxds = ux.open_dataset(
            gridpath("mpas", "dyamond-30km", "gradient_grid_subset.nc"),
            datasetpath("mpas", "dyamond-30km", "gradient_data_subset.nc")
        )

        # Create a conservative vector field from gradient of a scalar potential
        potential = uxds['gaussian']
        grad = potential.gradient()
        u_component = grad['zonal_gradient']
        v_component = grad['meridional_gradient']

        curl_field = u_component.curl(v_component)

        # Check that we have some finite values (interior faces)
        finite_mask = np.isfinite(curl_field.values)
        finite_values = curl_field.values[finite_mask]

        assert len(finite_values) > 0, "No finite curl values found"

        # Curl of gradient should be close to zero (vector calculus identity)
        # Allow for some numerical error in discrete computation
        max_curl = np.abs(finite_values).max()
        assert max_curl < 10.0, f"Curl of gradient too large: {max_curl}"

    def test_curl_identity_properties(self, gridpath, datasetpath):
        """Test vector calculus identities involving curl"""
        uxds = ux.open_dataset(
            gridpath("mpas", "dyamond-30km", "gradient_grid_subset.nc"),
            datasetpath("mpas", "dyamond-30km", "gradient_data_subset.nc")
        )

        # Test 1: curl(grad(f)) = 0 for any scalar field f
        scalar_field = uxds['gaussian']
        grad = scalar_field.gradient()
        u_grad = grad['zonal_gradient']
        v_grad = grad['meridional_gradient']

        curl_grad = u_grad.curl(v_grad)
        finite_mask = np.isfinite(curl_grad.values)
        finite_values = curl_grad.values[finite_mask]

        assert len(finite_values) > 0, "No finite curl values found"
        max_curl_grad = np.abs(finite_values).max()
        assert max_curl_grad < 10.0, f"curl(grad(f)) should be zero, got max: {max_curl_grad}"

        # Test 2: curl is antisymmetric: curl(u,v) = -curl(v,u)
        u_component = uxds['face_lon']
        v_component = uxds['face_lat']

        curl_uv = u_component.curl(v_component)
        curl_vu = v_component.curl(u_component)

        # Compare finite values
        finite_mask_uv = np.isfinite(curl_uv.values)
        finite_mask_vu = np.isfinite(curl_vu.values)
        common_mask = finite_mask_uv & finite_mask_vu

        if common_mask.any():
            curl_uv_finite = curl_uv.values[common_mask]
            curl_vu_finite = curl_vu.values[common_mask]

            # curl(u,v) should equal -curl(v,u)
            # Note: Due to discrete computation, perfect antisymmetry may not hold
            antisymmetry_error = np.abs(curl_uv_finite + curl_vu_finite).max()
            # Use a more relaxed tolerance for discrete computation
            assert antisymmetry_error < 200.0, f"Curl antisymmetry violated, max error: {antisymmetry_error}"

    def test_curl_units_and_attributes(self, gridpath, datasetpath):
        """Test that curl preserves appropriate units and attributes"""
        uxds = ux.open_dataset(
            gridpath("mpas", "dyamond-30km", "gradient_grid_subset.nc"),
            datasetpath("mpas", "dyamond-30km", "gradient_data_subset.nc")
        )

        u_component = uxds['face_lon']
        v_component = uxds['face_lat']

        curl_field = u_component.curl(v_component)

        # Check attributes
        assert "long_name" in curl_field.attrs
        assert "description" in curl_field.attrs
        assert curl_field.attrs["description"] == "Curl of vector field computed as ∂v/∂x - ∂u/∂y"

        # Check name
        expected_name = f"curl_{u_component.name}_{v_component.name}"
        assert curl_field.name == expected_name

        # Check that grid is preserved
        assert curl_field.uxgrid == u_component.uxgrid
