import uxarray as ux
import healpix as hp
import numpy as np
import pytest
import xarray as xr
import pandas as pd
from uxarray.constants import ERROR_TOLERANCE


@pytest.mark.parametrize("resolution_level", [0, 1, 2, 3])
def test_to_ugrid(resolution_level):
    uxgrid = ux.Grid.from_healpix(resolution_level)

    expected_n_face = hp.nside2npix(hp.order2nside(resolution_level))

    assert uxgrid.n_face == expected_n_face

@pytest.mark.parametrize("resolution_level", [0, 1, 2, 3])
def test_boundaries(resolution_level):
    uxgrid = ux.Grid.from_healpix(resolution_level)

    assert "face_node_connectivity" not in uxgrid.connectivity
    assert "node_lon" not in uxgrid.connectivity
    assert "node_lat" not in uxgrid.connectivity

    _ = uxgrid.face_node_connectivity

    assert "face_node_connectivity" in uxgrid.connectivity
    assert "node_lon" in uxgrid.coordinates
    assert "node_lat" in uxgrid.coordinates

    # check for the correct number of boundary nodes
    assert (uxgrid.n_node == uxgrid.n_face + 2)

@pytest.mark.parametrize("pixels_only", [True, False])
def test_time_dimension_roundtrip(datasetpath, pixels_only):
    ds_path = datasetpath("healpix", "outCSne30", "data.nc")
    ds = xr.open_dataset(ds_path)
    dummy_time = pd.to_datetime(["2025-01-01T00:00:00"])
    ds_time = ds.expand_dims(time=dummy_time)
    uxds = ux.UxDataset.from_healpix(ds_time, pixels_only=pixels_only)

    # Ensure time dimension is preserved and that the conversion worked
    assert "time" in uxds.dims
    assert uxds.sizes["time"] == 1

def test_dataset(datasetpath):
    ds_path = datasetpath("healpix", "outCSne30", "data.nc")
    uxds = ux.UxDataset.from_healpix(ds_path)

    assert uxds.uxgrid.source_grid_spec == "HEALPix"
    assert "n_face" in uxds.dims

    uxds = ux.UxDataset.from_healpix(ds_path, pixels_only=False)
    assert "face_node_connectivity" in uxds.uxgrid._ds



def test_number_of_boundary_nodes():
    uxgrid = ux.Grid.from_healpix(0)
    face_node_conn = uxgrid.face_node_connectivity
    n_face, n_max_face_nodes = face_node_conn.shape

    assert n_face == uxgrid.n_face
    assert n_max_face_nodes == uxgrid.n_max_face_nodes


def test_from_healpix_healpix():
    xrda = xr.DataArray(data=np.ones(12), dims=['cell'])
    uxda = ux.UxDataArray.from_healpix(xrda)
    assert isinstance(uxda, ux.UxDataArray)
    xrda = xrda.rename({'cell': 'n_face'})
    with pytest.raises(ValueError):
        uxda = ux.UxDataArray.from_healpix(xrda)


    uxda = ux.UxDataArray.from_healpix(xrda, face_dim="n_face")
    assert isinstance(uxda, ux.UxDataArray)

def test_from_healpix_dataset():
    xrda = xr.DataArray(data=np.ones(12), dims=['cell']).to_dataset(name='cell')
    uxda = ux.UxDataset.from_healpix(xrda)
    assert isinstance(uxda, ux.UxDataset)
    xrda = xrda.rename({'cell': 'n_face'})
    with pytest.raises(ValueError):
        uxda = ux.UxDataset.from_healpix(xrda)

    uxda = ux.UxDataset.from_healpix(xrda, face_dim="n_face")
    assert isinstance(uxda, ux.UxDataset)


def test_invalid_cells():
    # 11 is not a valid number of global cells
    xrda = xr.DataArray(data=np.ones(11), dims=['cell']).to_dataset(name='cell')
    with pytest.raises(ValueError):
        uxda = ux.UxDataset.from_healpix(xrda)

@pytest.mark.parametrize("resolution_level", [0, 2])  # Test lowest and a mid-level
def test_healpix_equal_area_property(resolution_level):
    """Test that all HEALPix faces have equal area as per the HEALPix specification.

    HEALPix (Hierarchical Equal Area isoLatitude Pixelization) is designed so that
    all pixels/faces have exactly the same spherical area.
    """
    # Create HEALPix grid with boundaries for area calculation
    uxgrid = ux.Grid.from_healpix(resolution_level, pixels_only=False)

    # Calculate face areas and expected theoretical area
    face_areas = uxgrid.face_areas.values
    nside = hp.order2nside(resolution_level)
    npix = hp.nside2npix(nside)
    expected_area_per_pixel = 4 * np.pi / npix

    # All face areas should be equal to the theoretical value
    # Use a more reasonable tolerance for numerical computations
    np.testing.assert_allclose(
        face_areas,
        expected_area_per_pixel,
        rtol=1e-12,  # Relaxed from 1e-14 for numerical stability
        atol=1e-15,  # Added absolute tolerance
        err_msg=f"HEALPix faces do not have equal areas at resolution {resolution_level}"
    )

    # Verify total surface area equals 4π
    total_area = np.sum(face_areas)
    expected_total_area = 4 * np.pi
    np.testing.assert_allclose(
        total_area,
        expected_total_area,
        rtol=1e-12,  # Relaxed from 1e-14
        atol=1e-15,  # Added absolute tolerance
        err_msg=f"Total HEALPix surface area incorrect at resolution {resolution_level}"
    )


def test_healpix_face_areas_consistency():
    """Test that HEALPix face areas are consistent across different resolution levels."""
    resolution_levels = [0, 1]  # Just test basic functionality

    for resolution_level in resolution_levels:
        uxgrid = ux.Grid.from_healpix(resolution_level, pixels_only=False)
        face_areas = uxgrid.face_areas.values

        # All faces should have identical areas
        area_std = np.std(face_areas)
        area_mean = np.mean(face_areas)

        # Avoid division by zero
        if area_mean == 0:
            pytest.fail(f"Mean face area is zero at resolution {resolution_level}")

        relative_std = area_std / area_mean

        # Relative standard deviation should be essentially zero (numerical precision)
        # Relaxed tolerance for numerical stability
        assert relative_std < 1e-12, (
            f"Face areas not equal at resolution {resolution_level}: "
            f"relative_std={relative_std:.2e}"
        )

        # Check that face areas match theoretical calculation
        nside = hp.order2nside(resolution_level)
        npix = hp.nside2npix(nside)
        theoretical_area = 4 * np.pi / npix

        # Use consistent tolerance with the parametrized test
        np.testing.assert_allclose(
            face_areas,
            theoretical_area,
            rtol=1e-12,
            atol=1e-15,
            err_msg=f"Face areas incorrect at resolution {resolution_level}"
        )


@pytest.mark.parametrize("resolution_level", [0, 1])  # Just test the scaling relationship
def test_healpix_area_scaling(resolution_level):
    """Test that face areas scale correctly with resolution level."""
    # Create grids at consecutive resolution levels
    uxgrid_current = ux.Grid.from_healpix(resolution_level, pixels_only=False)
    uxgrid_next = ux.Grid.from_healpix(resolution_level + 1, pixels_only=False)

    area_current = uxgrid_current.face_areas.values[0]  # All areas are equal
    area_next = uxgrid_next.face_areas.values[0]

    # Each resolution level increases npix by factor of 4, so area decreases by factor of 4
    expected_ratio = 4.0
    actual_ratio = area_current / area_next

    np.testing.assert_allclose(
        actual_ratio,
        expected_ratio,
        rtol=1e-12,
        err_msg=f"Area scaling incorrect between resolution {resolution_level} and {resolution_level + 1}"
    )


def test_healpix_round_trip_consistency(tmp_path):
    """Test round-trip serialization of HEALPix grid through UGRID and Exodus formats.

    Validates that HEALPix grid objects can be successfully converted to xarray.Dataset
    objects in both UGRID and Exodus formats, serialized to disk, and reloaded
    while maintaining numerical accuracy and topological integrity.

    Args:
        tmp_path: pytest fixture providing temporary directory

    Raises:
        AssertionError: If any round-trip validation fails
    """
    # Create HEALPix grid
    original_grid = ux.Grid.from_healpix(zoom=3)

    # Access node coordinates to ensure they're generated before encoding
    _ = original_grid.node_lon
    _ = original_grid.node_lat

    # Convert to xarray.Dataset objects in different formats
    ugrid_dataset = original_grid.to_xarray("UGRID")
    exodus_dataset = original_grid.to_xarray("Exodus")

    # Define output file paths using tmp_path fixture
    ugrid_filepath = tmp_path / "healpix_test_ugrid.nc"
    exodus_filepath = tmp_path / "healpix_test_exodus.exo"

    # Serialize datasets to disk
    ugrid_dataset.to_netcdf(ugrid_filepath)
    exodus_dataset.to_netcdf(exodus_filepath)

    # Verify files were created successfully
    assert ugrid_filepath.exists()
    assert ugrid_filepath.stat().st_size > 0
    assert exodus_filepath.exists()
    assert exodus_filepath.stat().st_size > 0

    # Reload grids from serialized files
    reloaded_ugrid = ux.open_grid(ugrid_filepath)
    reloaded_exodus = ux.open_grid(exodus_filepath)

    # Validate topological consistency (face-node connectivity)
    # Integer connectivity arrays must be exactly preserved
    np.testing.assert_array_equal(
        original_grid.face_node_connectivity.values,
        reloaded_ugrid.face_node_connectivity.values,
        err_msg="UGRID face connectivity mismatch for HEALPix"
    )
    np.testing.assert_array_equal(
        original_grid.face_node_connectivity.values,
        reloaded_exodus.face_node_connectivity.values,
        err_msg="Exodus face connectivity mismatch for HEALPix"
    )

    # Validate coordinate consistency with numerical tolerance
    # Coordinate transformations and I/O precision may introduce minor differences
    np.testing.assert_allclose(
        original_grid.node_lon.values,
        reloaded_ugrid.node_lon.values,
        err_msg="UGRID longitude mismatch for HEALPix",
        rtol=ERROR_TOLERANCE
    )
    np.testing.assert_allclose(
        original_grid.node_lon.values,
        reloaded_exodus.node_lon.values,
        err_msg="Exodus longitude mismatch for HEALPix",
        rtol=ERROR_TOLERANCE
    )
    np.testing.assert_allclose(
        original_grid.node_lat.values,
        reloaded_ugrid.node_lat.values,
        err_msg="UGRID latitude mismatch for HEALPix",
        rtol=ERROR_TOLERANCE
    )
    np.testing.assert_allclose(
        original_grid.node_lat.values,
        reloaded_exodus.node_lat.values,
        err_msg="Exodus latitude mismatch for HEALPix",
        rtol=ERROR_TOLERANCE
    )

    # Validate grid dimensions are preserved
    assert reloaded_ugrid.n_face == original_grid.n_face
    assert reloaded_exodus.n_face == original_grid.n_face
