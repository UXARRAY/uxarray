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
