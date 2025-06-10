import uxarray as ux
import healpix as hp
import numpy as np
import pytest
import os
import xarray as xr
import pandas as pd
from pathlib import Path


current_path = Path(os.path.dirname(os.path.realpath(__file__)))


ds_path = current_path / "meshfiles" / "healpix" / "outCSne30" / "data.nc"


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
def test_time_dimension_roundtrip(pixels_only):
    ds = xr.open_dataset(ds_path)
    dummy_time = pd.to_datetime(["2025-01-01T00:00:00"])
    ds_time = ds.expand_dims(time=dummy_time)
    uxds = ux.UxDataset.from_healpix(ds_time, pixels_only=pixels_only)

    # Ensure time dimension is preserved and that the conversion worked
    assert "time" in uxds.dims
    assert uxds.sizes["time"] == 1

def test_dataset():
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

def test_healpix_to_netcdf(tmp_path):
    """Test that HEALPix grid can be encoded as UGRID and saved to netCDF.
       Using pytest tmp_path fixture to create a temporary file.
    """
    # Create HEALPix grid
    h = ux.Grid.from_healpix(zoom=3)
    uxa_ugrid = h.encode_as("UGRID")
    uxa_exodus = h.encode_as("Exodus")


    tmp_filename_ugrid = tmp_path / "healpix_test_ugrid.nc"
    tmp_filename_exodus = tmp_path / "healpix_test_exodus.exo"

    # Save to netCDF
    uxa_ugrid.to_netcdf(tmp_filename_ugrid)
    uxa_exodus.to_netcdf(tmp_filename_exodus)

    # Assertions
    assert tmp_filename_ugrid.exists()
    assert tmp_filename_ugrid.stat().st_size > 0
    assert tmp_filename_exodus.exists()
    assert tmp_filename_exodus.stat().st_size > 0

    loaded_grid_ugrid = ux.open_grid(tmp_filename_ugrid)
    loaded_grid_exodus = ux.open_grid(tmp_filename_exodus)
    assert loaded_grid_ugrid.n_face == h.n_face
    assert loaded_grid_exodus.n_face == h.n_face
