import uxarray as ux
import healpix as hp
import pytest
import os
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

def test_dataset():
    uxds = ux.UxDataset.from_healpix(ds_path)

    assert uxds.uxgrid.source_grid_spec == "HEALPix"
    assert "n_face" in uxds.dims



def test_number_of_boundary_nodes():
    uxgrid = ux.Grid.from_healpix(0)
    face_node_conn = uxgrid.face_node_connectivity
    n_face, n_max_face_nodes = face_node_conn.shape

    assert n_face == uxgrid.n_face
    assert n_max_face_nodes == uxgrid.n_max_face_nodes
