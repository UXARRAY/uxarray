import uxarray as ux
import healpix as hp
import pytest
import numpy as np



@pytest.mark.parametrize("resolution_level", [1, 2, 3, 4])
def test_to_ugrid(resolution_level):
    uxgrid = ux.Grid.from_healpix(resolution_level)

    expected_n_face = hp.nside2npix(hp.order2nside(resolution_level))

    assert uxgrid.n_face == expected_n_face

@pytest.mark.parametrize("resolution_level", [1, 2, 3, 4])
def test_boundaries(resolution_level):
    uxgrid = ux.Grid.from_healpix(resolution_level)

    assert "face_node_connectivity" not in uxgrid.connectivity
    assert "node_lon" not in uxgrid.connectivity
    assert "node_lat" not in uxgrid.connectivity

    _ = uxgrid.face_node_connectivity

    assert "face_node_connectivity" in uxgrid.connectivity
    assert "node_lon" in uxgrid.coordinates
    assert "node_lat" in uxgrid.coordinates
