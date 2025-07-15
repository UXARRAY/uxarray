import pytest

import os
from pathlib import Path

import uxarray as ux
import numpy.testing as nt

current_path = Path(os.path.dirname(os.path.realpath(__file__)))

quad_hex_grid_path = current_path / "meshfiles" / "ugrid" / "quad-hexagon" / "grid.nc"
quad_hex_data_path = current_path / "meshfiles" / "ugrid" / "quad-hexagon" / "data.nc"


mpas_ocean_grid_path = current_path / "meshfiles" / "mpas" / "QU" / "480" / "grid.nc"
mpas_ocean_data_path = current_path / "meshfiles" / "mpas" / "QU" / "480" / "data.nc"

# TODO: pytest fixtures


class TestQuadHex:

    def test_gradient_output_format(self):
        """Tests the output format of gradient functionality"""
        uxds = ux.open_dataset(quad_hex_grid_path,quad_hex_data_path)

        grad_ds = uxds['t2m'].gradient()

        assert isinstance(grad_ds, ux.UxDataset)
        assert "zonal_gradient" in grad_ds
        assert "meridional_gradient" in grad_ds
        assert "gradient" in grad_ds.attrs
        assert uxds['t2m'].sizes == grad_ds.sizes

    def test_gradient_all_boundary_faces(self):
        """Quad hexagon grid has 4 faces, each of which are on the boundary, so the expected gradients are zero for both components"""
        uxds = ux.open_dataset(quad_hex_grid_path, quad_hex_data_path)

        grad_ds = uxds['t2m'].gradient()

        # TODO: maybe use a nt function
        assert (grad_ds['zonal_gradient'] == 0.0).all()
        assert (grad_ds['meridional_gradient'] == 0.0).all()


class TestMPASOcean:

    def test_gradient(self):
        uxds = ux.open_dataset(mpas_ocean_grid_path, mpas_ocean_data_path)

        grad = uxds['bottomDepth'].gradient()

        # TODO:



# TODO: Test nodal gradient









# TODO: Write test for gradient functionality here
