import numpy as np
import pytest

import os
from pathlib import Path

import uxarray as ux
import numpy.testing as nt

# Import centralized paths
import sys
sys.path.append(str(Path(__file__).parent.parent))
from paths import *



# Use centralized paths for dyamond files
dyamond_subset_grid_path = MPAS_DYAMOND_GRID
dyamond_subset_data_path = MPAS_DYAMOND_DATA




# TODO: pytest fixtures


class TestQuadHex:

    def test_gradient_output_format(self):
        """Tests the output format of gradient functionality"""
        uxds = ux.open_dataset(QUAD_HEXAGON_GRID, QUAD_HEXAGON_DATA)

        grad_ds = uxds['t2m'].gradient()

        assert isinstance(grad_ds, ux.UxDataset)
        assert "zonal_gradient" in grad_ds
        assert "meridional_gradient" in grad_ds
        assert "gradient" in grad_ds.attrs
        assert uxds['t2m'].sizes == grad_ds.sizes

    def test_gradient_all_boundary_faces(self):
        """Quad hexagon grid has 4 faces, each of which are on the boundary, so the expected gradients are zero for both components"""
        uxds = ux.open_dataset(QUAD_HEXAGON_GRID, QUAD_HEXAGON_DATA)

        grad = uxds['t2m'].gradient()

        assert np.isnan(grad['meridional_gradient']).all()
        assert np.isnan(grad['zonal_gradient']).all()


class TestMPASOcean:

    def test_gradient(self):
        uxds = ux.open_dataset(MPAS_QU_GRID, MPAS_QU_DATA)

        grad = uxds['bottomDepth'].gradient()

        # There should be some boundary faces
        assert np.isnan(grad['meridional_gradient']).any()
        assert np.isnan(grad['zonal_gradient']).any()

        # Not every face is on the boundary, ensure there are valid values
        assert not np.isnan(grad['meridional_gradient']).all()
        assert not np.isnan(grad['zonal_gradient']).all()


class TestDyamondSubset:

    center_fidx = 153
    left_fidx   = 100
    right_fidx  = 164
    top_fidx    = 154
    bottom_fidx = 66

    def test_lat_field(self):
        """Gradient of a latitude field. All vectors should be pointing east."""
        uxds =  ux.open_dataset(dyamond_subset_grid_path, dyamond_subset_data_path)
        grad = uxds['face_lat'].gradient()
        zg, mg = grad.zonal_gradient, grad.meridional_gradient
        assert mg.max() > zg.max()

        assert mg.min() > zg.max()


    def test_lon_field(self):
        """Gradient of a longitude field. All vectors should be pointing north."""
        uxds =  ux.open_dataset(dyamond_subset_grid_path, dyamond_subset_data_path)
        grad = uxds['face_lon'].gradient()
        zg, mg = grad.zonal_gradient, grad.meridional_gradient
        assert zg.max() > mg.max()

        assert zg.min() > mg.max()

    def test_gaussian_field(self):
        """Gradient of a gaussian field. All vectors should be pointing toward the center"""
        uxds =  ux.open_dataset(dyamond_subset_grid_path, dyamond_subset_data_path)
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



    def test_inverse_gaussian_field(self):
        """Gradient of an inverse gaussian field. All vectors should be pointing outward from the center."""
        uxds =  ux.open_dataset(dyamond_subset_grid_path, dyamond_subset_data_path)
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
