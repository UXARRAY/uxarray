import os
import uxarray as ux

from unittest import TestCase
from pathlib import Path
from uxarray.grid.regridding import generate_nearest_neighbor_map, remap_variable

current_path = Path(os.path.dirname(os.path.realpath(__file__)))


class TestRegridding(TestCase):

    def test_remap_variable(self):
        source_large_ocean = current_path / "meshfiles" / "netcdf" / "oQU120.230424.nc"
        target_small_ocean = current_path / "meshfiles" / "netcdf" / "oQU480.230422.nc"
        # load our vertices into a UXarray Grid object
        source_verts_grid = ux.open_grid(source_large_ocean)
        target_verts_grid = ux.open_grid(target_small_ocean)

        remapped_target_ds = generate_nearest_neighbor_map(
            source_verts_grid, target_verts_grid)

        nearest = current_path / "meshfiles" / "nearest_neighbor_map.nc"
        remap_variable(nearest, source_large_ocean, target_small_ocean,
                       "bottomDepth")
