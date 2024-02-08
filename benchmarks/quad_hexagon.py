import uxarray as ux

grid_path = "../test/meshfiles/ugrid/quad-hexagon/grid.nc"
data_path = "../test/meshfiles/ugrid/quad-hexagon/data.nc"


class QuadHexagon:
    def time_open_grid(self):
        ux.open_grid(grid_path)

    def mem_open_grid(self):
        return ux.open_grid(grid_path)

    def peakmem_open_grid(self):
        uxgrid = ux.open_grid(grid_path)
