import uxarray as ux

grid_path = "../tests/meshfiles/ugrid/quad-hexagon/grid.nc"
data_path = grid_path = "../tests/meshfiles/ugrid/quad-hexagon/grid.nc"


class QuadHexagon:
    def time_open_grid(self):
        ux.open_grid(data_path)

    def mem_open_grid(self):
        return ux.open_grid(data_path)

    def peakmem_open_grid(self):
        uxgrid = ux.open_grid(data_path)
