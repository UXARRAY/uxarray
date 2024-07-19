import uxarray as ux

# from "./grid/integrate.py" import _get_zonal_faces_weight_at_constLat
print("ux.__version__ =", ux.__version__)


# base_path = "./test/meshfiles/ugrid/outCSne30/"
# grid_path = base_path + "outCSne30.ug"
# data_path = base_path + "outCSne30_vortex.nc"

grid_path = "./test/meshfiles/ugrid/outCSne30/outCSne30.ug"
data_path = "./test/meshfiles/ugrid/outCSne30/outCSne30_vortex.nc"
uxds = ux.open_dataset(grid_path, data_path)
grid = uxds.uxgrid
res = uxds["psi"].zonal_mean()
print(res)

# Access the zonal mean for a specific latitude, e.g., 30 degrees
latitude_value = 30
zonal_mean_at_latitude = res.sel(latitude=latitude_value).values

print(
    f"The zonal mean at {latitude_value} degrees latitude is {zonal_mean_at_latitude}"
)

res = uxds["psi"].zonal_mean(25)
print(res)

res = uxds["psi"].zonal_mean((0, 90, 1))
print(res)
