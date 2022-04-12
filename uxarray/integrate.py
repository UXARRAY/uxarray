"""xarray accessor functions for integration with uxarray."""
import xarray as xr
import uxarray as ux
from .helpers import calculate_face_area


@xr.register_dataset_accessor('integrate')
class IntegrateAccessor:
    """Integrate Accessor works on xarray Dataset called "integrate".

    Integrates over all the faces of the given mesh.
    Note: The dataset must have both the mesh and the variable to integrate on.
    """

    def __init__(self, dataarray):
        self.dataarray = dataarray

    def __call__(self, var_key):
        integral = 0

        # area of a face call needs the units for coordinate conversion if spherical grid is used
        units = "spherical"
        if not "degree" in self.dataarray.Mesh2_node_x.units:
            units = "cartesian"

        num_faces = self.dataarray.get(var_key).data.size
        for i in range(num_faces):
            x = []
            y = []
            z = []
            for j in range(len(self.dataarray.Mesh2_face_nodes[i])):
                node_id = self.dataarray.Mesh2_face_nodes.data[i][j]

                x.append(self.dataarray.Mesh2_node_x.data[node_id])
                y.append(self.dataarray.Mesh2_node_y.data[node_id])
                if self.dataarray.Mesh2.topology_dimension > 2:
                    z.append(self.dataarray.Mesh2_node_z.data[node_id])
                else:
                    z.append(0)

            # After getting all the nodes of a face assembled call the  cal. face area routine
            face_area = calculate_face_area(x, y, z, units)
            # get the value from the data file
            face_val = self.dataarray.get(var_key).to_numpy().data[i]

            integral += face_area * face_val

        # print("Integral of ", var_key, " over the surface is ", integral)
        return integral
