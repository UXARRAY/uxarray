import numpy as np
import math
from scipy.spatial import cKDTree
from netCDF4 import Dataset


def generate_nearest_neighbor_map(source, target):
    source_faces = source.nMesh2_face
    target_faces = target.nMesh2_face
    face_number = math.ceil(source_faces / target_faces)
    source_coordinates = np.array(
        [source.Mesh2_face_x.data, source.Mesh2_face_y.data])
    source_coordinates = source_coordinates.T  # Transpose the array to have the shape (N, D)
    source_kd_tree = cKDTree(source_coordinates)

    # Target Grid
    target_coordinates = np.array(
        [target.Mesh2_face_x.data, target.Mesh2_face_y.data])
    target_coordinates = target_coordinates.reshape(
        -1, 2)  # Reshape to (N, 2) for the face centroids
    target_kd_tree = cKDTree(target_coordinates)

    # Find the nearest neighbors in the source mesh for each centroid in the target mesh
    _, indices = source_kd_tree.query(target_coordinates, k=face_number)

    # Reshape indices to have the shape (n_target_faces, face_number)
    indices = indices.reshape(-1, face_number)

    # Write the output map to NetCDF
    output_map_file = "meshfiles/nearest_neighbor_map.nc"
    with Dataset(output_map_file, "w", format="NETCDF4") as ncmap:
        ncmap.Title = "Nearest Neighbor Map"

        dimension_target_faces = ncmap.createDimension("n_target_faces",
                                                       target_faces)
        dimension_neighbors = ncmap.createDimension("k", face_number)

        variable_row = ncmap.createVariable("row", "i4", ("n_target_faces",))
        variable_column = ncmap.createVariable("column", "i4",
                                               ("n_target_faces", "k"))

        variable_row[:] = np.arange(1, target_faces + 1)
        variable_column[:] = indices


def remap_variable(mapping_file, source_mesh_file, target_mesh_file,
                   variable_to_remap):
    # Read the mapping file
    with Dataset(mapping_file, "r") as nc_map:
        mapping_column = nc_map.variables["column"][:]

    # Read the depth variable from the source mesh
    with Dataset(source_mesh_file, "r") as source_nc:
        depth_source = source_nc.variables[variable_to_remap][:]

    # Read the target mesh (smaller mesh) for writing interpolated depth values
    with Dataset(target_mesh_file, "r+") as target_nc:
        # Initialize an empty array for the remapped depths
        depth_target = np.empty(len(target_nc.variables[variable_to_remap]),
                                dtype=depth_source.dtype)

        # Interpolate depth values using the mapping information
        for target_index, source_indices in enumerate(mapping_column):
            # Get the depth values from the source mesh for all nearest neighbors
            depths = depth_source[source_indices -
                                  1]  # Subtract 1 for 0-based indexing

            # Take the average depth for each target face
            depth_target[target_index] = np.mean(depths)

        # Save the updated depth values to the target mesh file
        target_nc.variables[variable_to_remap][:] = depth_target
