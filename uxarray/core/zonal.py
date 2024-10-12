# import numpy as np
# import dask.array as da
#
#
# def _compute_zonal_mean(
#     uxda, data_mapping, latitudes, conservative=True, method="fast"
# ):
#     # Select the appropriate function based on the data_mapping
#
#     # Create a result array based on whether Dask is being used
#     shape = uxda.shape[:-1] + (len(latitudes),)
#     if isinstance(uxda.data, da.Array):
#         # Create a Dask array for storing results
#         result = da.empty(shape, dtype=uxda.dtype)
#     else:
#         # Create a NumPy array for storing results
#         result = np.empty(shape, dtype=uxda.dtype)
#
#     if data_mapping == "n_face":
#         _face_centered_zonal_mean(uxda, result, latitudes, conservative, method)
#     else:
#         pass
#
#     # if conservative:
#     #     # weighted by the area of each cell
#     #     weights = uxda.uxgrid.face_areas.data
#     # else:
#     #     # weighted by the magnitude of each edge
#     #     weights = uxda.uxgrid.edge_node_distances.data
#
#     for i, lat in enumerate(latitudes):
#         if conservative:
#             pass
#         else:
#             pass
#
#         face_indices = get_indices_func(lat, method)
#         averages[..., i] = uxda[..., face_indices].data.mean(axis=-1)
#
#     return averages
#
#
# def _face_centered_zonal_mean(uxda, result, latitudes, conservative, method):
#     if conservative:
#         weights = uxda.uxgrid.face_areas
#     else:
#         weights = uxda.uxgrid.edge_magnitudes
#
#     for i, lat in enumerate(latitudes):
#         face_indices = uxda.uxgrid.get
#         if conservative:
#             pass
#         else:
#             pass
#
#         result[..., u] = uxda.data[..., face_indices].data.mean(axis=-1)
#
#     pass
#
#
# def _edge_centered_zonal_mean():
#     pass
