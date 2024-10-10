import numpy as np
import dask.array as da


def _compute_zonal_mean(uxda, data_mapping, latitudes, method="fast"):
    # Select the appropriate function based on the data_mapping
    if data_mapping == "n_face":
        get_indices_func = uxda.uxgrid.get_faces_at_constant_latitude
    else:
        get_indices_func = uxda.uxgrid.get_edges_at_constant_latitude

    # Create a result array based on whether Dask is being used
    shape = uxda.shape[:-1] + (len(latitudes),)
    if isinstance(uxda.data, da.Array):
        # Create a Dask array for storing results
        averages = da.empty(shape, dtype=uxda.dtype)
    else:
        # Create a NumPy array for storing results
        averages = np.empty(shape, dtype=uxda.dtype)

    for i, lat in enumerate(latitudes):
        indices = get_indices_func(lat, method)
        averages[..., i] = uxda[..., indices].data.mean(axis=-1)

    return averages
