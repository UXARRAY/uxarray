from warnings import warn
import geopandas as gpd

def _read_shpfile(filepath):
    """Read shape file, use geopandas.

    Parameters: xarray.Dataset, required
    Returns: ugrid aware xarray.Dataset
    """
    raise RuntimeError(
        "Function not implemented yet. FYI, attempted to read SHAPE file: "
        + str(filepath)
    )
    # TODO: create ds
