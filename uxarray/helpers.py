import os
import xarray as xr
from pathlib import PurePath


# helper function to find file type
def determine_file_type(filepath):
    """Checks file path and contents to determine file type. Supports detection
    of UGrid, SCRIP, Exodus and shape file.

    Parameters: string, required
       Filepath of the file for which the filetype is to be determined.

    Returns: string
       File type: ug, exo, scrip or shp

    Raises:
       RuntimeError: Invalid file type
    """
    msg = ""
    mesh_filetype = "unknown"
    # exodus with coord
    try:
        # extract the file name and extension
        path = PurePath(filepath)
        file_extension = path.suffix

        # try to open file with xarray and test for exodus
        ext_ds = xr.open_dataset(filepath, mask_and_scale=False)["coord"]
        mesh_filetype = "exo"
    except KeyError as e:
        # exodus with coordx
        try:
            ext_ds = xr.open_dataset(filepath, mask_and_scale=False)["coordx"]
            mesh_filetype = "exo"
        except KeyError as e:
            # scrip with grid_center_lon
            try:
                ext_ds = xr.open_dataset(
                    filepath, mask_and_scale=False)["grid_center_lon"]
                mesh_filetype = "scrip"
            except KeyError as e:

                # check mesh topology and dimension
                try:
                    standard_name = lambda v: v is not None
                    # getkeys_filter_by_attribute(filepath, attr_name, attr_val)
                    # return type KeysView
                    base_dv_nc = list(
                        xr.open_dataset(
                            filepath, mask_and_scale=False).filter_by_attrs(
                                node_coordinates=standard_name).keys())[0]
                    base_dv_fc = list(
                        xr.open_dataset(
                            filepath, mask_and_scale=False).filter_by_attrs(
                                face_node_connectivity=standard_name).keys())[0]
                    base_dv_td = list(
                        xr.open_dataset(
                            filepath, mask_and_scale=False).filter_by_attrs(
                                topology_dimension=standard_name).keys())[0]
                    base_dv_mt = list(
                        xr.open_dataset(filepath,
                                        mask_and_scale=False).filter_by_attrs(
                                            cf_role="mesh_topology").keys())[0]
                    if base_dv_mt != "" and base_dv_td != "" and base_dv_fc != "" and base_dv_nc != "":
                        mesh_filetype = "ugrid"
                    else:
                        raise ValueError(
                            "cf_role is other than mesh_topology, the input NetCDF file is not UGRID format"
                        )
                except KeyError as e:
                    msg = str(e) + ': {}'.format(filepath)
    except (TypeError, AttributeError) as e:
        msg = str(e) + ': {}'.format(filepath)
    except (RuntimeError, OSError) as e:
        # check if this is a shp file
        # we won't use xarray to load that file
        if file_extension == ".shp":
            mesh_filetype = "shp"
        else:
            msg = str(e) + ': {}'.format(filepath)
    except ValueError as e:
        # check if this is a shp file
        # we won't use xarray to load that file
        if file_extension == ".shp":
            mesh_filetype = "shp"
        else:
            msg = str(e) + ': {}'.format(filepath)
    finally:
        if msg != "":  # we did not catch this above
            msg = "Unable to determine file type, mesh file not supported" + ': {}'.format(
                filepath)
            print(msg)
            os._exit(0)

    return mesh_filetype
