import xarray as xr
import numpy as np
from pathlib import PurePath
from datetime import datetime


# Exodus Number is one-based.
def read_exodus(self, ext_ds):
    print("Reading exodus file: ", self.filepath)
    # populate self.in_ds

    self._init_mesh2()

    for key, value in ext_ds.variables.items():
        if key == "qa_records":
            # print(value)
            pass
        elif key == "coord":
            self.in_ds.Mesh2.attrs['topology_dimension'] = ext_ds.dims[
                'num_dim']
            self.in_ds["Mesh2_node_x"] = xr.DataArray(
                data=ext_ds.coord[0],
                dims=["nMesh2_node"],
                attrs={
                    # nothing specified set to spherical
                    "standard_name": "spherical",
                    "long_name": ext_ds.title,
                    # nothing specified set to m
                    "units": "deg",
                })
            self.in_ds["Mesh2_node_y"] = xr.DataArray(
                data=ext_ds.coord[1],
                dims=["nMesh2_node"],
                attrs={
                    # nothing specified set to spherical
                    "standard_name": "spherical",
                    "long_name": ext_ds.title,
                    # nothing specified set to m
                    "units": "deg",
                })
            self.in_ds["Mesh2_node_z"] = xr.DataArray(
                data=ext_ds.coord[2],
                dims=["nMesh2_node"],
                attrs={
                    # nothing specified set to spherical
                    "standard_name": "spherical",
                    "long_name": ext_ds.title,
                    # nothing specified set to m
                    "units": "deg",
                })
        elif key == "coordx":
            self.in_ds["Mesh2_node_x"] = xr.DataArray(
                data=ext_ds.coordx,
                dims=["nMesh2_node"],
                attrs={
                    # nothing specified set to spherical
                    "standard_name": "spherical",
                    "long_name": ext_ds.title,
                    # nothing specified set to m
                    "units": "deg",
                })
        elif key == "coordy":
            self.in_ds["Mesh2_node_y"] = xr.DataArray(
                data=ext_ds.coordx,
                dims=["nMesh2_node"],
                attrs={
                    # nothing specified set to spherical
                    "standard_name": "spherical",
                    "long_name": ext_ds.title,
                    # nothing specified set to m
                    "units": "deg",
                })
        elif key == "coordz":
            self.in_ds["Mesh2_node_z"] = xr.DataArray(
                data=ext_ds.coordx,
                dims=["nMesh2_node"],
                attrs={
                    # nothing specified set to spherical
                    "standard_name": "spherical",
                    "long_name": ext_ds.title,
                    # nothing specified set to m
                    "units": "deg",
                })
        elif key == "connect1":
            for k, v in value.attrs.items():
                if k == "elem_type":
                    etype = v
            self.in_ds["Mesh2_face_nodes"] = xr.DataArray(
                data=(ext_ds.connect1[:] - 1),  #Note: -1
                dims=["nMesh2_face", "nMaxMesh2_face_nodes"],
                attrs={
                    "cf_role": "face_node_connectivity",
                    "_FillValue": -1,
                    "start_index":
                        0  #NOTE: This might cause an error if numbering has holes
                })
    print("Finished reading exodus file.")


def write_exodus(self, outfile):
    # Note this is 1-based unlike native Mesh2 construct
    print("Writing exodus file: ", outfile)

    self.exo_ds = xr.Dataset()

    path = PurePath(outfile)
    out_filename = path.name

    now = datetime.now()
    date = now.strftime("%m/%d/%Y")
    time = now.strftime("%H:%M:%S")

    title = f"uxarray(" + str(out_filename) + ")" + date + ": "+ time
    fp_word = np.int32(8)
    version = np.float32(5.0)
    api_version = np.float32(5.0)
    self.exo_ds.attrs = {
        "api_version": api_version,
        "version": version,
        "floating_point_word_size": fp_word,
        "file_size": 0,
        "title": title
    }

    self.exo_ds["time_whole"] = xr.DataArray(data=[], dims=["time_step"])

    # qa_records
    qa_records = [["uxarray"], ["1.0"], [date], [time]]
    self.exo_ds["qa_records"] = xr.DataArray(data=xr.DataArray(
        np.array(qa_records, dtype="S33")),
                                             dims=["four", "num_qa_rec"])

    # get orig dimention from Mesh2 attribute topology dimension
    dim = self.in_ds["Mesh2"].topology_dimension

    c_data = []
    if dim == 2:
        c_data = xr.DataArray([
            self.in_ds.Mesh2_node_x.data.tolist(),
            self.in_ds.Mesh2_node_y.data.tolist()
        ])
    elif dim == 3:
        c_data = xr.DataArray([
            self.in_ds.Mesh2_node_x.data.tolist(),
            self.in_ds.Mesh2_node_y.data.tolist(),
            self.in_ds.Mesh2_node_z.data.tolist()
        ])

    self.exo_ds["coord"] = xr.DataArray(data=c_data,
                                        dims=["num_dim", "num_nodes"])

    # passing the max node for element type
    # might not work for mixed elements, same for reading
    element_type = get_element_type(self.in_ds.nMaxMesh2_face_nodes.size)

    self.exo_ds["connect1"] = xr.DataArray(
        data=(self.in_ds.Mesh2_face_nodes[:] + 1),
        dims=["num_el_in_blk1", "num_nod_per_el1"],
        attrs={"elem_type": element_type})

    self.exo_ds["edge_type1"] = xr.DataArray(data=xr.DataArray(
        np.zeros(
            (self.in_ds.nMesh2_face.size, self.in_ds.nMaxMesh2_face_nodes.size),
            "i4")),
                                             dims=[
                                                 "num_el_in_blk1",
                                                 "num_nod_per_el1"
                                             ])

    self.exo_ds["global_id1"] = xr.DataArray(data=(self.in_ds.nMesh2_face[:] +
                                                   1),
                                             dims=["num_el_in_blk1"])

    # TODO: fix num attr
    num_attr = 1
    self.exo_ds["attrib1"] = xr.DataArray(
        data=xr.DataArray(
            np.zeros((self.in_ds.nMesh2_face.size, num_attr), float)),
        dims=["num_el_in_blk1", "num_att_in_blk1"])

    self.exo_ds["eb_prop1"] = xr.DataArray(data=xr.DataArray(np.ones((1),
                                                                     "i4")),
                                           dims=["num_el_blk"],
                                           attrs={"name": "ID"})

    self.exo_ds["eb_status"] = xr.DataArray(data=xr.DataArray(
        np.array([num_attr], dtype="i4")),
                                            dims=["num_el_blk"])

    self.exo_ds["eb_names"] = xr.DataArray(data=xr.DataArray(
        np.array([''], dtype="S33")),
                                           dims=["num_el_blk"])

    if dim == 2:
        cnames = ["x", "y"]
    elif dim == 3:
        cnames = ["x", "y", "z"]

    self.exo_ds["coor_names"] = xr.DataArray(data=xr.DataArray(
        np.array(cnames, dtype="S33")),
                                             dims=["num_dim"])

    # done processing write the file to disk
    self.exo_ds.to_netcdf(outfile)
    print("Wrote: ", outfile)


def get_element_type(num_nodes):
    if num_nodes == 2:
        element_type = "BEAM"
    elif num_nodes == 3:
        element_type = "TRI"
    elif num_nodes == 4:
        element_type = "SHELL4"
    elif num_nodes == 8:
        element_type = "SHELL8"
    elif num_nodes == 9:
        element_type = "SHELL9"
    elif num_nodes == 6:
        element_type = "TRI6"
    elif num_nodes == 7:
        element_type = "TRI7"

    return element_type
