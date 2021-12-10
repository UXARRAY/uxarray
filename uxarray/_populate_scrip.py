def populate_scrip_data(self, ds):
    print("populating scrip data..")
    ds["mesh2_node_x"] = ds.coords.values
    # print(ds)
