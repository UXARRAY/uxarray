from __future__ import annotations
from html import escape
from functools import lru_cache, partial

import xarray as xr

# from uxarray.core.formatting_html import (collapsible_section,
#                                           _get_indexes_dict,
#                                           _mapping_section,
#                                           summarize_vars,
#                                           dim_section,
#                                           coord_section,
#                                           index_section,
#                                           attr_section,
#                                           _obj_repr,
#                                           datavar_section)

# ================= Grid Coordinate Variables =================
grid_datavar_section = partial(
    xr.core.formatting_html._mapping_section,
    name="Grid Connectivity Variables",
    details_func=xr.core.formatting_html.summarize_vars,
    max_items_collapse=15,
    expand_option_name="display_expand_data_vars",  # TODO
)


def dataset_repr(ds):
    obj_type = f"xarray.{type(ds).__name__}"

    header_components = [f"<div class='xr-obj-type'>{escape(obj_type)}</div>"]

    sections = [
        xr.core.formatting_html.dim_section(ds),
        xr.core.formatting_html.coord_section(ds.coords),
        xr.core.formatting_html.datavar_section(ds.data_vars),
        grid_datavar_section(ds.uxgrid._ds.data_vars),
        xr.core.formatting_html.index_section(
            xr.core.formatting_html._get_indexes_dict(ds.xindexes)),
        xr.core.formatting_html.attr_section(ds.attrs),
    ]

    return xr.core.formatting_html._obj_repr(ds, header_components, sections)
