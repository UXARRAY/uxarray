from html import escape
import xarray.core.formatting_html as xrfm


def grid_repr(grid) -> str:
    obj_type = f"uxarray.{type(grid).__name__}"

    header_components = [f"<div class='xr-obj-type'>{escape(obj_type)}</div>"]

    sections = [xrfm.dim_section(grid._ds)]

    return xrfm._obj_repr(grid, header_components, sections)
