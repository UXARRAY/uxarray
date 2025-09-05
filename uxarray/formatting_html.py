import warnings
from collections import OrderedDict
from functools import partial
from html import escape

import xarray.core.formatting_html as xrfm

from uxarray.conventions import descriptors, ugrid


def _grid_header(grid, header_name=None):
    if header_name is None:
        obj_type = f"uxarray.{type(grid).__name__}"
    else:
        obj_type = f"{header_name}"

    header_components = [f"<div class='xr-obj-type'>{escape(obj_type)}</div>"]

    return header_components


def _grid_sections(grid, max_items_collapse=15):
    if grid._ds is None:
        warnings.warn(
            "Grid repr requested but the grid is None. "
            "This often means it was dropped by an unsupported Xarray operation.",
            UserWarning,
        )
        return []
    cartesian_coordinates = list(
        [coord for coord in ugrid.CARTESIAN_COORDS if coord in grid._ds]
    )
    spherical_coordinates = list(
        [coord for coord in ugrid.SPHERICAL_COORDS if coord in grid._ds]
    )
    descriptor = list(
        [desc for desc in descriptors.DESCRIPTOR_NAMES if desc in grid._ds]
    )
    connectivity = grid.connectivity

    sections = [xrfm.dim_section(grid._ds)]

    sections.append(
        grid_spherical_coordinates_section(
            grid._ds[spherical_coordinates],
            max_items_collapse=max_items_collapse,
            name="Spherical Coordinates",
        )
    )
    sections.append(
        grid_cartesian_coordinates_section(
            grid._ds[cartesian_coordinates],
            max_items_collapse=max_items_collapse,
            name="Cartesian Coordinates",
        )
    )

    sections.append(
        grid_connectivity_section(
            grid._ds[connectivity],
            max_items_collapse=max_items_collapse,
            name="Connectivity",
        )
    )

    sections.append(
        grid_descriptor_section(
            grid._ds[descriptor],
            max_items_collapse=max_items_collapse,
            name="Descriptors",
        )
    )

    sections.append(
        grid_attr_section(
            grid._ds.attrs, max_items_collapse=max_items_collapse, name="Attributes"
        )
    )

    return sections


def grid_repr(grid, max_items_collapse=15, header_name=None) -> str:
    """HTML repr for ``Grid`` class."""
    header_components = _grid_header(grid, header_name)

    sections = _grid_sections(grid, max_items_collapse)

    return xrfm._obj_repr(grid, header_components, sections)


grid_spherical_coordinates_section = partial(
    xrfm._mapping_section,
    details_func=xrfm.summarize_vars,
    expand_option_name="display_expand_data_vars",
)

grid_cartesian_coordinates_section = partial(
    xrfm._mapping_section,
    details_func=xrfm.summarize_vars,
    expand_option_name="display_expand_data_vars",
)

grid_connectivity_section = partial(
    xrfm._mapping_section,
    details_func=xrfm.summarize_vars,
    expand_option_name="display_expand_data_vars",
)

grid_descriptor_section = partial(
    xrfm._mapping_section,
    details_func=xrfm.summarize_vars,
    expand_option_name="display_expand_data_vars",
)

grid_attr_section = partial(
    xrfm._mapping_section,
    details_func=xrfm.summarize_attrs,
    expand_option_name="display_expand_attrs",
)


def _obj_repr_with_grid(obj, header_components, sections):
    """Return HTML repr of an uxarray object.

    If CSS is not injected (untrusted notebook), fallback to the plain
    text repr.
    """
    # Check if uxgrid exists and is not None
    if not hasattr(obj, "uxgrid") or obj.uxgrid is None:
        # Fallback to standard xarray representation without grid info
        icons_svg, css_style = xrfm._load_static_files()
        sections_html = "".join(
            f"<li class='xr-section-item'>{s}</li>" for s in sections
        )
        return (
            "<div>"
            f"{icons_svg}<style>{css_style}</style>"
            f"<pre class='xr-text-repr-fallback'>{escape(repr(obj))}</pre>"
            "<div class='xr-wrap' style='display:none'>"
            f"<div class='xr-header'>{''.join(h for h in header_components)}</div>"
            f"<ul class='xr-sections'>{sections_html}</ul>"
            "</div>"
            "</div>"
        )

    # Construct header and sections for the main object
    header = f"<div class='xr-header'>{''.join(h for h in header_components)}</div>"
    sections = "".join(f"<li class='xr-section-item'>{s}</li>" for s in sections)

    grid_html_repr = grid_repr(
        obj.uxgrid,
        max_items_collapse=0,
        header_name=f"uxarray.{type(obj).__name__}.uxgrid",
    )

    icons_svg, css_style = xrfm._load_static_files()
    obj_repr_html = (
        "<div>"
        f"{icons_svg}<style>{css_style}</style>"
        f"<pre class='xr-text-repr-fallback'>{escape(repr(obj))}</pre>"
        "<div class='xr-wrap' style='display:none'>"
        f"{header}"
        f"<ul class='xr-sections'>{sections}</ul>"
        "</div>"
        "</div>"
    )

    return (
        "<div>"
        f"{icons_svg}<style>{css_style}</style>"
        f"<pre class='xr-text-repr-fallback'>{escape(repr(obj))}</pre>"
        "<div class='xr-wrap' style='display:none'>"
        f"{obj_repr_html}"
        "<details>"
        "<summary>Show Grid Information</summary>"
        f"{grid_html_repr}"
        "</details>"
        "</div>"
        "</div>"
    )


def dataset_repr(ds) -> str:
    """HTML repr for ``UxDataset`` class."""
    obj_type = f"uxarray.{type(ds).__name__}"

    header_components = [f"<div class='xr-obj-type'>{escape(obj_type)}</div>"]

    sections = [
        xrfm.dim_section(ds),
        xrfm.coord_section(ds.coords),
        xrfm.datavar_section(ds.data_vars),
        xrfm.index_section(xrfm._get_indexes_dict(ds.xindexes)),
        xrfm.attr_section(ds.attrs),
    ]

    return _obj_repr_with_grid(ds, header_components, sections)


def array_repr(arr) -> str:
    """HTML repr for ``UxDataArray`` class."""

    dims = OrderedDict((k, v) for k, v in zip(arr.dims, arr.shape))
    if hasattr(arr, "xindexes"):
        indexed_dims = arr.xindexes.dims
    else:
        indexed_dims = {}

    obj_type = f"uxarray.{type(arr).__name__}"
    arr_name = f"'{arr.name}'" if getattr(arr, "name", None) else ""

    header_components = [
        f"<div class='xr-obj-type'>{obj_type}</div>",
        f"<div class='xr-array-name'>{arr_name}</div>",
        xrfm.format_dims(dims, indexed_dims),
    ]

    sections = [xrfm.array_section(arr)]

    if hasattr(arr, "coords"):
        sections.append(xrfm.coord_section(arr.coords))

    if hasattr(arr, "xindexes"):
        indexes = xrfm._get_indexes_dict(arr.xindexes)
        sections.append(xrfm.index_section(indexes))

    sections.append(xrfm.attr_section(arr.attrs))

    return _obj_repr_with_grid(arr, header_components, sections)
