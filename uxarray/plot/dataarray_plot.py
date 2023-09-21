from __future__ import annotations

from typing import Optional, TYPE_CHECKING
if TYPE_CHECKING:
    from uxarray.core.dataarray import UxDataArray


def plot(uxda, **kwargs):
    return raster(uxda, **kwargs)


def raster(uxda: UxDataArray,
           plot_height: Optional[int] = 300,
           plot_width: Optional[int] = 600,
           cmap: Optional[str] = "blue",
           agg: Optional[str] = "mean"):
    """TODO: Docstring & additional params"""
    import datashader as ds
    import datashader.transfer_functions as tf

    cvs = ds.Canvas(plot_width, plot_height)
    gdf = uxda.to_geodataframe()
    if agg == "mean":
        aggregated = cvs.polygons(gdf,
                                  geometry='geometry',
                                  agg=ds.mean(uxda.name))
    elif agg == "sum":
        aggregated = cvs.polygons(gdf,
                                  geometry='geometry',
                                  agg=ds.sum(uxda.name))
    else:
        raise ValueError
    return tf.shade(aggregated, cmap=cmap)
