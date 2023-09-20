from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from uxarray.grid import Grid

import hvplot.pandas


def plot(grid: Grid, **kwargs):
    # default to edge plot for now
    return edges(grid, **kwargs)


def edges(grid: Grid, **kwargs):
    return grid.to_geodataframe().hvplot.paths(**kwargs)


def nodes(grid: Grid, **kwargs):
    return grid.to_geodataframe().hvplot.points(**kwargs)
