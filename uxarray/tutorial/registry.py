from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Component = Literal["grid", "data"]


@dataclass(frozen=True)
class TutorialDataset:
    """Metadata for a UXarray tutorial dataset."""

    description: str
    grid: tuple[str, ...]
    data: tuple[str, ...] | None = None
    data_files: tuple[tuple[str, ...], ...] | None = None


DATASETS: dict[str, TutorialDataset] = {
    "quad-hexagon": TutorialDataset(
        description="Small UGRID grid with sample face, edge, and node data.",
        grid=("ugrid", "quad-hexagon", "grid.nc"),
        data=("ugrid", "quad-hexagon", "data.nc"),
    ),
    "outCSne30-vortex": TutorialDataset(
        description="Cubed-sphere UGRID grid with sample vortex data.",
        grid=("ugrid", "outCSne30", "outCSne30.ug"),
        data=("ugrid", "outCSne30", "outCSne30_vortex.nc"),
    ),
    "outCSne30": TutorialDataset(
        description="Cubed-sphere UGRID grid.",
        grid=("ugrid", "outCSne30", "outCSne30.ug"),
    ),
    "quad-hexagon-random-node": TutorialDataset(
        description="Small UGRID grid with random node-centered sample data.",
        grid=("ugrid", "quad-hexagon", "grid.nc"),
        data=("ugrid", "quad-hexagon", "random-node-data.nc"),
    ),
    "quad-hexagon-random-edge": TutorialDataset(
        description="Small UGRID grid with random edge-centered sample data.",
        grid=("ugrid", "quad-hexagon", "grid.nc"),
        data=("ugrid", "quad-hexagon", "random-edge-data.nc"),
    ),
    "quad-hexagon-random-face": TutorialDataset(
        description="Small UGRID grid with random face-centered sample data.",
        grid=("ugrid", "quad-hexagon", "grid.nc"),
        data=("ugrid", "quad-hexagon", "random-face-data.nc"),
    ),
    "outCSne30-timeseries": TutorialDataset(
        description="Cubed-sphere UGRID grid with sample time series data.",
        grid=("ugrid", "outCSne30", "outCSne30.ug"),
        data=("ugrid", "outCSne30", "outCSne30_sel_timeseries.nc"),
    ),
    "ne120-tcsubset": TutorialDataset(
        description="Regional NE120 tropical cyclone subset with sample data.",
        grid=("ugrid", "ne120_TCsubset", "ne120_TCsubset.ug"),
        data=("ugrid", "ne120_TCsubset", "ne120_TCsubset.nc"),
    ),
    "scrip-ne30pg2": TutorialDataset(
        description="SCRIP NE30 grid with relative humidity sample data.",
        grid=("scrip", "ne30pg2", "grid.nc"),
        data=("scrip", "ne30pg2", "data.nc"),
    ),
    "mpas-QU-480": TutorialDataset(
        description="MPAS QU 480 km grid with sample data.",
        grid=("mpas", "QU", "480", "grid.nc"),
        data=("mpas", "QU", "480", "data.nc"),
    ),
    "mpas-dyamond-30km-gradient": TutorialDataset(
        description="MPAS DYAMOND 30 km subset used for gradient examples.",
        grid=("mpas", "dyamond-30km", "gradient_grid_subset.nc"),
        data=("mpas", "dyamond-30km", "gradient_data_subset.nc"),
    ),
    "mpas-QU-oQU480": TutorialDataset(
        description="MPAS ocean QU 480 km mesh stored as a single file.",
        grid=("mpas", "QU", "oQU480.231010.nc"),
        data=("mpas", "QU", "oQU480.231010.nc"),
    ),
    "mpas-QU-1920": TutorialDataset(
        description="MPAS QU 1920 km mesh stored as a single file.",
        grid=("mpas", "QU", "mesh.QU.1920km.151026.nc"),
        data=("mpas", "QU", "mesh.QU.1920km.151026.nc"),
    ),
    "geoflow-small-v1": TutorialDataset(
        description="Small GeoFlow UGRID grid with v1 sample data.",
        grid=("ugrid", "geoflow-small", "grid.nc"),
        data=("ugrid", "geoflow-small", "v1.nc"),
    ),
    "scrip-outCSne8": TutorialDataset(
        description="Small SCRIP cubed-sphere grid.",
        grid=("scrip", "outCSne8", "outCSne8.nc"),
    ),
    "quad-hexagon-random": TutorialDataset(
        description="Small UGRID grid with random node-, edge-, and face-centered sample data.",
        grid=("ugrid", "quad-hexagon", "grid.nc"),
        data_files=(
            ("ugrid", "quad-hexagon", "random-node-data.nc"),
            ("ugrid", "quad-hexagon", "random-edge-data.nc"),
            ("ugrid", "quad-hexagon", "random-face-data.nc"),
        ),
    ),
}
