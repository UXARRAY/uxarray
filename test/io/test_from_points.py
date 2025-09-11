import uxarray as ux
import os
import pytest
from pathlib import Path

def test_spherical_delaunay(gridpath):
    uxgrid = ux.open_grid(gridpath("ugrid", "outCSne30", "outCSne30.ug"))
    points_xyz = (uxgrid.node_x.values, uxgrid.node_y.values, uxgrid.node_z.values)
    points_latlon = (uxgrid.node_lon.values, uxgrid.node_lat.values)

    uxgrid_dt_xyz = ux.Grid.from_points(points_xyz, method='spherical_delaunay')
    uxgrid_dt_latlon = ux.Grid.from_points(points_latlon, method='spherical_delaunay')
    uxgrid_dt_xyz.validate()
    uxgrid_dt_latlon.validate()

    assert uxgrid_dt_xyz.n_node == uxgrid_dt_latlon.n_node == len(points_xyz[0])
    assert uxgrid_dt_xyz.triangular
    assert uxgrid_dt_latlon.triangular

def test_regional_delaunay(gridpath):
    uxgrid = ux.open_grid(gridpath("ugrid", "outCSne30", "outCSne30.ug"))

    uxgrid_regional = uxgrid.subset.nearest_neighbor((0.0, 0.0), k=50)

    points_xyz = (uxgrid_regional.node_x.values, uxgrid_regional.node_y.values, uxgrid_regional.node_z.values)
    points_latlon = (uxgrid_regional.node_lon.values, uxgrid_regional.node_lat.values)

    uxgrid_dt_xyz = ux.Grid.from_points(points_xyz, method='regional_delaunay')
    uxgrid_dt_latlon = ux.Grid.from_points(points_latlon, method='regional_delaunay')

    assert uxgrid_dt_xyz.n_node == uxgrid_dt_latlon.n_node == len(points_xyz[0])
    assert uxgrid_dt_xyz.triangular
    assert uxgrid_dt_latlon.triangular

def test_spherical_voronoi(gridpath):
    uxgrid = ux.open_grid(gridpath("ugrid", "outCSne30", "outCSne30.ug"))
    points_xyz = (uxgrid.node_x.values, uxgrid.node_y.values, uxgrid.node_z.values)
    points_latlon = (uxgrid.node_lon.values, uxgrid.node_lat.values)

    uxgrid_sv_xyz = ux.Grid.from_points(points_xyz, method='spherical_voronoi')
    uxgrid_sv_latlon = ux.Grid.from_points(points_latlon, method='spherical_voronoi')
    uxgrid_sv_xyz.validate()
    uxgrid_sv_latlon.validate()

    assert uxgrid_sv_xyz.n_face == uxgrid_sv_latlon.n_face == len(points_xyz[0])
