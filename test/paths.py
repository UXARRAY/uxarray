"""
Centralized test file paths for all mesh files.
This module provides consistent paths to test data files for all test modules.
"""

import os
from pathlib import Path

# Base path to test directory
TEST_PATH = Path(os.path.dirname(os.path.realpath(__file__)))
MESHFILES_PATH = TEST_PATH / "meshfiles"

# UGRID format files
UGRID_PATH = MESHFILES_PATH / "ugrid"
QUAD_HEXAGON_GRID = UGRID_PATH / "quad-hexagon" / "grid.nc"
QUAD_HEXAGON_DATA = UGRID_PATH / "quad-hexagon" / "data.nc"
QUAD_HEXAGON_MULTI_DIM_DATA = UGRID_PATH / "quad-hexagon" / "multi_dim_data.nc"
QUAD_HEXAGON_RANDOM_NODE_DATA = UGRID_PATH / "quad-hexagon" / "random-node-data.nc"
OUTCSNE30_GRID = UGRID_PATH / "outCSne30" / "outCSne30.ug"
OUTCSNE30_VORTEX = UGRID_PATH / "outCSne30" / "outCSne30_vortex.nc"
OUTCSNE30_VAR2 = UGRID_PATH / "outCSne30" / "outCSne30_var2.nc"
OUTCSNE30_TEST2 = UGRID_PATH / "outCSne30" / "outCSne30_test2.nc"
OUTCSNE30_TEST3 = UGRID_PATH / "outCSne30" / "outCSne30_test3.nc"
OUTCSNE30_VAR2_ALT = UGRID_PATH / "outCSne30" / "var2.nc"
OUTRLL1DEG_GRID = UGRID_PATH / "outRLL1deg" / "outRLL1deg.ug"
OUTRLL1DEG_VORTEX = UGRID_PATH / "outRLL1deg" / "outRLL1deg_vortex.nc"
OV_RLL10DEG_CSNE4_GRID = UGRID_PATH / "ov_RLL10deg_CSne4" / "ov_RLL10deg_CSne4.ug"
OV_RLL10DEG_CSNE4_VORTEX = UGRID_PATH / "ov_RLL10deg_CSne4" / "ov_RLL10deg_CSne4_vortex.nc"
GEOFLOW_GRID = UGRID_PATH / "geoflow-small" / "grid.nc"
GEOFLOW_V1 = UGRID_PATH / "geoflow-small" / "v1.nc"
GEOFLOW_V2 = UGRID_PATH / "geoflow-small" / "v2.nc"
GEOFLOW_V3 = UGRID_PATH / "geoflow-small" / "v3.nc"
FESOM_MESH_DIAG = UGRID_PATH / "fesom" / "fesom.mesh.diag.nc"

# MPAS format files
MPAS_PATH = MESHFILES_PATH / "mpas"
MPAS_QU_MESH = MPAS_PATH / "QU" / "mesh.QU.1920km.151026.nc"
MPAS_QU_GRID = MPAS_PATH / "QU" / "480" / "grid.nc"
MPAS_QU_DATA = MPAS_PATH / "QU" / "480" / "data.nc"
MPAS_OCEAN_MESH = MPAS_PATH / "QU" / "oQU480.231010.nc"

# ESMF format files
ESMF_PATH = MESHFILES_PATH / "esmf"
ESMF_NE30_GRID = ESMF_PATH / "ne30" / "ne30pg3.grid.nc"
ESMF_NE30_DATA = ESMF_PATH / "ne30" / "ne30pg3.data.nc"

# Exodus format files
EXODUS_PATH = MESHFILES_PATH / "exodus"
EXODUS_OUTCSNE8 = EXODUS_PATH / "outCSne8" / "outCSne8.g"
EXODUS_MIXED = EXODUS_PATH / "mixed" / "mixed.exo"

# SCRIP format files
SCRIP_PATH = MESHFILES_PATH / "scrip"
SCRIP_OUTCSNE8 = SCRIP_PATH / "outCSne8" / "outCSne8.nc"
SCRIP_NE30PG2_GRID = SCRIP_PATH / "ne30pg2" / "grid.nc"
SCRIP_NE30PG2_DATA = SCRIP_PATH / "ne30pg2" / "data.nc"

# ICON format files
ICON_PATH = MESHFILES_PATH / "icon"
ICON_R02B04_GRID = ICON_PATH / "R02B04" / "icon_grid_0010_R02B04_G.nc"

# FESOM format files
FESOM_PATH = MESHFILES_PATH / "fesom"
FESOM_PI_PATH = FESOM_PATH / "pi"
FESOM_SOUFFLET_PATH = FESOM_PATH / "soufflet"
FESOM_SOUFFLET_NETCDF_GRID = FESOM_PATH / "soufflet-netcdf" / "grid.nc"

# HEALPix format files
HEALPIX_PATH = MESHFILES_PATH / "healpix"
HEALPIX_OUTCSNE30_DATA = HEALPIX_PATH / "outCSne30" / "data.nc"

# Points format files
POINTS_PATH = MESHFILES_PATH / "points"
POINTS_OUTCSNE30_PSI = POINTS_PATH / "outCSne30" / "psi-points.nc"

# Structured format files
STRUCTURED_PATH = MESHFILES_PATH / "structured"
STRUCTURED_OUTCSNE30_VORTEX = STRUCTURED_PATH / "outCSne30_vortex_structured.nc"

# Shapefile format files
SHP_PATH = MESHFILES_PATH / "shp"
SHP_5POLY_PATH = SHP_PATH / "5poly"
SHP_5POLY_FILE = SHP_5POLY_PATH / "5poly.shp"
SHP_MULTIPOLY_PATH = SHP_PATH / "multipoly"
SHP_MULTIPOLY_FILE = SHP_MULTIPOLY_PATH / "multipoly.shp"
SHP_CHICAGO_PATH = SHP_PATH / "chicago_neighborhoods"
SHP_US_NATION_PATH = SHP_PATH / "cb_2018_us_nation_20m"
SHP_US_NATION_FILE = SHP_US_NATION_PATH / "cb_2018_us_nation_20m.shp"

# GeoJSON format files
GEOJSON_PATH = MESHFILES_PATH / "geojson"
GEOJSON_CHICAGO_BUILDINGS = GEOJSON_PATH / "sample_chicago_buildings.geojson"

# GEOS-CS format files
GEOS_CS_PATH = MESHFILES_PATH / "geos-cs"
GEOS_CS_C12_GRID = GEOS_CS_PATH / "c12" / "test-c12.native.nc4"
