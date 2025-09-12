import uxarray as ux
import cartopy.crs as ccrs


def test_geodataframe_projection(gridpath):
    """Test the projection of a GeoDataFrame."""
    uxgrid = ux.open_grid(gridpath("geos-cs", "c12", "test-c12.native.nc4"))
    gdf = uxgrid.to_geodataframe(projection=ccrs.Robinson(), periodic_elements='exclude')

    # Example assertion to check if gdf is not None
    assert gdf is not None
