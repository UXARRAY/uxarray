import uxarray as ux
import xarray as xr
import pytest
from unittest import TestCase

class TestStructured(TestCase):
    def test_read_structured_grid_from_ds(self):
        for ds_name in ["air_temperature", "ersstv5"]:
            with self.subTest(ds_name=ds_name):
                ds = xr.tutorial.open_dataset(ds_name)
                uxgrid = ux.Grid.from_structured(ds)
                self.assertEqual(uxgrid.n_face, ds.sizes['lon'] * ds.sizes['lat'])
                self.assertTrue(uxgrid.validate())

    def test_read_structured_grid_from_latlon(self):
        for ds_name in ["air_temperature", "ersstv5"]:
            with self.subTest(ds_name=ds_name):
                ds = xr.tutorial.open_dataset(ds_name)
                uxgrid = ux.Grid.from_structured(lon=ds.lon, lat=ds.lat)
                self.assertEqual(uxgrid.n_face, ds.sizes['lon'] * ds.sizes['lat'])
                self.assertTrue(uxgrid.validate())

    def test_read_structured_uxds_from_ds(self):
        for ds_name in ["air_temperature", "ersstv5"]:
            with self.subTest(ds_name=ds_name):
                 ds = xr.tutorial.open_dataset(ds_name)
                 uxds = ux.UxDataset.from_structured(ds)
                 self.assertIn("n_face", uxds.dims)
                 self.assertNotIn("lon", uxds.dims)
                 self.assertNotIn("lat", uxds.dims)
                 self.assertTrue(uxds.uxgrid.validate())
