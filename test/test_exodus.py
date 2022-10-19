import os
import numpy as np

from unittest import TestCase
from pathlib import Path

import uxarray as ux

current_path = Path(os.path.dirname(os.path.realpath(__file__)))


class TestExodus(TestCase):

    def test_read_exodus(self):
        """Read an exodus file and writes a exodus file."""

        exo2_filename = current_path / "meshfiles" / "outCSne8.g"
        tgrid = ux.open_dataset(str(exo2_filename))

    def test_init_verts(self):
        """Create a uxarray grid from vertices and saves a 1 face exodus
        file."""
        verts = np.array([[0, 0], [2, 0], [0, 2], [2, 2]])
        vgrid = ux.Grid(verts)

    def test_encode_exodus(self):
        """Read a UGRID dataset and encode that as an Exodus format."""

        exo2_filename = current_path / "meshfiles" / "outCSne30.ug"
        tgrid = ux.open_dataset(str(exo2_filename))

        tgrid.encode_as("exodus")
