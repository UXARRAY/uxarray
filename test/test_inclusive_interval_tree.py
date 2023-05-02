import os
import numpy as np
import xarray as xr

from unittest import TestCase
from pathlib import Path

import xarray as xr
import uxarray as ux
from uxarray.inclusive_interval_tree import InclusiveIntervalTree, InclusiveNode, InclusiveInterval
import numpy.testing as nt

current_path = Path(os.path.dirname(os.path.realpath(__file__)))


class TestInclusiveIntervalTree(TestCase):

    def test_point_query(self):
        ivs = [(1, 3), (2, 7), (7, 9), (9, 13)]
        tree = InclusiveIntervalTree()

        for i, iv in enumerate(ivs):
            tree.add(InclusiveInterval(iv[0], iv[1], i))

        res = list(tree.at(3))
        expected = [InclusiveInterval(1, 3, 0), InclusiveInterval(2, 7, 1)]
        self.assertTrue(all([r == e for r, e in zip(res, expected)]))

    def test_range_query(self):
        """
        The range query is only inclusive on the beginning of the input range. It will search up to, but not including the input end.
        """
        ivs = [(1, 2), (4, 7), (5, 9), (9, 13)]
        tree = InclusiveIntervalTree()

        for i, iv in enumerate(ivs):
            tree.add(InclusiveInterval(iv[0], iv[1], i))
        res = list(tree[2:4])
        self.assertEqual(InclusiveInterval(1, 2, 0), res[0])

    def test_envelope(self):
        ivs = [(1, 2), (4, 7), (5, 9), (9, 13)]
        tree = InclusiveIntervalTree()

        for i, iv in enumerate(ivs):
            tree.add(InclusiveInterval(iv[0], iv[1], i))
        res = list(tree.envelop(4,9))
        expected = [InclusiveInterval(5, 9, 2), InclusiveInterval(4, 7, 1)]
        self.assertTrue(all([r == e for r, e in zip(res, expected)]))


