import os
import random

from unittest import TestCase
from pathlib import Path

import uxarray as ux

current_path = Path(os.path.dirname(os.path.realpath(__file__)))


class TestEdge(TestCase):

    # Test whether the edge class constructor works properly
    def test_init(self):
        edges_0_list = [1, 2, 3, 4, 5]
        edges_1_list = [2, 3, 4, 5, 6]
        for i in range(0, 5):
            edge = ux.Edge([edges_0_list[i], edges_1_list[i]])
            self.assertEqual(edge.node0, edges_0_list[i])
            self.assertEqual(edge.node1, edges_1_list[i])

        # Test if the indexes are sorted in ascending order
        edges_0_list = [2, 3, 4, 5, 6]
        edges_1_list = [1, 2, 3, 4, 5]
        for i in range(0, 5):
            edge = ux.Edge([edges_0_list[i], edges_1_list[i]])
            self.assertEqual(edge.node0, edges_1_list[i])
            self.assertEqual(edge.node1, edges_0_list[i])

    # Test if the edge is undirected
    def test_equal(self):
        for i in range(0, 10):
            node1 = random.randint(0, 100000)
            node2 = random.randint(0, 100000)
            while node2 == node1:
                node2 = random.randint(0, 100000)

            self.assertEqual(ux.Edge([node1, node2]), ux.Edge([node2, node1]))
