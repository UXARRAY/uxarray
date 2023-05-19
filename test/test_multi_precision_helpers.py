

import os
import numpy as np
import gmpy2
from gmpy2 import mpfr

from unittest import TestCase
from pathlib import Path

import uxarray as ux

from uxarray.constants import INT_DTYPE, INT_FILL_VALUE, FLOAT_PRECISION_BITS
from  uxarray.multi_precision_helpers import convert_to_mpfr, unique_coordinates_mpfr, precision_bits_to_decimal_digits, decimal_digits_to_precision_bits
import math

try:
    import constants
except ImportError:
    from . import constants

# Data files
current_path = Path(os.path.dirname(os.path.realpath(__file__)))

exodus = current_path / "meshfiles" / "exodus" / "outCSne8" / "outCSne8.g"
ne8 = current_path / 'meshfiles' / "scrip" / "outCSne8" / 'outCSne8.nc'
err_tolerance = 1.0e-12


class TestMultiPrecision(TestCase):
    def test_convert_to_mpfr(self):
        """Tests if the convert_to_mpfr() helper function converts a numpy
        array to a numpy array of the correct dtype."""
        # test different datatypes for face_nodes
        test_precision = 64
        f0_deg = np.array([np.array([120, -20]), np.array([130, -10]), np.array([120, 0]),
                           np.array([105, 0]), np.array([95, -10]), np.array([105, -20])])
        f1_deg = np.array([np.array([120, 0]), np.array([120, 10]), np.array([115, 0]),
                           np.array([ux.INT_FILL_VALUE, ux.INT_FILL_VALUE]),
                           np.array([ux.INT_FILL_VALUE, ux.INT_FILL_VALUE]),
                           np.array([ux.INT_FILL_VALUE, ux.INT_FILL_VALUE])])
        f2_deg = np.array([np.array([115, 0]), np.array([120, 10]), np.array([100, 10]),
                           np.array([105, 0]), np.array([ux.INT_FILL_VALUE, ux.INT_FILL_VALUE]),
                           np.array([ux.INT_FILL_VALUE, ux.INT_FILL_VALUE])])
        f3_deg = np.array([np.array([95, -10]), np.array([105, 0]), np.array([95, 30]),
                           np.array([80, 30]), np.array([70, 0]), np.array([75, -10])])
        f4_deg = np.array([np.array([65, -20]), np.array([75, -10]), np.array([70, 0]),
                           np.array([55, 0]), np.array([45, -10]), np.array([55, -20])])
        f5_deg = np.array([np.array([70, 0]), np.array([80, 30]), np.array([70, 30]),
                           np.array([60, 0]), np.array([ux.INT_FILL_VALUE, ux.INT_FILL_VALUE]),
                           np.array([ux.INT_FILL_VALUE, ux.INT_FILL_VALUE])])
        f6_deg = np.array([np.array([60, 0]), np.array([70, 30]), np.array([40, 30]),
                           np.array([45, 0]), np.array([ux.INT_FILL_VALUE, ux.INT_FILL_VALUE]),
                           np.array([ux.INT_FILL_VALUE, ux.INT_FILL_VALUE])])

        verts = np.array([f0_deg, f1_deg, f2_deg, f3_deg, f4_deg, f5_deg, f6_deg])

        verts_mpfr = convert_to_mpfr(verts, str_mode=False,precision=test_precision)

        # Test if every object in the verts_mpfr array is of type mpfr
        for i in range(verts_mpfr.shape[0]):
            for j in range(verts_mpfr.shape[1]):
                self.assertEqual(verts_mpfr[i, j][0].precision, test_precision)
                self.assertEqual(verts_mpfr[i, j][1].precision, test_precision)

                # Then compare the values between verts and verts_mpfr up to the 53 bits of precision
                self.assertAlmostEqual(verts[i, j][0], verts_mpfr[i, j][0], places=FLOAT_PRECISION_BITS)
                self.assertAlmostEqual(verts[i, j][1], verts_mpfr[i, j][1], places=FLOAT_PRECISION_BITS)

    def test_mpfr_unique_normal_case(self):
        """
        The input cartesian coordinates represents 8 vertices on a cube
             7---------6
            /|        /|
           / |       / |
          3---------2  |
          |  |      |  |
          |  4------|--5
          | /       | /
          |/        |/
          0---------1
        """

        test_precision = 64
        cart_x = [
            0.577340924821405, 0.577340924821405, 0.577340924821405,
            0.577340924821405, -0.577345166204668, -0.577345166204668,
            -0.577345166204668, -0.577345166204668
        ]
        cart_y = [
            0.577343045516932, 0.577343045516932, -0.577343045516932,
            -0.577343045516932, 0.577338804118089, 0.577338804118089,
            -0.577338804118089, -0.577338804118089
        ]
        cart_z = [
            0.577366836872017, -0.577366836872017, 0.577366836872017,
            -0.577366836872017, 0.577366836872017, -0.577366836872017,
            0.577366836872017, -0.577366836872017
        ]

        # The order of the vertexes is irrelevant, the following indexing is just for forming a face matrix
        face_vertices = [
            [0, 1, 2, 3],  # front face
            [1, 5, 6, 2],  # right face
            [5, 4, 7, 6],  # back face
            [4, 0, 3, 7],  # left face
            [3, 2, 6, 7],  # top face
            [4, 5, 1, 0]  # bottom face
        ]

        # Pack the cart_x/y/z into the face matrix using the index from face_vertices
        faces_coords = []
        for face in face_vertices:
            face_coords = []
            for vertex_index in face:
                x, y, z = cart_x[vertex_index], cart_y[vertex_index], cart_z[
                    vertex_index]
                face_coords.append([x, y, z])
            faces_coords.append(face_coords)

        # Now consturct the grid using the faces_coords
        verts_cart = np.array(faces_coords)
        verts_cart_mpfr = convert_to_mpfr(verts_cart, str_mode=False, precision=test_precision)
        verts_cart_mpfr_unique, unique_inverse = unique_coordinates_mpfr(verts_cart_mpfr.reshape(
            -1, verts_cart_mpfr.shape[-1]), precision=test_precision)
        recovered_verts_cart_mpfr = verts_cart_mpfr_unique[unique_inverse]

        # Compare the recovered verts_cart_mpfr with the original verts_cart_mpfr.reshape(-1, verts_cart_mpfr.shape[-1])
        expected = verts_cart_mpfr.reshape(-1, verts_cart_mpfr.shape[-1])
        for i in range(recovered_verts_cart_mpfr.shape[0]):
            for j in range(recovered_verts_cart_mpfr.shape[1]):
                self.assertEqual(recovered_verts_cart_mpfr[i, j].precision, test_precision)
                # Then compare the values between verts and verts_mpfr up to the 53 bits of precision
                self.assertAlmostEqual(expected[i, j], recovered_verts_cart_mpfr[i, j], places=FLOAT_PRECISION_BITS)

    def test_mpfr_unique_extreme_case(self):
        coord1 = [gmpy2.mpfr('1.23456789'), gmpy2.mpfr('2.34567890'), gmpy2.mpfr('3.45678901')]
        coord2 = [gmpy2.mpfr('1.23456789'), gmpy2.mpfr('2.34567890'), gmpy2.mpfr('3.45678901')]
        # Convert the coordinates to string format
        coord1_str = ["{0:.17f}".format(coord) for coord in coord1]
        coord2_str = ["{0:.17f}".format(coord) for coord in coord2]

        # Create the final coordinates with the differing 64th bit
        coord1_final = [coord + '0' for coord in coord1_str]
        coord2_final = [coord + '1' for coord in coord2_str]
        verts = np.array([coord1_final, coord2_final])
        bit_precision = decimal_digits_to_precision_bits(18)
        verts_60 = convert_to_mpfr(verts, str_mode=True, precision=bit_precision)
        verts_57 = convert_to_mpfr(verts, str_mode=True, precision=bit_precision - 1)

        verts_cart_mpfr_unique_64, unique_inverse_64 = unique_coordinates_mpfr(verts_60, precision=bit_precision)
        verts_cart_mpfr_unique_57, unique_inverse_57 = unique_coordinates_mpfr(verts_57, precision=bit_precision - 1)
        self.assertTrue(len(verts_cart_mpfr_unique_64) == 2)
        self.assertTrue(len(verts_cart_mpfr_unique_57) == 1)

        coord1 = [gmpy2.mpfr('1.23456789'), gmpy2.mpfr('2.34567890'), gmpy2.mpfr('3.45678901')]
        coord2 = [gmpy2.mpfr('1.23456789'), gmpy2.mpfr('2.34567890'), gmpy2.mpfr('3.45678901')]
        # Convert the coordinates to string format
        precision_bits = 200
        decimal_digits = precision_bits_to_decimal_digits(precision_bits - 1)
        check = decimal_digits_to_precision_bits(decimal_digits)

        format_str = "{0:." + str(decimal_digits) + "f}"
        coord1_str = [format_str.format(coord) for coord in coord1]
        coord2_str = [format_str.format(coord) for coord in coord2]

        # Create the final coordinates with the differing 64th bit
        coord1_final = [coord + '0' for coord in coord1_str]
        coord2_final = [coord + '1' for coord in coord2_str]
        verts = np.array([coord1_final, coord2_final])

        verts_200 = convert_to_mpfr(verts, str_mode=True, precision=precision_bits)
        verts_199 = convert_to_mpfr(verts, str_mode=True, precision=check - 1)

        verts_cart_mpfr_unique_200, unique_inverse_200 = unique_coordinates_mpfr(verts_200, precision=precision_bits)
        verts_cart_mpfr_unique_199, unique_inverse_199 = unique_coordinates_mpfr(verts_199, precision=check - 1)
        self.assertTrue(len(verts_cart_mpfr_unique_200) == 2)
        self.assertTrue(len(verts_cart_mpfr_unique_199) == 1)




