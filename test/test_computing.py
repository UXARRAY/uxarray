from unittest import TestCase
import numpy.testing as nt
import numpy as np
from uxarray.grid.coordinates import normalize_in_place
import uxarray.utils.computing as ac_utils
from uxarray.constants import ERROR_TOLERANCE


class TestCrossProduct(TestCase):
    """Since we don't have the multiprecision in current release, we're just
    going to test if the FMA enabled dot product is similar to the np.dot
    one."""

    def test_cross_fma(self):
        v1 = np.array(normalize_in_place([1.0, 2.0, 3.0]))
        v2 = np.array(normalize_in_place([4.0, 5.0, 6.0]))

        np_cross = np.cross(v1, v2)
        fma_cross = ac_utils.cross_fma(v1, v2)
        nt.assert_allclose(np_cross, fma_cross, atol=ERROR_TOLERANCE)


class TestDotProduct(TestCase):
    """Since we don't have the multiprecision in current release, we're just
    going to test if the FMA enabled dot product is similar to the np.dot
    one."""

    def test_dot_fma(self):
        v1 = np.array(normalize_in_place([1.0, 0.0, 0.0]), dtype=np.float64)
        v2 = np.array(normalize_in_place([1.0, 0.0, 0.0]), dtype=np.float64)

        np_dot = np.dot(v1, v2)
        fma_dot = ac_utils.dot_fma(v1, v2)
        nt.assert_allclose(np_dot, fma_dot, atol=ERROR_TOLERANCE)


class TestFMAOperations(TestCase):

    def test_two_sum(self):
        """Test the two_sum function."""
        a = 1.0
        b = 2.0
        s, e = ac_utils._two_sum(a, b)
        self.assertAlmostEquals(a + b, s + e, places=15)

    def test_fast_two_sum(self):
        """Test the fase_two_sum function."""
        a = 2.0
        b = 1.0
        s, e = ac_utils._two_sum(a, b)
        sf, ef = ac_utils._fast_two_sum(a, b)
        self.assertEquals(s, sf)
        self.assertEquals(e, ef)

    def test_two_prod_fma(self):
        """Test the two_prod_fma function."""
        import pyfma
        a = 1.0
        b = 2.0
        x, y = ac_utils._two_prod_fma(a, b)
        self.assertEquals(x, a * b)
        self.assertEquals(y, pyfma.fma(a, b, -x))
        self.assertAlmostEquals(a * b, x + y, places=15)

    def test_fast_two_mult(self):
        """Test the two_prod_fma function."""
        a = 1.0
        b = 2.0
        x, y = ac_utils._two_prod_fma(a, b)
        xf, yf = ac_utils._fast_two_mult(a, b)
        self.assertEquals(x, xf)
        self.assertEquals(y, yf)

    def test_err_fmac(self):
        """Test the _err_fmac function."""
        import pyfma
        a = 1.0
        b = 2.0
        c = 3.0
        x, y, z = ac_utils._err_fmac(a, b, c)
        self.assertEquals(x, pyfma.fma(a, b, c))
        self.assertAlmostEquals(a * b + c, x + y + z, places=15)


class TestAccurateSum(TestCase):

    def test_vec_sum(self):
        """Test the _vec_sum function."""
        a = np.array([1.0, 2.0, 3.0])
        res = ac_utils._vec_sum(a)
        self.assertAlmostEquals(6.0, res, places=15)
        import gmpy2
        a = gmpy2.mpfr('2.28888888888')
        b = gmpy2.mpfr('-2.2888889999')
        c = gmpy2.mpfr('0.000000000001')
        d = gmpy2.mpfr('-0.000000000001')

        a_float = float(a)
        b_float = float(b)
        c_float = float(c)
        d_float = float(d)

        res = ac_utils._vec_sum(np.array([a_float, b_float, c_float, d_float]))
        res_mp = gmpy2.mpfr(a_float) + gmpy2.mpfr(b_float) + gmpy2.mpfr(
            c_float) + gmpy2.mpfr(d_float)
        abs_res = abs(res - res_mp)
        self.assertTrue(
            gmpy2.cmp(abs_res, gmpy2.mpfr(np.finfo(np.float64).eps)) == -1)


class TestNorm(TestCase):

    def test_norm_faithful(self):
        """Test the norm_faithful function."""
        a = np.array([1.0, 2.0, 3.0])
        res = ac_utils._norm_faithful(a)
        self.assertAlmostEquals(np.linalg.norm(a), res, places=15)

    def test_sqrt_faithful(self):
        """Test the sqrt_faithful function."""
        a = 10.0
        res = ac_utils._acc_sqrt(a, 0.0)
        self.assertAlmostEquals(np.sqrt(a), res, places=15)

    def test_two_square(self):
        """Test the _two_square function."""
        a = 10.0
        res = ac_utils._two_square(a)
        self.assertAlmostEquals(a * a, res[0], places=15)
