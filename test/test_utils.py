import sys

from uxarray.utils.numba import is_numba_function_cached
from unittest import TestCase

import unittest
from unittest.mock import MagicMock, patch, mock_open
import os
import pickle
import numba
from uxarray.utils import computing
import numpy as np


class TestNumba(TestCase):
    @patch("os.path.isfile")
    @patch("builtins.open", new_callable=mock_open)
    def test_numba_function_cached_valid_cache(self, mock_open_func, mock_isfile):
        # Mock setup
        mock_isfile.return_value = True
        mock_open_func.return_value.__enter__.return_value.read = MagicMock(return_value=pickle.dumps(("stamp", None)))

        mock_func = MagicMock()
        mock_func._cache._cache_path = "/mock/cache/path"
        mock_func._cache._cache_file._index_name = "mock_cache.pkl"
        mock_func._cache._impl.locator.get_source_stamp.return_value = "stamp"

        # Mock pickle version load
        with patch("pickle.load", side_effect=[numba.__version__]):
            result = is_numba_function_cached(mock_func)

        self.assertTrue(result)

    @patch("os.path.isfile")
    @patch("builtins.open", new_callable=mock_open)
    def test_numba_function_cached_invalid_cache_version(self, mock_open_func, mock_isfile):
        # Mock setup
        mock_isfile.return_value = True
        mock_open_func.return_value.__enter__.return_value.read = MagicMock(return_value=pickle.dumps(("stamp", None)))

        mock_func = MagicMock()
        mock_func._cache._cache_path = "/mock/cache/path"
        mock_func._cache._cache_file._index_name = "mock_cache.pkl"
        mock_func._cache._impl.locator.get_source_stamp.return_value = "stamp"

        # Mock pickle version load with mismatched version
        with patch("pickle.load", side_effect=["invalid_version"]):
            result = is_numba_function_cached(mock_func)

        self.assertFalse(result)

    @patch("os.path.isfile")
    def test_numba_function_cached_no_cache_file(self, mock_isfile):
        # Mock setup
        mock_isfile.return_value = False

        mock_func = MagicMock()
        mock_func._cache._cache_path = "/mock/cache/path"
        mock_func._cache._cache_file._index_name = "mock_cache.pkl"

        result = is_numba_function_cached(mock_func)
        self.assertFalse(result)

    def test_numba_function_cached_no_cache_attribute(self):
        mock_func = MagicMock()
        del mock_func._cache  # Ensure _cache attribute does not exist

        result = is_numba_function_cached(mock_func)
        self.assertTrue(result)

    @patch("os.path.isfile")
    @patch("builtins.open", new_callable=mock_open)
    def test_numba_function_cached_invalid_stamp(self, mock_open_func, mock_isfile):
        # Mock setup
        mock_isfile.return_value = True
        mock_open_func.return_value.__enter__.return_value.read = MagicMock(
            return_value=pickle.dumps(("invalid_stamp", None)))

        mock_func = MagicMock()
        mock_func._cache._cache_path = "/mock/cache/path"
        mock_func._cache._cache_file._index_name = "mock_cache.pkl"
        mock_func._cache._impl.locator.get_source_stamp.return_value = "stamp"

        # Mock pickle version load
        with patch("pickle.load", side_effect=[numba.__version__]):
            result = is_numba_function_cached(mock_func)

        self.assertFalse(result)
