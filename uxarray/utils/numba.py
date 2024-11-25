import os
import pickle
import numba


def is_numba_function_cached(func):
    """
    Determines if a numba function is cached and up-to-date.

    Returns:
        - True if cache exists and is valid or the input is not a Numba function.
        - False if cache doesn't exist or needs recompilation
    """

    if not hasattr(func, "_cache"):
        return True

    cache = func._cache
    cache_file = cache._cache_file

    # Check if cache file exists
    full_path = os.path.join(cache._cache_path, cache_file._index_name)
    if not os.path.isfile(full_path):
        return False

    try:
        # Load and check version
        with open(full_path, "rb") as f:
            version = pickle.load(f)
            if version != numba.__version__:
                return False

            # Load and check source stamp
            data = f.read()
            stamp, _ = pickle.loads(data)

            # Get current source stamp
            current_stamp = cache._impl.locator.get_source_stamp()

            # Compare stamps
            return stamp == current_stamp

    except (OSError, pickle.PickleError):
        return False
