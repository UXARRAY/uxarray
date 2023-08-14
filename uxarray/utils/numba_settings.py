import uxarray.constants


def enable_jit_cache():
    """Allows Numba's JIT cache to be turned on.

    This cache variable lets @njit cache the machine code generated
    between runs, allowing for faster run times due to the fact that the
    code doesn't need to regenerate the machine code every run time. Our
    use case here was to study performance, in regular usage one might
    never turn off caching as it will only help if frequently modifying
    the code or because users have very limited disk space. The default
    is on (True)
    """
    uxarray.constants.ENABLE_JIT_CACHE = True


def disable_jit_cache():
    """Allows Numba's JIT cache to be turned on off.

    This cache variable lets @njit cache the machine code generated
    between runs, allowing for faster run times due to the fact that the
    code doesn't need to regenerate the machine code every run time. Our
    use case here was to study performance, in regular usage one might
    never turn off caching as it will only help if frequently modifying
    the code or because users have very limited disk space. The default
    is on (True)
    """
    uxarray.constants.ENABLE_JIT_CACHE = False


def enable_jit():
    """Allows Numba's JIT application to be turned on.

    This lets users choose whether they want machine code to be
    generated to speed up the performance of the code on large files.
    The default is on (True)
    """
    uxarray.constants.ENABLE_JIT = True


def disable_jit():
    """Allows Numba's JIT application to be turned off.

    This lets users choose whether they want machine code to be
    generated to speed up the performance of the code on large files.
    The default is on (True)
    """
    uxarray.constants.ENABLE_JIT = False
