import numpy as np
from gmpy2 import mpz

# numpy indexing code is written for np.intp
INT_DTYPE = np.intp
INT_FILL_VALUE = np.iinfo(INT_DTYPE).min
INT_FILL_VALUE_MPZ = mpz(str(INT_FILL_VALUE))
FLOAT_PRECISION_BITS = 53
ERROR_TOLERANCE = 1.0e-15
