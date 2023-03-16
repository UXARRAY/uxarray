import numpy as np

# is 32-bit safe or should we use 64-bit?
INT_DTYPE = np.uint32
FILL_VALUE = np.iinfo(INT_DTYPE).max
