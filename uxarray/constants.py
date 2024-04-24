import numpy as np

# numpy indexing code is written for np.intp
INT_DTYPE = np.intp
INT_FILL_VALUE = np.iinfo(INT_DTYPE).min


# According to Ogita, Takeshi & Rump, Siegfried & Oishi, Shin’ichi. (2005). Accurate Sum and Dot Product.
#     SIAM J. Scientific Computing. 26. 1955-1988. 10.1137/030601818.
# half of the working precision is already the most optimal value for the error tolerance,
# more tailored values will be used in the future.

ERROR_TOLERANCE = 1.0e-8

ENABLE_JIT_CACHE = True
ENABLE_JIT = True

GRID_DIMS = ["n_node", "n_edge", "n_face"]
