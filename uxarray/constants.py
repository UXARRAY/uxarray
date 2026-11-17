import numpy as np

# numpy indexing code is written for np.intp
INT_DTYPE = np.intp
INT_FILL_VALUE = np.iinfo(INT_DTYPE).min


# According to Ogita, Takeshi & Rump, Siegfried & Oishi, Shin’ichi. (2005). Accurate Sum and Dot Product.
#     SIAM J. Scientific Computing. 26. 1955-1988. 10.1137/030601818.
# half of the working precision is already the most optimal value for the error tolerance,
# more tailored values will be used in the future.

ERROR_TOLERANCE = np.float64(1.0e-8)

# The below value is the machine epsilon for the float64 data type, it will be used in the most basic operations as a
# error tolerance, mainly in the intersection calculations.
MACHINE_EPSILON = np.float64(np.finfo(float).eps)

ENABLE_FMA = False

GRID_DIMS = ["n_node", "n_edge", "n_face"]

WGS84_CRS = "EPSG:4326"


# Threshold the adaptive ``edge_node`` sort in ``uxarray.grid.utils`` is built around, in two
# related roles. Above a bucket of ``2 * EDGE_NODE_SORT_THRESHOLD``, an insertion sort is abandoned
# for a heap sort once it has done ``EDGE_NODE_SORT_THRESHOLD`` shifts per edge, keeping
# that path linear while leaving ample room for the near-sorted input it is chosen for. At or below
# that size, the bookkeeping would never pay for itself: a bucket of ``size`` holds at most
# ``size * (size - 1) / 2`` inversions, too few to exhaust the budget, so the plain insertion
# sort is used instead.
EDGE_NODE_SORT_THRESHOLD = 8


# Minimum thresholds for whether to cull _point_in_face_query candidate faces by their own radii
# (a face can't contain a point if the point is outside the face's radius).
# This is for efficiency, only; changing these values will not affect correctness.
POINT_IN_FACE_CULL_THRESHOLDS = {
    # Values here were picked as rough estimates without any fine-tuning;
    # there may be room to optimize further with more tuning, if needed.
    "CANDS_PER_POINT": 10,  # minimum n_candidates / n_points
    "CANDS_PER_FACE": 1.0,  # minimum n_candidates / n_face
}
