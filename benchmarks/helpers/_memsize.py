
__all__ = ["grid_nbytes", "dataset_nbytes"]


def grid_nbytes(uxgrid):
    """Total size of the arrays a ``Grid`` currently holds, in bytes."""
    return uxgrid._ds.nbytes


def dataset_nbytes(uxds):
    """Total size of a ``UxDataset``, in bytes, including its grid."""
    return uxds.nbytes + grid_nbytes(uxds.uxgrid)
