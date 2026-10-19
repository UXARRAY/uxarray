"""Conventions related to names of attrs"""

# list of attrs which describe the values contained in the array,
# and thus should probably be dropped by default for a derived quantity.
_VALUE_DESCRIBING_ATTRS = [
    # What the quantity is
    "standard_name",
    "long_name",
    "units",
    "units_metadata",
    # Valid/observed values (may cause masking on netcdf read-back)
    "valid_min",
    "valid_max",
    "valid_range",
    "actual_range",
    # Missing-data sentinels
    "_FillValue",
    "missing_value",
    # Packing / scaling
    "scale_factor",
    "add_offset",
    "_Unsigned",
    # Categorical flags
    "flag_values",
    "flag_masks",
    "flag_meanings",
    # Provenance tied to the input's values
    "cell_methods",
    "ancillary_variables",
    "standard_error_multiplier",
    # might also consider "description" and "comment",
    # but not including those here yet.
]


def _drop_value_describing_attrs(obj):
    """returns obj but with value-describing attrs removed from obj.attrs.
    if obj contained no value-describing attrs, result is obj, exactly,
    else returns a copy of obj with modified attrs dict.

    obj: xarray.DataArray, xarray.Dataset, or subclass.
        must have obj.attrs, and must be able to call obj.copy(deep=False).
    """
    new_attrs = {k: v for k, v in obj.attrs.items() if k not in _VALUE_DESCRIBING_ATTRS}
    if len(new_attrs) != len(obj.attrs):
        obj = obj.copy(deep=False)
        obj.attrs = new_attrs
    return obj
