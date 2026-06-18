.. currentmodule:: uxarray

Remap with Weights
==================

UXarray can apply precomputed offline remapping weights produced outside of UXarray.
This is useful when weights are generated once with tools such as ESMF or
TempestRemap and then reused many times across multiple ensemble members, time
slices, or variables.

The core workflow is:

1. Generate a weight file for a specific source grid and destination grid.
2. Apply it with :meth:`UxDataArray.remap.apply_weights` or :meth:`UxDataset.remap.apply_weights`.

Basic Usage
-----------

.. code-block:: python

   import uxarray as ux

   src = ux.open_dataset("source_grid.nc", "source_data.nc")
   dst = ux.open_grid("destination_grid.nc")

   remapped_temperature = src["temperature"].remap.apply_weights(dst, "map.nc")
   remapped_dataset = src.remap.apply_weights(dst, "map.nc")

Repeated calls with the same path reuse a cached sparse operator, so applying the
same file again in one Python session avoids rebuilding the matrix.

What A Weight File Represents
-----------------------------

A remap weight file represents a linear operator from one grid to another:

.. code-block:: text

   target_values = W @ source_values

If the source grid has ``4800`` elements and the destination grid has ``11000``
elements, then:

- ``source_values.shape = (4800,)``
- ``W.shape = (11000, 4800)``
- ``target_values.shape = (11000,)``

So the weight file necessarily encodes both the source grid and the destination
grid. It is specific to that grid pair and to the ordering of the source and
destination degrees of freedom.

Supported File Structure
------------------------

UXarray currently supports the standard sparse offline-map structure used by
ESMF-style and TempestRemap-style map files. The essential pieces are:

- ``n_a``: source size
- ``n_b``: destination size
- ``n_s``: number of nonzero entries
- ``row``: destination indices
- ``col``: source indices
- ``S``: sparse weight values

Common aliases are also accepted:

- ``src_grid_size`` and ``dst_grid_size``
- ``src_address`` and ``dst_address``
- ``weights`` instead of ``S``

In full offline map files, these sparse arrays are typically accompanied by
source and destination metadata such as center coordinates, corner coordinates,
areas, masks, and grid-dimension metadata.

Tool Compatibility
------------------

This implementation was verified against real files from both families:

- ESMF-generated offline map files created with ``ESMF_RegridWeightGen``
- TempestRemap-generated offline map files created with ``GenerateOfflineMap``

In practice, UXarray supports the standard full offline map format used by both
tools.

Currently, this API applies externally generated sparse remap files. Generating reusable UXarray weight maps can be added as a future extension.

Current caveats:

- The source data ordering must match the source ordering encoded in the weight file.
- Not every possible file variant is guaranteed yet.
- ESMF ``weight_only`` outputs may require additional handling if they omit
  source and destination size metadata.

How It Applies Data
-------------------

When remapping a :class:`UxDataArray` or :class:`UxDataset`, UXarray identifies a
single spatial dimension whose size matches the source size in the loaded
weights. That dimension is remapped to the requested destination dimension
(``faces``, ``edges``, or ``nodes``).

Non-spatial dimensions are preserved, which makes this workflow suitable for
reusing one operator across many time steps, ensemble members, or variables.

Why Use This Workflow
---------------------

This path is useful when:

- weight generation is expensive and should be done once
- remapping needs to be repeated many times
- external tools already produce trusted offline maps
- you want to stay in Python for applying the map and preserving array metadata
