Core (tier 1) functionality is indicated using regular text/list
item style. \
Secondary (tier 2) functionality is indicated using (*) in front.

# 1. class ``uxarray.UxDataset``

The ``uxarray.UxDataset`` class inherits from ``xarray.Dataset``
and A xarray.Dataset-like, and it is multi-dimensional, in
memory, array database. It has the ``Grid`` object, ``uxgrid``, as a
property to be unstructured grid-aware.

## 1.1. UxDataset IO

- uxarray.open_dataset(grid_filename_or_obj, [kwargs]) \
  Open a single dataset, given a grid topology definition.

- uxarray.open_mfdataset(grid_filename_or_obj, paths) \
  Open multiple datasets, given a grid topology definition.

## 1.2. UxDataset Attributes

- UXDataset.uxgrid: ``uxarray.Grid`` \
  ``uxarray.Grid`` property to make ``UxDataset`` unstructured grid-aware

- UxDataset.source_datasets: str \
  Property to keep track of the source data set files or object used to
  instantiate this ``UxDataset`` (For diagnostics and reporting purposes).

## 1.3. UxDataset Methods

### 1.3.1. Implemented UxDataset Methods

- UxDataset.info(self) \
  Concise summary of UxDataset variables and attributes

- np.float64 UxDataset.integrate(self, [quadrature_rule, order]) \
  Integrate dataset variables over all the faces of the given mesh.

### 1.3.2. Future UxDataset Methods

- np.float64 UxDataset.integrate(self, [quadrature_rule, order, Grid region]) \
  Integrate the dataset variables over a region, if specified.

- UxDataset UxDataset.regrid(self, uxarray.Grid target_grid, opts) \
  Regrid the dataset to the target_grid (by default via 1st order FV).

- UxDataset UxDataset.zonal_mean(self, integer bin_count or bin_lats) \
  Compute global zonal means over bincount latitudinal bands.

- (*) UxDataset uxarray.UxDataset.composite(self, nodes) \
  Produce composites of the DataArray at the specific locations via
  stereographic projection.

- (*) UxDataset UxDataset.snapshot(self, nodes) \
  Produce snapshots of the DataArray at the specific locations via
  stereographic projection.

This list will be further populated ...

# 2. class ``uxarray.UxDataArray``

N-dimensional, ``xarray.DataArray``-like array. The
``uxarray.UxDataArray`` class inherits from ``xarray.DataArray``. It has
the ``Grid`` object, `uxgrid`, as a property to be unstructured grid-aware.

## 2.1. UxDataArray Attributes

### 2.1.1. Implemented UxDataArray Attributes

- UXDataArray.uxgrid: ``uxarray.Grid`` \
  ``uxarray.Grid`` property to make ``UxDataArray`` unstructured grid-aware

### 2.1.2. Future UxDataArray Attributes

- UxDataArray.type: enumeration {“vertexcentered”, “facecentered”,
  “faceaverage”, “edgecentered”, “edgeorthogonal”, “edgeparallel”,
  “cgll”, “dgll”} \
  Where data is stored in this UxDataArray.

- (*) UxDataArray.np: integer \
  Polynomial order of data (when using UxDataArray.type = “cgll” or “dgll”)

## 2.2. UxDataArray Methods

### 2.2.1 Implemented UxDataArray Methods

- UxDataArray.integrate(self, [quadrature_rule, order]) \
  Integrate over all the faces of the given mesh.

- ### 2.2.2 Future UxDataArray Methods

- UxDataArray UxDataArray.divergence(self, UxDataArray other) \
  Compute the divergence of the vector field defined by self and other.

- UxDataArray UxDataArray.relative_vorticity(self, UxDataArray other) \
  Compute the vertical component of the vorticity of the vector field defined
  by self and other.

- UxDataArray UxDataArray.laplacian(self) \
  Compute the scalar Laplacian of the scalar field defined by self.

- (UxDataArray, UxDataArray) UxDataArray.gradient(self) \
  Compute the gradient of the scalar field defined by self in spherical
  coordinates.

- UxDataArray UxDataArray.scalardotgradient(self, UxDataArray v, UxDataArray q) \
  Compute the dot product between a vector field (self, v) and the gradient
  of a scalar field q.

This list will be further populated ...

# 3. class uxarray.Grid

Describes an unstructured grid topology. It can be used standalone to explore
an unstructured grid topology, or can be seen as the property of
``uxarray.UxDataset`` and ``uxarray.DataArray`` to make them unstructured
grid-aware data sets and arrays, respectively.

## 3.1. Grid IO
- uxarray.open_grid(grid_filename_or_obj, gridspec, [kwargs]) \
  Create a ``Grid`` object from a grid topology definition.

## 3.2. Grid Attributes

### 3.2.1. Implemented Grid Attributes

- Grid.isconcave: boolean \
  A flag indicating the grid contains concave faces. If this flag is set,
  then alternative algorithms may be needed for some of the operations below.

- Grid.islatlon: boolean \
  A flag indicating the grid is a latitude longitude grid.

- Grid.Mesh2: np.float64 xarray.DataArray \
  UGRID Attribute. Indicates the topology data of a 2D unstructured mesh (just
  like the dummy variable "Mesh2" in the UGRID conventions).

- Grid.Mesh2_node_x: np.float64 xarray.DataArray of size (nMesh2_node) \
  UGRID Coordinate Variable. 2D longitude coordinate or 3D x coordinates for
  nodes on the sphere in degrees.

- Grid.Mesh2_node_y: np.float64 xarray.DataArray of size (nMesh2_node) \
  UGRID Coordinate Variable. 2D latitude coordinate or 3D y coordinates for
  nodes on the sphere in degrees.

- Grid.nMesh2_node: int
  UGRID Dimension. Represents the total number of nodes.

- Grid.parsed_attrs: dict
  Dictionary of parsed attributes from the source grid.

- Grid.Mesh2_face_nodes: int xarray.DataArray of size
  (nMesh2_face, MaxNumNodesPerFace) \
  A xarray.DataArray of indices for each face node, corresponding to coordinates
  in uxarray.Grid.node_*.  Faces can have arbitrary length, with
  _FillValue=-1 used when faces have fewer nodes than MaxNumNodesPerFace.
  Nodes are in counter-clockwise order.

- uxarray.Grid.Mesh2_edge_nodes: int xarray.DataArray of size (nMesh2_edge, Two)
  (optional) \
  A xarray.DataArray of indices for each edge.  Nodes are in arbitrary order.

- Grid.source_grid: str \
  The source file or object for this Grid's definition (For diagnostics and
  reporting purposes).

- Grid.use_dual: boolean \
  A flag indicating if the grid is a MPAS dual mesh.

- Grid.vertices: boolean \
  A flag indicating if the grid is built via vertices.

- (*) uxarray.Grid.edge_dual: uxarray.Grid \
  The edge dual grid.

- (*) uxarray.Grid.vertex_dual: uxarray.Grid \
  The vertex dual grid.

- (*) uxarray.Grid.Mesh2_edge_types: int DataArray of size (nMesh2_edge)
  (optional; not in UGRID standard) \
  A DataArray indicating the type of edge (0 = great circle arc, 1 = line of
  constant latitude)

- uxarray.Grid.Mesh2_face_areas: np.float64 DataArray of size (nMesh2_face)
  (optional; not in UGRID standard) \
  A DataArray providing face areas for each face.

- (*) uxarray.Grid.Mesh2_imask: int DataArray of size (nMesh2_face)
  (optional; not in UGRID standard) \
  The int mask for this grid (1 = face is active; 0 = face is inactive)

- uxarray.Grid.Mesh2_face_edges: int DataArray of size (nMesh2_face,
  MaxNumNodesPerFace) (optional) \
  A DataArray of indices indicating edges that are neighboring each face.

- uxarray.Grid.Mesh2_face_links: int DataArray of size (nMesh2_face,
  MaxNumNodesPerFace) (optional) \
  A DataArray of indices indicating faces that are neighboring each face.

- uxarray.Grid.Mesh2_edge_faces: int DataArray of size (nMesh2_edge,
  Two) (optional) \
  A DataArray of indices indicating faces that are neighboring each edge.

- uxarray.Grid.Mesh2_node_faces: int DataArray of size (nMesh2_node,
  MaxNumFacesPerNode) (optional) \
  A DataArray of indices indicating faces that are neighboring each node.

- (*) uxarray.Grid.Mesh2_latlon_bounds: np.float64 DataArray of size
  (nMesh2_face, Four) (optional; not in UGRID standard) \
  A DataArray of values indicating the latitude-longitude boundaries of
  each face.

- (*) uxarray.Grid.Mesh2_overlapfaces_a: int DataArray of size
  (nMesh2_face) (optional; not in UGRID standard) \
  A DataArray of indices storing the indices of the parent face from
  grid A, available when this Grid is a supermesh.

- (*) uxarray.Grid.Mesh2_overlapfaces_b: int DataArray of size
  (nMesh2_face) (optional; not in UGRID standard) \
  A DataArray of indices storing the indices of the parent face from
  grid B, available when this Grid is a supermesh.

### 3.2.2. Future Grid Attributes

- Grid.Mesh2_node_z: np.float64 xarray.DataArray of size (nMesh2_node) \
  (optional)
  3D z coordinates for nodes on the sphere.

## 3.3. Grid Methods

### 3.3.1. Implemented Grid Methods

- uxarray.Grid.__init__(self, input_obj) \
  Populate Grid object with input_object that can be one of xarray.Dataset,
  ndarray, list, or tuple.  The routine will automatically recognize if it is
  a UGRID, MPAS, SCRIP, or Exodus, or shape file.

- uxarray.Grid.encode_as(self, str grid_type) \
  Encode a `uxarray.Grid` as a `xarray.Dataset`in the specified grid type
  (UGRID, SCRIP, Exodus, or SHP).

- uxarray.Grid.compute_face_areas(self, [quadrature_rule, order]) \
  Calculate the areas of all faces.

### 3.3.2. Future Grid Methods

- (*) uxarray.Grid.__init__(self, str gridspec) \
  Define a grid specified by gridspec str (analogous to the gridspec
  used in ncremap for grid generation).

- uxarray.Grid.build_node_face_connectivity(self) \
  Build the node-face connectivity array.

- Grid.build_edge_face_connectivity(self) \
  Build the edge-face connectivity array.

- Grid.buildlatlon_bounds(self) \
  Build the array of latitude-longitude bounding boxes.

- Grid.encode_as(self, str grid_type) \
  Encode a `uxarray.Grid` as a `xarray.Dataset`in the MPAS grid type.

- Grid.validate(self) \
  Validate that the grid conforms to the UGRID standards.

- (*) Grid Grid.supermesh(self, Grid other) \
  Construct the supermesh, consisting of all face edges from Grids self and
  other.
