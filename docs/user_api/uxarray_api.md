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

### 2.2.2 Future UxDataArray Methods

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

- Grid.source_grid: str \
  The source file or object for this Grid's definition (For diagnostics and
  reporting purposes).

- Grid.use_dual: boolean \
  A flag indicating if the grid is a MPAS dual mesh.

- Grid.vertices: boolean \
  A flag indicating if the grid is built via vertices.

- Grid.Mesh2: np.float64 xarray.DataArray \
  UGRID Attribute. Indicates the topology data of a 2D unstructured mesh (just
  like the dummy variable "Mesh2" in the UGRID conventions).

- Grid.Mesh2_face_x: np.float64 xarray.DataArray of size (nMesh2_face) \
  UGRID Coordinate Variable. 2D longitude coordinate of face centers in degrees.

- Grid.Mesh2_face_y: np.float64 xarray.DataArray of size (nMesh2_face) \
  UGRID Coordinate Variable. 2D latitude coordinate of face centers in degrees.

- Grid.Mesh2_node_x: np.float64 xarray.DataArray of size (nMesh2_node) \
  UGRID Coordinate Variable. 2D longitude coordinate for nodes on the sphere in
  degrees.

- Grid.Mesh2_node_y: np.float64 xarray.DataArray of size (nMesh2_node) \
  UGRID Coordinate Variable. 2D latitude coordinates for nodes on the sphere in
  degrees.

- Grid.Mesh2_node_cart_x: np.float64 xarray.DataArray of size (nMesh2_node) \
  Coordinate Variable. x coordinates for nodes in meters.

- Grid.Mesh2_node_cart_y: np.float64 xarray.DataArray of size (nMesh2_node) \
  Coordinate Variable. y coordinates for nodes in meters.

- Grid.Mesh2_node_cart_z: np.float64 xarray.DataArray of size (nMesh2_node) \
  Coordinate Variable. z coordinates for nodes in meters.

- Grid.nMaxMesh2_face_nodes: int
  UGRID Dimension. Represents the maximum number of nodes that a face may contain.

- Grid.nMaxMesh2_face_edges: int
  Dimension. Represents the maximum number of edges per face.

- Grid.nMesh2_edge: int
  UGRID Dimension. Represents the total number of edges.

- Grid.nMesh2_face: int
  UGRID Dimension. Represents the total number of faces.

- Grid.nMesh2_node: int
  UGRID Dimension. Represents the total number of nodes.

- Grid.nNodes_per_face: int
  Dimension. Represents the number of non-fill-value nodes per face.

- Grid.Mesh2_edge_nodes: int xarray.DataArray of size (nMesh2_edge, Two)
  (optional) \
  UGRID Connectivity Variable. Maps every edge to the two nodes that it connects

- Grid.Mesh2_face_edges: int xarray.DataArray of size (nMesh2_face,
  nMaxMesh2_face_nodes) (optional) \
  UGRID Connectivity Variable. Maps every face to its edges.

- Grid.Mesh2_face_nodes: int xarray.DataArray of size
  (nMesh2_face, MaxNumNodesPerFace) \
  UGRID Connectivity Variable. Maps each face to its corner nodes.

- Grid.face_areas: np.float64 xarray.DataArray of size (nMesh2_face) \
  Provides areas for each face.

- Grid.parsed_attrs: dict
  Dictionary of parsed attributes from the source grid.

### 3.2.2. Future Grid Attributes

- Grid.Mesh2_node_z: np.float64 xarray.DataArray of size (nMesh2_node) \
  (optional)
  3D z coordinates for nodes on the sphere.

- (*) Grid.edge_dual: Grid \
  The edge dual grid.

- (*) Grid.vertex_dual: Grid \
  The vertex dual grid.

- (*) Grid.Mesh2_edge_types: int DataArray of size (nMesh2_edge)
  (optional; not in UGRID standard) \
  A DataArray indicating the type of edge (0 = great circle arc, 1 = line of
  constant latitude)

- (*) Grid.Mesh2_imask: int DataArray of size (nMesh2_face)
  (optional; not in UGRID standard) \
  The int mask for this grid (1 = face is active; 0 = face is inactive)

- Grid.Mesh2_face_links: int DataArray of size (nMesh2_face,
  MaxNumNodesPerFace) (optional) \
  A DataArray of indices indicating faces that are neighboring each face.

- Grid.Mesh2_edge_faces: int DataArray of size (nMesh2_edge,
  Two) (optional) \
  A DataArray of indices indicating faces that are neighboring each edge.

- Grid.Mesh2_node_faces: int DataArray of size (nMesh2_node,
  MaxNumFacesPerNode) (optional) \
  A DataArray of indices indicating faces that are neighboring each node.

- (*) Grid.Mesh2_latlon_bounds: np.float64 DataArray of size
  (nMesh2_face, Four) (optional; not in UGRID standard) \
  A DataArray of values indicating the latitude-longitude boundaries of
  each face.

- (*) Grid.Mesh2_overlapfaces_a: int DataArray of size
  (nMesh2_face) (optional; not in UGRID standard) \
  A DataArray of indices storing the indices of the parent face from
  grid A, available when this Grid is a supermesh.

- (*) Grid.Mesh2_overlapfaces_b: int DataArray of size
  (nMesh2_face) (optional; not in UGRID standard) \
  A DataArray of indices storing the indices of the parent face from
  grid B, available when this Grid is a supermesh.

## 3.3. Grid Methods

### 3.3.1. Implemented Grid Methods

- Grid.__init__(self, input_obj) \
  Populate Grid object with input_object that can be one of xarray.Dataset,
  ndarray, list, or tuple.  The routine will automatically recognize if it is
  a UGRID, MPAS, SCRIP, or Exodus, or shape file.

- Grid.copy(self) \
  Return a deep copy of self.

- Grid.encode_as(self, str grid_type) \
  Encode a `uxarray.Grid` as a `xarray.Dataset`in the specified grid type
  (UGRID, SCRIP, Exodus).

- Grid.calculate_total_face_area(self, [quadrature_rule, order]) \
  Calculate the total surface area of the whole mesh, i.e. sum of face areas.

- Grid.compute_face_areas(self, [quadrature_rule, order]) \
  Calculate the individual areas of all faces.

### 3.3.2. Future Grid Methods

- (*) Grid.__init__(self, str gridspec) \
  Define a grid specified by gridspec str (analogous to the gridspec
  used in ncremap for grid generation).

- Grid.build_node_face_connectivity(self) \
  Build the node-face connectivity array.

- Grid.build_edge_face_connectivity(self) \
  Build the edge-face connectivity array.

- Grid.build_lat_lon_bounds(self) \
  Build the array of latitude-longitude bounding boxes.

- Grid.encode_as(self, str grid_type) \
  Encode a `uxarray.Grid` as a `xarray.Dataset`in the MPAS or SHP grid type.

- Grid.validate(self) \
  Validate that the grid conforms to the UGRID standards.

- (*) Grid Grid.super_mesh(self, Grid other) \
  Construct the super mesh, consisting of all face edges from Grids self and
  other.
