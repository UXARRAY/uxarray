Core (tier 1) functionality is indicated using regular text/list
item style. \
Secondary (tier 2) functionality is indicated using (*) in front.

# class uxarray.Grid
Describes an unstructured grid.

## uxarray.Grid Attributes

- uxarray.Grid.ds: DataSet\
  DataSet containing uxarray.Grid properties
  `dims={nMesh2_node (total number of nodes),
  nMesh2_face (number of faces),
  MaxNumNodesPerFace (maximum number of nodes per face)}` \
  `optional_dims={nMesh2_edge (number of edges, optional),
  MaxNumFacesPerNode (max number of faces per node),Two, Three, Four}`


- uxarray.Grid.filename: string \
  Original filename for this uxarray.Grid.


- (*) uxarray.Grid.islatlon: boolean \
  A flag indicating the grid is a latitude longitude grid.


- (*) uxarray.Grid.isconcave: boolean \
  A flag indicating the grid contains concave faces.  If
  this flag is set then alternative algorithms may be needed
  for some of the operations below.


- (*) uxarray.Grid.edgedual: uxarray.Grid \
  The edge dual grid.


- (*) uxarray.Grid.vertexdual: uxarray.Grid \
  The vertex dual grid.


According to the UGRID specification, the UGRID file should
contain a dummy variable with attribute cf_role and value
“mesh_topology”.  This variable stores information on mesh
topology, including relevant variable names.  The API will
need to search for the variable containing this attribute
and throw an error if it is missing.  Following the UGRID
specification guide, the code below uses the name “Mesh2”
for the dummy variable, but this could be different. The
other names below are the ones used in the UGRID standards
document, but they could be different.


- uxarray.Grid.Mesh2_node_x: np.float64 DataArray of size (nMesh2_node) \
  2D longitude coordinate or 3D x coordinates for nodes on the sphere.


- uxarray.Grid.Mesh2_node_y: np.float64 DataArray of size (nMesh2_node) \
  2D latitude coordinate or 3D y coordinates for nodes on the sphere.


- uxarray.Grid.Mesh2_node_z: np.float64 DataArray of size (nMesh2_node) \
  (optional)
  3D z coordinates for nodes on the sphere.


- (*) uxarray.Grid.Mesh2_node_coordinates: np.float64 DataArray of size
  (nMesh2_node, Two or Three) \
  Alternative storage mechanism for node information.


- uxarray.Grid.Mesh2_face_nodes: integer DataArray of size
  (nMesh2_face, MaxNumNodesPerFace) \
  A DataArray of indices for each face node, corresponding to coordinates
  in uxarray.Grid.node_*.  Faces can have arbitrary length, with
  _FillValue=-1 used when faces have fewer nodes than MaxNumNodesPerFace.
  Nodes are in counter-clockwise order.


- uxarray.Grid.Mesh2_edge_nodes: integer DataArray of size (nMesh2_edge, Two)
  (optional) \
  A DataArray of indices for each edge.  Nodes are in arbitrary order.


- (*) uxarray.Grid.Mesh2_edge_types: integer DataArray of size (nMesh2_edge)
  (optional; not in UGRID standard) \
  A DataArray indicating the type of edge (0 = great circle arc, 1 = line of
  constant latitude)


- uxarray.Grid.Mesh2_face_areas: np.float64 DataArray of size (nMesh2_face)
  (optional; not in UGRID standard) \
  A DataArray providing face areas for each face.


- (*) uxarray.Grid.Mesh2_imask: integer DataArray of size (nMesh2_face)
  (optional; not in UGRID standard) \
  The integer mask for this grid (1 = face is active; 0 = face is inactive)


- uxarray.Grid.Mesh2_face_edges: integer DataArray of size (nMesh2_face,
  MaxNumNodesPerFace) (optional) \
  A DataArray of indices indicating edges that are neighboring each face.


- uxarray.Grid.Mesh2_face_links: integer DataArray of size (nMesh2_face,
  MaxNumNodesPerFace) (optional) \
  A DataArray of indices indicating faces that are neighboring each face.


- uxarray.Grid.Mesh2_edge_faces: integer DataArray of size (nMesh2_edge,
  Two) (optional) \
  A DataArray of indices indicating faces that are neighboring each edge.


- uxarray.Grid.Mesh2_node_faces: integer DataArray of size (nMesh2_node,
  MaxNumFacesPerNode) (optional) \
  A DataArray of indices indicating faces that are neighboring each node.


- (*) uxarray.Grid.Mesh2_latlon_bounds: np.float64 DataArray of size
  (nMesh2_face, Four) (optional; not in UGRID standard) \
  A DataArray of values indicating the latitude-longitude boundaries of
  each face.


- (*) uxarray.Grid.Mesh2_overlapfaces_a: integer DataArray of size
  (nMesh2_face) (optional; not in UGRID standard) \
  A DataArray of indices storing the indices of the parent face from
  grid A, available when this Grid is a supermesh.


- (*) uxarray.Grid.Mesh2_overlapfaces_b: integer DataArray of size
  (nMesh2_face) (optional; not in UGRID standard) \
  A DataArray of indices storing the indices of the parent face from
  grid B, available when this Grid is a supermesh.


## uxarray.Grid  Functions

- uxarray.Grid.__init__(self, string file) \
  Load the grid file specified by file.  The routine will automatically
  detect if it is a UGrid, SCRIP, Exodus, or shape file.


- (*) uxarray.Grid.__init__(self, string gridspec) \
  Define a grid specified by gridspec string (analogous to the gridspec
  used in ncremap for grid generation).


- uxarray.Grid.__init__(self, np.float64.list vertices) \
  Create a grid with one face with vertices specified by the given argument.


- uxarray.Grid.write(self, string file, string format) \
  Write a uxgrid to a file with specified format (UGRID, SCRIP, Exodus,
  or SHP).


- uxarray.Grid.calculatefaceareas(self) \
  Calculate the area of all faces.


- uxarray.Grid.build_node_face_connectivity(self) \
  Build the node-face connectivity array.


- uxarray.Grid.build_edge_face_connectivity(self) \
  Build the edge-face connectivity array.


- uxarray.Grid.buildlatlon_bounds(self) \
  Build the array of latitude-longitude bounding boxes.


- uxarray.Grid.validate(self) \
  Validate that the grid conforms to the UGRID standards.


## Additional xarray.DataArray Attributes

- xarray.DataArray.grid: uxarray.Grid \
  The grid associated with this xarray.DataArray.


- xarray.type: enumeration {“vertexcentered”, “facecentered”,
  “faceaverage”, “edgecentered”, “edgeorthogonal”, “edgeparallel”,
  “cgll”, “dgll”} \
  Where data is stored in this DataArray.


- (*) xarray.np: integer \
  Polynomial order of data (when using xarray.type = “cgll” or “dgll”)


## Helper Functions

- np.float64 uxarray.integrate(self, xarray.DataArray q,
  uxarray.Grid region (optional)) \
  Integrate the DataArray globally or over a specified region
  (if specified).


- xarray.DataSet uxarray.zonalmean(self, xarray.DataArray q,
  integer bincount or binlats) \
  Compute global zonal means over bincount latitudinal bands.


- xarray.DataSet uxarray.regrid(self, xarray.DataArray q,
  uxarray.Grid targetgrid, opts) \
  Regrid the data to the target grid (by default via 1st order FV).


- xarray.DataArray uxarray.divergence(xarray.DataArray u,
  xarray.DataArray v) \
  Compute the divergence of the vector field defined by u and v.


- xarray.DataArray uxarray.relative_vorticity(xarray.DataArray u,
  xarray.DataArray v) \
  Compute the vertical component of the vorticity of the vector field defined by u and v.


- xarray.DataArray uxarray.laplacian(xarray.DataArray q) \
  Compute the scalar Laplacian of the scalar field q.


- (xarray.DataArray, xarray.DataArray) uxarray.gradient(xarray.DataArray q) \
  Compute the gradient of the scalar field q in spherical coordinates.


- xarray.DataArray uxarray.scalardotgradient(xarray.DataArray u,
  xarray.DataArray v, xarray.DataArray q) \
  Compute the dot product between a vector field (u,v) and the gradient of a scalar field q.


- (*) xarray.Grid uxarray.supermesh(uxarray.Grid a, uxarray.Grid b) \
  Construct the supermesh, consisting of all face edges from Grids a and b.


- (*) xarray.DataSet xarray.DataSet.snapshot(self, nodes) \
  Produce snapshots of the DataArray at the specific locations via stereographic projection.


- (*) xarray.DataSet xarray.DataSet.composite(self, nodes) \
  Produce composites of the DataArray at the specific locations via stereographic projection.
