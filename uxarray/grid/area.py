import numpy as np
from numba import njit

from uxarray.constants import ERROR_TOLERANCE


@njit(cache=True)
def calculate_face_area(
    x,
    y,
    z,
    quadrature_rule="gaussian",
    order=4,
    latitude_adjusted_area=False,
):
    """Calculate area of a face on sphere.

    Parameters
    ----------
    x : list, required
        x-coordinate of all the nodes forming the face

    y : list, required
        y-coordinate of all the nodes forming the face

    z : list, required
        z-coordinate of all the nodes forming the face

    quadrature_rule : str, optional
        triangular and Gaussian quadrature supported, expected values: "triangular" or "gaussian"

    order: int, optional
        Order of the quadrature rule. Default is 4.

        Supported values:
            - Gaussian Quadrature: 1 to 10
            - Triangular: 1, 4, 8, 10 and 12

    latitude_adjusted_area : bool, optional
        If True, performs the check if any face consists of an edge that has constant latitude, modifies the area of that face by applying the correction term due to that edge. Default is False.

    Returns
    -------
    area : double
    jacobian: double
    """
    area = 0.0  # set area to 0
    jacobian = 0.0  # set jacobian to 0
    order = order

    if quadrature_rule == "gaussian":
        dG, dW = get_gauss_quadrature_dg(order)
    elif quadrature_rule == "triangular":
        dG, dW = get_tri_quadrature_dg(order)
    else:
        raise ValueError("Invalid quadrature rule, specify gaussian or triangular")

    num_nodes = len(x)

    # num triangles is two less than the total number of nodes
    num_triangles = num_nodes - 2
    # Using tempestremap GridElements: https://github.com/ClimateGlobalChange/tempestremap/blob/master/src/GridElements.cpp
    # loop through all sub-triangles of face
    total_correction = 0.0
    for j in range(0, num_triangles):
        node1 = np.array([x[0], y[0], z[0]], dtype=x.dtype)
        node2 = np.array([x[j + 1], y[j + 1], z[j + 1]], dtype=x.dtype)
        node3 = np.array([x[j + 2], y[j + 2], z[j + 2]], dtype=x.dtype)

        for p in range(len(dW)):
            if quadrature_rule == "gaussian":
                for q in range(len(dW)):
                    dA = dG[0][p]
                    dB = dG[0][q]
                    jacobian = calculate_spherical_triangle_jacobian(
                        node1, node2, node3, dA, dB
                    )
                    area += dW[p] * dW[q] * jacobian
                    jacobian += jacobian
            elif quadrature_rule == "triangular":
                dA = dG[p][0]
                dB = dG[p][1]
                jacobian = calculate_spherical_triangle_jacobian_barycentric(
                    node1, node2, node3, dA, dB
                )
                area += dW[p] * jacobian
                jacobian += jacobian

    # check if the any edge is on the line of constant latitude
    # which means we need to check edges for same z-coordinates and call area correction routine
    correction = 0.0
    # TODO: Make this work when latitude_adjusted_area is False and each edge has a flag that indicates if it is a constant latitude edge
    if latitude_adjusted_area:
        for i in range(num_nodes):
            node1 = np.array([x[i], y[i], z[i]], dtype=x.dtype)
            node2 = np.array(
                [
                    x[(i + 1) % num_nodes],
                    y[(i + 1) % num_nodes],
                    z[(i + 1) % num_nodes],
                ],
                dtype=x.dtype,
            )
            # Check if z-coordinates are approximately equal
            if np.isclose(node1[2], node2[2], atol=ERROR_TOLERANCE):
                # Check if z-coordinates are approximately 0 - Equator
                if np.abs(node1[2]) < ERROR_TOLERANCE:
                    continue

                # Check if the edge passes through a pole
                passes_through_pole = edge_passes_through_pole(node1, node2)
                if passes_through_pole:
                    # Skip the edge if it passes through a pole
                    continue

                z_sign = np.sign(node1[2])
                # Convert Cartesian coordinates to longitude
                lon1 = np.arctan2(node1[1], node1[0])
                lon2 = np.arctan2(node2[1], node2[0])

                # Calculate the longitude difference in radians, handling wraparound
                lon_diff = lon2 - lon1
                # Normalize longitude difference to [-π, π]
                while lon_diff > np.pi:
                    lon_diff -= 2 * np.pi
                while lon_diff < -np.pi:
                    lon_diff += 2 * np.pi

                # Skip the edge if it spans more than 180 degrees of longitude
                if abs(lon_diff) > np.pi:
                    continue

                # Calculate the correction term
                correction = area_correction(node1, node2)

                # Check if the longitude is increasing in the northern hemisphere or decreasing in the southern hemisphere
                if (z_sign > 0 and lon_diff > 0) or (z_sign < 0 and lon_diff < 0):
                    correction = -correction

                total_correction += correction

    if total_correction != 0.0:
        area += total_correction

    return area, jacobian


@njit(cache=True)
def edge_passes_through_pole(node1, node2):
    """
    Check if the edge passes through a pole.

    Parameters:
    - node1: first node of the edge (normalized).
    - node2: second node of the edge (normalized).

    Returns:
    - bool: True if the edge passes through a pole, False otherwise.
    """
    # Calculate the normal vector to the plane defined by the origin, node1, and node2
    n = np.cross(node1, node2)

    # Check for numerical stability issues with the normal vector
    if np.allclose(n, 0):
        # Handle cases where the cross product is near zero, such as when nodes are nearly identical or opposite
        return False

    # North and South Pole vectors
    p_north = np.array([0.0, 0.0, 1.0])
    p_south = np.array([0.0, 0.0, -1.0])

    # Check if the normal vector is orthogonal to either pole
    return np.isclose(np.dot(n, p_north), 0, atol=ERROR_TOLERANCE) or np.isclose(
        np.dot(n, p_south), 0, atol=ERROR_TOLERANCE
    )


@njit(cache=True)
def get_all_face_area_from_coords(
    x,
    y,
    z,
    face_nodes,
    face_geometry,
    quadrature_rule="triangular",
    order=4,
    latitude_adjusted_area=False,
):
    """Given coords, connectivity and other area calculation params, this
    routine loop over all faces and return an numpy array with areas of each
    face.

    Parameters
    ----------
    x : ndarray, required
        x-coordinate of all the nodes

    y : ndarray, required
        y-coordinate of all the nodes

    z : ndarray, required
        z-coordinate of all the nodes

    face_nodes : 2D ndarray, required
         node ids of each face

    quadrature_rule : str, optional
        "triangular" or "gaussian". Defaults to triangular

    order : int, optional
        count or order for Gaussian or spherical resp. Defaults to 4 for spherical.

    latitude_adjusted_area : bool, optional
        If True, performs the check if any face consists of an edge that has constant latitude, modifies the area of that face by applying the correction term due to that edge. Default is False.

    Returns
    -------
    area and jacobian of all faces : ndarray, ndarray
    """
    # this casting helps to prevent the type mismatch
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)

    n_face, n_max_face_nodes = face_nodes.shape

    # set initial area of each face to 0
    area = np.zeros(n_face)
    jacobian = np.zeros(n_face)

    for face_idx, max_nodes in enumerate(face_geometry):
        face_x = x[face_nodes[face_idx, 0:max_nodes]]
        face_y = y[face_nodes[face_idx, 0:max_nodes]]
        face_z = z[face_nodes[face_idx, 0:max_nodes]]

        # After getting all the nodes of a face assembled call the  cal. face area routine
        face_area, face_jacobian = calculate_face_area(
            face_x,
            face_y,
            face_z,
            quadrature_rule,
            order,
            latitude_adjusted_area,
        )
        # store current face area
        area[face_idx] = face_area
        jacobian[face_idx] = face_jacobian

    return area, jacobian


@njit(cache=True)
def area_correction(node1, node2):
    """
    Calculate the area correction A using the given formula.

    Parameters:
    - node1: first node of the edge (normalized).
    - node2: second node of the edge (normalized).
    - z: z-coordinate (shared by both points and part of the formula, normalized).

    Returns:
    - A: correction term of the area, when one of the edges is a line of constant latitude
    """
    x1, y1, z = node1
    x2, y2, _ = node2

    # Calculate terms
    term1 = x1 * y2 - x2 * y1
    den2 = x1 * x2 + y1 * y2
    den1 = x1**2 + y1**2 + den2

    # Compute angles using arctan2
    angle1 = np.arctan2(z * term1, den1)
    angle2 = np.arctan2(term1, den2)

    # Compute A
    A = np.abs(2 * angle1 - z * angle2)

    return A


@njit(cache=True)
def calculate_spherical_triangle_jacobian(node1, node2, node3, d_a, d_b):
    """Calculate Jacobian of a spherical triangle. This is a helper function
    for calculating face area.

    Parameters
    ----------
    node1 : list, required
        First node of the triangle

    node2 : list, required
        Second node of the triangle

    node3 : list, required
        Third node of the triangle

    d_a : float, required
        quadrature point

    d_b : float, required
        quadrature point

    Returns
    -------
    jacobian : float
    """
    d_f = np.array(
        [
            (1.0 - d_b) * ((1.0 - d_a) * node1[0] + d_a * node2[0]) + d_b * node3[0],
            (1.0 - d_b) * ((1.0 - d_a) * node1[1] + d_a * node2[1]) + d_b * node3[1],
            (1.0 - d_b) * ((1.0 - d_a) * node1[2] + d_a * node2[2]) + d_b * node3[2],
        ]
    )

    d_da_f = np.array(
        [
            (1.0 - d_b) * (node2[0] - node1[0]),
            (1.0 - d_b) * (node2[1] - node1[1]),
            (1.0 - d_b) * (node2[2] - node1[2]),
        ]
    )

    d_db_f = np.array(
        [
            -(1.0 - d_a) * node1[0] - d_a * node2[0] + node3[0],
            -(1.0 - d_a) * node1[1] - d_a * node2[1] + node3[1],
            -(1.0 - d_a) * node1[2] - d_a * node2[2] + node3[2],
        ]
    )

    d_inv_r = 1.0 / np.sqrt(d_f[0] * d_f[0] + d_f[1] * d_f[1] + d_f[2] * d_f[2])

    d_da_g = np.array(
        [
            d_da_f[0] * (d_f[1] * d_f[1] + d_f[2] * d_f[2])
            - d_f[0] * (d_da_f[1] * d_f[1] + d_da_f[2] * d_f[2]),
            d_da_f[1] * (d_f[0] * d_f[0] + d_f[2] * d_f[2])
            - d_f[1] * (d_da_f[0] * d_f[0] + d_da_f[2] * d_f[2]),
            d_da_f[2] * (d_f[0] * d_f[0] + d_f[1] * d_f[1])
            - d_f[2] * (d_da_f[0] * d_f[0] + d_da_f[1] * d_f[1]),
        ]
    )

    d_db_g = np.array(
        [
            d_db_f[0] * (d_f[1] * d_f[1] + d_f[2] * d_f[2])
            - d_f[0] * (d_db_f[1] * d_f[1] + d_db_f[2] * d_f[2]),
            d_db_f[1] * (d_f[0] * d_f[0] + d_f[2] * d_f[2])
            - d_f[1] * (d_db_f[0] * d_f[0] + d_db_f[2] * d_f[2]),
            d_db_f[2] * (d_f[0] * d_f[0] + d_f[1] * d_f[1])
            - d_f[2] * (d_db_f[0] * d_f[0] + d_db_f[1] * d_f[1]),
        ]
    )

    d_denom_term = d_inv_r * d_inv_r * d_inv_r

    d_da_g *= d_denom_term
    d_db_g *= d_denom_term

    #  Cross product gives local Jacobian
    node_cross = np.cross(d_da_g, d_db_g)
    d_jacobian = np.sqrt(
        node_cross[0] * node_cross[0]
        + node_cross[1] * node_cross[1]
        + node_cross[2] * node_cross[2]
    )

    return d_jacobian


@njit(cache=True)
def calculate_spherical_triangle_jacobian_barycentric(node1, node2, node3, d_a, d_b):
    """Calculate Jacobian of a spherical triangle. This is a helper function
    for calculating face area.

    Parameters
    ----------
    node1 : list, required
        First node of the triangle

    node2 : list, required
        Second node of the triangle

    node3 : list, required
        Third node of the triangle

    d_a : float, required
        first component of barycentric coordinates of quadrature point

    d_b : float, required
        second component of barycentric coordinates of quadrature point

    Returns
    -------
    jacobian : float
    """
    # Calculate the position vector d_f
    d_f = np.array(
        [
            d_a * node1[0] + d_b * node2[0] + (1.0 - d_a - d_b) * node3[0],
            d_a * node1[1] + d_b * node2[1] + (1.0 - d_a - d_b) * node3[1],
            d_a * node1[2] + d_b * node2[2] + (1.0 - d_a - d_b) * node3[2],
        ]
    )

    # Calculate the gradients d_da_f and d_db_f
    d_da_f = np.array([node1[0] - node3[0], node1[1] - node3[1], node1[2] - node3[2]])
    d_db_f = np.array([node2[0] - node3[0], node2[1] - node3[1], node2[2] - node3[2]])

    # Calculate the inverse radius
    d_inv_r = 1.0 / np.sqrt(d_f[0] * d_f[0] + d_f[1] * d_f[1] + d_f[2] * d_f[2])

    # Calculate the gradients d_da_g and d_db_g
    d_da_g = np.array(
        [
            d_da_f[0] * (d_f[1] * d_f[1] + d_f[2] * d_f[2])
            - d_f[0] * (d_da_f[1] * d_f[1] + d_da_f[2] * d_f[2]),
            d_da_f[1] * (d_f[0] * d_f[0] + d_f[2] * d_f[2])
            - d_f[1] * (d_da_f[0] * d_f[0] + d_da_f[2] * d_f[2]),
            d_da_f[2] * (d_f[0] * d_f[0] + d_f[1] * d_f[1])
            - d_f[2] * (d_da_f[0] * d_f[0] + d_da_f[1] * d_f[1]),
        ]
    )

    d_db_g = np.array(
        [
            d_db_f[0] * (d_f[1] * d_f[1] + d_f[2] * d_f[2])
            - d_f[0] * (d_db_f[1] * d_f[1] + d_db_f[2] * d_f[2]),
            d_db_f[1] * (d_f[0] * d_f[0] + d_f[2] * d_f[2])
            - d_f[1] * (d_db_f[0] * d_f[0] + d_db_f[2] * d_f[2]),
            d_db_f[2] * (d_f[0] * d_f[0] + d_f[1] * d_f[1])
            - d_f[2] * (d_db_f[0] * d_f[0] + d_db_f[1] * d_f[1]),
        ]
    )

    # Calculate the denominator term
    d_denom_term = d_inv_r * d_inv_r * d_inv_r

    # Scale the gradients
    d_da_g *= d_denom_term
    d_db_g *= d_denom_term

    # Calculate the cross product
    node_cross = np.cross(d_da_g, d_db_g)

    # Calculate the Jacobian
    d_jacobian = np.sqrt(
        node_cross[0] * node_cross[0]
        + node_cross[1] * node_cross[1]
        + node_cross[2] * node_cross[2]
    )

    return 0.5 * d_jacobian


@njit(cache=True)
def get_gauss_quadrature_dg(n_count):
    """Gauss Quadrature Points for integration.

    Parameters
    ----------
    n_count : int, required
         Degree of quadrature points required, supports: 1 to 10.

    Returns
    -------
        d_g : double
            numpy array of size n_count, quadrature points. Scaled before returning.
        d_w : double
            numpy array of size n_count x 3, weights. Scaled before returning.

    Raises
    ------
       RuntimeError: Invalid degree
    """
    # Degree 1
    if n_count == 1:
        d_g = np.array([[0.0]])
        d_w = np.array([2.0])

    # Degree 2
    elif n_count == 2:
        d_g = np.array([[-0.5773502691896257, 0.5773502691896257]])
        d_w = np.array([1.0, 1.0])

    # Degree 3
    elif n_count == 3:
        d_g = np.array([[-0.7745966692414834, 0.0, 0.7745966692414834]])
        d_w = np.array([0.5555555555555556, 0.8888888888888888, 0.5555555555555556])

    # Degree 4
    elif n_count == 4:
        d_g = np.array(
            [
                [
                    -0.8611363115940526,
                    -0.3399810435848563,
                    0.3399810435848563,
                    0.8611363115940526,
                ]
            ]
        )
        d_w = np.array(
            [
                0.3478548451374538,
                0.6521451548625461,
                0.6521451548625461,
                0.3478548451374538,
            ]
        )

    # Degree 5
    elif n_count == 5:
        d_g = np.array(
            [
                [
                    -0.9061798459386640,
                    -0.5384693101056831,
                    0.0,
                    0.5384693101056831,
                    0.9061798459386640,
                ]
            ]
        )
        d_w = np.array(
            [
                0.2369268850561891,
                0.4786286704993665,
                0.5688888888888889,
                0.4786286704993665,
                0.2369268850561891,
            ]
        )

    # Degree 6
    elif n_count == 6:
        d_g = np.array(
            [
                [
                    -0.9324695142031521,
                    -0.6612093864662645,
                    -0.2386191860831969,
                    0.2386191860831969,
                    0.6612093864662645,
                    0.9324695142031521,
                ]
            ]
        )
        d_w = np.array(
            [
                0.1713244923791704,
                0.3607615730481386,
                0.4679139345726910,
                0.4679139345726910,
                0.3607615730481386,
                0.1713244923791704,
            ]
        )

    # Degree 7
    elif n_count == 7:
        d_g = np.array(
            [
                [
                    -0.9491079123427585,
                    -0.7415311855993945,
                    -0.4058451513773972,
                    0.0,
                    0.4058451513773972,
                    0.7415311855993945,
                    0.9491079123427585,
                ]
            ]
        )
        d_w = np.array(
            [
                0.1294849661688697,
                0.2797053914892766,
                0.3818300505051189,
                0.4179591836734694,
                0.3818300505051189,
                0.2797053914892766,
                0.1294849661688697,
            ]
        )

    # Degree 8
    elif n_count == 8:
        d_g = np.array(
            [
                [
                    -0.9602898564975363,
                    -0.7966664774136267,
                    -0.5255324099163290,
                    -0.1834346424956498,
                    0.1834346424956498,
                    0.5255324099163290,
                    0.7966664774136267,
                    0.9602898564975363,
                ]
            ]
        )
        d_w = np.array(
            [
                0.1012285362903763,
                0.2223810344533745,
                0.3137066458778873,
                0.3626837833783620,
                0.3626837833783620,
                0.3137066458778873,
                0.2223810344533745,
                0.1012285362903763,
            ]
        )

    # Degree 9
    elif n_count == 9:
        d_g = np.array(
            [
                [
                    -1.0,
                    -0.899757995411460,
                    -0.677186279510738,
                    -0.363117463826178,
                    0.0,
                    0.363117463826178,
                    0.677186279510738,
                    0.899757995411460,
                    1.0,
                ]
            ]
        )
        d_w = np.array(
            [
                0.0277777777777778,
                0.1654953615608055,
                0.2745387125001617,
                0.3464285109730464,
                0.3715192743764172,
                0.3464285109730464,
                0.2745387125001617,
                0.1654953615608055,
                0.0277777777777778,
            ]
        )

    # Degree 10
    elif n_count == 10:
        d_g = np.array(
            [
                [
                    -0.9739065285171717,
                    -0.8650633666889845,
                    -0.6794095682990244,
                    -0.4333953941292472,
                    -0.1488743389816312,
                    0.1488743389816312,
                    0.4333953941292472,
                    0.6794095682990244,
                    0.8650633666889845,
                    0.9739065285171717,
                ]
            ]
        )
        d_w = np.array(
            [
                0.0666713443086881,
                0.1494513491505806,
                0.2190863625159820,
                0.2692667193099963,
                0.2955242247147529,
                0.2955242247147529,
                0.2692667193099963,
                0.2190863625159820,
                0.1494513491505806,
                0.0666713443086881,
            ]
        )

    # Scale quadrature points
    d_xi0 = 0.0
    d_xi1 = 1.0
    for i in range(n_count):
        d_g[0][i] = d_xi0 + 0.5 * (d_xi1 - d_xi0) * (d_g[0][i] + 1.0)
        d_w[i] = 0.5 * (d_xi1 - d_xi0) * d_w[i]

    return d_g, d_w


@njit(cache=True)
def get_tri_quadrature_dg(n_order):
    """Triangular Quadrature Points for integration.

    Parameters
    ----------
    n_order : int
        Integration order, supports: 12, 10, 8, 4 and 1

    Returns
    -------
        d_g, d_w : ndarray
            points and weights, with dimension order x 3
    """
    # 12th order quadrature rule (33 points)
    if n_order == 12:
        d_g = np.array(
            [
                [0.023565220452390, 0.488217389773805, 0.488217389773805],
                [0.488217389773805, 0.023565220452390, 0.488217389773805],
                [0.488217389773805, 0.488217389773805, 0.023565220452390],
                [0.120551215411079, 0.439724392294460, 0.439724392294460],
                [0.439724392294460, 0.120551215411079, 0.439724392294460],
                [0.439724392294460, 0.439724392294460, 0.120551215411079],
                [0.457579229975768, 0.271210385012116, 0.271210385012116],
                [0.271210385012116, 0.457579229975768, 0.271210385012116],
                [0.271210385012116, 0.271210385012116, 0.457579229975768],
                [0.744847708916828, 0.127576145541586, 0.127576145541586],
                [0.127576145541586, 0.744847708916828, 0.127576145541586],
                [0.127576145541586, 0.127576145541586, 0.744847708916828],
                [0.957365299093576, 0.021317350453210, 0.021317350453210],
                [0.021317350453210, 0.957365299093576, 0.021317350453210],
                [0.021317350453210, 0.021317350453210, 0.957365299093576],
                [0.115343494534698, 0.275713269685514, 0.608943235779788],
                [0.115343494534698, 0.608943235779788, 0.275713269685514],
                [0.275713269685514, 0.115343494534698, 0.608943235779788],
                [0.275713269685514, 0.608943235779788, 0.115343494534698],
                [0.608943235779788, 0.115343494534698, 0.275713269685514],
                [0.608943235779788, 0.275713269685514, 0.115343494534698],
                [0.022838332222257, 0.281325580989940, 0.695836086787803],
                [0.022838332222257, 0.695836086787803, 0.281325580989940],
                [0.281325580989940, 0.022838332222257, 0.695836086787803],
                [0.281325580989940, 0.695836086787803, 0.022838332222257],
                [0.695836086787803, 0.022838332222257, 0.281325580989940],
                [0.695836086787803, 0.281325580989940, 0.022838332222257],
                [0.025734050548330, 0.116251915907597, 0.858014033544073],
                [0.025734050548330, 0.858014033544073, 0.116251915907597],
                [0.116251915907597, 0.025734050548330, 0.858014033544073],
                [0.116251915907597, 0.858014033544073, 0.025734050548330],
                [0.858014033544073, 0.025734050548330, 0.116251915907597],
                [0.858014033544073, 0.116251915907597, 0.025734050548330],
            ]
        )

        d_w = np.array(
            [
                0.025731066440455,
                0.025731066440455,
                0.025731066440455,
                0.043692544538038,
                0.043692544538038,
                0.043692544538038,
                0.062858224217885,
                0.062858224217885,
                0.062858224217885,
                0.034796112930709,
                0.034796112930709,
                0.034796112930709,
                0.006166261051559,
                0.006166261051559,
                0.006166261051559,
                0.040371557766381,
                0.040371557766381,
                0.040371557766381,
                0.040371557766381,
                0.040371557766381,
                0.040371557766381,
                0.022356773202303,
                0.022356773202303,
                0.022356773202303,
                0.022356773202303,
                0.022356773202303,
                0.022356773202303,
                0.017316231108659,
                0.017316231108659,
                0.017316231108659,
                0.017316231108659,
                0.017316231108659,
                0.017316231108659,
            ]
        )

    # 10th order quadrature rule (25 points)
    elif n_order == 10:
        d_g = np.array(
            [
                [0.333333333333333, 0.333333333333333, 0.333333333333333],
                [0.028844733232685, 0.485577633383657, 0.485577633383657],
                [0.485577633383657, 0.028844733232685, 0.485577633383657],
                [0.485577633383657, 0.485577633383657, 0.028844733232685],
                [0.781036849029926, 0.109481575485037, 0.109481575485037],
                [0.109481575485037, 0.781036849029926, 0.109481575485037],
                [0.109481575485037, 0.109481575485037, 0.781036849029926],
                [0.141707219414880, 0.307939838764121, 0.550352941820999],
                [0.141707219414880, 0.550352941820999, 0.307939838764121],
                [0.307939838764121, 0.141707219414880, 0.550352941820999],
                [0.307939838764121, 0.550352941820999, 0.141707219414880],
                [0.550352941820999, 0.141707219414880, 0.307939838764121],
                [0.550352941820999, 0.307939838764121, 0.141707219414880],
                [0.025003534762686, 0.246672560639903, 0.728323904597411],
                [0.025003534762686, 0.728323904597411, 0.246672560639903],
                [0.246672560639903, 0.025003534762686, 0.728323904597411],
                [0.246672560639903, 0.728323904597411, 0.025003534762686],
                [0.728323904597411, 0.025003534762686, 0.246672560639903],
                [0.728323904597411, 0.246672560639903, 0.025003534762686],
                [0.009540815400299, 0.066803251012200, 0.923655933587500],
                [0.009540815400299, 0.923655933587500, 0.066803251012200],
                [0.066803251012200, 0.009540815400299, 0.923655933587500],
                [0.066803251012200, 0.923655933587500, 0.009540815400299],
                [0.923655933587500, 0.009540815400299, 0.066803251012200],
                [0.923655933587500, 0.066803251012200, 0.009540815400299],
            ]
        )

        d_w = np.array(
            [
                0.090817990382754,
                0.036725957756467,
                0.036725957756467,
                0.036725957756467,
                0.045321059435528,
                0.045321059435528,
                0.045321059435528,
                0.072757916845420,
                0.072757916845420,
                0.072757916845420,
                0.072757916845420,
                0.072757916845420,
                0.072757916845420,
                0.028327242531057,
                0.028327242531057,
                0.028327242531057,
                0.028327242531057,
                0.028327242531057,
                0.028327242531057,
                0.009421666963733,
                0.009421666963733,
                0.009421666963733,
                0.009421666963733,
                0.009421666963733,
                0.009421666963733,
            ]
        )

    # 8th order quadrature rule (16 points)
    elif n_order == 8:
        d_g = np.array(
            [
                [0.333333333333333, 0.333333333333333, 0.333333333333333],
                [0.081414823414554, 0.459292588292723, 0.459292588292723],
                [0.459292588292723, 0.081414823414554, 0.459292588292723],
                [0.459292588292723, 0.459292588292723, 0.081414823414554],
                [0.658861384496480, 0.170569307751760, 0.170569307751760],
                [0.170569307751760, 0.658861384496480, 0.170569307751760],
                [0.170569307751760, 0.170569307751760, 0.658861384496480],
                [0.898905543365938, 0.050547228317031, 0.050547228317031],
                [0.050547228317031, 0.898905543365938, 0.050547228317031],
                [0.050547228317031, 0.050547228317031, 0.898905543365938],
                [0.008394777409958, 0.263112829634638, 0.728492392955404],
                [0.008394777409958, 0.728492392955404, 0.263112829634638],
                [0.263112829634638, 0.008394777409958, 0.728492392955404],
                [0.263112829634638, 0.728492392955404, 0.008394777409958],
                [0.728492392955404, 0.263112829634638, 0.008394777409958],
                [0.728492392955404, 0.008394777409958, 0.263112829634638],
            ]
        )

        d_w = np.array(
            [
                0.144315607677787,
                0.095091634267285,
                0.095091634267285,
                0.095091634267285,
                0.103217370534718,
                0.103217370534718,
                0.103217370534718,
                0.032458497623198,
                0.032458497623198,
                0.032458497623198,
                0.027230314174435,
                0.027230314174435,
                0.027230314174435,
                0.027230314174435,
                0.027230314174435,
                0.027230314174435,
            ]
        )

    # 4th order quadrature rule (6 points)
    elif n_order == 4:
        d_g = np.array(
            [
                [0.108103018168070, 0.445948490915965, 0.445948490915965],
                [0.445948490915965, 0.108103018168070, 0.445948490915965],
                [0.445948490915965, 0.445948490915965, 0.108103018168070],
                [0.816847572980458, 0.091576213509771, 0.091576213509771],
                [0.091576213509771, 0.816847572980458, 0.091576213509771],
                [0.091576213509771, 0.091576213509771, 0.816847572980458],
            ]
        )

        d_w = np.array(
            [
                0.223381589678011,
                0.223381589678011,
                0.223381589678011,
                0.109951743655322,
                0.109951743655322,
                0.109951743655322,
            ]
        )
    # 1st order quadrature rule (1 point)
    elif n_order == 1:
        d_g = np.array([[0.333333333333333, 0.333333333333333, 0.333333333333333]])
        d_w = np.array([1.000000000000000])

    return d_g, d_w
