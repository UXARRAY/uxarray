import numpy as np

from uxarray.grid.coordinates import _lonlat_rad_to_xyz

from numba import njit


@njit(cache=True)
def calculate_face_area(
    x, y, z, quadrature_rule="gaussian", order=4, coords_type="spherical"
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

    coords_type : str, optional
        coordinate type, default is spherical, can be cartesian also.

    Returns
    -------
    area : double
    jacobian: double
    """
    area = 0.0  # set area to 0
    jacobian = 0.0  # set jacobian to 0
    order = order

    if quadrature_rule == "gaussian":
        dG, dW = get_gauss_quadratureDG(order)
    elif quadrature_rule == "triangular":
        dG, dW = get_tri_quadratureDG(order)
    else:
        raise ValueError("Invalid quadrature rule, specify gaussian or triangular")

    num_nodes = len(x)

    # num triangles is two less than the total number of nodes
    num_triangles = num_nodes - 2

    # Using tempestremap GridElements: https://github.com/ClimateGlobalChange/tempestremap/blob/master/src/GridElements.cpp
    # loop through all sub-triangles of face
    for j in range(0, num_triangles):
        node1 = np.array([x[0], y[0], z[0]], dtype=x.dtype)
        node2 = np.array([x[j + 1], y[j + 1], z[j + 1]], dtype=x.dtype)
        node3 = np.array([x[j + 2], y[j + 2], z[j + 2]], dtype=x.dtype)

        if coords_type == "spherical":
            node1 = _lonlat_rad_to_xyz(np.deg2rad(x[0]), np.deg2rad(y[0]))
            node1 = np.asarray(node1)

            node2 = _lonlat_rad_to_xyz(np.deg2rad(x[j + 1]), np.deg2rad(y[j + 1]))
            node2 = np.asarray(node2)

            node3 = _lonlat_rad_to_xyz(np.deg2rad(x[j + 2]), np.deg2rad(y[j + 2]))
            node3 = np.asarray(node3)

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

    return area, jacobian


@njit(cache=True)
def get_all_face_area_from_coords(
    x,
    y,
    z,
    face_nodes,
    face_geometry,
    dim,
    quadrature_rule="triangular",
    order=4,
    coords_type="spherical",
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

    dim : int, required
         dimension

    quadrature_rule : str, optional
        "triangular" or "gaussian". Defaults to triangular

    order : int, optional
        count or order for Gaussian or spherical resp. Defaults to 4 for spherical.

    coords_type : str, optional
        coordinate type, default is spherical, can be cartesian also.

    Returns
    -------
    area of all faces : ndarray
    """

    n_face, n_max_face_nodes = face_nodes.shape

    # set initial area of each face to 0
    area = np.zeros(n_face)
    jacobian = np.zeros(n_face)

    for face_idx, max_nodes in enumerate(face_geometry):
        face_x = x[face_nodes[face_idx, 0:max_nodes]]
        face_y = y[face_nodes[face_idx, 0:max_nodes]]

        # check if z dimension

        if dim > 2:
            face_z = z[face_nodes[face_idx, 0:max_nodes]]
        else:
            face_z = face_x * 0.0

        # After getting all the nodes of a face assembled call the  cal. face area routine
        face_area, face_jacobian = calculate_face_area(
            face_x, face_y, face_z, quadrature_rule, order, coords_type
        )
        # store current face area
        area[face_idx] = face_area
        jacobian[face_idx] = face_jacobian

    return area, jacobian


@njit(cache=True)
def calculate_spherical_triangle_jacobian(node1, node2, node3, dA, dB):
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

    dA : float, required
        quadrature point

    dB : float, required
        quadrature point

    Returns
    -------
    jacobian : float
    """
    dF = np.array(
        [
            (1.0 - dB) * ((1.0 - dA) * node1[0] + dA * node2[0]) + dB * node3[0],
            (1.0 - dB) * ((1.0 - dA) * node1[1] + dA * node2[1]) + dB * node3[1],
            (1.0 - dB) * ((1.0 - dA) * node1[2] + dA * node2[2]) + dB * node3[2],
        ]
    )

    dDaF = np.array(
        [
            (1.0 - dB) * (node2[0] - node1[0]),
            (1.0 - dB) * (node2[1] - node1[1]),
            (1.0 - dB) * (node2[2] - node1[2]),
        ]
    )

    dDbF = np.array(
        [
            -(1.0 - dA) * node1[0] - dA * node2[0] + node3[0],
            -(1.0 - dA) * node1[1] - dA * node2[1] + node3[1],
            -(1.0 - dA) * node1[2] - dA * node2[2] + node3[2],
        ]
    )

    dInvR = 1.0 / np.sqrt(dF[0] * dF[0] + dF[1] * dF[1] + dF[2] * dF[2])

    dDaG = np.array(
        [
            dDaF[0] * (dF[1] * dF[1] + dF[2] * dF[2])
            - dF[0] * (dDaF[1] * dF[1] + dDaF[2] * dF[2]),
            dDaF[1] * (dF[0] * dF[0] + dF[2] * dF[2])
            - dF[1] * (dDaF[0] * dF[0] + dDaF[2] * dF[2]),
            dDaF[2] * (dF[0] * dF[0] + dF[1] * dF[1])
            - dF[2] * (dDaF[0] * dF[0] + dDaF[1] * dF[1]),
        ]
    )

    dDbG = np.array(
        [
            dDbF[0] * (dF[1] * dF[1] + dF[2] * dF[2])
            - dF[0] * (dDbF[1] * dF[1] + dDbF[2] * dF[2]),
            dDbF[1] * (dF[0] * dF[0] + dF[2] * dF[2])
            - dF[1] * (dDbF[0] * dF[0] + dDbF[2] * dF[2]),
            dDbF[2] * (dF[0] * dF[0] + dF[1] * dF[1])
            - dF[2] * (dDbF[0] * dF[0] + dDbF[1] * dF[1]),
        ]
    )

    dDenomTerm = dInvR * dInvR * dInvR

    dDaG *= dDenomTerm
    dDbG *= dDenomTerm

    #  Cross product gives local Jacobian
    nodeCross = np.cross(dDaG, dDbG)
    dJacobian = np.sqrt(
        nodeCross[0] * nodeCross[0]
        + nodeCross[1] * nodeCross[1]
        + nodeCross[2] * nodeCross[2]
    )

    return dJacobian


@njit(cache=True)
def calculate_spherical_triangle_jacobian_barycentric(node1, node2, node3, dA, dB):
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

    dA : float, required
        first component of barycentric coordinates of quadrature point

    dB : float, required
        second component of barycentric coordinates of quadrature point

    Returns
    -------
    jacobian : float
    """

    dF = np.array(
        [
            dA * node1[0] + dB * node2[0] + (1.0 - dA - dB) * node3[0],
            dA * node1[1] + dB * node2[1] + (1.0 - dA - dB) * node3[1],
            dA * node1[2] + dB * node2[2] + (1.0 - dA - dB) * node3[2],
        ]
    )

    dDaF = np.array([node1[0] - node3[0], node1[1] - node3[1], node1[2] - node3[2]])

    dDbF = np.array([node2[0] - node3[0], node2[1] - node3[1], node2[2] - node3[2]])

    dInvR = 1.0 / np.sqrt(dF[0] * dF[0] + dF[1] * dF[1] + dF[2] * dF[2])

    dDaG = np.array(
        [
            dDaF[0] * (dF[1] * dF[1] + dF[2] * dF[2])
            - dF[0] * (dDaF[1] * dF[1] + dDaF[2] * dF[2]),
            dDaF[1] * (dF[0] * dF[0] + dF[2] * dF[2])
            - dF[1] * (dDaF[0] * dF[0] + dDaF[2] * dF[2]),
            dDaF[2] * (dF[0] * dF[0] + dF[1] * dF[1])
            - dF[2] * (dDaF[0] * dF[0] + dDaF[1] * dF[1]),
        ]
    )

    dDbG = np.array(
        [
            dDbF[0] * (dF[1] * dF[1] + dF[2] * dF[2])
            - dF[0] * (dDbF[1] * dF[1] + dDbF[2] * dF[2]),
            dDbF[1] * (dF[0] * dF[0] + dF[2] * dF[2])
            - dF[1] * (dDbF[0] * dF[0] + dDbF[2] * dF[2]),
            dDbF[2] * (dF[0] * dF[0] + dF[1] * dF[1])
            - dF[2] * (dDbF[0] * dF[0] + dDbF[1] * dF[1]),
        ]
    )

    dDenomTerm = dInvR * dInvR * dInvR

    dDaG *= dDenomTerm
    dDbG *= dDenomTerm

    #  Cross product gives local Jacobian
    nodeCross = np.cross(dDaG, dDbG)
    dJacobian = np.sqrt(
        nodeCross[0] * nodeCross[0]
        + nodeCross[1] * nodeCross[1]
        + nodeCross[2] * nodeCross[2]
    )

    return 0.5 * dJacobian


@njit(cache=True)
def get_gauss_quadratureDG(nCount):
    """Gauss Quadrature Points for integration.

    Parameters
    ----------
    nCount : int, required
         Degree of quadrature points required, supports: 1 to 10.

    Returns
    -------
        dG : double
            numpy array of size ncount, quadrature points. Scaled before returning.
        dW : double
            numpy array of size ncount x 3, weights. Scaled before returning.

    Raises
    ------
       RuntimeError: Invalid degree
    """
    # Degree 1
    if nCount == 1:
        dG = np.array([[0.0]])
        dW = np.array([+2.0])

    # Degree 2
    elif nCount == 2:
        dG = np.array([[-0.5773502691896257, +0.5773502691896257]])
        dW = np.array([+1.0, +1.0])

    # Degree 3
    elif nCount == 3:
        dG = np.array([[-0.7745966692414834, 0.0, +0.7745966692414834]])

        dW = np.array([+0.5555555555555556, +0.8888888888888888, +0.5555555555555556])

    # Degree 4
    elif nCount == 4:
        dG = np.array(
            [
                [
                    -0.8611363115940526,
                    -0.3399810435848563,
                    +0.3399810435848563,
                    +0.8611363115940526,
                ]
            ]
        )

        dW = np.array(
            [
                0.3478548451374538,
                0.6521451548625461,
                0.6521451548625461,
                0.3478548451374538,
            ]
        )

    # Degree 5
    elif nCount == 5:
        dG = np.array(
            [
                [
                    -0.9061798459386640,
                    -0.5384693101056831,
                    0.0,
                    +0.5384693101056831,
                    +0.9061798459386640,
                ]
            ]
        )

        dW = np.array(
            [
                0.2369268850561891,
                0.4786286704993665,
                0.5688888888888889,
                0.4786286704993665,
                0.2369268850561891,
            ]
        )

    # Degree 6
    elif nCount == 6:
        dG = np.array(
            [
                [
                    -0.9324695142031521,
                    -0.6612093864662645,
                    -0.2386191860831969,
                    +0.2386191860831969,
                    +0.6612093864662645,
                    +0.9324695142031521,
                ]
            ]
        )

        dW = np.array(
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
    elif nCount == 7:
        dG = np.array(
            [
                [
                    -0.9491079123427585,
                    -0.7415311855993945,
                    -0.4058451513773972,
                    0.0,
                    +0.4058451513773972,
                    +0.7415311855993945,
                    +0.9491079123427585,
                ]
            ]
        )

        dW = np.array(
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
    elif nCount == 8:
        dG = np.array(
            [
                [
                    -0.9602898564975363,
                    -0.7966664774136267,
                    -0.5255324099163290,
                    -0.1834346424956498,
                    +0.1834346424956498,
                    +0.5255324099163290,
                    +0.7966664774136267,
                    +0.9602898564975363,
                ]
            ]
        )

        dW = np.array(
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
    elif nCount == 9:
        dG = np.array(
            [
                [
                    -1.0,
                    -0.899757995411460,
                    -0.677186279510738,
                    -0.363117463826178,
                    0.0,
                    +0.363117463826178,
                    +0.677186279510738,
                    +0.899757995411460,
                    +1.0,
                ]
            ]
        )

        dW = np.array(
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
    elif nCount == 10:
        dG = np.array(
            [
                [
                    -0.9739065285171717,
                    -0.8650633666889845,
                    -0.6794095682990244,
                    -0.4333953941292472,
                    -0.1488743389816312,
                    +0.1488743389816312,
                    +0.4333953941292472,
                    +0.6794095682990244,
                    +0.8650633666889845,
                    +0.9739065285171717,
                ]
            ]
        )

        dW = np.array(
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
    # else:
    #     msg = "quadrature order 1 to 10 is supported: ", nCount, " is invalid\n"
    #     raise ValueError(msg)

    # Scale quadrature points
    dXi0 = 0.0
    dXi1 = 1.0
    for i in range(nCount):
        dG[0][i] = dXi0 + 0.5 * (dXi1 - dXi0) * (dG[0][i] + 1.0)
        dW[i] = 0.5 * (dXi1 - dXi0) * dW[i]

    return dG, dW


@njit(cache=True)
def get_tri_quadratureDG(nOrder):
    """Triangular Quadrature Points for integration.

    Parameters
    ----------
    nOrder : int
        Integration order, supports: 12, 10, 8, 4 and 1

    Returns
    -------
        dG, dW : ndarray
            points and weights, with dimension order x 3
    """
    # 12th order quadrature rule (33 points)
    if nOrder == 12:
        dG = np.array(
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

        dW = np.array(
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
    elif nOrder == 10:
        dG = np.array(
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

        dW = np.array(
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
    elif nOrder == 8:
        dG = np.array(
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

        dW = np.array(
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
    elif nOrder == 4:
        dG = np.array(
            [
                [0.108103018168070, 0.445948490915965, 0.445948490915965],
                [0.445948490915965, 0.108103018168070, 0.445948490915965],
                [0.445948490915965, 0.445948490915965, 0.108103018168070],
                [0.816847572980458, 0.091576213509771, 0.091576213509771],
                [0.091576213509771, 0.816847572980458, 0.091576213509771],
                [0.091576213509771, 0.091576213509771, 0.816847572980458],
            ]
        )

        dW = np.array(
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
    elif nOrder == 1:
        dG = np.array([[0.333333333333333, 0.333333333333333, 0.333333333333333]])
        dW = np.array([1.000000000000000])

    return dG, dW
