import numpy as np


def __parallel_to_plane_eqn(x0, x1, y0, y1, z0, z1, x_i_old, y_i_old, z_lat):
    # X0 x X1 dot Xi=0
    return (y0 * z1 - z0 * y1) * x_i_old + (-x0 * z1 + z0 * x1) * y_i_old + (x0 * y1 - y0 * x1) * z_lat


def __sphere_eqn(x_i_old, y_i_old, z_lat):
    # xi^2+yi^2+zi^2=1
    return x_i_old * x_i_old + y_i_old * y_i_old + z_lat * z_lat - 1


def __inv_jacobian(x0, x1, y0, y1, z0, z1, x_i_old, y_i_old):
    J = np.ones((2, 2))
    d_dx = (x0 * x_i_old - x1 * x_i_old * z0 + y0 * y_i_old * z1 - y1 * y_i_old * z0 - y1 * y_i_old * z0)
    d_dy = 2 * (x0 * x_i_old * z1 - x1 * x_i_old * z0 + y0 * y_i_old * z1 - y1 * y_i_old * z0)

    # row 1
    J[0, 0] = y_i_old / d_dx
    J[0, 1] = (x0 * z1 - z0 * x1) / d_dy
    J[1, 0] = x_i_old / d_dx
    J[1, 1] = (y0 * z1 - z0 * y1) / d_dy
    return J


def _newton_raphson_solver_for_intersection_pts(init_cart, w0_cart, w1_cart, max_iter=1000):
    tolerance = 1.0e-17
    error = 9e9
    alpha = 1


    y_new = np.ones((2, 1))
    y_old = np.array(init_cart[0:2])
    y_guess = np.array(init_cart[0:2])

    # numpy column matrix for the F
    F = np.copy(y_guess)

    _iter = 0

    while error > tolerance and _iter < max_iter:
        F[0] = __parallel_to_plane_eqn(w0_cart[0], w1_cart[0], w0_cart[1], w1_cart[1], w0_cart[2], w1_cart[2],
                                       y_guess[0], y_guess[1], y_guess[2])
        F[1] = __sphere_eqn(y_guess[0], y_guess[1], y_guess[2])

        J_inv = __inv_jacobian(w0_cart[0], w1_cart[0], w0_cart[1], w1_cart[1], w0_cart[2], w1_cart[2], y_guess[0],
                               y_guess[1])
        y_new = y_guess - alpha * np.matmul(J_inv, F)
        error = np.max(np.abs(y_guess - y_new))
        y_guess = y_new
        _iter += 1

    return y_new
