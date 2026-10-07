import numpy as np

def cart_to_pol(cartesian_state):
    t, x, y = cartesian_state
    r = np.sqrt(x*x+y*y)
    theta = np.arctan2(y, x)
    return t, r, theta

def pol_to_cart(polar_position):
    t, r, theta = polar_position
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return t, x, y

def jacobian_pol_to_cart(polar_position):
    t, r, theta = polar_position
    jacobian = np.zeros((3,3), dtype = np.float64)
    jacobian[0,0] = 1.0
    jacobian[1,1] = np.cos(theta)
    jacobian[1,2] = -r * np.sin(theta)
    jacobian[2,1] = np.sin(theta)
    jacobian[2,2] = r * np.cos(theta)
    return jacobian

def mom_cart_to_pol(cartesian_state):
    t, x, y, p_t, p_x, p_y = cartesian_state
    polar_position = cart_to_pol((t, x, y))
    
    jacobian = jacobian_pol_to_cart(polar_position)
    p_t_new = jacobian[0,0] * p_t
    p_r = jacobian[1,1] * p_x + jacobian[1,2] * p_y
    p_theta = jacobian[2,1] * p_x + jacobian[2,2] * p_y
    
    return np.array([polar_position[0], polar_position[1], polar_position[2],
                     p_t_new, p_r, p_theta], dtype=np.float64)