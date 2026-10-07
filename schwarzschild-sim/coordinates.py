import numpy as np

def cart_to_pol(cartesian_positions):
    t = cartesian_positions[..., 0]
    x = cartesian_positions[..., 1]
    y = cartesian_positions[..., 2]
    
    polar_positions = np.empty(cartesian_positions.shape)
    polar_positions[..., 0] = t
    polar_positions[..., 1] = np.sqrt(x * x + y * y)
    polar_positions[..., 2] = np.arctan2(y, x)
    return polar_positions

def pol_to_cart(polar_positions):
    t = polar_positions[..., 0]
    r = polar_positions[..., 1]
    theta  = polar_positions[..., 2]
    
    cartesian_positions = np.empty(polar_positions.shape)
    cartesian_positions[..., 0] = t
    cartesian_positions[..., 1] = r * np.cos(theta)
    cartesian_positions[..., 2] = r * np.sin(theta)
    return cartesian_positions

def jacobian_pol_to_cart(polar_positions):
    r = polar_positions[..., 1]
    theta = polar_positions[..., 2]
    
    jacobian = np.zeros(polar_positions.shape[:-1] + (3, 3), dtype = np.float64)
    jacobian[..., 0, 0] = 1.0
    jacobian[..., 1, 1] = np.cos(theta)
    jacobian[..., 1, 2] = -r * np.sin(theta)
    jacobian[..., 2, 1] = np.sin(theta)
    jacobian[..., 2, 2] = r * np.cos(theta)
    return jacobian

def mom_cart_to_pol(cartesian_states):
    p_t = cartesian_states[..., 3]
    p_x = cartesian_states[..., 4]
    p_y = cartesian_states[..., 5]
    
    polar_positions = cart_to_pol(cartesian_states[..., :3])
    jacobian = jacobian_pol_to_cart(polar_positions)
    
    polar_states = np.empty(cartesian_states.shape)
    polar_states[..., :3] = polar_positions
    polar_states[..., 3] = jacobian[..., 0, 0] * p_t
    polar_states[..., 4] = jacobian[..., 1, 1] * p_x + jacobian[..., 2, 1] * p_y
    polar_states[..., 5] = jacobian[..., 1, 2] * p_x + jacobian[..., 2, 2] * p_y
    
    return polar_states