import numpy as np
from metric import metric
from coordinates import cart_to_pol, mom_cart_to_pol

def create_initial_positions(y_rays, y_size, x_0, t_min):
    t_initials = np.full(y_rays, t_min)
    x_initials = np.full(y_rays, x_0)
    y_initials = np.linspace(-y_size, y_size, y_rays)
    rays_initial_positions = np.column_stack((t_initials, x_initials, y_initials))
    return rays_initial_positions

def create_initial_states(y_rays, y_size, x_0, t_min):
    rays_initial_positions = create_initial_positions(y_rays, y_size, x_0, t_min)
    number_of_rays = rays_initial_positions.shape[0]
    initial_states = np.empty((number_of_rays, 6), dtype=np.float64)
    
    p_t = -1.0
    p_x = -1.0
    p_y = 0.0
    
    for i, position in enumerate(rays_initial_positions):
        
        cartesian_state = np.array([position[0], position[1], position[2],
                                    p_t, p_x, p_y])
        initial_states[i] = mom_cart_to_pol(cartesian_state)
    
    return initial_states