import numpy as np
from metric import metric
from coordinates import mom_cov_cart_to_pol, mom_contra_cart_to_pol

def create_initial_positions(y_rays, y_size, x_0, t_min):
    t_initials = np.full(y_rays, t_min)
    x_initials = np.full(y_rays, x_0)
    y_initials = np.linspace(-y_size, y_size, y_rays)
    rays_initial_positions = np.column_stack((t_initials, x_initials, y_initials))
    return rays_initial_positions

def create_initial_states(y_rays, y_size, x_0, t_min):
    cartesian_initial_positions = create_initial_positions(y_rays, y_size, x_0, t_min)
    number_of_rays = cartesian_initial_positions.shape[0]
    
    # g = metric(polar_initial_positions, 1.0)
    
    # r = polar_initial_positions[:, 1]
    # theta = polar_initial_positions[:, 2]
    
    px0 = -1.0
    py0 = 0.0
    pt0 = 0.0 # Placeholder for array shape
    
    cartesian_states = np.column_stack((cartesian_initial_positions,
                                        np.full(number_of_rays, pt0),
                                        np.full(number_of_rays, px0),
                                        np.full(number_of_rays, py0)))
    initial_states = mom_contra_cart_to_pol(cartesian_states)
    
    g = metric(initial_states[:, :3], 1.0)
    initial_states[:, 3] = np.sqrt((g[:, 1] * initial_states[:, 4]**2
                                    + g[:, 2] * initial_states[:, 5]**2) / -g[:, 0])
    
    initial_states[:, 3:] = g * initial_states[:, 3:]
    
    
    
    return initial_states