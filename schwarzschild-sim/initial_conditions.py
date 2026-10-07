import numpy as np
from metric import metric
from coordinates import cart_to_pol

def create_initial_positions(y_rays, y_size, x_0):
    t_initials = np.zeros(y_rays)
    x_initials = np.full(y_rays, x_0)
    y_initials = np.linspace(-y_size, y_size, y_rays)
    rays_initial_positions = np.column_stack((t_initials, x_initials, y_initials))
    return rays_initial_positions

def create_initial_state(rays_initial_positions):
    number_of_rays = rays_initial_positions.shape[0]
    initial_states = np.empty((number_of_rays, 6), dtype=np.float64)
    
    for i, initial_ray in enumerate(rays_initial_positions):
        polar_pos = cart_to_pol(initial_ray)
        metric_values = metric(polar_pos, 1)
        
        t0, r0, theta0 = cart_to_pol(initial_ray)
        
        pmu_cart = np.array([0,-1,0])
        # we should not use pmu_cart, but pmu_polar via jacobian...
        p_t0 = -1
        p_r0 = metric_values[1] * pmu_cart[1]
        p_theta0 = metric_values[2] * pmu_cart[2]
        
        initial_states[i] = (t0, r0, theta0, p_t0, p_r0, p_theta0)
    print(initial_states)
    return initial_states