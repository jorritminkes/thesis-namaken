import numpy as np

def create_initial_grid(y_rays, y_size, x_0):
    x_initials = np.full(y_rays, x_0)
    y_initials = np.linspace(-y_size, y_size, y_rays)
    rays_initial_positions = np.column_stack((x_initials, y_initials))
    print(rays_initial_positions)
    return rays_initial_positions