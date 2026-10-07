import numpy as np
import matplotlib.pyplot as plt

from metric import metric, inv_metric
from coordinates import cart_to_pol, pol_to_cart
from initial_conditions import create_initial_positions, create_initial_states
from geodesics import derivatives_ddt, integrate_rays
from animation import plot_rays
from analysis import find_qstar, find_closest_turning_ray

y_rays = 3000
y_size = 6
x_0 = 20

t_min = 0.0
t_max = 100.0
t_steps = 2000

initial_states = create_initial_states(y_rays, y_size, x_0, t_min)
ray_states = integrate_rays(initial_states, t_min, t_max, t_steps)
# plot_rays(ray_states)












ray_states = integrate_rays(
    initial_states,
    t_min,
    t_max,
    t_steps,
)

has_both_signs, only_negative, only_positive = find_qstar(ray_states)

lower_index, upper_index, lower_y, upper_y = find_closest_turning_ray(
    ray_states,
    y_size,
)

print("Rays with both signs:", np.count_nonzero(has_both_signs))
print("Rays with only negative p_r:", np.count_nonzero(only_negative))
print("Rays with only positive p_r:", np.count_nonzero(only_positive))

print("Closest negative-y turning ray:")
print("  index =", lower_index)
print("  y     =", lower_y)

print("Closest positive-y turning ray:")
print("  index =", upper_index)
print("  y     =", upper_y)








