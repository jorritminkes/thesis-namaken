import numpy as np
import matplotlib.pyplot as plt

from metric import metric, inv_metric
from coordinates import cart_to_pol, pol_to_cart
from initial_conditions import create_initial_positions, create_initial_states
from geodesics import derivatives_ddt, integrate_rays
from animation import plot_rays

y_rays = 100
y_size = 10
x_0 = 20

t_min = 0.0
t_max = 100.0
t_steps = 2000

initial_states = create_initial_states(y_rays, y_size, x_0, t_min)
ray_states = integrate_rays(initial_states, t_min, t_max, t_steps)

plot_rays(ray_states)