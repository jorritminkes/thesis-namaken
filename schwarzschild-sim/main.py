import numpy as np
import matplotlib.pyplot as plt

from metric import metric, inv_metric
from coordinates import cart_to_pol
from initial_conditions import create_initial_positions, create_initial_state

y_rays = 10
y_size = 10
x_0 = 20

rays_initial_positions = create_initial_positions(y_rays, y_size, x_0)
initial_states = create_initial_state(rays_initial_positions)
