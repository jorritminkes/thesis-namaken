import numpy as np
import matplotlib.pyplot as plt

from metric import metric, inv_metric
from coordinates import cart_to_pol, pol_to_cart
from initial_conditions import create_initial_positions, create_initial_states
from geodesics import derivatives_ddt

y_rays = 10
y_size = 10
x_0 = 20

initial_states = create_initial_states(y_rays, y_size, x_0)
