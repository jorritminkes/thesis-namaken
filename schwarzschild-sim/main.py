import numpy as np
import matplotlib.pyplot as plt

from metric import metric, inv_metric
from coordinates import cart_to_pol
from initial_conditions import create_initial_grid

y_rays = 10
y_size = 10
x_0 = 20

create_initial_grid(y_rays, y_size, x_0)