import numpy as np

def cart_to_pol(cartesian_position):
    t, x, y = cartesian_position
    r = np.sqrt(x*x+y*y)
    theta = np.arctan2(y, x)
    return t, r, theta