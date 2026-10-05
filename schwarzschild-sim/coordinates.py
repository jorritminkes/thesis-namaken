import numpy as np

def cart_to_pol(x, y):
    r = np.sqrt(x*x+y*y)
    theta = np.arctan2(y, x)
    return r, theta