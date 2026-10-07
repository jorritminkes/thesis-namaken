import numpy as np
import matplotlib.pyplot as plt

from coordinates import pol_to_cart

def plot_rays(ray_states):
    positions = pol_to_cart(ray_states[..., :3])
    x = positions[..., 1]
    y = positions[..., 2]
    
    
    fig, ax = plt.subplots(figsize=(12, 12))
    
    for i in range(x.shape[1]):
        ax.plot(x[:, i], y[:, i])
    
    hole = plt.Circle((0, 0), 2, color="black")
    ax.add_artist(hole)
    
    ax.set_xlim(-10, 20)
    ax.set_ylim(-15, 15)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    
    plt.show()