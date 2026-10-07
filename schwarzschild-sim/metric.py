import numpy as np

def metric(position, M):
    r = position[..., 1]
    
    f = 1 - 2*M/r
    
    g = np.empty(position.shape[:-1] + (3,))
    
    g[...,0] = -f
    g[...,1] = 1/f
    g[...,2] = r*r
    
    return g

def inv_metric(position, M):
    r = position[..., 1]
    
    f = 1 - 2*M/r
    
    g = np.empty(position.shape[:-1] + (3,))
    
    g[...,0] = -1/f
    g[...,1] = f
    g[...,2] = 1/(r*r)
    
    return g