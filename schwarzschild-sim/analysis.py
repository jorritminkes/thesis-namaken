import numpy as np

def find_qstar(ray_states, tol=1e-12):
    p_r = ray_states[:, :, 4]
    has_positive = np.any(p_r > tol, axis=0)
    has_negative = np.any(p_r < -tol, axis=0)
    
    has_both_signs = has_positive & has_negative
    only_negative = has_negative & ~has_positive
    only_positive = has_positive & ~has_negative
    return has_both_signs, only_negative, only_positive