import numpy as np

def find_qstar(ray_states, tol=1e-12):
    r = ray_states[:, :, 1]
    theta = ray_states[:, :, 2]
    p_r = ray_states[:, :, 4]

    # Cartesian y from the polar state
    y = r * np.sin(theta)
    y0 = y[0, :]

    # True while the ray is still on its starting side of y=0.
    # cumprod keeps it False after the first crossing, even if the
    # ray winds around and returns to its original side.
    same_side = (y * y0 > 0)
    before_crossing = np.cumprod(same_side, axis=0).astype(bool)

    has_positive = np.any((p_r > tol) & before_crossing, axis=0)
    has_negative = np.any((p_r < -tol) & before_crossing, axis=0)

    has_both_signs = has_positive & has_negative
    only_negative = has_negative & ~has_positive
    only_positive = has_positive & ~has_negative
    return has_both_signs, only_negative, only_positive

def find_closest_turning_ray(ray_states, y_size, tol=1e-12):
    n_rays = ray_states.shape[1]
    y_rays = np.linspace(-y_size, y_size, n_rays)

    has_both_signs, _, _ = find_qstar(ray_states, tol=tol)

    negative_candidates = np.flatnonzero(has_both_signs & (y_rays < 0))
    positive_candidates = np.flatnonzero(has_both_signs & (y_rays > 0))

    lower_index = None
    upper_index = None
    lower_y = None
    upper_y = None

    if negative_candidates.size > 0:
        lower_index = negative_candidates[np.argmin(np.abs(y_rays[negative_candidates]))]
        lower_y = y_rays[lower_index]

    if positive_candidates.size > 0:
        upper_index = positive_candidates[np.argmin(np.abs(y_rays[positive_candidates]))]
        upper_y = y_rays[upper_index]

    return lower_index, upper_index, lower_y, upper_y