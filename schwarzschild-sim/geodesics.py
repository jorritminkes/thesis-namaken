import numpy as np
from metric import inv_metric

def derivatives_ddt(states):
    M = 1.0
    r_s = 2.0 * M
    
    r = states[:,1]
    
    p_t = states[:,3]
    p_r = states[:,4]
    p_theta = states[:,5]
    
    position = states[:, :3]
    g_inv = inv_metric(position, M)
    
    f = 1.0 - r_s/r
    
    d_r_gtt = r_s / (r**2 * f**2)
    d_r_grr = r_s/r**2
    d_r_gthetatheta = -2.0/r**3
    
    dt_dlambda = g_inv[:,0] * p_t
    dr_dlambda = g_inv[:,1] * p_r
    dtheta_dlambda = g_inv[:,2] * p_theta
    
    dp_t_dlambda = 0.0
    # dp_r_dlambda = 0.5 * (d_r_gtt * p_t**2 + d_r_grr * p_r**2 + d_r_gthetatheta * p_theta**2)
    dp_r_dlambda = - 0.5 * (d_r_gtt * p_t**2 + d_r_grr * p_r**2 + d_r_gthetatheta * p_theta**2)
    dp_theta_dlambda = 0.0
    
    dlambda_dt = 1.0 / dt_dlambda
    
    dr_dt = dlambda_dt * dr_dlambda
    dtheta_dt = dlambda_dt * dtheta_dlambda
    dp_t_dt = dlambda_dt * dp_t_dlambda
    dp_r_dt = dlambda_dt * dp_r_dlambda
    dp_theta_dt = dlambda_dt * dp_theta_dlambda
    
    derivatives = np.empty_like(states)
    
    derivatives[:, 0] = 1.0
    derivatives[:, 1] = dr_dt
    derivatives[:, 2] = dtheta_dt
    derivatives[:, 3] = dp_t_dt
    derivatives[:, 4] = dp_r_dt
    derivatives[:, 5] = dp_theta_dt
    
    return derivatives

def rk4_single_step(states, h):
    k1 = derivatives_ddt(states)
    k2 = derivatives_ddt(states + 0.5 * h * k1)
    k3 = derivatives_ddt(states + 0.5 * h * k2)
    k4 = derivatives_ddt(states + h * k3)
    return states + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

def integrate_rays(initial_states, t_min, t_max, t_steps, R_max = 100, horizon_eps=1e-2):
    M = 1.0
    r_s = 2.0 * M
    
    h = (t_max - t_min) / t_steps
    
    states = np.asarray(initial_states, dtype=np.float64).copy()
    n_rays = states.shape[0]
    
    history = np.empty((t_steps + 1, n_rays, 6), dtype=np.float64)
    history[0] = states
    
    active = np.ones(n_rays, dtype=bool)
    
    for i in range(1, t_steps + 1):
        if active.any():
            idx = np.flatnonzero(active)
            new = rk4_single_step(states[idx], h)
            
            ok = np.isfinite(new).all(axis=1)
            idx = np.flatnonzero(active)
            states[idx[ok]] = new[ok]
            active[idx[~ok]] = False
            
            r = states[:, 1]
            active &= (r > r_s * (1.0 + horizon_eps)) & (r < R_max)
        
        history[i] = states
    
    return history














