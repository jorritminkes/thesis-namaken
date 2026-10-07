import numpy as np
from metric import inv_metric

def derivatives_ddt(states):
    M = 1.0
    r_s = 2.0 * M
    # t = states[:,0]
    r = states[:,1]
    # theta = states[:,2]
    
    p_t = states[:,3]
    p_r = states[:,4]
    p_theta = states[:,5]
    
    position = states[:, :3]
    g_inv = inv_metric(position, M)
    
    f = 1.0 - r_s/r
    
    d_r_gtt = -1/(f**2) * r_s/r**2
    d_r_grr = r_s/r**2
    d_r_gthetatheta = -2.0/r**3
    
    dt_dlambda = g_inv[:,0] * p_t
    dr_dlambda = g_inv[:,1] * p_r
    dtheta_dlambda = g_inv[:,2] * p_theta
    
    dp_t_dlambda = 0.0
    dp_r_dlambda = 0.5 * (d_r_gtt * p_t**2 + d_r_grr * p_r**2 + d_r_gthetatheta * p_theta**2)
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
    