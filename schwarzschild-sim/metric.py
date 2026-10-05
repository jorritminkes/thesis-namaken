def metric(t, r, theta, M):
    r_s = 2*M
    f = 1 - r_s/r
    g_tt = -f
    g_rr = 1/f
    g_thth = r*r
    return g_tt, g_rr, g_thth

def inv_metric(t, r, theta, M):
    r_s = 2*M
    f = 1 - r_s/r
    gtt = -1/f
    grr = f
    gthth = 1/(r*r)
    return gtt, grr, gthth

