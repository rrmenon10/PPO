import numpy as np

def conjugate_gradient(f_Ax,
                       b,
                       x_0=None,
                       cg_iters=10,
                       residual_tot=1e-10):
    x = np.zeros_like(b)
    r = b.copy()
    p = r.copy()
    rdotr = r.dot(r)

    for i in range(cg_iters):
        z = f_Ax(p)
        v = rdotr / p.dot(z)
        x += v * p
        r -= v * z
        newrdotr = r.dot(r)
        mu = newrdotr / rdotr
        p = r + mu * p

        rdotr = newrdotr
        if rdotr < residual_tot:
            break
    return x
