import numpy as np

def get_linearized_dynamics(m_base=1.0, mw=0.432, mp=5.0, l=0.4, r=0.1, Ip=0.2, g=9.81, damping_x=0.0, damping_theta=0.0):
    """
    Returns (A, B) matrices for the Segway model.
    State: [x, x_dot, theta, theta_dot]
    Input: u (torque/force coupled)
    """
    M = m_base + 2 * mw
    Dx = damping_x
    Dtheta = damping_theta

    det = (M + mp) * (Ip + mp * l**2) - (mp * l)**2

    # A Matrix
    a22 = -Dx * (Ip + mp * l**2) / det
    a23 = -g * (mp * l)**2 / det
    a24 = Dtheta * mp * l / det
    
    a42 = Dx * mp * l / det
    a43 = g * mp * l * (M + mp) / det
    a44 = -Dtheta * (M + mp) / det

    A = np.array([
        [0, 1, 0, 0],
        [0, a22, a23, a24],
        [0, 0, 0, 1],
        [0, a42, a43, a44]
    ])

    # B Matrix
    b_x = (mp * l + (Ip + mp * l**2) / r) / det
    b_theta = -(mp * l / r + (M + mp)) / det

    B = np.array([
        [0],
        [b_x],
        [0],
        [b_theta]
    ])
    
    return A, B
