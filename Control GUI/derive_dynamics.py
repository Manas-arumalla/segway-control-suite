import sympy as sp

# Define symbols
g, mw, mp, l, r, Ip, Dx, Dtheta = sp.symbols('g mw mp l r Ip Dx Dtheta')
x, theta = sp.symbols('x theta')
x_dot, theta_dot = sp.symbols('x_dot theta_dot')
x_ddot, theta_ddot = sp.symbols('x_ddot theta_ddot')
tau = sp.symbols('tau')

# Total Cart Mass (Base + 2 Wheels)
# Note: The wheel's moment of inertia Iw also contributes to effective mass IP term if we were doing full contact dynamics,
# but using the "force/reaction" model:
M = sp.symbols('M') # Total base mass

# Equations of Motion for Segway (Linearized)
# 1. Horizontal Motion: (M + m) * x_ddot + m*l*theta_ddot = F_traction - Dx * x_dot
#    where F_traction = tau / r
# 2. Rotational Motion: m*l*x_ddot + (m*l^2 + Ip) * theta_ddot = -tau + m*g*l*theta - Dtheta * theta_dot
#    (Reaction torque -tau opposes motion, Gravity assists theta)

numerator_x = tau/r - Dx*x_dot
numerator_theta = -tau + mp*g*l*theta - Dtheta*theta_dot

# System Matrix Form for [x_ddot, theta_ddot]^T
# [ (M+mp)      mp*l       ] [x_ddot    ] = [ numerator_x    ]
# [ mp*l       (mp*l^2+Ip) ] [theta_ddot]   [ numerator_theta]

# Mass Matrix
J = sp.Matrix([
    [M + mp, mp*l],
    [mp*l, mp*l**2 + Ip]
])

# RHS
RHS = sp.Matrix([
    numerator_x,
    numerator_theta
])

# Solve for accelerations
accels = J.inv() * RHS

# Extract coeffecients for A and B matrices
# state = [x, x_dot, theta, theta_dot]
# x_ddot = f1(state, tau)
# theta_ddot = f2(state, tau)

f_x = accels[0]
f_theta = accels[1]

# Gradients (Jacobians)
A_rows = []
B_rows = []

vars = [x, x_dot, theta, theta_dot]

# Row 1: dot(x) = x_dot
A_rows.append([0, 1, 0, 0])
B_rows.append([0])

# Row 2: dot(x_dot) = x_ddot
A_rows.append([sp.diff(f_x, v) for v in vars])
B_rows.append([sp.diff(f_x, tau)])

# Row 3: dot(theta) = theta_dot
A_rows.append([0, 0, 0, 1])
B_rows.append([0])

# Row 4: dot(theta_dot) = theta_ddot
A_rows.append([sp.diff(f_theta, v) for v in vars])
B_rows.append([sp.diff(f_theta, tau)])

print("A Matrix:")
for row in A_rows:
    print(row)

print("\nB Matrix:")
for row in B_rows:
    print(row)
