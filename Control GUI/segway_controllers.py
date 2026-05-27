import numpy as np
import control as ctrl
import cvxpy as cp
from scipy.signal import cont2discrete
from scipy.linalg import solve_discrete_are
import segway_dynamics

class BaseController:
    def __init__(self, params):
        self.params = params
        self.A, self.B = segway_dynamics.get_linearized_dynamics(
            m_base=params.get('m_base', 1.0),
            mw=params.get('mw', 0.432),
            mp=params.get('mp', 5.0),
            l=params.get('l', 0.4),
            r=params.get('r', 0.1),
            Ip=params.get('Ip', 0.2),
            damping_x=params.get('damping_x', 0.05),
            damping_theta=params.get('damping_theta', 0.2)
        )

    def reset(self):
        pass

class LQRController(BaseController):
    def __init__(self, params, Q_diag, R_val):
        super().__init__(params)
        Q = np.diag(Q_diag)
        R = np.array([[R_val]])
        try:
            self.K, _, _ = ctrl.lqr(self.A, self.B, Q, R)
            self.K = np.asarray(self.K)
            print(f"[LQR] K = {self.K}")
        except Exception as e:
            print(f"[LQR] Failed: {e}")
            self.K = np.zeros((1, 4))

    def update(self, state, t=0):
        u = -self.K @ state
        return u.item()
    
    def reset(self):
        pass

class PolePlacementController(BaseController):
    def __init__(self, params, poles):
        super().__init__(params)
        try:
            # poles is a list of 4 complex/real numbers
            print(f"[PP] Placing poles at: {poles}")
            self.K = ctrl.place(self.A, self.B, poles)
            self.K = np.asarray(self.K)
            print(f"[PP] K = {self.K}")
        except Exception as e:
            print(f"[PP] Failed: {e}")
            self.K = np.zeros((1, 4))
            
    def update(self, state, t=0):
        u = -self.K @ state
        return u.item()
    
    def reset(self):
        pass

class SlidingModeController(BaseController):
    def __init__(self, params, lambda_vec, K_smc, phi):
        super().__init__(params)
        # lambda_vec: [lambda1, 1.0, lambda2, lambda3] (coeffs for x, xd, th, thd)
        # Typically xd coeff is 1.0
        self.coeffs = np.array(lambda_vec) 
        self.K_smc = K_smc
        self.phi = phi
        
        # Pre-compute terms for equivalent control
        # Need system params for this specific derivation
        # u_eq = -(CB)^-1 * CA * x
        # S = C_surf * x
        # C_surf = coeffs
        
        # We assume S = coeffs @ x
        # S_dot = coeffs @ (Ax + Bu) = coeffs@A@x + coeffs@B@u
        # For S_dot = -K sgn(S), we have
        # coeffs@A@x + coeffs@B@u = -K sgn(S)
        # u = -(coeffs@B)^-1 * (coeffs@A@x + K sgn(S))
        
        self.CB_inv = 1.0 / (self.coeffs @ self.B)
        self.CA = self.coeffs @ self.A
        
    def sat(self, s):
        return np.clip(s / self.phi, -1.0, 1.0)

    def update(self, state, t=0):
        s = self.coeffs @ state
        
        # Standard SMC: u = u_eq + u_sw
        # u = -(CB)^-1 * (CAx + K sat(S))
        
        term = self.CA @ state + self.K_smc * self.sat(s)
        u = -self.CB_inv * term
        return u.item()
    
    def reset(self):
        pass

class MPCController(BaseController):
    def __init__(self, params, Q_diag, R_val, horizon=10):
        super().__init__(params)
        self.N = int(horizon)
        self.dt = 0.05 # MPC stepping
        
        # Discretize
        sysd = cont2discrete((self.A, self.B, np.eye(4), np.zeros((4,1))), self.dt)
        self.Ad = sysd[0]
        self.Bd = sysd[1]
        
        # Weights
        self.Q = np.diag(Q_diag)
        self.R = np.array([[R_val]])
        
        # Setup CVXPY Problem (Warm start optimization)
        # We re-solve at each step, but can't pickle CVXPY easily if we pre-compile too strictly.
        # Ideally we define the structure once.
        self.x0_param = cp.Parameter(4)
        self.u_vars = cp.Variable((1, self.N))
        self.x_vars = cp.Variable((4, self.N + 1))
        
        cost = 0
        constraints = [self.x_vars[:, 0] == self.x0_param]
        
        for k in range(self.N):
            cost += cp.quad_form(self.x_vars[:, k], self.Q) + cp.quad_form(self.u_vars[:, k], self.R)
            constraints += [self.x_vars[:, k+1] == self.Ad @ self.x_vars[:, k] + self.Bd @ self.u_vars[:, k]]
            constraints += [self.u_vars[:, k] <= 100, self.u_vars[:, k] >= -100] # Torque limits
            
        cost += cp.quad_form(self.x_vars[:, self.N], self.Q) # Terminal cost
        
        self.prob = cp.Problem(cp.Minimize(cost), constraints)
        self.last_update_time = -1
        self.last_u = 0

    def update(self, state, t=0):
        # Basic MPC: Run every 0.05s roughly
        if t - self.last_update_time < self.dt and self.last_update_time >= 0:
             return self.last_u

        try:
            self.x0_param.value = state
            self.prob.solve(solver=cp.OSQP, warm_start=True)
            if self.u_vars.value is not None:
                self.last_u = self.u_vars[:, 0].value.item()
            else:
                self.last_u = 0
        except:
             pass # Fail safe
             
        self.last_update_time = t
        return self.last_u
    
    def reset(self):
        self.last_update_time = -1
        self.last_u = 0


