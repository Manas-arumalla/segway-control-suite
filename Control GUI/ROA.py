import mujoco as mj
import numpy as np
import matplotlib.pyplot as plt
import segway_controllers as controllers

def generate_roa_plot(
    # Physics
    m_base=1.0, mw=0.432, mp=5.0, l=0.4, r=0.1, Ip=0.2,
    damping_x=0.05, damping_theta=0.2,
    # Controller
    strat_name="LQR",
    strat_params=None
):
    """
    Generates generic ROA plot for any controller.
    """
    if strat_params is None: strat_params = {}
    
    # ----------------------------
    # 1. Setup Controller
    # ----------------------------
    phys_params = {
        'm_base': m_base, 'mw': mw, 'mp': mp, 'l': l, 'r': r, 'Ip': Ip,
        'damping_x': damping_x, 'damping_theta': damping_theta
    }
    
    print(f"[ROA] Initializing {strat_name}...")
    # NOTE: Code Duplication with segway_control_lqr.py :( 
    # But acceptable for now.
    
    if strat_name == "LQR":
        q = strat_params.get("Q_diag", [980, 1, 1, 1])
        r_val = strat_params.get("R_val", 0.01)
        controller = controllers.LQRController(phys_params, q, r_val)
    elif strat_name == "PolePlacement":
        poles = strat_params.get("poles", [-2, -3, -4, -5])
        controller = controllers.PolePlacementController(phys_params, poles)
    elif strat_name == "SMC":
        lam = strat_params.get("lambda", [2.0, 1.0, 5.0, 0.8])
        K = strat_params.get("K_smc", 20.0)
        phi = strat_params.get("phi", 0.1)
        controller = controllers.SlidingModeController(phys_params, lam, K, phi)
    elif strat_name == "MPC":
        q = strat_params.get("Q_diag", [100, 1, 1000, 1])
        r_val = strat_params.get("R_val", 0.01)
        hz = strat_params.get("horizon", 10)
        controller = controllers.MPCController(phys_params, q, r_val, hz)
    else:
        controller = controllers.LQRController(phys_params, [1,1,1,1], 1)

    # ----------------------------
    # 2. Load model & Update Physics
    # ----------------------------
    model = mj.MjModel.from_xml_path("segway.xml")
    
    # Update Mass/Inertia/Damping at runtime
    M = m_base + 2 * mw
    base_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "base")
    if base_id != -1: model.body_mass[base_id] = M

    pend_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "pendulum")
    if pend_id != -1:
        model.body_mass[pend_id] = mp
        model.body_inertia[pend_id][:] = [Ip, Ip, Ip/20.0]
        model.body_ipos[pend_id][2] = l 
        
    slide_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, "slide")
    hinge_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, "pend_hinge")
    
    if slide_id != -1: 
        model.dof_damping[model.jnt_dofadr[slide_id]] = damping_x
    if hinge_id != -1: 
        model.dof_damping[model.jnt_dofadr[hinge_id]] = damping_theta

    # ----------------------------
    # 3. RoA grid
    # ----------------------------
    print("Computing ROA... This might take time.")
    theta_vals = np.linspace(-1.2, 1.2, 30) # Reduced resolution for speed
    thetadot_vals = np.linspace(-5.0, 5.0, 30)
    
    ROA = np.zeros((len(theta_vals), len(thetadot_vals)))
    
    Tsim = 4.0
    dt = model.opt.timestep
    steps = int(Tsim / dt)
    
    for i, theta0 in enumerate(theta_vals):
        for j, thetadot0 in enumerate(thetadot_vals):
            
            data = mj.MjData(model)
            data.qpos[0] = 0.0
            data.qvel[0] = 0.0
            data.qpos[1] = theta0
            data.qvel[1] = thetadot0
            
            # CRITICAL: Reset controller state!
            controller.reset()
            
            success = True
            
            for k in range(steps):
                t = k * dt
                state = np.array([data.qpos[0], data.qvel[0], data.qpos[1], data.qvel[1]])
                
                u_val = controller.update(state, t)
                
                data.ctrl[0] = np.clip(u_val / r, -1000, 1000)
                data.ctrl[1] = np.clip(-u_val, -100, 100)
                
                mj.mj_step(model, data)
                
                if abs(data.qpos[1]) > 1.4: # Fell over (>80 deg)
                    success = False
                    break
            
            if success and abs(data.qpos[1]) < 0.2: # Stabilized
                ROA[i, j] = 1

    # ----------------------------
    # Plot RoA
    # ----------------------------
    plt.figure(figsize=(8, 6))
    plt.contourf(thetadot_vals, theta_vals, ROA, levels=[-0.1, 0.5, 1.1], colors=["red", "green"], alpha=0.7)
    plt.xlabel(r"$\dot{\theta}_0$ (rad/s)")
    plt.ylabel(r"$\theta_0$ (rad)")
    plt.title(f"Region of Attraction ({strat_name})")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    generate_roa_plot()
