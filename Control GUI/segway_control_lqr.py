import mujoco as mj
from mujoco.glfw import glfw
import numpy as np
import matplotlib.pyplot as plt
import time
import time
import segway_controllers as controllers

def run_segway_simulation(
    # Physics
    m_base=1.0, mw=0.432, mp=5.0, l=0.4, r=0.1, Ip=0.2,
    damping_x=0.05, damping_theta=0.2,
    # Controller
    strat_name="LQR",
    strat_params=None, # Dict of params (e.g. {Q:..., R:...} or {poles:...})
    # Simulation
    initial_tilt=1.0,
    initial_pos=0.0,
    disturbances=[],
    duration=30.0,
    sim_speed_factor=1.25,
    show_plots=True
):
    """
    Runs the Segway simulation with a specific control strategy.
    
    strat_name: "LQR", "MPC", "PolePlacement", "SMC", "HInf"
    strat_params: Dict containing necessary params for the chosen strategy.
    """
    if strat_params is None: strat_params = {}

    # ----------------------------
    # 1. Physics Setup
    # ----------------------------
    # Aggregate physics params for the controller
    phys_params = {
        'm_base': m_base, 'mw': mw, 'mp': mp, 'l': l, 'r': r, 'Ip': Ip,
        'damping_x': damping_x, 'damping_theta': damping_theta
    }
    
    # ----------------------------
    # 2. Instantiate Controller
    # ----------------------------
    print(f"Initializing {strat_name} Controller...")
    
    if strat_name == "LQR":
        # Expects Q_diag, R_val
        q = strat_params.get("Q_diag", [980, 1, 1, 1])
        r_val = strat_params.get("R_val", 0.01)
        controller = controllers.LQRController(phys_params, q, r_val)
        
    elif strat_name == "PolePlacement":
        # Expects 'poles' list
        poles = strat_params.get("poles", [-2, -3, -4, -5])
        controller = controllers.PolePlacementController(phys_params, poles)
        
    elif strat_name == "SMC":
        # Expects lambda, K, phi
        lam = strat_params.get("lambda", [2.0, 1.0, 5.0, 0.8])
        K = strat_params.get("K_smc", 20.0)
        phi = strat_params.get("phi", 0.1)
        controller = controllers.SlidingModeController(phys_params, lam, K, phi)
        
    elif strat_name == "MPC":
        q = strat_params.get("Q_diag", [100, 1, 1000, 1])
        r_val = strat_params.get("R_val", 0.01)
        horizon = strat_params.get("horizon", 20)
        controller = controllers.MPCController(phys_params, q, r_val, horizon)
        
    else:
        print(f"Unknown strategy {strat_name}, defaulting to LQR")
        controller = controllers.LQRController(phys_params, [1,1,1,1], 1)

    # ----------------------------
    # 3. Load MuJoCo & Update Physics
    # ----------------------------
    model = mj.MjModel.from_xml_path("segway.xml")
    
    # --- Runtime Physics Update ---
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
        dof_adr = model.jnt_dofadr[slide_id]
        model.dof_damping[dof_adr] = damping_x
    if hinge_id != -1:
        dof_adr = model.jnt_dofadr[hinge_id]
        model.dof_damping[dof_adr] = damping_theta

    data = mj.MjData(model)
    
    # Initial Condition
    data.qpos[1] = initial_tilt
    data.qpos[0] = initial_pos
    mj.mj_forward(model, data)

    # ----------------------------
    # 4. Simulation Loop
    # ----------------------------
    # Visualization setup
    if not glfw.init(): return
    window = glfw.create_window(1600, 1200, f"Segway - {strat_name}", None, None)
    glfw.make_context_current(window)
    cam = mj.MjvCamera()
    cam.type = mj.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = 1 
    cam.distance = 4.5
    cam.elevation = -30
    cam.azimuth = 135 # Isometric
    
    opt = mj.MjvOption()
    scene = mj.MjvScene(model, maxgeom=10000)
    context = mj.MjrContext(model, mj.mjtFontScale.mjFONTSCALE_150.value)

    time_data, pos_data, theta_data, ctrl_data = [], [], [], []
    dt = model.opt.timestep
    steps = int(duration / dt)
    
    last_render_time = time.time()
    
    for i in range(steps):
        t = i * dt
        
        # --- Disturbances ---
        for d in disturbances:
            # Apply impulse at specific time (duration of 1 timestep)
            if abs(t - d['time']) < dt/2:
                print(f"Kick at t={t:.2f}!")
                data.qvel[1] += d['impulse']
        
        # --- Control Step ---
        # State: x, x_dot, theta, theta_dot
        state = np.array([data.qpos[0], data.qvel[0], data.qpos[1], data.qvel[1]])
        
        u_val = controller.update(state, t)
        
        # Coupled Torque Application
        # Force on Wheel = u/r   (Pushing cart)
        # Torque on Pendulum = -u (Reaction)
        # Check sign convention: 
        # If u > 0 (torque to correct falling right), cart pushes right, body twists left?
        # Standard: data.ctrl[0]=u/r, data.ctrl[1]=-u is derived in dynamics.
        
        data.ctrl[0] = np.clip(u_val / r, -1000, 1000)
        data.ctrl[1] = np.clip(-u_val, -100, 100)
        
        mj.mj_step(model, data)
        
        # Data Rec
        if i % 10 == 0:
            time_data.append(t)
            pos_data.append(data.qpos[0])
            theta_data.append(data.qpos[1])
            ctrl_data.append(u_val)
            
        # Real-time Sync
        # We want t_sim to match t_wall
        # Check current wall time
        if i == 0: start_wall_time = time.time()
        
        target_wall_time = t / 1.0 # 1.0 = real speed
        current_wall_time = time.time() - start_wall_time
        
        drift = target_wall_time - current_wall_time
        if drift > 0:
            time.sleep(drift)
            
        # Render (Real-time sync)
        if time.time() - last_render_time > 0.033:
            viewport = mj.MjrRect(0, 0, 1600, 1200)
            mj.mjv_updateScene(model, data, opt, None, cam, mj.mjtCatBit.mjCAT_ALL.value, scene)
            mj.mjr_render(viewport, scene, context)
            
            # Dynamic Title
            title = f"Segway ({strat_name}) | Time: {t:.2f} s | Tilt: {data.qpos[1]:.3f} rad"
            glfw.set_window_title(window, title)
            
            glfw.swap_buffers(window)
            glfw.poll_events()
            last_render_time = time.time()
            
            if glfw.window_should_close(window): break
            
    glfw.terminate()
    
    # ----------------------------
    # 5. Plotting
    # ----------------------------
    if show_plots and len(time_data) > 0:
        # Increase plot size and font size
        plt.rcParams.update({'font.size': 12})
        plt.figure(figsize=(14, 12))
        
        plt.subplot(3, 1, 1)
        plt.plot(time_data, theta_data, 'r', label="Tilt (rad)", linewidth=2)
        plt.axhline(0, color='k', linestyle='--')
        plt.ylabel("Angle (rad)", fontsize=14)
        plt.title(f"Response - {strat_name}", fontsize=16, fontweight='bold')
        plt.legend(fontsize=12)
        plt.grid(True)
        
        plt.subplot(3, 1, 2)
        plt.plot(time_data, pos_data, 'b', label="Position (m)", linewidth=2)
        plt.ylabel("Position (m)", fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True)
        
        plt.subplot(3, 1, 3)
        plt.plot(time_data, ctrl_data, 'g', label="Control Torque (Nm)", linewidth=2)
        plt.ylabel("Torque (Nm)", fontsize=14)
        plt.xlabel("Time (s)", fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    # Default behavior if run directly
    print("Running with default parameters...")
    run_segway_simulation(
        Q_diag=[980.93, 1.10, 1.69, 1.00],
        R_val=0.0056,
        initial_tilt=1.0,
        disturbances=[
            {'time': 5.0, 'impulse': 0.2},
            {'time': 10.0, 'impulse': -0.2}
        ],
        duration=15.0
    )
