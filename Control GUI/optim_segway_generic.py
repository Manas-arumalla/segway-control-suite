import numpy as np
import control as ctrl
from deap import base, creator, tools, algorithms
import random
import random
import mujoco as mj
import segway_controllers as controllers
import segway_dynamics

# Initialize DEAP
# We need to maximize stability (minimize cost)
if not hasattr(creator, "FitnessMin"):
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
if not hasattr(creator, "Individual"):
    creator.create("Individual", list, fitness=creator.FitnessMin)

def get_bounds(strat_name):
    """
    Returns (min_bounds, max_bounds, ndim) for a given strategy.
    """
    if strat_name in ["LQR", "MPC"]:
        # [Q1, Q2, Q3, Q4, R] (and maybe Horizon later for MPC, but keeping simple for now)
        # Q diagonal: [1, 1000]
        # R: [0.0001, 1.0]
        lows = [1, 1, 1, 1, 0.0001]
        highs = [2000, 100, 2000, 100, 1.0]
        return lows, highs, 5
        
    elif strat_name == "PolePlacement":
        # 4 Poles (Real parts)
        # Range: -50 to -0.5 (must be stable)
        lows = [-50, -50, -50, -50]
        highs = [-0.5, -0.5, -0.5, -0.5]
        return lows, highs, 4
        
    elif strat_name == "SMC":
        # [lambda1, lambda2, lambda3, K, phi]
        # Lambdas typically > 0.
        # K > some bound
        # Phi > 0
        lows = [0.1, 0.1, 0.1, 1.0, 0.01]
        highs = [10.0, 10.0, 10.0, 100.0, 1.0]
        return lows, highs, 5
    
    return [], [], 0

def simulation_cost(individual, strat_name):
    """
    Evaluates an individual by running a short simulation.
    Cost = Settling Time + Penalty for Failure
    """
    # 1. Decode Individual -> Controller Params
    params = {}
    
    # Physics for controller
    phys_params = {
        'm_base': 1.0, 'mw': 0.432, 'mp': 5.0, 'l': 0.4, 'r': 0.1, 'Ip': 0.2,
        'damping_x': 0.05, 'damping_theta': 0.2
    }
    
    try:
        if strat_name in ["LQR", "MPC"]:
            q_diag = individual[0:4]
            r_val = individual[4]
            # Verify positive definiteness strictly
            if any(v <= 0 for v in q_diag) or r_val <= 0: return 1e6,
            
            if strat_name == "LQR":
                controller = controllers.LQRController(phys_params, q_diag, r_val)
            else:
                controller = controllers.MPCController(phys_params, q_diag, r_val, horizon=15)
                
        elif strat_name == "PolePlacement":
            poles = individual[0:4]
            controller = controllers.PolePlacementController(phys_params, poles)
            
        elif strat_name == "SMC":
            lam = [individual[0], 1.0, individual[1], individual[2]]
            K = individual[3]
            phi = individual[4]
            controller = controllers.SlidingModeController(phys_params, lam, K, phi)
            
        else:
            return 1e6,

    except Exception:
        return 1e6, # Controller creation failed (e.g., unstable poles)

    # 2. Run Simulation (Headless)
    model = mj.MjModel.from_xml_path("segway.xml")
    data = mj.MjData(model)
    
    # Initial Disturbance
    data.qpos[1] = 0.4 # 22 degrees tilt
    mj.mj_forward(model, data)
    
    T_sim = 4.0
    dt = model.opt.timestep
    steps = int(T_sim / dt)
    
    settling_time = T_sim * 2 # Default penalty
    stabilized = False
    
    for i in range(steps):
        t = i * dt
        state = np.array([data.qpos[0], data.qvel[0], data.qpos[1], data.qvel[1]])
        
        # Check Failure
        if abs(state[2]) > 1.2: # Fall
            return 1e4 - t, # Reward longer survival if fail
            
        u_val = controller.update(state, t)
        
        data.ctrl[0] = np.clip(u_val / 0.1, -1000, 1000)
        data.ctrl[1] = np.clip(-u_val, -100, 100)
        
        mj.mj_step(model, data)
        
        # Check Settling
        # Condition: Angle < 0.05 rad (~3 deg) and AngVel < 0.1
        if abs(state[2]) < 0.05 and abs(state[3]) < 0.1:
            if not stabilized:
                settling_time = t
                stabilized = True
        else:
            stabilized = False # Must stay settled
            
    if stabilized:
        return settling_time,
    else:
        return 100.0 + abs(data.qpos[1]), # Penalty based on final error

def optimize_parameters(strat_name="LQR"):
    print(f"\n[GA] Starting Optimization for {strat_name}...")
    
    lows, highs, ndim = get_bounds(strat_name)
    if ndim == 0:
        print("Strategy not supported for Auto-Tuning.")
        return []

    # Setup Toolbox
    toolbox = base.Toolbox()
    
    # Attribute generator
    for i in range(ndim):
        toolbox.register(f"attr_{i}", random.uniform, lows[i], highs[i])
    
    # Structure
    toolbox.register("individual", tools.initCycle, creator.Individual,
                     tuple(getattr(toolbox, f"attr_{i}") for i in range(ndim)), n=1)
                     
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    # Evaluation
    toolbox.register("evaluate", simulation_cost, strat_name=strat_name)
    
    # Operators
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutPolynomialBounded, eta=10, 
                     low=lows, up=highs, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)
    
    # Run
    random.seed(42)
    pop = toolbox.population(n=30) # Small population for speed
    hof = tools.HallOfFame(1)
    
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", np.min)
    
    # 20 Generations
    algorithms.eaSimple(pop, toolbox, cxpb=0.6, mutpb=0.3, ngen=15, 
                        stats=stats, halloffame=hof, verbose=True)
    
    best = hof[0]
    print(f"[GA] Best Fitness: {best.fitness.values[0]}")
    print(f"[GA] Best Params: {best}")
    
    return list(best)

if __name__ == "__main__":
    # Test LQR
    optimize_parameters("LQR")
    # Test SMC
    optimize_parameters("SMC")
