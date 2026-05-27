import tkinter as tk
from tkinter import ttk, messagebox, font
import segway_control_lqr
import optim_segway_generic as optim_segway
import ROA

class SegwayLauncher(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Segway Control Center")
        self.geometry("800x950")
        
        # --- THEME CONFIGURATION ---
        self.colors = {
            "bg_dark": "#1e1e1e",        # Main Window Background
            "bg_card": "#2d2d2d",        # Section Background
            "accent":  "#00e5ff",        # Cyan Highlight
            "text":    "#ffffff",        # White Text
            "text_dim": "#aaaaaa",       # Grey Text
            "btn_bg":  "#3d3d3d",        # Button Normal
            "btn_fg":  "#ffffff",
            "success": "#00e676",
        }
        
        self.configure(bg=self.colors["bg_dark"])
        
        # Configure Styles
        style = ttk.Style()
        style.theme_use('clam') # Clam supports custom colors best
        
        # Default Font
        default_font = ("Segoe UI", 10)
        title_font = ("Segoe UI", 11, "bold")
        
        # Frame & Label Styles
        style.configure("TFrame", background=self.colors["bg_dark"])
        style.configure("Card.TFrame", background=self.colors["bg_card"])
        
        style.configure("TLabel", background=self.colors["bg_dark"], foreground=self.colors["text"], font=default_font)
        style.configure("Card.TLabel", background=self.colors["bg_card"], foreground=self.colors["text"], font=default_font)
        
        # Labelframe Styles
        style.configure("TLabelframe", background=self.colors["bg_card"], bordercolor=self.colors["bg_dark"])
        style.configure("TLabelframe.Label", background=self.colors["bg_card"], foreground=self.colors["accent"], font=title_font)
        
        # Button Styles
        style.configure("TButton", 
                        background=self.colors["accent"], 
                        foreground="#000000", 
                        font=("Segoe UI", 10, "bold"),
                        borderwidth=0, 
                        focuscolor=self.colors["accent"])
        style.map("TButton", 
                  background=[('active', "#00b8cc")], 
                  foreground=[('active', 'black')])
        
        style.configure("Create.TButton", background=self.colors["btn_bg"], foreground=self.colors["btn_fg"])
        
        # Entry Style
        style.configure("TEntry", fieldbackground="#404040", foreground=self.colors["text"], insertcolor="white", borderwidth=0)
        
        # Radiobutton
        style.configure("TRadiobutton", background=self.colors["bg_card"], foreground=self.colors["text"], font=default_font, indicatorcolor=self.colors["accent"])
        style.map("TRadiobutton", indicatorcolor=[('selected', self.colors["accent"])])

        self.create_widgets()
        
    def create_widgets(self):
        # Main Container with padding
        main_container = ttk.Frame(self)
        main_container.pack(fill="both", expand=True, padx=20, pady=20)
        
        # --- Mode Selection ---
        mode_frame = ttk.LabelFrame(main_container, text="OPERATION MODE")
        mode_frame.pack(fill="x", pady=(0, 15), ipady=5)
        
        self.mode_var = tk.StringVar(value="Manual")
        
        rb1 = ttk.Radiobutton(mode_frame, text="Manual Tuning (Custom Parameters)", variable=self.mode_var, value="Manual", command=self.toggle_inputs)
        rb1.pack(anchor="w", padx=15, pady=5)
        
        rb2 = ttk.Radiobutton(mode_frame, text="Auto-Tune (Genetic Algorithm Optimization) - *Supports LQR, MPC, PP, SMC*", variable=self.mode_var, value="Auto", command=self.toggle_inputs)
        rb2.pack(anchor="w", padx=15, pady=5)

        # --- Controller Selection ---
        ctrl_select_frame = ttk.Frame(main_container)
        ctrl_select_frame.pack(fill="x", pady=(0, 5))
        
        ttk.Label(ctrl_select_frame, text="Control Strategy:", style="TLabel").pack(side="left", padx=5)
        
        self.strat_var = tk.StringVar(value="LQR")
        self.strat_combo = ttk.Combobox(ctrl_select_frame, textvariable=self.strat_var, state="readonly", width=15)
        self.strat_combo['values'] = ("LQR", "MPC", "PolePlacement", "SMC")
        self.strat_combo.pack(side="left", padx=5)
        self.strat_combo.bind("<<ComboboxSelected>>", self.update_ctrl_inputs)

        # --- Dynamic Control Parameters ---
        self.ctrl_frame = ttk.LabelFrame(main_container, text="CONTROL PARAMETERS")
        self.ctrl_frame.pack(fill="x", pady=(0, 15), ipady=5)
        
        self.param_vars = {} # Dict to hold entry widgets/vars
        self.update_ctrl_inputs() # Init with LQR

        # --- Robot Properties ---
        robot_frame = ttk.LabelFrame(main_container, text="ROBOT PHYSICAL PROPERTIES")
        robot_frame.pack(fill="x", pady=(0, 15), ipady=5)
        
        # Using a sub-frame for grid to manage background
        rf = ttk.Frame(robot_frame, style="Card.TFrame")
        rf.pack(fill="x", padx=10)
        
        props = [
            ("Base Mass (kg)", "1.0", "mbase_entry"),
            ("Wheel Mass (kg)", "0.432", "mw_entry"),
            ("Pendulum Mass (kg)", "5.0", "mp_entry"),
            ("Inertia (kg·m²)", "0.2", "ip_entry"),
            ("COM Length (m)", "0.4", "l_entry"),
            ("Wheel Radius (m)", "0.1", "r_entry_phys")
        ]
        
        for i, (text, val, attr) in enumerate(props):
            r, c = i // 2, (i % 2) * 2
            ttk.Label(rf, text=text, style="Card.TLabel").grid(row=r, column=c, padx=10, pady=5, sticky="w")
            e = ttk.Entry(rf, width=10)
            e.insert(0, val)
            e.grid(row=r, column=c+1, padx=5, pady=5)
            setattr(self, attr, e)

        # --- Damping & Init ---
        misc_frame = ttk.LabelFrame(main_container, text="ENVIRONMENT & INITIAL STATE")
        misc_frame.pack(fill="x", pady=(0, 15), ipady=5)
        mf = ttk.Frame(misc_frame, style="Card.TFrame")
        mf.pack(fill="x", padx=10)
        
        # Damping
        ttk.Label(mf, text="Damping (Slide)", style="Card.TLabel").grid(row=0, column=0, padx=10, sticky="w")
        self.dx_entry = ttk.Entry(mf, width=8)
        self.dx_entry.insert(0, "0.05")
        self.dx_entry.grid(row=0, column=1, padx=5)
        
        ttk.Label(mf, text="Damping (Hinge)", style="Card.TLabel").grid(row=0, column=2, padx=10, sticky="w")
        self.dtheta_entry = ttk.Entry(mf, width=8)
        self.dtheta_entry.insert(0, "0.2")
        self.dtheta_entry.grid(row=0, column=3, padx=5)
        
        # Init
        ttk.Label(mf, text="Initial Position (m)", style="Card.TLabel").grid(row=1, column=0, padx=10, pady=5, sticky="w")
        self.pos_entry = ttk.Entry(mf, width=8)
        self.pos_entry.insert(0, "0.0")
        self.pos_entry.grid(row=1, column=1, padx=5, pady=5)

        ttk.Label(mf, text="Initial Tilt (rad)", style="Card.TLabel").grid(row=1, column=2, padx=10, pady=5, sticky="w")
        self.tilt_entry = ttk.Entry(mf, width=8)
        self.tilt_entry.insert(0, "0.1")
        self.tilt_entry.grid(row=1, column=3, padx=5, pady=5)
        
        ttk.Label(mf, text="Sim Duration (s)", style="Card.TLabel").grid(row=2, column=0, padx=10, pady=5, sticky="w")
        self.duration_entry = ttk.Entry(mf, width=8)
        self.duration_entry.insert(0, "15.0")
        self.duration_entry.grid(row=2, column=1, padx=5, pady=5)

        # --- Disturbances ---
        dist_frame = ttk.LabelFrame(main_container, text="DISTURBANCES (KICKS)")
        dist_frame.pack(fill="both", expand=True, pady=(0, 15), ipady=5)
        
        self.dist_list_frame = ttk.Frame(dist_frame, style="Card.TFrame")
        self.dist_list_frame.pack(fill="both", expand=True, padx=10)
        
        self.dist_rows = []
        
        # Header for list
        hdr = ttk.Frame(self.dist_list_frame, style="Card.TFrame")
        hdr.pack(fill="x")
        ttk.Label(hdr, text="Time (s)", style="Card.TLabel", width=15).pack(side="left", padx=5)
        ttk.Label(hdr, text="Impulse (rad/s)", style="Card.TLabel", width=15).pack(side="left", padx=5)
        
        # Buttons
        btn_frame = ttk.Frame(dist_frame, style="Card.TFrame")
        btn_frame.pack(fill="x", padx=10, pady=5)
        ttk.Button(btn_frame, text="+ ADD KICK", style="Create.TButton", command=self.add_disturbance_row).pack(side="left")
        
        # Add default rows
        self.add_disturbance_row(time="5.0", impulse="0.3")
        self.add_disturbance_row(time="10.0", impulse="-0.3")
        
        # --- Action Buttons ---
        action_frame = ttk.Frame(main_container)
        action_frame.pack(fill="x", pady=10)
        
        self.inspect_btn = ttk.Button(action_frame, text="VIEW MODEL (3D)", command=self.show_model)
        self.inspect_btn.pack(side="left", fill="x", expand=True, padx=5, ipady=10)
        
        self.run_btn = ttk.Button(action_frame, text="RUN SIMULATION", command=self.run_sim)
        self.run_btn.pack(side="left", fill="x", expand=True, padx=5, ipady=10)
        
        self.roa_btn = ttk.Button(action_frame, text="GENERATE ROA", command=self.generate_roa)
        self.roa_btn.pack(side="right", fill="x", expand=True, padx=5, ipady=10)
        
    def update_ctrl_inputs(self, event=None):
        # Clear existing
        for widget in self.ctrl_frame.winfo_children():
            widget.destroy()
        
        self.param_vars = {}
        strat = self.strat_var.get()
        
        # Helper to add entry row
        def add_entry(row, col, label, key, default):
            ttk.Label(self.ctrl_frame, text=label, style="Card.TLabel").grid(row=row, column=col, padx=10, pady=5, sticky="w")
            e = ttk.Entry(self.ctrl_frame, width=10)
            e.insert(0, default)
            e.grid(row=row, column=col+1, padx=5, pady=5)
            self.param_vars[key] = e
        
        if strat in ["LQR", "MPC"]:
            # Q Diagonal
            ttk.Label(self.ctrl_frame, text="Q (State Weights)", style="Card.TLabel").grid(row=0, column=0, columnspan=4, sticky="w", padx=10)
            q_lbls = ["x", "x_dot", "theta", "theta_dot"]
            q_defs = ["980.93", "1.10", "1.69", "1.00"]
            for i, (l, d) in enumerate(zip(q_lbls, q_defs)):
                add_entry(1, i*2, f"Q_{l}", f"q{i}", d)
                
            # R
            add_entry(2, 0, "R (Control Cost)", "r_val", "0.0056")
            
            if strat == "MPC":
                add_entry(2, 2, "Horizon (Steps)", "horizon", "20")
                
        elif strat == "PolePlacement":
            ttk.Label(self.ctrl_frame, text="Desired Closed-Loop Poles", style="Card.TLabel").grid(row=0, column=0, columnspan=4, sticky="w", padx=10)
            # Default poles from file
            defs = ["-43.9360", "-0.7630", "-2.0485", "-48.7724"]
            for i, d in enumerate(defs):
                r = 1 + i // 2
                c = (i % 2) * 2
                add_entry(r, c, f"Pole {i+1}", f"p{i}", d)
                
        elif strat == "SMC":
            ttk.Label(self.ctrl_frame, text="Sliding Surface (Lambda) & Gains", style="Card.TLabel").grid(row=0, column=0, columnspan=4, sticky="w", padx=10)
            # lambda1, lambda2, lambda3 (x, th, thd)
            add_entry(1, 0, "Lambda_x", "lam0", "2.398")
            add_entry(1, 2, "Lambda_theta", "lam2", "5.388")
            add_entry(2, 0, "Lambda_th_dot", "lam3", "0.872")
            
            add_entry(2, 2, "Gain (K)", "K_smc", "18.62")
            add_entry(3, 0, "Phi (Boundary)", "phi", "0.1")
            

    def toggle_inputs(self):
        # Disable/Enable current entries in ctrl_frame
        state = "disabled" if self.mode_var.get() == "Auto" else "normal"
        for child in self.ctrl_frame.winfo_children():
            if isinstance(child, ttk.Entry):
                child.configure(state=state)

    def add_disturbance_row(self, time="", impulse=""):
        row_frame = ttk.Frame(self.dist_list_frame, style="Card.TFrame")
        row_frame.pack(fill="x", pady=2)
        
        e_time = ttk.Entry(row_frame, width=15)
        e_time.insert(0, time)
        e_time.pack(side="left", padx=5)
        
        e_imp = ttk.Entry(row_frame, width=15)
        e_imp.insert(0, impulse)
        e_imp.pack(side="left", padx=5)
        
        def delete_row():
            row_frame.destroy()
            self.dist_rows.remove((e_time, e_imp))
            
        del_btn = ttk.Button(row_frame, text="×", width=3, style="Create.TButton", command=delete_row)
        del_btn.pack(side="left", padx=5)
        
        self.dist_rows.append((e_time, e_imp))
    
    def get_common_params(self):
        dx = float(self.dx_entry.get())
        dtheta = float(self.dtheta_entry.get())
        m_base_val = float(self.mbase_entry.get())
        mw_val = float(self.mw_entry.get())
        mp_val = float(self.mp_entry.get())
        l_val = float(self.l_entry.get())
        r_phys_val = float(self.r_entry_phys.get())
        ip_val = float(self.ip_entry.get())
        return dx, dtheta, m_base_val, mw_val, mp_val, l_val, r_phys_val, ip_val

    def get_strat_params(self):
        strat = self.strat_var.get()
        params = {}
        
        if strat in ["LQR", "MPC"]:
            q_vals = [float(self.param_vars[f"q{i}"].get()) for i in range(4)]
            r_val = float(self.param_vars["r_val"].get())
            params["Q_diag"] = q_vals
            params["R_val"] = r_val
            if strat == "MPC":
                try: params["horizon"] = int(self.param_vars["horizon"].get())
                except: params["horizon"] = 20
        elif strat == "PolePlacement":
            poles = [float(self.param_vars[f"p{i}"].get()) for i in range(4)]
            params["poles"] = poles
        elif strat == "SMC":
            l0 = float(self.param_vars["lam0"].get())
            l2 = float(self.param_vars["lam2"].get())
            l3 = float(self.param_vars["lam3"].get())
            # lambda vector: [l0, 1.0, l2, l3]
            params["lambda"] = [l0, 1.0, l2, l3]
            params["K_smc"] = float(self.param_vars["K_smc"].get())
            params["phi"] = float(self.param_vars["phi"].get())
            
        return strat, params

    def show_model(self):
        try:
            dx, dtheta, m_base_val, mw_val, mp_val, l_val, r_phys_val, ip_val = self.get_common_params()
            strat, sp = self.get_strat_params()
            try: init_pos = float(self.pos_entry.get())
            except: init_pos = 0.0
            
            self.destroy() 
            segway_control_lqr.run_segway_simulation(
                m_base=m_base_val, mw=mw_val, mp=mp_val, l=l_val, r=r_phys_val, Ip=ip_val,
                damping_x=dx, damping_theta=dtheta,
                strat_name=strat, strat_params=sp, # Pass strategy even to viewer
                initial_tilt=0.0, initial_pos=init_pos, disturbances=[], duration=3600.0, show_plots=False
            )
        except ValueError as e:
            messagebox.showerror("Input Error", f"Check inputs: {e}")

    def generate_roa(self):
        try:
            dx, dtheta, m_base_val, mw_val, mp_val, l_val, r_phys_val, ip_val = self.get_common_params()
            strat, sp = self.get_strat_params()
            
            messagebox.showinfo("Generating ROA", f"Generating ROA for {strat}.\nThis takes 30-60s.")
            self.update()
            
            ROA.generate_roa_plot(
                m_base=m_base_val, mw=mw_val, mp=mp_val, l=l_val, r=r_phys_val, Ip=ip_val,
                damping_x=dx, damping_theta=dtheta,
                strat_name=strat, strat_params=sp
            )
        except ValueError as e:
            messagebox.showerror("Input Error", f"Check inputs: {e}")

    def run_sim(self):
        try:
            # Check Mode
            if self.mode_var.get() == "Auto":
                strat = self.strat_var.get()
                
                if strat not in ["LQR", "MPC", "PolePlacement", "SMC"]:
                     messagebox.showwarning("Auto-Tune", f"Auto-Tuning not fully implemented for {strat} yet.")
                else:
                    messagebox.showinfo("Optimization", f"Running Genetic Algorithm for {strat} parameters...")
                    self.update()
                    
                    # RUN OPTIMIZATION
                    best_params = optim_segway.optimize_parameters(strat)
                    
                    if not best_params:
                         messagebox.showerror("Optimization Failed", "GA could not find stable parameters or failed.")
                         return
                    
                    # POPULATE GUI
                    if strat in ["LQR", "MPC"]:
                        for i in range(4):
                            self.param_vars[f"q{i}"].delete(0, tk.END)
                            self.param_vars[f"q{i}"].insert(0, f"{best_params[i]:.2f}")
                        self.param_vars["r_val"].delete(0, tk.END)
                        self.param_vars["r_val"].insert(0, f"{best_params[4]:.4f}")
                        
                    elif strat == "PolePlacement":
                        for i in range(4):
                            self.param_vars[f"p{i}"].delete(0, tk.END)
                            self.param_vars[f"p{i}"].insert(0, f"{best_params[i]:.4f}")
                            
                    elif strat == "SMC":
                        # [lam0, lam2, lam3, K, phi]
                        self.param_vars["lam0"].delete(0, tk.END)
                        self.param_vars["lam0"].insert(0, f"{best_params[0]:.3f}")
                        self.param_vars["lam2"].delete(0, tk.END)
                        self.param_vars["lam2"].insert(0, f"{best_params[1]:.3f}")
                        self.param_vars["lam3"].delete(0, tk.END)
                        self.param_vars["lam3"].insert(0, f"{best_params[2]:.3f}")
                        self.param_vars["K_smc"].delete(0, tk.END)
                        self.param_vars["K_smc"].insert(0, f"{best_params[3]:.3f}")
                        self.param_vars["phi"].delete(0, tk.END)
                        self.param_vars["phi"].insert(0, f"{best_params[4]:.3f}")
                        
                    messagebox.showinfo("Optimization Complete", f"Found optimized values for {strat}!")
            
            # Parse params (manual or just-filled)
            dx, dtheta, m_base_val, mw_val, mp_val, l_val, r_phys_val, ip_val = self.get_common_params()
            strat, sp = self.get_strat_params()
            
            init_tilt = float(self.tilt_entry.get())
            init_pos = float(self.pos_entry.get())
            sim_duration = float(self.duration_entry.get())
            
            # Parse Disturbances
            disturbances = []
            for e_t, e_f in self.dist_rows:
                t_str = e_t.get()
                f_str = e_f.get()
                if t_str and f_str:
                    disturbances.append({'time': float(t_str), 'impulse': float(f_str)})
            
            # Launch
            self.destroy() 
            segway_control_lqr.run_segway_simulation(
                m_base=m_base_val, mw=mw_val, mp=mp_val, l=l_val, r=r_phys_val, Ip=ip_val,
                damping_x=dx, damping_theta=dtheta,
                strat_name=strat, strat_params=sp,
                initial_tilt=init_tilt, initial_pos=init_pos, disturbances=disturbances, duration=sim_duration
            )
            
        except ValueError as e:
            messagebox.showerror("Input Error", f"Please check your numbers.\n{e}")

if __name__ == "__main__":
    app = SegwayLauncher()
    app.mainloop()
