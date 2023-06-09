import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy.linalg import block_diag
import openlab

from hareland_model import *
from heave_prediction import *
from qp_formulation import *

cvxopt.solvers.options['show_progress'] = False

def setup():
    # Read in drilling data
    df = pd.read_csv("drillingdata_1Hz.csv")
    t = df["t"]
    ROP_input = df["ROP"]
    W_input = df["WOB"]
    Nr_input = df["RPM"]
    Flow_input = df["Flow"]
    MD_input = df["MD"]
    Bpos_input = df["BlockPos"]

    # Convert input data from metric to imperial units for algorithm
    Nr = Nr_input # Rotational speed [rpm]
    Q = Flow_input * 1.67e-5 # Flow rate [m3/s]
    W = W_input * 2.205 # Weight on bit [klb]
    y = ROP_input * 3.2808399 # Rate of Penetration [ft/hr]
    h = MD_input * 3.2808399 # Measured Depth [ft]

    # Specify Simulation Properties
    nsim = 800 # Length of simulation 
    L = 60 # Length of sliding window phi
    sample_time = 1 # Sample time of MPC
    dt = sample_time / 3600 # Delta t for linearized state space model

    # Hareland Model Coefficients
    a = 1493.08128 # Initial guess lithology coefficient a (Range: [0.001, 10e4])
    b = 0.99 # Initial guess WOB coefficient b (Range: [0.5, 1])
    c = 0.6 # Initial guess RPM coefficient b (Range: [0.5, 1.5])
    Db = 12.25 # Bit diameter [in]
    Nc = 50 # Number of cutters [-]
    dc = 0.63 # Diameter of cutters [in]
    alpha = np.deg2rad(30) # Cutter backrake angle [rad]
    theta = np.deg2rad(10) # Cutter siderake angle [rad]
    Wf = 1 # Bit wear factor [-]
    sigma_c = 80 # UCS [psi]
    eta_corr = 0.001 # Heave Correction Factor [-]

    # MPC Design Parameters
    P_h = 12 # Prediction Horizon
    n_x = 2 # Number of states
    n_u = 2 # Number of inputs
    n_y = 1 # Number of outputs

    ROP_ub = 60 * 3.2808399 # Maximum ROP [ft/hr]
    uk_lb = np.zeros((P_h*n_u,1)) # Input lower bounds
    uk_ub = np.zeros((P_h*n_u,1)) # Input upper bounds
    uk_lb[::2] = 24 # Upper bound WOB [klbs]  
    uk_lb[1::2] = 100 # Lower bound RPM [rpm]
    uk_ub[::2] = 1 # Lower bound WOB [klbs]
    uk_ub[1::2] = 190 # Upper bound RPM [rpm]
    duk_lb = np.zeros((P_h*n_u,1)) # Upper bounds to rate of change in inputs
    duk_ub = np.zeros((P_h*n_u,1)) # Upper bounds to rate of change in inputs
    duk_lb[::2] = -3 # Upper bound rate of change WOB [klbs]
    duk_lb[1::2] = -3 # Lower bound rate of change RPM [rpm]
    duk_ub[::2] = 3 # Upper bound rate of change WOB [klbs]
    duk_ub[1::2] = 3 # Upper bound rate of change RPM [rpm]
    r = np.zeros((P_h*n_u,1)) # Input reference vector
    r[1::2] = 140 # RPM reference [rpm]
    
    Q = np.eye(P_h*n_y)*1e3 # Weight matrix
    r = np.array(
        [[5, 0],
         [0, 5]])
    R = block_diag(*([r] * P_h)) # Weight matrix
    W = np.eye(P_h)*1e2 # Weight matrix quadratic slack
    w = np.ones((P_h,1)) # Weight matrix linear slack

    rop_sp = 30 # ROP setpoint [ft/hr]
    y_r = np.ones(((nsim+100),1))*rop_sp*3.2808399 # Reference ROP [ft/hr]

    # Memory arrays
    lin_rop = np.array([])  # ROP estimate based on linearized Hareland Model
    lin_md = np.array([]) # MD estimate based on linearized Hareland Model
    u_prev = np.array([[W[0]], [Nr[0]]]) # Previous set of WOB and RPM

    U_opt_memory = [] # Optimum set of inputs
    W_opt = [] # Optimum set of WOB
    Nr_opt = [] # Optimum set of RPM

def run_mpc(i, L, y, Nr, W, Flow, h, dt):
    y_win, Nr_win, W_win, Q_win = partition_drillingdata(i, L, y, Nr, W, Flow)
    eta_win = partition_responsedata(i, Np, waveres)
    a = 1493.08128 # Initial guess a
    b = 0.6 # Initial guess b
    c = 0.6 # Initial guess c
    if i > 40:
        a, b, c = drilling_scipy_opt(y_win+0.01, Nr_win+0.01, W_win+0.01, a, b, c)
    else:
        return 4, 100
    if Nr[i] == 0 or W[i] == 0:
        ROP_est = 0
    else:
        ROP_est = hareland(Nr[i], W[i], a, b, c)

    
    # Linearization
    WOB = W[i]
    RPM = Nr[i]
    if ROP_est == 0:
        At, Bt, Ct, Dt = get_linearized_model(dt, 0.01, a, b, c, 0.01, 0.01)
    else:
        At, Bt, Ct, Dt = get_linearized_model(dt, ROP_est, a, b, c, WOB, RPM)
    
    # Simulate system
    x = np.array([[h[i]], [y[i] - ROP_est]])
    u = np.array([[W[i]+0.0001], [Nr[i]+0.0001]])
 
    x_prev = x
    u_prev = u
        
    f0 = np.array([[x_prev[0][0] + dt*(ROP_est + x_prev[1][0])], [x_prev[1][0]]])
    g0 = ROP_est + x_prev[1][0]

    F = f0 - np.dot(At, x_prev) - np.dot(Bt, u_prev)
    G = g0 - np.dot(Ct, x_prev) - np.dot(Dt, u_prev)
    
    x_k1 = f0 + np.dot(At, x-x_prev) + np.dot(Ct, u-u_prev)
    y_k = g0 + np.dot(Ct, x-x_prev) + np.dot(Dt, u-u_prev)
    
    h_k1 = x_k1[0]
    x0 = x
    U_1 = np.tile(u_prev,[Np, 1])

    # Solving the open loop optimization problem
    U_opt = get_opt_drillparams(i, P_h, At, Bt, Ct, Dt, r_y, r_u, gamma, phi, Q, R, x0, u0, ymax, eta, fac, uk_lb, uk_ub, duk_ub, duk_lb, w, W)
    
    W_opt.append(U_opt[0])
    Nr_opt.append(U_opt[1])
    U_opt_memory.append(U_opt[:20])
    
    u_prev = u
    #print(U_opt[0].shape)
    x_prev = x
    if U_opt[0][0] < 0:
        return 0.01, U_opt[1][0]
    return U_opt[0][0], U_opt[1][0]

def run_openlab(username, apikey, licenseguid, config_name, sim_name, initial_bit_depth, step_duration, simulation_time):
    session = openlab.http_client(username=username, apikey=apikey, licenseguid=licenseguid)
    sim = session.create_simulation(config_name, sim_name, initial_bit_depth, StepDuration=step_duration)
    sim.setpoints.RPM = 0 #Hz
    sim.setpoints.DesiredROP = 0 #m/s
    sim.setpoints.WOBAutoDriller = True
    sim.setpoints.TopOfStringVelocity = 0.1 #m/s
    sim.setpoints.FlowRateIn = 0.045 #60000 is to convert l/min to m^3/s
    sim.setpoints.WOBProportionalGain = 19e-8
    sim.setpoints.WOBIntegralGain = 6e-8

    h_memory = np.array([])
    ROP_memory = np.array([])
    WOB_memory = np.array([])
    RPM_memory = np.array([])
    WOBOpt_memory = np.array([])
    RPMOpt_memory = np.array([])
    starti = 1
    for i in range(starti, starti+simulation_time):
        if i == starti:
            # Circulation
            sim.setpoints.DesiredWOB = 0 
            sim.setpoints.SurfaceRPM = 0 
            sim.setpoints.TopOfStringVelocity = 0.1
            sim.setpoints.FlowRateIn = 0.045
        else:
            WOB_opt, RPM_opt = run_mpc(i-2, L, ROP_memory, RPM_memory, WOB_memory, Flow_input, h_memory)
            WOBOpt_memory = np.append(WOBOpt_memory, (WOB_opt / 2.205) * 1000)
            RPMOpt_memory = np.append(RPMOpt_memory, RPM_opt / 60)
            sim.setpoints.DesiredWOB = (WOB_opt / 2.205) * 1000 +0.01 #5000
            sim.setpoints.SurfaceRPM = RPM_opt / 60 #3
            
        sim.step(i)
        
        insth = sim.get_results(i,["TD"])['TD'][i] * 3.2808399 # m to ft
        instRop = sim.get_results(i,["InstantaneousROP"])['InstantaneousROP'][i] * 11811 #* 3.2808399  # m/s to ft/hr 
        instRPM = sim.get_results(i,["SurfaceRPM"])['SurfaceRPM'][i] * 60 # rps to rpm
        instWOB = sim.get_results(i,["WOB"])['WOB'][i] * 0.001 * 2.205 # kilos to klbs
        
        #print(instRop, instRPM, instWOB)
        h_memory = np.append(h_memory, insth)
        WOB_memory = np.append(WOB_memory, instWOB)
        RPM_memory = np.append(RPM_memory, instRPM)
        ROP_memory = np.append(ROP_memory, instRop)