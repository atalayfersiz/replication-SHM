import numpy as np
import pandas as pd
import pickle
import json
import os

# Set parameters
L = 17.9 # m
x_sensors = np.array([2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0]) # sorted ascending
h = 0.164 # m
Ec = 38.0e9 # Pa
b = 2.6 # m
d = 0.28 # m
I = (b * (d**3)) / 12.0 # m^4
EcI = Ec * I / 1e3 # kNm^2

# Load models and data
print("Loading trained ML baseline models...")
with open("backend/all_models.pkl", "rb") as f:
    save_dict = pickle.load(f)
    models = save_dict['models']
    scaler_X = save_dict['scaler_X']
    scaler_Y = save_dict['scaler_Y']

print("Loading test data...")
test_df = pd.read_csv("backend/load_test_data.csv")

# Input features for test baseline prediction
X_cols = ['TMP_02m', 'TMP_07m', 'TMP_12m', 'TMP_17m']
X_test = test_df[X_cols].values
X_test_scaled = scaler_X.transform(X_test)

# Get raw strains
strain_cols = [col for col in test_df.columns if col.startswith("STR_")]
Y_raw = test_df[strain_cols].values

# Create lists to store computed values for JSON output
timestamps = test_df['Timestamp'].values.tolist()
load_positions = (2.0 + (15.0 / 960.0) * np.arange(960)).tolist()

# Paper values for comparison at critical load step t=13 (load at x=14m)
M_paper_FE_A = [0.80, 0.53, 0.26, 0.25, 0.25, -0.54, -0.94, -4.89, 0.42, 4.05, 7.78, 7.78, 6.71, 1.79, -3.81, -9.27]
M_paper_FE_B = [0.80, 0.53, 0.26, 0.25, 0.25, -0.54, -0.94, -4.89, 0.42, 4.05, 7.78, 7.78, 6.71, 1.79, -3.81, -9.27]
M_paper_DT_A = [0.06, 0.19, 0.03, -0.71, -1.52, -2.86, -5.02, -3.96, -0.94, 2.24, 5.11, 6.69, 7.56, 3.94, 0.82, -1.69]
M_paper_DT_B = [0.17, 0.32, 0.28, -0.52, -1.52, -2.51, -2.93, -3.83, -0.89, 1.95, 5.02, 7.03, 6.47, 2.95, 0.75, -1.81]

D_paper_FE_A = [0.01, 0.03, 0.03, 0.03, 0.02, 0.01, -0.03, -0.08, -0.14, -0.20, -0.22, -0.21, -0.20, -0.15, -0.08, -0.02]
D_paper_FE_B = [0.04, 0.06, 0.06, 0.06, 0.05, 0.04, 0.00, -0.05, -0.12, -0.18, -0.21, -0.19, -0.18, -0.14, -0.07, -0.02]
D_paper_DT_A = [0.04, 0.06, 0.07, 0.08, 0.08, 0.07, 0.05, 0.03, 0.00, -0.05, -0.09, -0.01, -0.11, -0.09, -0.07, -0.03]
D_paper_DT_B = [0.03, 0.04, 0.06, 0.06, 0.06, 0.05, 0.04, 0.02, 0, -0.04, -0.07, -0.09, -0.08, -0.07, -0.05, -0.03]

out_data = {
    'timestamps': timestamps,
    'load_positions': load_positions,
    'x_sensors': x_sensors.tolist(),
    'temperatures': {
        'TMP_02m': test_df['TMP_02m'].values.tolist(),
        'TMP_07m': test_df['TMP_07m'].values.tolist(),
        'TMP_12m': test_df['TMP_12m'].values.tolist(),
        'TMP_17m': test_df['TMP_17m'].values.tolist()
    },
    'models': {
        'ANN': {'Array_A': [], 'Array_B': [], 'Array_A_Half': [], 'Array_B_Half': [], 'Array_A_Sparse': [], 'Array_B_Sparse': []},
        'Linear': {'Array_A': [], 'Array_B': [], 'Array_A_Half': [], 'Array_B_Half': [], 'Array_A_Sparse': [], 'Array_B_Sparse': []},
        'Ridge': {'Array_A': [], 'Array_B': [], 'Array_A_Half': [], 'Array_B_Half': [], 'Array_A_Sparse': [], 'Array_B_Sparse': []},
        'Lasso': {'Array_A': [], 'Array_B': [], 'Array_A_Half': [], 'Array_B_Half': [], 'Array_A_Sparse': [], 'Array_B_Sparse': []}
    },
    'Paper_FE_A': {
        'bending_moments': M_paper_FE_A,
        'deflections': D_paper_FE_A
    },
    'Paper_FE_B': {
        'bending_moments': M_paper_FE_B,
        'deflections': D_paper_FE_B
    },
    'Paper_DT_A': {
        'bending_moments': M_paper_DT_A,
        'deflections': D_paper_DT_A
    },
    'Paper_DT_B': {
        'bending_moments': M_paper_DT_B,
        'deflections': D_paper_DT_B
    }
}

# Helper to integrate curvature twice to get deflection (with continuous beam boundary conditions)
def double_integrate_curvature(poly_coeffs, L):
    # Midspan pier support location
    xp = 8.95
    p_curv = np.poly1d(poly_coeffs)
    p_int1 = np.polyint(p_curv, 1)
    p_int2 = np.polyint(p_int1, 1)
    
    # Span 1: x in [0, xp]
    # w1(0) = 0 => C2 = 0
    # w1(xp) = 0 => C1 = -w_raw(xp) / xp
    w_raw_xp = p_int2(xp)
    C1 = -w_raw_xp / xp
    
    # Span 2: x in [xp, L]
    # w2(xp) = 0
    # w2(L) = 0
    w_raw_L = p_int2(L)
    D1 = (w_raw_xp - w_raw_L) / (L - xp)
    D2 = -w_raw_L - D1 * L
    
    # Return a function to evaluate deflection (converted to mm)
    def deflection(x):
        if isinstance(x, np.ndarray):
            res = np.zeros_like(x)
            mask1 = (x <= xp)
            mask2 = ~mask1
            res[mask1] = p_int2(x[mask1]) + C1 * x[mask1]
            res[mask2] = p_int2(x[mask2]) + D1 * x[mask2] + D2
            return res * 1.0e3
        else:
            if x <= xp:
                val = p_int2(x) + C1 * x
            else:
                val = p_int2(x) + D1 * x + D2
            return val * 1.0e3
        
    return deflection

# Helper to calculate stiffness retention percentage using Branson's Equation
def calculate_stiffness_retention(moments):
    M_cr = 5.0 # kNm cracking threshold
    I_g = 1.0 # Gross section
    I_cr = 0.35 # Cracked section (35% of Ig)
    ratios = []
    for M in moments:
        abs_M = abs(M)
        if abs_M > M_cr:
            I_eff = ((M_cr / abs_M) ** 3) * I_g + (1.0 - (M_cr / abs_M) ** 3) * I_cr
            I_eff = max(min(I_eff, 1.0), 0.35)
            ratios.append(I_eff * 100.0)
        else:
            ratios.append(100.0)
    return ratios

# Run structural analysis for each machine learning model
for model_name, model in models.items():
    print(f"Processing model: {model_name}...")
    if model_name == 'ANN':
        # Predict on scaled features
        Y_baseline_scaled = model.predict(X_test_scaled)
        Y_baseline_pred = scaler_Y.inverse_transform(Y_baseline_scaled)
    else:
        # Linear models predict on original features
        Y_baseline_pred = model.predict(X_test)
        
    Y_delta = Y_raw - Y_baseline_pred
    
    for t in range(960):
        # Extract strains
        strain_A_top = []
        strain_A_bot = []
        raw_A_top = []
        raw_A_bot = []
        base_A_top = []
        base_A_bot = []
        
        strain_B_top = []
        strain_B_bot = []
        raw_B_top = []
        raw_B_bot = []
        base_B_top = []
        base_B_bot = []
        
        for x in x_sensors:
            idx_A_top = strain_cols.index(f"STR_A_Top_{int(x):02d}m")
            idx_A_bot = strain_cols.index(f"STR_A_Bot_{int(x):02d}m")
            idx_B_top = strain_cols.index(f"STR_B_Top_{int(x):02d}m")
            idx_B_bot = strain_cols.index(f"STR_B_Bot_{int(x):02d}m")
            
            strain_A_top.append(Y_delta[t, idx_A_top])
            strain_A_bot.append(Y_delta[t, idx_A_bot])
            raw_A_top.append(Y_raw[t, idx_A_top])
            raw_A_bot.append(Y_raw[t, idx_A_bot])
            base_A_top.append(Y_baseline_pred[t, idx_A_top])
            base_A_bot.append(Y_baseline_pred[t, idx_A_bot])
            
            strain_B_top.append(Y_delta[t, idx_B_top])
            strain_B_bot.append(Y_delta[t, idx_B_bot])
            raw_B_top.append(Y_raw[t, idx_B_top])
            raw_B_bot.append(Y_raw[t, idx_B_bot])
            base_B_top.append(Y_baseline_pred[t, idx_B_top])
            base_B_bot.append(Y_baseline_pred[t, idx_B_bot])
            
        strain_A_top = np.array(strain_A_top)
        strain_A_bot = np.array(strain_A_bot)
        strain_B_top = np.array(strain_B_top)
        strain_B_bot = np.array(strain_B_bot)
        
        # Curvature κ = (ε_bot - ε_top) / h (microstrain to strain)
        kappa_A = (strain_A_bot - strain_A_top) * 1.0e-6 / h
        kappa_B = (strain_B_bot - strain_B_top) * 1.0e-6 / h
        
        # Configurations: (suffix, x_coords_fit, kappa_A_fit, kappa_B_fit, poly_degree)
        configs = [
            ('', x_sensors, kappa_A, kappa_B, 4),
            ('_Half', x_sensors[::2], kappa_A[::2], kappa_B[::2], 4),
            ('_Sparse', x_sensors[::4], kappa_A[::4], kappa_B[::4], 2)
        ]
        
        for suffix, x_fit, kap_A_fit, kap_B_fit, deg in configs:
            poly_A = np.polyfit(x_fit, kap_A_fit, deg)
            poly_B = np.polyfit(x_fit, kap_B_fit, deg)
            
            # Evaluate bending moment at all 16 sensor locations
            M_A = (EcI * np.polyval(poly_A, x_sensors)).tolist()
            M_B = (EcI * np.polyval(poly_B, x_sensors)).tolist()
            
            # Double integrate for deflections
            w_A_func = double_integrate_curvature(poly_A, L)
            w_B_func = double_integrate_curvature(poly_B, L)
            
            # Evaluate deflection at all 16 sensor locations
            w_A = [w_A_func(x) for x in x_sensors]
            w_B = [w_B_func(x) for x in x_sensors]
            # Calculate stiffness retention ratios using Branson's Equation
            stiffness_ratios_A = calculate_stiffness_retention(M_A)
            stiffness_ratios_B = calculate_stiffness_retention(M_B)
            
            # Append outputs
            out_data['models'][model_name]['Array_A' + suffix].append({
                'bending_moments': M_A,
                'deflections': w_A,
                'strains_top': strain_A_top.tolist(),
                'strains_bot': strain_A_bot.tolist(),
                'raw_top': raw_A_top,
                'raw_bot': raw_A_bot,
                'base_top': base_A_top,
                'base_bot': base_A_bot,
                'stiffness_ratios': stiffness_ratios_A
            })
            out_data['models'][model_name]['Array_B' + suffix].append({
                'bending_moments': M_B,
                'deflections': w_B,
                'strains_top': strain_B_top.tolist(),
                'strains_bot': strain_B_bot.tolist(),
                'raw_top': raw_B_top,
                'raw_bot': raw_B_bot,
                'base_top': base_B_top,
                'base_bot': base_B_bot,
                'stiffness_ratios': stiffness_ratios_B
            })

# Export to JSON for frontend
with open("backend/frontend_data.json", "w") as f:
    json.dump(out_data, f)
print("Saved frontend_data.json")

# Verification table at t=768s (load at x=14m, corresponding to t=13 min)
idx_13 = 768
calc_M_A = out_data['models']['ANN']['Array_A'][idx_13]['bending_moments']
calc_D_A = out_data['models']['ANN']['Array_A'][idx_13]['deflections']

print("\n" + "="*80)
print("VERIFICATION AT CRITICAL LOAD STEP (t=13 min, load at 14m, ANN MODEL)")
print("="*80)
print(f"{'Sensor':<10} | {'Paper M (kNm)':<15} | {'Calc M (kNm)':<15} | {'Paper D (mm)':<15} | {'Calc D (mm)':<15}")
print("-"*80)
for i, x in enumerate(x_sensors):
    print(f"Str_{int(x):02d}m    | {M_paper_DT_A[i]:<15.2f} | {calc_M_A[i]:<15.2f} | {D_paper_DT_A[i]:<15.2f} | {calc_D_A[i]:<15.2f}")
print("="*80)

# Calculate MAE error between paper DT and computed DT values
mae_M = np.mean(np.abs(np.array(calc_M_A) - np.array(M_paper_DT_A)))
mae_D = np.mean(np.abs(np.array(calc_D_A) - np.array(D_paper_DT_A)))
print(f"Bending Moment Mean Absolute Difference: {mae_M:.4f} kNm")
print(f"Deflection Mean Absolute Difference: {mae_D:.4f} mm")
