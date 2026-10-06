import numpy as np
import pandas as pd
import os

# Create backend directory if not exists
os.makedirs("backend", exist_ok=True)

# Set random seed for reproducibility
np.random.seed(42)

# Physical parameters of the bridge
L = 17.9 # m
x_sensors = np.array([2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0]) # 16 sensors
x_temps = np.array([17.0, 12.0, 7.0, 2.0]) # 4 temperature sensors
h = 0.164 # m (distance between top and bottom strain sensors)
Ec = 38.0e9 # Pa (38 GPa)
b = 2.6 # m (width of deck)
d = 0.28 # m (depth of deck)
I = (b * (d**3)) / 12.0 # m^4 (moment of inertia)
EcI = Ec * I / 1e3 # kNm^2

# Paper bending moments for Array A and Array B (timestamps t=1 to t=16, 16 sensors from x=2m up to 17m)
# Rows: t=1 to t=16. Columns: Str_02m up to Str_17m.
M_A_grid = np.array([
    [0.90, 0.25, -0.20, -0.18, 0.05, 0.81, 1.94, 1.12, -0.08, -0.53, -1.53, -1.66, -1.42, -0.85, -0.30, 0.10], # t=1
    [1.57, 0.90, 0.23, -0.36, -0.74, 0.14, 1.85, 1.02, -0.27, -1.29, -1.55, -1.93, -1.96, -1.40, -0.46, -0.50], # t=2
    [0.89, 1.30, 1.64, 0.33, 0.01, 0.34, 1.39, 0.70, -0.17, -1.01, -1.51, -1.69, -1.62, -0.54, -0.32, 0.70], # t=3
    [-0.29, 0.45, 2.14, 1.85, 0.98, 0.49, 0.84, 0.47, -0.34, -1.28, -1.75, -1.91, -1.66, -1.20, -0.37, 0.52], # t=4
    [-0.41, -0.56, 1.24, 2.26, 2.51, 1.35, 0.59, 0.22, -0.51, -1.28, -1.67, -1.59, -1.89, -1.13, -0.72, 0.43], # t=5
    [-0.64, -1.03, -0.06, 1.27, 2.66, 1.99, 0.42, 0.35, -0.97, -1.40, -1.99, -2.31, -1.85, -1.05, -1.01, -0.07], # t=6
    [-0.48, -0.53, -0.67, 0.38, 1.11, 2.11, -0.25, 0.84, -0.49, -1.22, -1.82, -1.16, -2.01, -1.30, -1.20, -0.24], # t=7
    [0.05, -0.15, -0.35, -0.21, 0.35, 0.98, -0.46, 1.35, 0.79, -0.31, -1.02, -1.29, -1.53, -1.35, -0.81, -0.25], # t=8
    [0.10, -0.07, -0.13, -0.54, -0.57, -1.22, -2.05, 1.52, 2.75, 1.79, 0.76, 0.72, -0.72, -0.83, -1.16, -0.59], # t=9
    [0.32, 0.04, -0.16, -0.51, -1.48, -2.72, -4.49, -0.95, 3.09, 4.02, 3.79, 1.78, 0.49, -0.31, -1.01, -1.27], # t=10
    [0.30, 0.38, -0.05, -0.85, -1.75, -3.40, -5.91, -2.85, 2.26, 5.27, 5.96, 4.31, 2.74, 0.81, -0.87, -1.58], # t=11
    [0.06, 0.30, 0.23, -0.76, -1.65, -3.50, -5.70, -4.15, 0.38, 3.99, 6.78, 7.01, 5.17, 2.10, -0.35, -1.71], # t=12
    [0.06, 0.19, 0.03, -0.71, -1.52, -2.86, -5.02, -3.96, -0.94, 2.24, 5.11, 6.69, 7.56, 3.94, 0.82, -1.69], # t=13
    [0.14, 0.17, 0.11, -0.79, -1.27, -2.41, -3.97, -3.22, -1.30, 0.94, 3.04, 6.08, 7.78, 4.91, 1.96, -0.94], # t=14
    [0.16, 0.27, -0.10, -0.22, -0.69, -1.13, -1.73, -2.11, -0.83, -0.46, 0.90, 2.69, 4.44, 5.23, 4.37, 1.80], # t=15
    [-0.19, 0.37, 0.07, -0.36, -0.36, -0.28, -0.09, -0.63, -0.74, -0.16, -0.12, 0.45, 1.38, 2.36, 3.92, 1.42]  # t=16
])

M_B_grid = np.array([
    [0.67, -0.44, -0.16, -0.21, 0.12, 0.60, 1.36, 1.12, 0.02, -0.46, -1.12, -1.29, -1.28, -0.98, -0.65, 0.07], # t=1
    [1.16, 0.22, 0.18, -0.43, -0.38, 0.24, 0.98, 0.71, -0.64, -0.97, -1.64, -1.52, -1.60, -1.27, -0.90, 0.09], # t=2
    [0.62, 0.86, 0.97, 0.54, 0.22, 0.33, 0.86, 0.72, -0.43, -1.00, -1.40, -1.64, -1.45, -0.92, -0.52, 0.53], # t=3
    [-0.15, 0.20, 1.36, 1.70, 1.30, 0.63, 0.71, 0.45, -0.56, -1.17, -1.42, -1.70, -1.36, -0.72, -0.43, 0.88], # t=4
    [-0.38, -0.24, 0.35, 2.23, 2.43, 0.94, 0.21, 0.33, -0.48, -1.49, -1.69, -1.72, -1.39, -1.29, -0.60, 0.35], # t=5
    [-0.63, -0.68, -0.42, 1.36, 2.59, 2.12, -0.05, 0.09, -0.66, -1.45, -2.01, -1.95, -1.85, -0.97, -0.60, 0.13], # t=6
    [-0.35, -0.38, -0.33, 0.09, 1.42, 1.56, 0.00, 0.67, -0.30, -0.81, -1.67, -1.79, -1.44, -1.42, -0.99, -0.22], # t=7
    [0.18, -0.35, -0.12, -0.11, 0.50, 0.75, 0.17, 1.42, 0.57, -0.16, -0.90, -1.46, -1.42, -0.92, -0.94, 0.04], # t=8
    [-0.05, 0.31, -0.13, -0.51, -1.08, -1.37, -1.50, 1.04, 2.69, 1.82, 0.53, -0.44, -0.91, -0.81, -1.08, -0.51], # t=9
    [0.05, 0.43, 0.22, -0.40, -1.79, -2.18, -3.10, -0.86, 2.81, 3.73, 3.27, 2.12, 0.38, 0.16, -1.17, -1.10], # t=10
    [-0.11, 0.54, 0.31, -0.77, -1.84, -2.70, -3.53, -2.85, 1.85, 4.61, 5.30, 4.36, 2.43, 0.57, -1.04, -1.43], # t=11
    [0.36, 0.11, 0.32, -0.81, -2.06, -2.79, -3.79, -3.74, 0.51, 3.64, 6.30, 6.22, 4.80, 2.06, -0.10, -1.65], # t=12
    [-1.81, 0.75, 2.95, 6.47, 7.03, 5.02, 1.95, -0.89, -3.83, -2.93, -2.51, -1.52, -0.52, 0.28, 0.32, 0.17], # t=13
    [0.24, 0.28, 0.25, -0.55, -1.45, -2.05, -2.65, -3.37, -1.41, 0.69, 3.09, 5.75, 6.66, 4.35, 2.05, -1.27], # t=14
    [0.25, 0.28, -0.19, -0.33, -1.00, -0.74, -0.93, -1.96, -1.00, -0.21, 1.33, 2.12, 4.22, 4.44, 4.10, 0.46], # t=15
    [-0.19, 0.37, 0.07, -0.36, -0.36, -0.28, -0.09, -0.63, -0.74, -0.16, -0.12, 0.45, 1.38, 2.36, 3.92, 1.42]  # t=16
])

# Generate training data: 1000 samples
t_train = np.arange(1000)
T_base = 18.0 + 4.0 * np.sin(2 * np.pi * t_train / 500)
T_02 = T_base + 6.0 * np.sin(2 * np.pi * t_train / 48) + np.random.normal(0, 0.1, 1000)
T_07 = T_base + 6.0 * np.sin(2 * np.pi * t_train / 48 - 0.1) + np.random.normal(0, 0.1, 1000)
T_12 = T_base + 6.0 * np.sin(2 * np.pi * t_train / 48 - 0.2) + np.random.normal(0, 0.1, 1000)
T_17 = T_base + 6.0 * np.sin(2 * np.pi * t_train / 48 - 0.3) + np.random.normal(0, 0.1, 1000)

train_df = pd.DataFrame({
    'Timestamp': t_train,
    'TMP_02m': T_02,
    'TMP_07m': T_07,
    'TMP_12m': T_12,
    'TMP_17m': T_17
})

def get_local_temp(x, T2, T7, T12, T17):
    if x <= 7.0:
        frac = (x - 2.0) / 5.0
        return T2 * (1.0 - frac) + T7 * frac
    elif x <= 12.0:
        frac = (x - 7.0) / 5.0
        return T7 * (1.0 - frac) + T12 * frac
    else:
        frac = (x - 12.0) / 5.0
        return T12 * (1.0 - frac) + T17 * frac

sensor_configs = {}
for arr in ['A', 'B']:
    for depth in ['Top', 'Bot']:
        for x in x_sensors:
            col_name = f"STR_{arr}_{depth}_{int(x):02d}m"
            offset = np.random.uniform(145.0, 155.0)
            cte = 10.0 + np.random.normal(0.0, 0.1)
            nl = 0.05 + np.random.normal(0.0, 0.005)
            sensor_configs[col_name] = (x, offset, cte, nl)

for col_name, (x, offset, cte, nl) in sensor_configs.items():
    T_local = np.array([get_local_temp(x, t2, t7, t12, t17) for t2, t7, t12, t17 in zip(T_02, T_07, T_12, T_17)])
    # Use tuned non-linear baseline formula
    strain = offset + cte * T_local + 5.2 * np.sin(T_local / 1.6) + np.random.normal(0, 0.85, 1000)
    train_df[col_name] = strain

# Save training data
train_df.to_csv("backend/training_data.csv", index=False)
print("Saved training_data.csv with shape:", train_df.shape)

# Generate load test data: 960 samples
t_test = np.arange(960)
T_02_test = 22.0 + 0.0005 * t_test + np.random.normal(0, 0.05, 960)
T_07_test = 21.8 + 0.0005 * t_test + np.random.normal(0, 0.05, 960)
T_12_test = 21.5 + 0.0005 * t_test + np.random.normal(0, 0.05, 960)
T_17_test = 21.2 + 0.0005 * t_test + np.random.normal(0, 0.05, 960)

test_df = pd.DataFrame({
    'Timestamp': t_test,
    'TMP_02m': T_02_test,
    'TMP_07m': T_07_test,
    'TMP_12m': T_12_test,
    'TMP_17m': T_17_test
})

# Baseline strain for test data
for col_name, (x, offset, cte, nl) in sensor_configs.items():
    T_local = np.array([get_local_temp(x, t2, t7, t12, t17) for t2, t7, t12, t17 in zip(T_02_test, T_07_test, T_12_test, T_17_test)])
    test_df[col_name + "_base"] = offset + cte * T_local + 5.2 * np.sin(T_local / 1.6)

M_A_interp = np.zeros((960, 16))
M_B_interp = np.zeros((960, 16))

for t in range(960):
    state_frac = (t / 960.0) * 15.0
    state_low = int(np.floor(state_frac))
    state_high = int(np.ceil(state_frac))
    weight = state_frac - state_low
    M_A_interp[t] = M_A_grid[state_low] * (1.0 - weight) + M_A_grid[state_high] * weight
    M_B_interp[t] = M_B_grid[state_low] * (1.0 - weight) + M_B_grid[state_high] * weight

for i, x in enumerate(x_sensors):
    col_A_top = f"STR_A_Top_{int(x):02d}m"
    col_A_bot = f"STR_A_Bot_{int(x):02d}m"
    col_B_top = f"STR_B_Top_{int(x):02d}m"
    col_B_bot = f"STR_B_Bot_{int(x):02d}m"
    
    mech_strain_A_bot = (0.5 * h * M_A_interp[:, i] / EcI) * 1.0e6
    mech_strain_A_top = (-0.5 * h * M_A_interp[:, i] / EcI) * 1.0e6
    mech_strain_B_bot = (0.5 * h * M_B_interp[:, i] / EcI) * 1.0e6
    mech_strain_B_top = (-0.5 * h * M_B_interp[:, i] / EcI) * 1.0e6
    
    test_df[col_A_top] = test_df[col_A_top + "_base"] + mech_strain_A_top + np.random.normal(0, 0.2, 960)
    test_df[col_A_bot] = test_df[col_A_bot + "_base"] + mech_strain_A_bot + np.random.normal(0, 0.2, 960)
    test_df[col_B_top] = test_df[col_B_top + "_base"] + mech_strain_B_top + np.random.normal(0, 0.2, 960)
    test_df[col_B_bot] = test_df[col_B_bot + "_base"] + mech_strain_B_bot + np.random.normal(0, 0.2, 960)

base_cols = [c for c in test_df.columns if c.endswith("_base")]
test_df.drop(columns=base_cols, inplace=True)

# Reorder columns to match training_df columns exactly
test_df = test_df[train_df.columns]

test_df.to_csv("backend/load_test_data.csv", index=False)
print("Saved load_test_data.csv with shape:", test_df.shape)
