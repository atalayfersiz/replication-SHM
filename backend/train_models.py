import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Ensure backend directory exists
os.makedirs("backend", exist_ok=True)

# Load training data
df = pd.read_csv("backend/training_data.csv")

# Input features (temperatures)
X_cols = ['TMP_02m', 'TMP_07m', 'TMP_12m', 'TMP_17m']
X = df[X_cols].values

# Output targets (strains)
strain_cols = [col for col in df.columns if col.startswith("STR_")]
Y = df[strain_cols].values

# Split into train and test sets (80% train, 20% test)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Scalers for ANN
scaler_X = StandardScaler()
scaler_Y = StandardScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)
Y_train_scaled = scaler_Y.fit_transform(Y_train)
Y_test_scaled = scaler_Y.transform(Y_test)

# Models dictionary
models = {
    'ANN': MLPRegressor(hidden_layer_sizes=(16, 8), activation='tanh', solver='lbfgs', max_iter=3000, random_state=42),
    'Linear': LinearRegression(),
    'Lasso': Lasso(alpha=0.01, random_state=42),
    'Ridge': Ridge(alpha=1.0, random_state=42)
}

# Dictionary to store performance metrics
results = {}
fitted_models = {}

for name, model in models.items():
    print(f"Training {name} model...")
    if name == 'ANN':
        # Train on scaled data
        model.fit(X_train_scaled, Y_train_scaled)
        fitted_models[name] = model
        
        # Predict and inverse transform
        Y_pred_train_s = model.predict(X_train_scaled)
        Y_pred_test_s = model.predict(X_test_scaled)
        
        Y_pred_train = scaler_Y.inverse_transform(Y_pred_train_s)
        Y_pred_test = scaler_Y.inverse_transform(Y_pred_test_s)
    else:
        # Train on original data
        model.fit(X_train, Y_train)
        fitted_models[name] = model
        
        # Predict
        Y_pred_train = model.predict(X_train)
        Y_pred_test = model.predict(X_test)
    
    # Calculate metrics (average across all 64 sensors)
    train_mae = mean_absolute_error(Y_train, Y_pred_train)
    train_mse = mean_squared_error(Y_train, Y_pred_train)
    train_rmse = np.sqrt(train_mse)
    train_r2 = r2_score(Y_train, Y_pred_train)
    
    test_mae = mean_absolute_error(Y_test, Y_pred_test)
    test_mse = mean_squared_error(Y_test, Y_pred_test)
    test_rmse = np.sqrt(test_mse)
    test_r2 = r2_score(Y_test, Y_pred_test)
    
    results[name] = {
        'Train_MAE': train_mae, 'Train_MSE': train_mse, 'Train_RMSE': train_rmse, 'Train_R2': train_r2,
        'Test_MAE': test_mae, 'Test_MSE': test_mse, 'Test_RMSE': test_rmse, 'Test_R2': test_r2
    }

# Display Table 4 Comparison
print("\n" + "="*95)
print("TABLE 4: Statistical metrics of machine learning models (Average across 64 sensors)")
print("="*95)
print(f"{'Model':<10} | {'Train MAE':<10} | {'Train MSE':<10} | {'Train RMSE':<10} | {'Train R2':<10} | {'Test MAE':<10} | {'Test MSE':<10} | {'Test RMSE':<10} | {'Test R2':<10}")
print("-"*95)
for name, metrics in results.items():
    print(f"{name:<10} | {metrics['Train_MAE']:<10.3f} | {metrics['Train_MSE']:<10.3f} | {metrics['Train_RMSE']:<10.3f} | {metrics['Train_R2']:<10.4f} | {metrics['Test_MAE']:<10.3f} | {metrics['Test_MSE']:<10.3f} | {metrics['Test_RMSE']:<10.3f} | {metrics['Test_R2']:<10.4f}")
print("="*95)

# Save the trained ANN and its scalers for baseline prediction
with open("backend/ann_model.pkl", "wb") as f:
    pickle.dump({
        'model': fitted_models['ANN'],
        'scaler_X': scaler_X,
        'scaler_Y': scaler_Y
    }, f)
print("Saved trained ANN model and scalers to backend/ann_model.pkl")

# Save all models for reference
with open("backend/all_models.pkl", "wb") as f:
    pickle.dump({
        'models': fitted_models,
        'scaler_X': scaler_X,
        'scaler_Y': scaler_Y
    }, f)

# Generate plots for a representative sensor (STR_A_Bot_02m) to replicate Paper Figures
target_sensor = "STR_A_Bot_02m"
sensor_idx = strain_cols.index(target_sensor)

y_train_sensor = Y_train[:, sensor_idx]
y_test_sensor = Y_test[:, sensor_idx]

# Create Figure 9: Error Density Plots
fig9, axes9 = plt.subplots(2, 2, figsize=(14, 10))
axes9 = axes9.ravel()

plot_order = ['Ridge', 'Lasso', 'Linear', 'ANN']
colors = ['#2b5c8f', '#2b5c8f', '#2b5c8f', '#2b5c8f']

for idx, name in enumerate(plot_order):
    model = fitted_models[name]
    if name == 'ANN':
        y_pred_test_s = model.predict(X_test_scaled)
        y_pred_test_sensor = scaler_Y.inverse_transform(y_pred_test_s)[:, sensor_idx]
    else:
        y_pred_test_sensor = model.predict(X_test)[:, sensor_idx]
        
    errors = y_test_sensor - y_pred_test_sensor
    
    # Plot histogram
    n, bins, patches = axes9[idx].hist(errors, bins=25, density=True, alpha=0.6, label=target_sensor, color=colors[idx], edgecolor='black')
    
    # Fit Normal Distribution
    mu, std = stats.norm.fit(errors)
    xmin, xmax = axes9[idx].get_xlim()
    x = np.linspace(xmin, xmax, 200)
    
    # Overlay theoretical normal distribution N(0, 1) or fitted normal
    axes9[idx].plot(x, stats.norm.pdf(x, 0, 1), 'r-', linewidth=2, label=r'$\mathcal{N}(\mu=0, \sigma=1)$')
    
    # KDE curve
    kde = stats.gaussian_kde(errors)
    axes9[idx].plot(x, kde(x), 'k-', linewidth=1.5, label='Kernel density estimate')
    
    axes9[idx].set_title(name, fontsize=14, fontweight='bold')
    axes9[idx].set_xlabel('Prediction Error (με)', fontsize=12)
    axes9[idx].set_ylabel('Density', fontsize=12)
    axes9[idx].legend(frameon=True)
    axes9[idx].set_xlim(-10, 10)
    if name == 'ANN':
        axes9[idx].set_xlim(-6, 6)

plt.tight_layout()
fig9.savefig("backend/figure9_error_density.png", dpi=150)
plt.close(fig9)
print("Saved Figure 9 to backend/figure9_error_density.png")

# Create Figure 10: Normal Q-Q Plots
fig10, axes10 = plt.subplots(2, 2, figsize=(14, 10))
axes10 = axes10.ravel()

for idx, name in enumerate(plot_order):
    model = fitted_models[name]
    if name == 'ANN':
        y_pred_test_s = model.predict(X_test_scaled)
        y_pred_test_sensor = scaler_Y.inverse_transform(y_pred_test_s)[:, sensor_idx]
    else:
        y_pred_test_sensor = model.predict(X_test)[:, sensor_idx]
        
    errors = y_test_sensor - y_pred_test_sensor
    
    # Plot Q-Q
    stats.probplot(errors, dist="norm", plot=axes10[idx])
    
    # Customize markers to look like the paper
    axes10[idx].get_lines()[0].set_markerfacecolor('none')
    axes10[idx].get_lines()[0].set_markeredgecolor('black')
    axes10[idx].get_lines()[0].set_markersize(4)
    axes10[idx].get_lines()[0].set_alpha(0.7)
    axes10[idx].get_lines()[1].set_color('red')
    axes10[idx].get_lines()[1].set_linewidth(1.5)
    
    axes10[idx].set_title(f"{name}\nNormal Q-Q Plot", fontsize=12, fontweight='bold')
    axes10[idx].set_xlabel('Theoretical quantiles', fontsize=10)
    axes10[idx].set_ylabel('Ordered Values', fontsize=10)

plt.tight_layout()
fig10.savefig("backend/figure10_qq_plot.png", dpi=150)
plt.close(fig10)
print("Saved Figure 10 to backend/figure10_qq_plot.png")

# Create Figure 11: Baseline Strain Prediction vs Test Data (ANN)
fig11, ax11 = plt.subplots(figsize=(12, 6))
model_ann = fitted_models['ANN']
y_pred_test_s = model_ann.predict(X_test_scaled)
y_pred_test_ann = scaler_Y.inverse_transform(y_pred_test_s)[:, sensor_idx]

# Sort by temperature to show a smooth curve
sort_idx = np.argsort(X_test[:, 0])
ax11.plot(y_test_sensor[sort_idx], 'r-', label='Test Data', linewidth=1.5)
ax11.plot(y_pred_test_ann[sort_idx], 'b-', label='Test Prediction', linewidth=1.2)
ax11.set_title("Strain Prediction (ANN) - STR_A_Bot_02m", fontsize=14, fontweight='bold')
ax11.set_xlabel("Sorted Temp Samples", fontsize=12)
ax11.set_ylabel("Strain (με)", fontsize=12)
ax11.legend(frameon=True, fontsize=12)

fig11.savefig("backend/figure11_baseline_prediction.png", dpi=150)
plt.close(fig11)
print("Saved Figure 11 to backend/figure11_baseline_prediction.png")
