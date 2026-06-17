import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')
import os
os.makedirs('plots', exist_ok=True)

df = pd.read_csv('electric_vehicle_analytics.csv')

# Features and target
features = ['Battery_Capacity_kWh', 'Battery_Health_%', 'Temperature_C',
            'Avg_Speed_kmh', 'Energy_Consumption_kWh_per_100km',
            'Charge_Cycles', 'Mileage_km', 'Charging_Power_kW']

X = df[features]
y = df['Range_km']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
}

results = {}
print("Model Comparison:")
print("-" * 55)

for name, model in models.items():
    if name == 'Linear Regression':
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    results[name] = {'MAE': mae, 'RMSE': rmse, 'R2': r2, 'pred': y_pred}
    print(f"{name}: MAE={mae:.2f}km | RMSE={rmse:.2f}km | R2={r2:.4f}")

print("-" * 55)

# Best model feature importance
rf = models['Random Forest']
importances = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=True)

BG = '#1a1a2e'

# Plot 1 - Feature importance
fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)
importances.plot(kind='barh', ax=ax, color='#00D2BE')
ax.set_title('Feature Importance - Random Forest\nWhat Predicts EV Range?', 
             color='white', fontsize=13, fontweight='bold')
ax.tick_params(colors='white')
ax.set_xlabel('Importance', color='white')
ax.spines[['top','right','left','bottom']].set_visible(False)
plt.tight_layout()
plt.savefig('plots/feature_importance.png', dpi=150, bbox_inches='tight', facecolor=BG)
print("\nPlot 1 saved")

# Plot 2 - Actual vs Predicted
fig, ax = plt.subplots(figsize=(8, 8))
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)
best_pred = results['Random Forest']['pred']
ax.scatter(y_test, best_pred, alpha=0.3, color='#00D2BE', s=10)
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
        'white', linewidth=1.5, linestyle='--')
ax.set_xlabel('Actual Range (km)', color='white')
ax.set_ylabel('Predicted Range (km)', color='white')
ax.set_title(f'Actual vs Predicted Range\nRandom Forest | R2={results["Random Forest"]["R2"]:.4f}',
             color='white', fontsize=13, fontweight='bold')
ax.tick_params(colors='white')
ax.spines[['top','right','left','bottom']].set_visible(False)
plt.tight_layout()
plt.savefig('plots/actual_vs_predicted.png', dpi=150, bbox_inches='tight', facecolor=BG)
print("Plot 2 saved")

# Plot 3 - Model comparison bar chart
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
fig.patch.set_facecolor(BG)
metrics = ['MAE', 'RMSE', 'R2']
colors = ['#E8002D', '#FFD700', '#00D2BE']

for i, metric in enumerate(metrics):
    ax = axes[i]
    ax.set_facecolor(BG)
    vals = [results[m][metric] for m in models.keys()]
    bars = ax.bar(list(models.keys()), vals, color=colors[i], width=0.5)
    ax.set_title(metric, color='white', fontsize=12, fontweight='bold')
    ax.tick_params(colors='white', labelsize=8)
    ax.spines[['top','right','left','bottom']].set_visible(False)
    ax.grid(axis='y', alpha=0.2, color='white')
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{bar.get_height():.2f}', ha='center', va='bottom', 
                color='white', fontsize=9)
    plt.setp(ax.get_xticklabels(), rotation=15, ha='right')

fig.suptitle('Model Comparison - EV Range Prediction', color='white', 
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('plots/model_comparison.png', dpi=150, bbox_inches='tight', facecolor=BG)
print("Plot 3 saved")

plt.show()
print("\nDone!")