import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings('ignore')
import os
os.makedirs('plots', exist_ok=True)

df = pd.read_csv('electric_vehicle_analytics.csv')

# ── FEATURE ENGINEERING ──
# Effective range accounting for battery degradation
df['Effective_Range'] = df['Range_km'] * (df['Battery_Health_%'] / 100)

# Energy efficiency score
df['Efficiency_Score'] = df['Battery_Capacity_kWh'] / df['Energy_Consumption_kWh_per_100km']

# Battery degradation factor
df['Degradation_Factor'] = (100 - df['Battery_Health_%']) / 100

# Power to capacity ratio
df['Power_Capacity_Ratio'] = df['Charging_Power_kW'] / df['Battery_Capacity_kWh']

# Temperature impact (EVs lose range in extreme temps)
df['Temp_Impact'] = np.abs(df['Temperature_C'] - 20)  # optimal temp ~20C

# Usage intensity
df['Usage_Intensity'] = df['Mileage_km'] / (df['Charge_Cycles'] + 1)

# Speed efficiency (higher speed = more consumption)
df['Speed_Efficiency'] = df['Avg_Speed_kmh'] / df['Energy_Consumption_kWh_per_100km']

print("New features created:")
print(df[['Efficiency_Score', 'Degradation_Factor', 'Power_Capacity_Ratio', 
          'Temp_Impact', 'Usage_Intensity', 'Speed_Efficiency']].describe().round(3))

# ── FEATURES AND TARGET ──
features_original = ['Battery_Capacity_kWh', 'Battery_Health_%', 'Temperature_C',
                     'Avg_Speed_kmh', 'Energy_Consumption_kWh_per_100km',
                     'Charge_Cycles', 'Mileage_km', 'Charging_Power_kW']

features_engineered = features_original + ['Efficiency_Score', 'Degradation_Factor',
                                            'Power_Capacity_Ratio', 'Temp_Impact',
                                            'Usage_Intensity', 'Speed_Efficiency']

X = df[features_engineered]
y = df['Range_km']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ── MODELS WITH CROSS VALIDATION ──
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, random_state=42),
    'XGBoost': XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=6, random_state=42, verbosity=0)
}

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

results = {}
print("\nModel Comparison with Cross Validation:")
print("-" * 70)

for name, model in models.items():
    if name == 'Linear Regression':
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    results[name] = {'MAE': mae, 'RMSE': rmse, 'R2': r2, 'CV_R2': cv_scores.mean(), 'pred': y_pred, 'model': model}
    print(f"{name}:")
    print(f"  MAE={mae:.2f}km | RMSE={rmse:.2f}km | R2={r2:.4f} | CV R2={cv_scores.mean():.4f} (+/-{cv_scores.std():.4f})")

print("-" * 70)

# ── FEATURE IMPORTANCE (XGBoost) ──
BG = '#1a1a2e'
xgb = results['XGBoost']['model']
importances = pd.Series(xgb.feature_importances_, index=features_engineered).sort_values(ascending=True)

fig, ax = plt.subplots(figsize=(10, 8))
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)
colors = ['#00D2BE' if 'Score' in i or 'Factor' in i or 'Ratio' in i or 'Impact' in i or 'Intensity' in i or 'Efficiency' in i 
          else '#C0C0C0' for i in importances.index]
importances.plot(kind='barh', ax=ax, color=colors)
ax.set_title('Feature Importance - XGBoost\nTeal = Engineered Features', 
             color='white', fontsize=13, fontweight='bold')
ax.tick_params(colors='white')
ax.set_xlabel('Importance', color='white')
ax.spines[['top','right','left','bottom']].set_visible(False)
plt.tight_layout()
plt.savefig('plots/engineered_feature_importance.png', dpi=150, bbox_inches='tight', facecolor=BG)
print("\nPlot saved!")
plt.show()

# Save best model info for Streamlit
import pickle
best_model = results['XGBoost']['model']
with open('model.pkl', 'wb') as f:
    pickle.dump(best_model, f)
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
with open('features.pkl', 'wb') as f:
    pickle.dump(features_engineered, f)

print("\nModel saved for deployment!")
print(f"\nBest model: XGBoost")
print(f"R2: {results['XGBoost']['R2']:.4f}")
print(f"MAE: {results['XGBoost']['MAE']:.2f}km")