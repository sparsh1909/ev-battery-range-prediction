import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

os.makedirs('plots', exist_ok=True)

df = pd.read_csv('electric_vehicle_analytics.csv')

print("Shape:", df.shape)
print("\nColumns:", df.columns.tolist())
print("\nData types:\n", df.dtypes)
print("\nMissing values:\n", df.isnull().sum())
print("\nBasic stats:\n", df[['Range_km', 'Battery_Capacity_kWh', 'Battery_Health_%', 
                               'Temperature_C', 'Avg_Speed_kmh', 'Energy_Consumption_kWh_per_100km',
                               'Charge_Cycles', 'Mileage_km']].describe().round(2))

BG = '#1a1a2e'

# Plot 1 - Range distribution
fig, ax = plt.subplots(figsize=(10, 5))
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)
ax.hist(df['Range_km'], bins=40, color='#00D2BE', edgecolor='white', alpha=0.8)
ax.set_title('Distribution of EV Range (km)', color='white', fontsize=13, fontweight='bold')
ax.set_xlabel('Range (km)', color='white')
ax.set_ylabel('Count', color='white')
ax.tick_params(colors='white')
ax.spines[['top','right','left','bottom']].set_visible(False)
plt.tight_layout()
plt.savefig('plots/range_distribution.png', dpi=150, bbox_inches='tight', facecolor=BG)
print("\nPlot 1 saved")

# Plot 2 - Correlation heatmap
fig, ax = plt.subplots(figsize=(12, 8))
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)
numeric_cols = ['Range_km', 'Battery_Capacity_kWh', 'Battery_Health_%', 
                'Temperature_C', 'Avg_Speed_kmh', 'Energy_Consumption_kWh_per_100km',
                'Charge_Cycles', 'Mileage_km', 'Charging_Power_kW']
corr = df[numeric_cols].corr()
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', ax=ax,
            annot_kws={'size': 9, 'color': 'white'})
ax.set_title('Feature Correlation with Range', color='white', fontsize=13, fontweight='bold')
ax.tick_params(colors='white')
plt.tight_layout()
plt.savefig('plots/correlation_heatmap.png', dpi=150, bbox_inches='tight', facecolor=BG)
print("Plot 2 saved")

# Plot 3 - Battery capacity vs Range
fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)
scatter = ax.scatter(df['Battery_Capacity_kWh'], df['Range_km'], 
                     c=df['Battery_Health_%'], cmap='RdYlGn', alpha=0.5, s=10)
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('Battery Health %', color='white')
cbar.ax.yaxis.set_tick_params(color='white')
plt.setp(cbar.ax.yaxis.get_ticklabels(), color='white')
ax.set_xlabel('Battery Capacity (kWh)', color='white')
ax.set_ylabel('Range (km)', color='white')
ax.set_title('Battery Capacity vs Range\nColored by Battery Health', color='white', fontsize=13, fontweight='bold')
ax.tick_params(colors='white')
ax.spines[['top','right','left','bottom']].set_visible(False)
plt.tight_layout()
plt.savefig('plots/capacity_vs_range.png', dpi=150, bbox_inches='tight', facecolor=BG)
print("Plot 3 saved")

plt.show()
print("\nEDA complete!")