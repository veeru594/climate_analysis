import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Step 1: Load original training data
data = pd.read_csv('data/global.csv')
data = data.dropna(subset=['Total']).fillna(method='ffill')

# Step 2: Feature Engineering
data['Total Fuel'] = data[['Gas Fuel', 'Liquid Fuel', 'Solid Fuel', 'Cement', 'Gas Flaring']].sum(axis=1)
data['Gas Fuel Ratio'] = data['Gas Fuel'] / data['Total Fuel']
data['Decade'] = (data['Year'] // 10) * 10

# Step 3: Select Features and Target
features = ['Gas Fuel', 'Liquid Fuel', 'Solid Fuel', 'Cement', 'Gas Flaring', 'Total Fuel', 'Gas Fuel Ratio', 'Decade']
X = data[features]
y = data['Total']

# Step 4: Train model (final linear regression)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = LinearRegression()
model.fit(X_scaled, y)

# Step 5: Create base scenario (e.g., 2025 with normal usage)
base = pd.DataFrame({
    'Year': [2025],
    'Gas Fuel': [1700],
    'Liquid Fuel': [2800],
    'Solid Fuel': [4000],
    'Cement': [500],
    'Gas Flaring': [90]
})
base['Total Fuel'] = base[['Gas Fuel', 'Liquid Fuel', 'Solid Fuel', 'Cement', 'Gas Flaring']].sum(axis=1)
base['Gas Fuel Ratio'] = base['Gas Fuel'] / base['Total Fuel']
base['Decade'] = (base['Year'] // 10) * 10
base_X = base[features]
base_scaled = scaler.transform(base_X)
base_pred = model.predict(base_scaled)[0]

# Step 6: Define scenarios
scenarios = {
    '🔺 Gas +20%': base.copy(),
    '🔻 Cement -50%': base.copy(),
    '⚡ All Fuels +10%': base.copy(),
    '🍃 Green Policy (-25%)': base.copy()
}

# Step 7: Apply changes to each scenario
scenarios['🔺 Gas +20%']['Gas Fuel'] *= 1.2
scenarios['🔻 Cement -50%']['Cement'] *= 0.5
for col in ['Gas Fuel', 'Liquid Fuel', 'Solid Fuel', 'Cement', 'Gas Flaring']:
    scenarios['⚡ All Fuels +10%'][col] *= 1.1
    scenarios['🍃 Green Policy (-25%)'][col] *= 0.75

# Step 8: Recalculate features and predict
results = []
for label, df in scenarios.items():
    df['Total Fuel'] = df[['Gas Fuel', 'Liquid Fuel', 'Solid Fuel', 'Cement', 'Gas Flaring']].sum(axis=1)
    df['Gas Fuel Ratio'] = df['Gas Fuel'] / df['Total Fuel']
    df['Decade'] = (df['Year'] // 10) * 10
    X_scenario = df[features]
    X_scaled = scaler.transform(X_scenario)
    pred = model.predict(X_scaled)[0]
    results.append({'Scenario': label, 'Predicted CO₂': round(pred, 2)})

# Include base scenario
results.insert(0, {'Scenario': '🔵 Base Scenario', 'Predicted CO₂': round(base_pred, 2)})

# Step 9: Display results
df_results = pd.DataFrame(results)
print("📊 Scenario Simulation Results:\n")
print(df_results)

# Step 10: Plot results
plt.figure(figsize=(10, 5))
plt.bar(df_results['Scenario'], df_results['Predicted CO₂'], color='skyblue')
plt.title("CO₂ Predictions for Various Scenarios")
plt.ylabel("Predicted CO₂ Emissions")
plt.xticks(rotation=30)
plt.tight_layout()
plt.grid(axis='y')
plt.show()
