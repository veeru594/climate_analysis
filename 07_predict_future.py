import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

# Step 1: Create hypothetical future data
future_data = pd.DataFrame({
    'Year': [2025, 2030, 2035],
    'Gas Fuel': [1700, 1750, 1800],
    'Liquid Fuel': [2900, 3000, 3100],
    'Solid Fuel': [3900, 4000, 4100],
    'Cement': [500, 520, 540],
    'Gas Flaring': [100, 105, 110]
})

# Step 2: Feature Engineering (same as training)
future_data['Total Fuel'] = future_data[['Gas Fuel', 'Liquid Fuel', 'Solid Fuel', 'Cement', 'Gas Flaring']].sum(axis=1)
future_data['Gas Fuel Ratio'] = future_data['Gas Fuel'] / future_data['Total Fuel']
future_data['Decade'] = (future_data['Year'] // 10) * 10

# Step 3: Select the same features used during training
features = ['Gas Fuel', 'Liquid Fuel', 'Solid Fuel', 'Cement', 'Gas Flaring', 'Total Fuel', 'Gas Fuel Ratio', 'Decade']
X_future = future_data[features]

# Step 4: Recreate scaler and model (same settings as training)
# ❗ IMPORTANT: These values should ideally be saved/reused from training
# For now, we'll retrain quickly with full data

# Load and preprocess original training data
train_df = pd.read_csv('data/global.csv')
train_df = train_df.dropna(subset=['Total']).fillna(method='ffill')
train_df['Total Fuel'] = train_df[['Gas Fuel', 'Liquid Fuel', 'Solid Fuel', 'Cement', 'Gas Flaring']].sum(axis=1)
train_df['Gas Fuel Ratio'] = train_df['Gas Fuel'] / train_df['Total Fuel']
train_df['Decade'] = (train_df['Year'] // 10) * 10

X_train = train_df[features]
y_train = train_df['Total']

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_future_scaled = scaler.transform(X_future)

# Train model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Step 5: Predict
future_predictions = model.predict(X_future_scaled)

# Step 6: Show Results
future_data['Predicted CO₂'] = np.round(future_predictions, 2)
print("📊 Future Predictions:\n")
print(future_data[['Year', 'Predicted CO₂']])

# Step 7: Plot
plt.figure(figsize=(8, 5))
plt.plot(future_data['Year'], future_data['Predicted CO₂'], marker='o', color='green')
plt.title("Predicted Global CO₂ Emissions")
plt.xlabel("Year")
plt.ylabel("CO₂ Emissions (million tonnes)")
plt.grid(True)
plt.tight_layout()
plt.show()
