import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Step 1: Load the dataset
data = pd.read_csv('data/global.csv')

# Step 2: Clean missing values
data = data.dropna(subset=['Total'])        # Drop rows with missing target
data = data.fillna(method='ffill')          # Forward fill for others

# Step 3: Feature Engineering
# Create new feature: Total Fuel Usage
data['Total Fuel'] = data[['Gas Fuel', 'Liquid Fuel', 'Solid Fuel', 'Cement', 'Gas Flaring']].sum(axis=1)

# Create Fuel Ratio: Gas Fuel / Total Fuel
data['Gas Fuel Ratio'] = data['Gas Fuel'] / data['Total Fuel']

# Optional: Decade feature (if 'Year' exists and is int)
if 'Year' in data.columns:
    data['Decade'] = (data['Year'] // 10) * 10

# Step 4: Choose Features (including new ones)
features = ['Gas Fuel', 'Liquid Fuel', 'Solid Fuel', 'Cement', 'Gas Flaring',
            'Total Fuel', 'Gas Fuel Ratio']
if 'Decade' in data.columns:
    features.append('Decade')

target = 'Total'

# Step 5: Prepare data
X = data[features]
y = data[target]

# Step 6: Scale the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 7: Split train and test
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Step 8: Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 9: Predict and evaluate
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("✅ Feature Engineering + Training Complete")
print("MAE:", round(mae, 4))
print("R² Score:", round(r2, 5))

# Step 10: Visualize
plt.figure(figsize=(10, 5))
plt.plot(y_test.values, label='Actual CO₂', marker='o')
plt.plot(y_pred, label='Predicted CO₂', marker='x')
plt.title("Actual vs Predicted CO₂ After Feature Engineering")
plt.xlabel("Test Sample Index")
plt.ylabel("CO₂ Emissions")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
