import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Step 1: Load the dataset
data = pd.read_csv('data/global.csv')

# Step 2: Clean and prepare the data
data = data.dropna(subset=['Total'])          # Remove rows with missing Total
data = data.fillna(method='ffill')            # Fill other missing values

# Step 3: Select features and target
features = ['Gas Fuel', 'Liquid Fuel', 'Solid Fuel', 'Cement', 'Gas Flaring']
target = 'Total'
X = data[features]
y = data[target]

# Step 4: Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 5: Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Step 6: Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 7: Predict using the test set
y_pred = model.predict(X_test)

# Step 8: Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("✅ Model Training Completed")
print("Mean Absolute Error (MAE):", mae)
print("R-squared Score:", r2)

# Step 9: Plot predictions vs actual values
plt.figure(figsize=(10, 5))
plt.plot(y_test.values, label='Actual CO₂ Emissions', marker='o')
plt.plot(y_pred, label='Predicted CO₂ Emissions', marker='x')
plt.title("Actual vs Predicted CO₂ Emissions")
plt.xlabel("Test Sample Index")
plt.ylabel("CO₂ Emissions (million tonnes)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
