import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Step 1: Load the dataset
file_path = 'data/global.csv'
data = pd.read_csv(file_path)

# Step 2: Drop rows with missing 'Total' CO2 emissions
data = data.dropna(subset=['Total'])

# Step 3: Fill other missing values (forward fill)
data = data.fillna(method='ffill')

# Step 4: Select Features and Target
features = ['Gas Fuel', 'Liquid Fuel', 'Solid Fuel', 'Cement', 'Gas Flaring']
target = 'Total'

X = data[features]
y = data[target]

# Step 5: Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 6: Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Step 7: Output for verification
print("✅ Data Preprocessing Completed")
print("Features used:", features)
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)
