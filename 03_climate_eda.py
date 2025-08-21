import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load the dataset
file_path = 'data/global.csv'
data = pd.read_csv(file_path)

# Step 2: Show basic info
print("\nFirst 5 rows:")
print(data.head())

print("\nColumn Names:")
print(data.columns)

print("\nMissing Values:")
print(data.isnull().sum())

print("\nSummary Stats:")
print(data.describe())

# Step 3: Convert 'Year' to integer (if it's not)
data['Year'] = pd.to_numeric(data['Year'], errors='coerce')

# Step 4: Line plot of Total CO2 emissions over time
plt.figure(figsize=(10, 5))
sns.lineplot(x='Year', y='Total', data=data)
plt.title("Total Global CO₂ Emissions Over Time")
plt.xlabel("Year")
plt.ylabel("CO₂ Emissions (million tonnes)")
plt.grid(True)
plt.show()

# Step 5: Line plot of Per Capita CO2
plt.figure(figsize=(10, 5))
sns.lineplot(x='Year', y='Per Capita', data=data)
plt.title("Global Per Capita CO₂ Emissions Over Time")
plt.xlabel("Year")
plt.ylabel("CO₂ per Person")
plt.grid(True)
plt.show()

# ✅ Corrected Per Capita Plot (only one clean version)

# Drop rows with missing 'Per Capita'
data_pc = data.dropna(subset=['Per Capita'])

# Plot Per Capita CO2 (Cleaned Data)
plt.figure(figsize=(10, 5))
sns.lineplot(x='Year', y='Per Capita', data=data_pc)
plt.title("Global Per Capita CO₂ Emissions Over Time")
plt.xlabel("Year")
plt.ylabel("CO₂ per Person")
plt.grid(True)
plt.show()
