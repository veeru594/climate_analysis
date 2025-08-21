# 02_eda.py — EDA on Climate Dataset

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load the dataset
file_path = 'data/climate_nasa.csv'
data = pd.read_csv(file_path)

print("\nColumn Names in Dataset:")
for col in data.columns:
    print(f"- {col}")


# Step 2: View the first few rows
print("First 5 rows:\n", data.head())

# Step 3: View column names and data types
print("\nColumn Names:", data.columns)
print("\nInfo:")
print(data.info())

# Step 4: Check for missing values
print("\nMissing Values:\n", data.isnull().sum())

# Step 5: Basic statistics
print("\nSummary Stats:\n", data.describe())






# # Correlation heatmap
# import seaborn as sns
# import matplotlib.pyplot as plt

# plt.figure(figsize=(12, 6))
# sns.heatmap(data.corr(numeric_only=True), annot=True, cmap='coolwarm')
# plt.title("Correlation Between Climate Variables")
# plt.show()




# # CO₂ over years
# plt.figure(figsize=(10, 5))
# sns.lineplot(x='Year', y='CO2', data=data)
# plt.title("CO₂ Emissions Over Time")
# plt.xlabel("Year")
# plt.ylabel("CO₂ (ppm)")
# plt.grid(True)
# plt.show()




# # Temperature over years
# plt.figure(figsize=(10, 5))
# sns.lineplot(x='Year', y='Temperature', data=data)
# plt.title("Global Temperature Rise Over Time")
# plt.xlabel("Year")
# plt.ylabel("Temperature (°C)")
# plt.grid(True)
# plt.show()
