# 01_intro_pandas.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
url = 'https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv'
data = pd.read_csv(url)

# Explore dataset
print("First 5 rows:\n", data.head())
print("\nColumn Names:\n", data.columns)
print("\nSummary Stats:\n", data.describe())
print("\nMissing Values:\n", data.isnull().sum())

# Simple Plot
sns.scatterplot(x='total_bill', y='tip', data=data)
plt.title('Total Bill vs Tip')
plt.show()
