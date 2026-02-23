import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration
DATA_FILE = 'US_flights_2023.csv'
OUTPUT_FILE = 'airport_performance_tiers_enriched.csv'

print("--- 1. Loading Flight Data ---")
# We only need specific columns for clustering
cols_to_load = ['FlightDate', 'Dep_Airport', 'Dep_Delay', 'Arr_Delay', 'Airline']
try:
    df = pd.read_csv(DATA_FILE, usecols=cols_to_load, low_memory=False)
except FileNotFoundError:
    print(f"Error: {DATA_FILE} not found.")
    exit(1)

# Drop rows with missing delays
df = df.dropna(subset=['Dep_Delay', 'Arr_Delay'])
print(f"Data Loaded: {len(df):,} flights.")

print("\n--- 2. Engineering Airport 'Report Cards' ---")
# Aggregate by Departure Airport
airport_df = df.groupby('Dep_Airport').agg(
    total_flights=('Dep_Delay', 'count'),
    avg_dep_delay=('Dep_Delay', 'mean'),
    avg_arr_delay=('Arr_Delay', 'mean'),
    delay_volatility=('Dep_Delay', 'std'),
    unique_airlines=('Airline', 'nunique')
).reset_index()

# Handle NaN in volatility
airport_df['delay_volatility'] = airport_df['delay_volatility'].fillna(0)

print("\n--- 3. Filtering Outliers ---")
# Keep only airports with significant traffic (> 1000 flights/year)
original_count = len(airport_df)
airport_df = airport_df[airport_df['total_flights'] > 1000].copy()
print(f"Removed {original_count - len(airport_df)} small airports.")
print(f"Final Dataset: {len(airport_df)} Airports ready for clustering.")

print("\n--- 4. Scaling Data ---")
features = ['total_flights', 'avg_dep_delay', 'avg_arr_delay', 'delay_volatility', 'unique_airlines']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(airport_df[features])

print("\n--- 5. Applying K-Means Clustering (k=5) ---")
# Based on previous analysis, k=5 is chosen
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
airport_df['Performance_Tier'] = kmeans.fit_predict(X_scaled)

print("Clusters assigned.")
print(airport_df['Performance_Tier'].value_counts().sort_index())

print(f"\n--- 6. Saving Results to {OUTPUT_FILE} ---")
airport_df.to_csv(OUTPUT_FILE, index=False)
print("Done.")
