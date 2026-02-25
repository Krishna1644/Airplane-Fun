# %%
# ==========================================
# CHUNK 0: MASTER SETUP & IMPORTS
# ==========================================
import pandas as pd
import numpy as np
import time 
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
SAMPLE_SIZE = 200000  # None = use logic for 6.7 million rows

print("Loading datasets...")
df_flights = pd.read_csv('US_flights_2023.csv', low_memory=False)
df_airports = pd.read_csv('airports_geolocation.csv')
df_weather = pd.read_csv('weather_meteo_by_airport.csv')

df_flights['FlightDate'] = pd.to_datetime(df_flights['FlightDate'])
df_weather['time'] = pd.to_datetime(df_weather['time'])

if SAMPLE_SIZE and len(df_flights) > SAMPLE_SIZE:
    print(f"   -> Sampling {SAMPLE_SIZE} rows...")
    df_flights = df_flights.sample(n=SAMPLE_SIZE, random_state=42)

# MERGE AIRPORTS
df_merged = df_flights.merge(
    df_airports[['IATA_CODE', 'STATE']], 
    left_on='Dep_Airport', 
    right_on='IATA_CODE', 
    how='left'
)
df_merged = df_merged.rename(columns={'STATE': 'Origin_State'}).drop(columns=['IATA_CODE'])

# MERGE WEATHER
df_final = df_merged.merge(
    df_weather[['airport_id', 'time', 'prcp', 'snow', 'wspd', 'tavg']],
    left_on=['Dep_Airport', 'FlightDate'],
    right_on=['airport_id', 'time'],
    how='left'
)
df_final = df_final.drop(columns=['airport_id', 'time'])

# Add month
df_final['Month'] = df_final['FlightDate'].dt.month

print(f"SETUP COMPLETE. 'df_final' is ready. Shape: {df_final.shape}")

# %%
# ==========================================
# CHUNK 1: Clean and Prepare Universal Setup
# ==========================================
print("\n--- Setting up Regression Data ---")

features = [
    'Dep_Airport',
    'Arr_Airport',
    'Month', 
    'Day_Of_Week', 
    'Airline',
    'DepTime_label',      
    'Flight_Duration',      
    'tavg', 
    'prcp', 
    'wspd',
    'Aicraft_age',
    'Dep_Delay',    # Target for Stage 1, Feature for Stage 2
    'Arr_Delay'     # Target for Stage 2
]

df_reg = df_final[features].copy()
df_reg = df_reg.dropna()

impossible_flights = df_reg[df_reg['Flight_Duration'] <= 0]
if len(impossible_flights) > 0:
    df_reg = df_reg[df_reg['Flight_Duration'] > 0]

print(f"Data Cleaned. Total viable rows: {len(df_reg)}")

# %%
# ==========================================
# CHUNK 2: Export Categorical Mappings
# ==========================================
print("\n--- Exporting Categorical Mappings ---")
cat_cols = ['Dep_Airport', 'Arr_Airport', 'Airline', 'DepTime_label']

# Fit OrdinalEncoder specifically to extract mappings BEFORE pipelines
oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
oe.fit(df_reg[cat_cols])

for i, col in enumerate(cat_cols):
    categories = oe.categories_[i]
    mapping_df = pd.DataFrame({
        'Category_String': categories,
        'Encoded_Integer': range(len(categories))
    })
    csv_name = f"{col}_mapping.csv"
    mapping_df.to_csv(csv_name, index=False)
    print(f" -> Saved {csv_name}")


# %%
# ==========================================
# CHUNK 3: STAGE 1 (Predict Departure Delay)
# ==========================================
print("\n==================================")
print("=== TRAINING STAGE 1 MODEL ===")
print("==================================")

features_stage1 = [
    'Dep_Airport', 'Arr_Airport', 'Month', 'Day_Of_Week', 'Airline',
    'DepTime_label', 'Flight_Duration', 'tavg', 'prcp', 'wspd', 'Aicraft_age'
]
target_stage1 = 'Dep_Delay'

X1 = df_reg[features_stage1]
y1 = df_reg[target_stage1]

X_train1, X_test1, y_train1, y_test1 = train_test_split(X1, y1, test_size=0.2, random_state=42)

num_cols1 = X1.select_dtypes(include=np.number).columns.tolist()
cat_cols1 = X1.select_dtypes(exclude=np.number).columns.tolist()

preprocessor1 = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', num_cols1),
        ('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), cat_cols1)
    ]
)

rf_stage1 = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
pipe1 = Pipeline(steps=[('preprocessor', preprocessor1), ('model', rf_stage1)])

start_time = time.time()
pipe1.fit(X_train1, y_train1)
print(f"Stage 1 Training Time: {(time.time() - start_time):.2f}s")

preds1 = pipe1.predict(X_test1)
print(f"Stage 1 RMSE (Dep_Delay): {np.sqrt(mean_squared_error(y_test1, preds1)):.4f} mins")
print(f"Stage 1 R2: {r2_score(y_test1, preds1):.4f}")

joblib.dump(pipe1, "stage1_model.joblib")
print("Saved stage1_model.joblib")


# %%
# ==========================================
# CHUNK 4: STAGE 2 (Predict Arrival Delay)
# ==========================================
print("\n==================================")
print("=== TRAINING STAGE 2 MODEL ===")
print("==================================")

features_stage2 = [
    'Dep_Airport', 'Arr_Airport', 'Month', 'Day_Of_Week', 'Airline',
    'DepTime_label', 'Flight_Duration', 'tavg', 'prcp', 'wspd', 'Aicraft_age',
    'Dep_Delay' # Now including Dep_Delay as an input!
]
target_stage2 = 'Arr_Delay'

X2 = df_reg[features_stage2]
y2 = df_reg[target_stage2]

X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y2, test_size=0.2, random_state=42)

num_cols2 = X2.select_dtypes(include=np.number).columns.tolist()
cat_cols2 = X2.select_dtypes(exclude=np.number).columns.tolist()

preprocessor2 = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', num_cols2),
        ('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), cat_cols2)
    ]
)

rf_stage2 = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
pipe2 = Pipeline(steps=[('preprocessor', preprocessor2), ('model', rf_stage2)])

start_time = time.time()
pipe2.fit(X_train2, y_train2)
print(f"Stage 2 Training Time: {(time.time() - start_time):.2f}s")

preds2 = pipe2.predict(X_test2)
print(f"Stage 2 RMSE (Arr_Delay): {np.sqrt(mean_squared_error(y_test2, preds2)):.4f} mins")
print(f"Stage 2 R2: {r2_score(y_test2, preds2):.4f}")

joblib.dump(pipe2, "stage2_model.joblib")
print("Saved stage2_model.joblib")

print("\nAll models trained and exported successfully!")
