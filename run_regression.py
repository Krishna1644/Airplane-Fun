import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# CONFIGURATION
SAMPLE_SIZE = 500000  # Using 500k rows for efficiency

print("--- 1. Loading Data ---")
# Load datasets
try:
    df_flights = pd.read_csv('US_flights_2023.csv', low_memory=False)
    df_airports = pd.read_csv('airports_geolocation.csv')
    df_weather = pd.read_csv('weather_meteo_by_airport.csv')
except FileNotFoundError as e:
    print(f"Error loading files: {e}")
    exit(1)

print("--- 2. Merging & Prepping ---")
# Convert dates
df_flights['FlightDate'] = pd.to_datetime(df_flights['FlightDate'])
df_weather['time'] = pd.to_datetime(df_weather['time'])

# SAMPLING
if SAMPLE_SIZE and len(df_flights) > SAMPLE_SIZE:
    print(f"   -> Sampling {SAMPLE_SIZE} rows from {len(df_flights)}...")
    df_flights = df_flights.sample(n=SAMPLE_SIZE, random_state=42)

# MERGE AIRPORTS (Get Origin_State)
df_merged = df_flights.merge(
    df_airports[['IATA_CODE', 'STATE']], 
    left_on='Dep_Airport', 
    right_on='IATA_CODE', 
    how='left'
)
df_merged = df_merged.rename(columns={'STATE': 'Origin_State'}).drop(columns=['IATA_CODE'])

# MERGE WEATHER (Get Rain/Wind)
df_final = df_merged.merge(
    df_weather[['airport_id', 'time', 'prcp', 'snow', 'wspd', 'tavg']],
    left_on=['Dep_Airport', 'FlightDate'],
    right_on=['airport_id', 'time'],
    how='left'
)
# Cleanup
df_final = df_final.drop(columns=['airport_id', 'time'])

print(f"SETUP COMPLETE. 'df_final' is ready. Shape: {df_final.shape}")

# ==========================================
# CHUNK 1: Feature Selection & Split
# ==========================================
print("--- Setting up Final Regression Data ---")

# 1. FINAL FEATURES
features_final = [
    'Dep_Delay',          # The primary driver
    'Flight_Duration',    # Proxy for Distance
    'Day_Of_Week', 
    'DepTime_label',      # Morning/Evening
    'Airline',
    'Aicraft_age',
    'Origin_State',
    'tavg', 'prcp', 'wspd' # Weather factors
]

target = 'Arr_Delay'

# 2. FILTER & CLEAN
# Ensure columns exist
missing_cols = [c for c in features_final if c not in df_final.columns]
if missing_cols:
    print(f"Warning: Missing columns {missing_cols}. Dropping them.")
    features_final = [c for c in features_final if c in df_final.columns]

df_reg = df_final[features_final + [target]].copy()
df_reg = df_reg.dropna()

# 4. SPLIT
X = df_reg.drop(columns=[target])
y = df_reg[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Data Ready. Training on {len(X_train)} rows, Testing on {len(X_test)} rows.")

# ==========================================
# CHUNK 2: The Pipeline
# ==========================================

# Identify columns by type
num_cols = X.select_dtypes(include=np.number).columns.tolist()
cat_cols = X.select_dtypes(exclude=np.number).columns.tolist()

# Build Transformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
    ]
)

print("Pipeline defined.")
print(f"Numerical Features: {num_cols}")
print(f"Categorical Features: {cat_cols}")

# ==========================================
# CHUNK 3: Model Training & Comparison
# ==========================================

results_data = {}

def train_and_score(name, model, save_preds=False):
    print(f"\nTraining {name}...")
    start = time.time()
    
    # Create full pipeline
    pipe = Pipeline(steps=[('preprocessor', preprocessor),
                           ('model', model)])
    
    # Fit
    pipe.fit(X_train, y_train)
    
    # Predict
    preds = pipe.predict(X_test)
    
    # Score
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    
    duration = time.time() - start
    print(f"  -> RMSE: {rmse:.4f} min")
    print(f"  -> R2:   {r2:.4f}")
    print(f"  -> Time: {duration:.2f}s")
    
    results_data[name] = {'RMSE': rmse, 'R2': r2, 'Time': duration}

    if save_preds:
        # Save a sample of predictions for the dashboard (Actual vs Predicted)
        # We'll take a random sample of 1000 points to keep the file size small and plotting fast
        output_df = pd.DataFrame({'Actual': y_test, 'Predicted': preds})
        sample_output = output_df.sample(n=min(1000, len(output_df)), random_state=42)
        sample_output.to_csv('regression_predictions_sample.csv', index=False)
        print("  -> Saved prediction sample to 'regression_predictions_sample.csv'")
        
        # Save Feature Importance if available (Random Forest)
        if hasattr(model, 'feature_importances_'):
            # Get feature names from preprocessor
            # This is a bit tricky with pipelines, we'll try our best
            try:
                # OneHotEncoder feature names
                ohe = pipe.named_steps['preprocessor'].named_transformers_['cat']
                ohe_cols = ohe.get_feature_names_out(cat_cols)
                all_cols = num_cols + list(ohe_cols)
                
                importances = model.feature_importances_
                feat_df = pd.DataFrame({'Feature': all_cols, 'Importance': importances})
                feat_df = feat_df.sort_values('Importance', ascending=False).head(20)
                feat_df.to_csv('feature_importance.csv', index=False)
                print("  -> Saved feature importance to 'feature_importance.csv'")
            except Exception as e:
                print(f"  -> Could not extract feature names: {e}")

    return pipe

# Model A: Linear Regression
train_and_score("Linear Regression", LinearRegression())

# Model B: Random Forest (Simplified for speed)
# Using fewer estimators and depth to keep it fast for this run
print("\nTraining Random Forest (Simplified)...")
rf_model = RandomForestRegressor(n_estimators=20, max_depth=10, n_jobs=-1, random_state=42)
train_and_score("Random Forest", rf_model, save_preds=True)

# Save Metrics
metrics_df = pd.DataFrame(results_data).T.reset_index().rename(columns={'index': 'Model'})
metrics_df.to_csv('regression_metrics.csv', index=False)
print("\nRegression Analysis Complete. Dashboard data saved.")
