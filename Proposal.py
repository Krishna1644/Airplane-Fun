# %%
# === TASK 1: REGRESSION ANALYSIS ===
# This code assumes your 'df_final_renamed' DataFrame is in memory
# from the previous cells you ran.

import numpy as np
import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

print("--- Task 1: Regression Analysis ---")
print(f"Using 'df_final_renamed' DataFrame with {len(df_final_renamed)} rows.")

# --- 1. Define Features and Target ---
# Based on your proposal and the columns you created
target = 'ArrDelay' # Using 'ArrDelay' as seen in the source CSVs.
                   # If your column is 'ARRIVAL_DELAY', please change this.

# Define Independent Variables (Features)
# We use the specific columns from your 'df_final_renamed'
numerical_features = [
    'DepDelay', # Using 'DepDelay'. If it's 'DEPARTURE_DELAY', change this.
    'Distance', # Using 'Distance'. If it's 'DISTANCE', change this.
    'Aircraft_Age',
    
    # Departure Weather Features
    'avg_temp_dep',
    'precipitation_dep',
    'snowfall_dep',
    'wind_speed_dep',
    
    # Arrival Weather Features
    'avg_temp_arr',
    'precipitation_arr',
    'snowfall_arr',
    'wind_speed_arr'
]

categorical_features = [
    'OP_UNIQUE_CARRIER', # Proposal "AIRLINE NAME"
    'Origin_State_Name', # Proposal "ORIGIN_STATE"
    'DayOfWeek'          # Proposal "DAY_OF_WEEK"
]

# --- 2. Data Preparation ---
# Select only the columns we need for this task
all_cols = numerical_features + categorical_features + [target]
try:
    df_task1 = df_final_renamed[all_cols].copy()
except KeyError as e:
    print(f"Error: A column name is incorrect. {e}")
    print("Please check the 'target', 'numerical_features', and 'categorical_features' lists")
    print("to match the exact column names in your 'df_final_renamed' DataFrame.")
    # Stop execution if columns are wrong
    raise e

print("Selected features and target for Task 1.")

# We will sample the data (500k) for efficient model training.
# Using all 6.7M+ rows will be very slow for iterative development.
if len(df_task1) > 500000:
    print(f"Sampling 500,000 rows from {len(df_task1)} for speed.")
    df_task1_sample = df_task1.sample(n=500000, random_state=42)
else:
    df_task1_sample = df_task1

# Split the data
X = df_task1_sample.drop(target, axis=1)
y = df_task1_sample[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 3. Build Pre-processing Pipeline ---
# This pipeline will:
# 1. Impute (fill) missing numerical data with the median
# 2. Impute (fill) missing categorical data with a 'Missing' placeholder
# 3. Scale numerical features
# 4. One-hot encode categorical features

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='Missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Bundle transformers into a ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ],
    remainder='passthrough' # Keep any other columns (though we have none)
)

# --- 4. Define Models (as requested in proposal) ---
# We will test Linear Regression and Random Forest
# Gradient Boosting is good but often very slow on this size data

models = {
    "Linear Regression": Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', LinearRegression())
    ]),
    
    "Random Forest": Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', RandomForestRegressor(
            n_estimators=100,  # A good balance of speed/accuracy
            max_depth=12,      # Prevents overfitting and speeds up training
            random_state=42,
            n_jobs=-1          # Use all available CPU cores
        ))
    ])
}

# --- 5. Train and Evaluate Models ---
print("\n--- Model Training and Evaluation ---")
for name, model in models.items():
    print(f"Training {name}...")
    start_time = time.time()
    
    model.fit(X_train, y_train)
    
    end_time = time.time()
    print(f"Training completed in {end_time - start_time:.2f} seconds.")
    
    # Evaluate on the test set
    y_pred = model.predict(X_test)
    
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print(f"Results for {name}:")
    print(f"  Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"  R-squared (R²): {r2:.4f}\n")

# --- 6. Feature Importance (as you requested) ---
print("\n--- Feature Importance (from Random Forest) ---")
try:
    # Get the pipeline steps
    rf_pipeline = models['Random Forest']
    preprocessor_step = rf_pipeline.named_steps['preprocessor']
    model_step = rf_pipeline.named_steps['model']
    
    # Get one-hot encoded feature names
    onehot_features = preprocessor_step.named_transformers_['cat'] \
                                     .named_steps['onehot'] \
                                     .get_feature_names_out(categorical_features)
    
    # Combine all feature names in the correct order
    all_feature_names = numerical_features + list(onehot_features)
    
    # Get importances
    importances = model_step.feature_importances_
    
    # Create a DataFrame
    feature_importance_df = pd.DataFrame({
        'feature': all_feature_names,
        'importance': importances
    }).sort_values(by='importance', ascending=False)
    
    print("Top 20 most important features:")
    print(feature_importance_df.head(20))

except Exception as e:
    print(f"Could not extract feature importances: {e}")

print("\nTask 1 (Regression) is complete.")