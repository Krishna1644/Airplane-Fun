# %%
import pandas as pd

# Load each of the CSV files into its own DataFrame
df_airports = pd.read_csv('airports_geolocation.csv')
df_flights_jan2024 = pd.read_csv('maj us flight - january 2024.csv')
df_flights_2023 = pd.read_csv('US_flights_2023.csv', low_memory=False)
df_weather = pd.read_csv('weather_meteo_by_airport.csv')
df_cancelled_diverted = pd.read_csv('Cancelled_Diverted_2023.csv')

# --- Verification Step ---
# You can print the first few rows of any dataframe to make sure it loaded correctly
print("Successfully loaded 'US_flights_2023.csv'. Here's a preview:")
print(df_flights_2023.info())

# %%
import pandas as pd

# --- Step 1: Prepare the main flights DataFrame ---
df_flights_2023['FlightDate'] = pd.to_datetime(df_flights_2023['FlightDate'])
df_weather['time'] = pd.to_datetime(df_weather['time'])

# --- Step 2: Create two distinct, renamed weather DataFrames ---

# Define the columns we want to rename
weather_cols = ['tavg', 'tmin', 'tmax', 'prcp', 'snow', 'wdir', 'wspd', 'pres', 'time', 'airport_id']

# Create a version for DEPARTURE weather
dep_rename_dict = {col: col + '_dep' for col in weather_cols}
df_weather_dep = df_weather.rename(columns=dep_rename_dict)

# Create a version for ARRIVAL weather
arr_rename_dict = {col: col + '_arr' for col in weather_cols}
df_weather_arr = df_weather.rename(columns=arr_rename_dict)


# --- Step 3: Perform the first merge with the departure weather data ---
df_merged = pd.merge(
    df_flights_2023,
    df_weather_dep,
    left_on=['FlightDate', 'Dep_Airport'],
    right_on=['time_dep', 'airport_id_dep'],
    how='left'
)

# --- Step 4: Perform the second merge with the arrival weather data ---
df_final = pd.merge(
    df_merged,
    df_weather_arr,
    left_on=['FlightDate', 'Arr_Airport'],
    right_on=['time_arr', 'airport_id_arr'],
    how='left'
)

# --- Step 5: Clean up the redundant key columns ---
df_final = df_final.drop(columns=['time_dep', 'airport_id_dep', 'time_arr', 'airport_id_arr'])


# --- Verification ---
# This will now show the correct '_dep' and '_arr' suffixes for all weather columns
print("Columns of the final merged DataFrame:")
print(df_final.info())

print("\nPreview of the final DataFrame with weather for both airports:")
print(df_final[['FlightDate', 'Dep_Airport', 'Arr_Airport', 'tavg_dep', 'tavg_arr']].head())

# %%
rename_mapping = {
    # Departure Weather Columns
    'tavg_dep': 'avg_temp_dep',
    'tmin_dep': 'min_temp_dep',
    'tmax_dep': 'max_temp_dep',
    'prcp_dep': 'precipitation_dep',
    'snow_dep': 'snowfall_dep',
    'wdir_dep': 'wind_direction_dep',
    'wspd_dep': 'wind_speed_dep',
    'pres_dep': 'pressure_dep',

    # Arrival Weather Columns
    'tavg_arr': 'avg_temp_arr',
    'tmin_arr': 'min_temp_arr',
    'tmax_arr': 'max_temp_arr',
    'prcp_arr': 'precipitation_arr',
    'snow_arr': 'snowfall_arr',
    'wdir_arr': 'wind_direction_arr',
    'wspd_arr': 'wind_speed_arr',
    'pres_arr': 'pressure_arr',
    
    # You could also rename other columns if you wish, for example:
    'Aicraft_age': 'Aircraft_Age'
}

# Apply the renaming to the DataFrame
df_final_renamed = df_final.rename(columns=rename_mapping)
print("Columns of the final merged DataFrame:")
print(df_final_renamed.info())

# %%
missing_values = df_final_renamed.isnull().sum()

# Filter to show only columns that have at least one missing value
missing_values = missing_values[missing_values > 0]

# Calculate the percentage of missing values for those columns
missing_percentage = (missing_values / len(df_final_renamed)) * 100

# Combine the count and percentage into a new DataFrame for a clean report
missing_report = pd.DataFrame({
    'Missing Count': missing_values,
    'Percentage (%)': missing_percentage
})

# Sort the report to show columns with the most missing values first
missing_report = missing_report.sort_values(by='Missing Count', ascending=False)

print("--- Missing Values Report ---")
if not missing_report.empty:
    print(missing_report)
else:
    print("Congratulations! No missing values found in the DataFrame.")
print("---------------------------\n")



# %%


# %%



