# uv run 1_extraction.py
import pandas as pd
import iesopt
import os

# model = 'base_model'
# run_name = 'base_200hh'
# model = 'tou_model'
# run_name = 'run_200hh'
# model = 'mpd-50_model'
# run_name = 'run_200hh'
# model = 'mpd-50_inc'
# run_name = 'run_200hh'
model = 'mpd-50_inc_4kW'
run_name = 'run_200hh'

# Initialize path to CSV folder
csv_path = f'results/{model}/{run_name}'

# Create the directory if it doesn't exist
os.makedirs(csv_path, exist_ok=True)

# Specify the start index (default to 1, but can be adjusted to resume processing)
start_index = 1 # <----- start with 1

# Loop through household files from start_index to 200
for i in range(start_index, 201):  # Ends with 201
    try:
        # Construct the filename dynamically
        file_name = f"results/{model}/household_{i}.iesopt.result.jld2"
        
        # Load results from the specified file
        results = iesopt.Results(file=file_name)
        
        # Extract time series data
        timeseries_data = results.to_pandas(
            filter=lambda c, t, f: (
                c in [
                    "demand", "connection_point", "el_market", 
                    "pv_generation", "pv_surplus", "ev_unit", 
                    "heat_unit", "water_unit"
                ]
                and t == "exp"
            ),
            orientation="wide"
        )
        
        timeseries_data_var = results.to_pandas(
            filter=lambda c, t, f: (
                c in [
                    "bat_charge", "bat_discharge"
                ]
                and t == "var"
            ),
            orientation="wide"
        )

        # Flatten multi-index into single column names if necessary for both DataFrames
        for df in [timeseries_data, timeseries_data_var]:
            if isinstance(df.columns, pd.MultiIndex):
                single_column_index = ['_'.join(col).strip() for col in df.columns.to_flat_index()]
                df.columns = single_column_index

        # Rename columns to include the component name and row index (keep lowercase)
        timeseries_data.columns = [f"{col}_{i}" for col in timeseries_data.columns]
        timeseries_data_var.columns = [f"{col}_{i}" for col in timeseries_data_var.columns]

        # Combine the two DataFrames horizontally (add timeseries_data_var as additional columns)
        combined_timeseries_data = pd.concat([timeseries_data, timeseries_data_var], axis=1)

        # Sort combined columns alphabetically
        combined_timeseries_data = combined_timeseries_data.sort_index(axis=1)

        # Save the combined time series DataFrame to a CSV file
        csv_file = f"{csv_path}/household_{i}.csv"
        
        # Check if a file already exists
        if os.path.exists(csv_file):
            print(f"Replacing existing CSV file: {csv_file}")
        
        combined_timeseries_data.to_csv(csv_file, index=False)
        print(f"Processed combined time series data for household {i} successfully")

        # Extract the objective value and grid cost
        objective_value = results.objectives["total_cost"]  # Fetch the objective value
        grid_cost = results.components["grid_tariff"].obj.cost  # Fetch the grid cost

        # Create a single DataFrame containing both values
        single_values_df = pd.DataFrame({
            "objective_value": [objective_value],
            "grid_cost": [grid_cost]
        })

        # Save the combined values to a single CSV file
        single_values_file = f"{csv_path}/single_values_{i}.csv"
        if os.path.exists(single_values_file):
            print(f"Replacing existing single values CSV file: {single_values_file}")

        single_values_df.to_csv(single_values_file, index=False)  # Save the combined DataFrame
        print(f"Processed single values (objective and grid cost) for household {i} successfully")

    except Exception as e:
        # Handle errors (e.g., file not found or processing issues)
        print(f"Error processing household {i}: {e}")

# #**************************************************************************************************************
# Load the results for a specific household (e.g., household 1) to check the components
# import iesopt
# file_name = f"results/base_model/household_1.iesopt.result.jld2"
# results = iesopt.Results(file=file_name)
# # Convert to pandas DataFrame and save as CSV
# df_household = results.to_pandas(orientation='wide') 
# df_household.to_csv('results/base_model/a_household.csv', index=False)
# # Get list of components in model
# print(results._components)

# #***************************************************************************************************************
# check for whole dataset
# import iesopt
# import pandas as pd

# # Load the household results
# file_name = f"results/base_model/household_12.iesopt.result.jld2"
# results = iesopt.Results(file=file_name)

# # Convert to pandas DataFrame and save as CSV
# df_household = results.to_pandas(orientation='wide') 
# df_household.to_csv('results/base_model/a_household.csv', index=False)

# print("CSV file has been saved successfully!")