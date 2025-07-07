# uv run 2_fusion.py
import pandas as pd
import os

# Define the run name and file paths
model = 'base_model'
run_name = 'base_200hh'
# model = 'tou_model'
# run_name = 'run_200hh'
# model = 'mpd-50_model'
# run_name = 'run_200hh'
# model = 'mpd-50_inc'
# run_name = 'run_200hh'
# model = 'mpd-50_inc_4kW'
# run_name = 'run_200hh'

csv_folder_path = f"results/{model}/{run_name}"  # Folder containing the CSV files
output_file_path = f"results/{model}/{run_name}/combined_household_data.csv"  # Output combined CSV file

# Remove the output file if it already exists (to start fresh)
if os.path.exists(output_file_path):
    os.remove(output_file_path)

# Initialize an empty DataFrame for combining column-wise data
combined_timeseries = pd.DataFrame()

start_index = 1  # Starting index 1
end_index = 201 # Ending index 201

pv_min = start_index  # Minimum index for individual net load calculation = 1
pv_max = 81  # Maximum index for individual net load calculation = 81

# Combine only household_1.csv to household_200.csv
for i in range(start_index, end_index):  # Loop through numbers 1 to 200
    try:
        # Construct the file name dynamically
        file_name = f"household_{i}.csv"
        file_path = os.path.join(csv_folder_path, file_name)
        
        # Check if the file exists
        if os.path.exists(file_path):
            # Read the individual CSV file
            household_data = pd.read_csv(file_path)

            # Combine data column-wise (add columns side-by-side)
            if combined_timeseries.empty:
                # Use the first file to initialize combined_timeseries
                combined_timeseries = household_data
            else:
                # Concatenate horizontally (adding columns side-by-side)
                combined_timeseries = pd.concat([combined_timeseries, household_data], axis=1)

            print(f"Added {file_name} as columns")
        else:
            print(f"File {file_name} does not exist. Skipping.")
    except Exception as e:
        print(f"Error processing file {file_name}: {e}")

# Save the combined data to the output file
if not combined_timeseries.empty:  # Ensure there is data to save
    combined_timeseries.to_csv(output_file_path, index=False)
    print(f"Combined data saved to {output_file_path}")
else:
    print("No data was combined. The combined file was not created.")
# uv run 2_fusion.py
# *************************************************************************************************
# Process the combined CSV file in chunks to calculate new columns
chunk_size = 10000  # Adjust chunk size based on memory limits

try:
    chunked_calculations = []
    for chunk in pd.read_csv(output_file_path, chunksize=chunk_size):
        # Calculate aggregated sums
        connection_point_sum = chunk.filter(like="connection_point_exp_value").sum(axis=1)
        demand_sum = chunk.filter(like="demand_exp_value").sum(axis=1)
        grid_load_sum = chunk.filter(like="el_market_exp_value").sum(axis=1)
        ev_unit = chunk.filter(like="ev_unit_exp_in_electricity").sum(axis=1)
        heat_unit = chunk.filter(like="heat_unit_exp_in_electricity").sum(axis=1)
        water_unit = chunk.filter(like="water_unit_exp_in_electricity").sum(axis=1)
        pv_sum = chunk.filter(like="pv_generation_exp_value").sum(axis=1)
        pv_surplus = chunk.filter(like="pv_surplus_exp_value").sum(axis=1)
        bat_charge = chunk.filter(like="bat_charge_var_flow").sum(axis=1)
        bat_discharge = chunk.filter(like="bat_discharge_var_flow").sum(axis=1)
        # Calculate household-level net load for each of the 200 households
        for i in range(pv_min, pv_max):
            chunk[f"net_load_hh_{i}"] = chunk[f"el_market_exp_value_{i}"] - chunk[f"pv_surplus_exp_value_{i}"]

        # Calculate additional time series
        space_heating_cum = demand_sum + heat_unit
        dhw_cum = space_heating_cum + water_unit
        ev_cum = dhw_cum + ev_unit
        bat_cum = ev_cum + bat_charge
        bat_dis_cum = bat_cum - bat_discharge
        net_load_lv_grid = grid_load_sum - pv_surplus
        pv_surplus_lv_grid = net_load_lv_grid[net_load_lv_grid < 0]
        net_load_lv_grid_abs = net_load_lv_grid.abs()

        # Insert general aggregated values
        chunk.insert(0, "demand", demand_sum)
        chunk.insert(1, "grid_load", grid_load_sum)
        chunk.insert(2, "connection_point", connection_point_sum)
        chunk.insert(3, "pv_generation", pv_sum)
        chunk.insert(4, "ev_unit", ev_unit)
        chunk.insert(5, "heat_unit", heat_unit)
        chunk.insert(6, "water_unit", water_unit)
        chunk.insert(7, "space_heating_cum", space_heating_cum)
        chunk.insert(8, "dhw_cum", dhw_cum)
        chunk.insert(9, "ev_cum", ev_cum)
        chunk.insert(10, "pv_surplus", pv_surplus)
        chunk.insert(11, "pv_surplus_lv_grid", pv_surplus_lv_grid)
        chunk.insert(12, "net_load_lv_grid", net_load_lv_grid)
        chunk.insert(13, "net_load_lv_grid_abs", net_load_lv_grid_abs)
        chunk.insert(14, "bat_charge", bat_charge)
        chunk.insert(15, "bat_cum", bat_cum)
        chunk.insert(16, "bat_discharge", bat_discharge)
        chunk.insert(17, "bat_dis_cum", bat_dis_cum)

        # Append processed chunk to the list
        chunked_calculations.append(chunk)

    # Concatenate all processed chunks and save the final output
    final_combined = pd.concat(chunked_calculations, ignore_index=True)
    final_combined.to_csv(output_file_path, index=False)
    print(f"Final combined file with calculated columns saved to {output_file_path}")
    print(f"The file has {final_combined.shape[0]} rows and {final_combined.shape[1]} columns.")

except Exception as e:
    print(f"Error while calculating additional columns: {e}")
# uv run 2_fusion.py
# *************************************************************************************************
# Combine the objective value and grid cost from single_values_{i}.csv files
combined_objective_file = f"{csv_folder_path}/combined_objective_value.csv"
combined_grid_cost_file = f"{csv_folder_path}/combined_grid_cost.csv"

# Lists to hold aggregated values
objective_values = []  # Aggregated objective values
grid_costs = []  # Aggregated grid costs

# Loop through the single_values_{i}.csv files
for i in range(1, 201):  # Adjust range as needed
    try:
        # Construct the file path dynamically
        file_name = f"single_values_{i}.csv"
        file_path = os.path.join(csv_folder_path, file_name)

        # Check if the file exists
        if os.path.exists(file_path):
            # Read the single_values_{i}.csv file
            single_values = pd.read_csv(file_path)

            # Extract the objective value and grid cost
            objective_value = single_values["objective_value"].iloc[0]
            grid_cost = single_values["grid_cost"].iloc[0]

            # Append the data to their respective lists
            objective_values.append({"household": i, "objective_value": objective_value})
            grid_costs.append({"household": i, "grid_cost": grid_cost})

            print(f"Processed {file_name} successfully")
        else:
            print(f"File {file_name} does not exist. Skipping.")
    except Exception as e:
        print(f"Error processing file {file_name}: {e}")

# Save the aggregated objective values to a single CSV file
if objective_values:  # Ensure there are values to save
    combined_objective_df = pd.DataFrame(objective_values)

    # Calculate the total objective value
    total_objective_value = combined_objective_df["objective_value"].sum()
    combined_objective_df = pd.concat(
        [combined_objective_df, pd.DataFrame([{"household": "Total", "objective_value": total_objective_value}])],
        ignore_index=True
    )

    # Save to combined CSV
    combined_objective_df.to_csv(combined_objective_file, index=False)
    print(f"Combined objective values saved to {combined_objective_file}")

# Save the aggregated grid costs to a single CSV file
if grid_costs:  # Ensure there are values to save
    combined_grid_cost_df = pd.DataFrame(grid_costs)

    # Calculate the total grid cost
    total_grid_cost = combined_grid_cost_df["grid_cost"].sum()
    combined_grid_cost_df = pd.concat(
        [combined_grid_cost_df, pd.DataFrame([{"household": "Total", "grid_cost": total_grid_cost}])],
        ignore_index=True
    )

    # Save to combined CSV
    combined_grid_cost_df.to_csv(combined_grid_cost_file, index=False)
    print(f"Combined grid costs saved to {combined_grid_cost_file}")

# If neither values were found
if not objective_values and not grid_costs:
    print("No data was processed. No combined files were created.")
# uv run 2_fusion.py
# *************************************************************************************************
