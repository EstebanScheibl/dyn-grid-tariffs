import pandas as pd
import iesopt
# When adding additional components ensure to filter it in the lambda filter.
# Make sure summary file state is enabled and other not used components are disabled.
#******************************************************************************************************************************************
# Define the run name
run_name = 'v1'

# Load the CSV file
file_path = r"C:\Users\ScheiblS\Documents\Repositories\dyn-grid-tariffs\files\household_summary_3.csv"
data = pd.read_csv(file_path)

# Initialize variables to store timeseries data
combined_timeseries = pd.DataFrame()  # DataFrame to store all timeseries data
total_objective_value = 0  # To accumulate the sum of objective values
snapshot_stored = False  # Flag to ensure the snapshot is stored only once

# Iterate through rows to solve the model for each row
for i in range(len(data)):  # Use len(data) to include all rows
    # Extract parameters for the current row
    demand = data.iloc[i, 2]  # Column 2 (zero-based index for demand)
    pv_size = data.iloc[i, 8]  # Column 8 (PV size)
    pv_enable = bool(data.iloc[i, 6])  # Column 6 (Convert PV enable to boolean)
    pv_param = data.iloc[i, 7]  # Column 7 (PV generation)
    ev_enable = bool(data.iloc[i, 3])  # Column 3 (Convert EV enable to boolean)
    ev_available = data.iloc[i, 4]  # Column 4 (EV availability)
    ev_distance_param = data.iloc[i, 5]  # Column 5 (EV distance)
    # bat_enable = bool(data.iloc[i, 9])  # Column 9 (Convert battery enable to boolean)
    bat_enable = bool(data.iloc[i, 9])
    ev_capacity = data.iloc[i, 10]  # Column 10 (EV capacity)
    ev_consumption = data.iloc[i, 11]  # Column 11 (EV consumption)
    heat_enable = bool(data.iloc[i, 12])  # Column 12 (Convert heat enable to boolean)
    q_tot = data.iloc[i, 16]  # Column 13 (Total heat demand)
    power_to_c = data.iloc[i, 17]  # Column 14 (Power to cool)
    heat_lb = data.iloc[i, 18]  # Column 15 (Heat load)
    heat_ub = data.iloc[i, 19]  # Column 16 (Heat up)
    heat_technology = data.iloc[i, 20]  # Column 17 (Heat technology)
    water_vol = data.iloc[i, 21]  # Column 18 (Water volume)
    water_tot = data.iloc[i, 23]  # Column 19 (Water total)
    water_technology = data.iloc[i, 24]  # Column 20 (Water technology)
    heat_capacity = data.iloc[i, 25]  # Column 21 (Heat capacity)
    water_capacity = data.iloc[i, 26]  # Column 22 (Water capacity)

    # Run the optimization model with the extracted parameters
    model = iesopt.run(
        "base_model.iesopt.yaml",
        parameters={
            "demand": demand,
            "pv_size": pv_size,
            "pv_enable": pv_enable,  # Ensure this is a boolean
            "pv_param": pv_param,
            "ev_enable": ev_enable,  # Ensure this is a boolean
            "ev_available": ev_available,
            "ev_distance_param": ev_distance_param,
            "bat_enable": bat_enable,  # Ensure this is a boolean
            "ev_capacity": ev_capacity,
            "ev_consumption": ev_consumption,
            "heat_enable": heat_enable,  # Ensure this is a boolean
            "q_tot": q_tot,
            "power_to_c": power_to_c,
            "heat_lb": heat_lb,
            "heat_ub": heat_ub,
            "heat_technology": heat_technology,
            "water_vol": water_vol,
            "water_tot": water_tot,
            "water_technology": water_technology,
            "heat_capacity": heat_capacity,
            "water_capacity": water_capacity,
        }
    )

    # Accumulate the objective value
    total_objective_value += model.objective_value

    # Extract timeseries data for demand, connection_point, and grid_load
    timeseries_data = model.results.to_pandas(
        filter=lambda c, t, f: (
            c in ["demand", "connection_point", "grid_load", "pv_generation", "pv_surplus", "ev_unit", "heat_unit", "water_unit", "pv_surplus"]  # expensive_load, charger
            and t == "exp"  # Ensure fieldtype is "exp"
        ),
        orientation="wide"
    )

    # Flatten multi-index into single column names if necessary
    if isinstance(timeseries_data.columns, pd.MultiIndex):
        single_column_index = ['_'.join(col).strip() for col in timeseries_data.columns.to_flat_index()]
        timeseries_data.columns = single_column_index

    # Rename columns to include the component name and row index (keep lowercase)
    timeseries_data.columns = [f"{col}_{i+1}" for col in timeseries_data.columns]

    # Sort columns alphabetically
    timeseries_data = timeseries_data.sort_index(axis=1)

    # Add the data as new columns to the combined DataFrame
    combined_timeseries = pd.concat([combined_timeseries, timeseries_data], axis=1)

    print("-" * 50)
    print(f"Household count: {i + 1}")

# Combine data by calculating row-wise sums for specific components
demand_sum = combined_timeseries.filter(like="demand_exp_value").sum(axis=1)
grid_load_sum = combined_timeseries.filter(like="grid_load_exp_value").sum(axis=1)
connection_point_sum = combined_timeseries.filter(like="connection_point_exp_value").sum(axis=1)
pv_sum = combined_timeseries.filter(like="pv_generation_exp_value").sum(axis=1)
ev_unit = combined_timeseries.filter(like="ev_unit_exp_out_electricity").sum(axis=1)
heat_unit = combined_timeseries.filter(like="heat_unit_exp_in_electricity").sum(axis=1)
water_unit = combined_timeseries.filter(like="water_unit_exp_in_electricity").sum(axis=1)
pv_surplus = combined_timeseries.filter(like="pv_surplus_exp_value").sum(axis=1)
# expensive_load_sum = combined_timeseries.filter(like="expensive_load_exp_value").sum(axis=1)

# Calculate additional time series
space_heating_cum = demand_sum + heat_unit  # Sum of demand_sum and heat_unit
dhw_cum = space_heating_cum + water_unit  # Sum of space_heating_cum and water_unit
ev_cum = dhw_cum + ev_unit  # Sum of dhw_cum and ev_unit
surplus = pv_surplus * (-1)  # Invert the surplus value
net_load_lv_grid = grid_load_sum - surplus # Calculate net load on the low-voltage grid

# Add the calculated sums as new columns at the beginning of the DataFrame
combined_timeseries.insert(0, "demand", demand_sum)
combined_timeseries.insert(1, "grid_load", grid_load_sum)
combined_timeseries.insert(2, "connection_point", connection_point_sum)
combined_timeseries.insert(3, "pv_generation", pv_sum)
combined_timeseries.insert(4, "ev_unit", ev_unit)
combined_timeseries.insert(5, "heat_unit", heat_unit)
combined_timeseries.insert(6, "water_unit", water_unit)
# combined_timeseries.insert(6, "expensive_load", expensive_load_sum)
combined_timeseries.insert(7, "space_heating_cum", space_heating_cum)  # Add space_heating_cum
combined_timeseries.insert(8, "dhw_cum", dhw_cum)  # Add dhw_cum
combined_timeseries.insert(9, "ev_cum", ev_cum)  # Add ev_cum
combined_timeseries.insert(10, "pv_surplus", surplus)  # Add surplus
combined_timeseries.insert(11, "net_load_lv_grid", net_load_lv_grid)  # Add net load on low-voltage grid

# Save the updated combined timeseries data to a CSV file
timeseries_output_file = f'results\\timeseries_{run_name}.csv'
combined_timeseries.to_csv(timeseries_output_file, index=False)

print("-" * 50)
print(model.results._components)  # Get list of components in model
print(f"Timeseries data saved to {timeseries_output_file}")
print(f"Total objective value: {total_objective_value}")

# ******************************************************************************************************************************************
import plotly.graph_objects as go

# Check if the 'net_load_lv_grid' column exists
if 'net_load_lv_grid' in combined_timeseries.columns:
    # Calculate the metrics
    max_load = round(combined_timeseries['net_load_lv_grid'].max(), 2) / 1000  # Convert to MW
    min_load = round(combined_timeseries['net_load_lv_grid'].min(), 2) / 1000  # Convert to MW
    integral = round(combined_timeseries['net_load_lv_grid'].sum() / 4) / 1000  # Convert to MW

    # Output the results
    print(f"Max Load: {max_load} MW")
    print(f"Min Load: {min_load} MW")
    print(f"Integral (Sum / 4, in MW): {integral} MW")

    # Create a DataFrame with the results
    results_df = pd.DataFrame({
        'Metric': ['Max Load', 'Min Load', 'Integral (Sum / 4)'],
        'Value': [max_load, min_load, integral]
    })

    # Save the DataFrame to a CSV file
    results_file_path = f'results\\net_load_metrics_{run_name}.csv'
    results_df.to_csv(results_file_path, index=False)

    print(f"Results saved to {results_file_path}")

    # Create the plot for 'net_load_lv_grid'
    fig = go.Figure()

    # Add 'net_load_lv_grid' data to the plot
    fig.add_trace(go.Scatter(
        x=combined_timeseries.index, 
        y=combined_timeseries['net_load_lv_grid'], 
        mode='lines', 
        name='Net load on low-voltage grid', 
        line=dict(color='red')
    ))

    # Add metrics as annotations
    metrics_text = (
        f"Max Load: {max_load:.2f} MW<br>"
        f"Min Load: {min_load:.2f} MW<br>"
        f"Integral: {integral:.2f} MWh"
    )
    fig.add_annotation(
        text=metrics_text,
        xref="paper",
        yref="paper",
        x=1,
        y=1,
        showarrow=False,
        align="left",
        font=dict(size=12, color="black"),
        bgcolor="lightyellow",
        bordercolor="black"
    )

    # Customize the layout of the plot
    fig.update_layout(
        title='Net load LV grid',
        xaxis_title='15-minute intervals',
        yaxis_title='Power in kW',
        template='plotly_white',
        legend_title='Legend'
    )

    # Save the interactive plot as an HTML file
    html_file_path = f'results\\net_load_{run_name}.html'
    fig.write_html(html_file_path)

    print(f"Plot for net load on low-voltage grid saved as HTML at {html_file_path}")


#******************************************************************************************************************************************
#LDC load duration curve

# Create a new figure for the Load Duration Curve (LDC)
fig_ldc = go.Figure()

# Sort the 'grid_load' column in descending order for LDC
if 'grid_load' in combined_timeseries.columns:
    sorted_grid_load = combined_timeseries['grid_load'].sort_values(ascending=False).reset_index(drop=True)
    fig_ldc.add_trace(go.Scatter(x=sorted_grid_load.index, y=sorted_grid_load, mode='lines', name='Load Duration Curve', line=dict(color='blue')))

# Customize the layout for the LDC plot
fig_ldc.update_layout(
    title='Load Duration Curve',
    xaxis_title='Sorted Time Intervals (Highest to Lowest)',
    yaxis_title='Grid Load (kW)',
    template='plotly_white',
    legend_title='Legend'
)

# Save the Load Duration Curve as an HTML file
html_file_path_ldc = f'results\\load_duration_curve_{run_name}.html'
fig_ldc.write_html(html_file_path_ldc)

print(f"Load Duration Curve saved as HTML at {html_file_path_ldc}")

#******************************************************************************************************************************************
# code from this section might be required some time in the future

# # extract single values operation data (marginal cost totals)
# df = results.to_pandas(filter=lambda c, t, f: not f.endswith("__dual"))
# df_single_values = df[df["snapshot"].str.match("None", na = True)]
# df_single_values.to_csv(f'results\\single_value_{run_name}.csv')

#Save the results to a CSV file in wide format
results_output_file = f'C:\\Users\\ScheiblS\\Documents\\Repositories\\dyn-grid-tariffs\\results\\results_raw_{run_name}.csv'
model.results.to_pandas(
    field_types=["exp", "var"], orientation="wide"
).to_csv(results_output_file, index=False)

print(f"Results saved to {results_output_file}")

# # Add 'grid_load' data to the plot
# if 'grid_load' in combined_timeseries.columns:
#     fig.add_trace(go.Scatter(x=combined_timeseries.index, y=combined_timeseries['grid_load'], mode='lines', name='Grid load', line=dict(color='red')))

# # Add 'demand' data to the plot
# if 'demand' in combined_timeseries.columns:
#     fig.add_trace(go.Scatter(x=combined_timeseries.index, y=combined_timeseries['demand'], mode='lines', name='Demand', line=dict(color='purple')))

# # Add 'pv' data to the plot if it exists
# if 'pv_generation' in combined_timeseries.columns:
#     fig.add_trace(go.Scatter(x=combined_timeseries.index, y=combined_timeseries['pv_generation'], mode='lines', name='PV', line=dict(color='yellow')))

# # Add 'ev-charger' data to the plot if it exists
# if 'ev_unit' in combined_timeseries.columns:
#     fig.add_trace(go.Scatter(x=combined_timeseries.index, y=combined_timeseries['ev_unit'], mode='lines', name='EV charger', line=dict(color='green')))

# # Add 'expensive_load' data to the plot if it exists
# if 'expensive_load' in combined_timeseries.columns:
#     fig.add_trace(go.Scatter(x=combined_timeseries.index, y=combined_timeseries['expensive_load'], mode='lines', name='Expensive load', line=dict(color='black')))

# # add heat_unit data to the plot if it exists
# if 'heat_unit' in combined_timeseries.columns:
#     fig.add_trace(go.Scatter(x=combined_timeseries.index, y=combined_timeseries['heat_unit'], mode='lines', name='Space heating', line=dict(color='orange')))

# # add water_unit data to the plot if it exists
# if 'water_unit' in combined_timeseries.columns:
#     fig.add_trace(go.Scatter(x=combined_timeseries.index, y=combined_timeseries['water_unit'], mode='lines', name='Domestic hot water', line=dict(color='blue')))
