# uv run main.py
import pandas as pd
import iesopt

# Load the CSV file
file_path = r"C:\Users\ScheiblS\Documents\Repositories\dyn-grid-tariffs\files\household_summary.csv"
data = pd.read_csv(file_path)

# Set the starting index to resume the loop
start_index = 150 # <----- start with 0
endswith = len(data)  # Set to the length of the DataFrame
# Iterate through rows to solve the model for each row
for i in range(start_index, endswith):  # Start from the specified index 
    # Extract parameters for the current row
    demand = data.iloc[i, 2]  # Column 2 (zero-based index for demand)
    pv_size = data.iloc[i, 8]  # Column 8 (PV size)
    pv_enable = bool(data.iloc[i, 6])  # Column 6 (Convert PV enable to boolean)
    pv_param = data.iloc[i, 7]  # Column 7 (PV generation)
    ev_enable = bool(data.iloc[i, 3])  # Column 3 (Convert EV enable to boolean)
    ev_available = data.iloc[i, 4]  # Column 4 (EV availability)
    ev_distance_param = data.iloc[i, 5]  # Column 5 (EV distance)
    bat_enable = bool(data.iloc[i, 9])  # Column 9 (Convert battery enable to boolean)
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
    name = data.iloc[i, 0]  # Column 0 (Name)

    # Run the optimization model with the extracted parameters
    model = iesopt.run(
        "cap_sub.iesopt.yaml",
        parameters={
            "demand": demand,
            "pv_size": pv_size,
            "pv_enable": pv_enable,
            "pv_param": pv_param,
            "ev_enable": ev_enable,
            "ev_available": ev_available,
            "ev_distance_param": ev_distance_param,
            "bat_enable": bat_enable,
            "ev_capacity": ev_capacity,
            "ev_consumption": ev_consumption,
            "heat_enable": heat_enable,
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
            "name": name
        }
    )

    print("-" * 50)
    print(f"Household count: {i + 1}")
#****************************************************************************************************************
# # Extract time series data
# timeseries_data = model.results.to_pandas(
#     filter=lambda c, t, f: (
#         c in [
#             "demand", "connection_point", "el_market", 
#             "pv_generation", "pv_surplus", "ev_unit", 
#             "heat_unit", "water_unit"
#         ]
#         and t == "exp"
#     ),
#     orientation="wide"
# )

# timeseries_data_var = model.results.to_pandas(
#     filter=lambda c, t, f: (
#         c in [
#             "bat_charge", "bat_discharge",
#         ]
#         and t == "var"
#     ),
#     orientation="wide"
# )

# # Flatten multi-index into single column names if necessary for both DataFrames
# for df in [timeseries_data, timeseries_data_var]:
#     if isinstance(df.columns, pd.MultiIndex):
#         single_column_index = ['_'.join(col).strip() for col in df.columns.to_flat_index()]
#         df.columns = single_column_index

# # # Rename columns to include the component name and row index (keep lowercase)
# # timeseries_data.columns = [f"{col}_{i}" for col in timeseries_data.columns]
# # timeseries_data_var.columns = [f"{col}_{i}" for col in timeseries_data_var.columns]

# # Combine the two DataFrames horizontally (add timeseries_data_var as additional columns)
# combined_timeseries_data = pd.concat([timeseries_data, timeseries_data_var], axis=1)

# # Sort combined columns alphabetically
# combined_timeseries_data = combined_timeseries_data.sort_index(axis=1)
# combined_timeseries_data.to_csv('df_debug_household.csv', index=False)
# #*****************************************************************************************************************
# #combined graph with all powers
# import plotly.graph_objects as go

# # Create the figure
# fig = go.Figure()

# # Add demand with light orange fill
# fig.add_trace(go.Scatter(
#     x=combined_timeseries_data.index, y=combined_timeseries_data['demand_exp_value'],
#     mode='lines', name='Demand', line=dict(color='orange'),
#     fill='tozeroy', fillcolor='rgba(255, 165, 0, 0.3)', stackgroup='one'
# ))

# # Add space heating with reddish-orange fill
# fig.add_trace(go.Scatter(
#     x=combined_timeseries_data.index, y=combined_timeseries_data['heat_unit_exp_in_electricity'],
#     mode='lines', name='Space heating', line=dict(color='violet'),
#     fill='tonexty', fillcolor='rgba(138, 43, 226, 0.3)', stackgroup='one'
# ))

# # Add domestic hot water with violet fill
# fig.add_trace(go.Scatter(
#     x=combined_timeseries_data.index, y=combined_timeseries_data['water_unit_exp_in_electricity'],
#     mode='lines', name='DHW', line=dict(color='purple'),
#     fill='tonexty', fillcolor='rgba(255, 99, 71, 0.3)', stackgroup='one'
# ))

# # Add EV charging with red fill
# fig.add_trace(go.Scatter(
#     x=combined_timeseries_data.index, y=combined_timeseries_data['ev_unit_exp_out_electricity'],
#     mode='lines', name='BEV charging', line=dict(color='red'),
#     fill='tonexty', fillcolor='rgba(220, 20, 60, 0.3)', stackgroup='one'
# ))

# # # Add BESS charging
# # fig.add_trace(go.Scatter(
# #     x=combined_timeseries_data.index, y=combined_timeseries_data["bat_charge_var_flow"],
# #     mode="none", name="BESS charging",
# #     fill="tonexty",  fillcolor="rgba(173, 216, 230, 0.3)",  # Light blue with 30% opacity
# #     fillpattern=dict(shape="."), stackgroup='one'
# # ))

# # # Add BESS discharging
# # fig.add_trace(go.Scatter(
# #     x=combined_timeseries_data.index, y=combined_timeseries_data["bat_discharge_var_flow"],
# #     mode="none", name="BESS discharging",
# #     fill="tonexty", fillcolor="rgba(144, 238, 144, 0.3)",  # Light green with 30% opacity
# #     fillpattern=dict(shape="x"), stackgroup='one'  # Correct way to add a pattern
# # ))

# # # Add PV generation with yellow fill for Schnittmenge
# # fig.add_trace(go.Scatter(
# #     x=combined_timeseries_data.index, y=combined_timeseries_data['pv_generation_exp_value'],
# #     mode='lines', name='PV generation',
# #     fill='tozeroy', fillcolor='rgba(255, 255, 0, 0.3)',
# #     line=dict(color='yellow')
# # ))

# # Add net load on low-voltage grid with a finer dashed line
# fig.add_trace(go.Scatter(
#     x=combined_timeseries_data.index, y=combined_timeseries_data['el_market_exp_value'],
#     mode='lines', name='Net load',
#     line=dict(color='blue', dash='dot')  # Change 'dash' to 'dot' or another style
# ))

# # Customize the layout of the plot
# fig.update_layout(
#     title='Load summary',
#     xaxis_title='15-minute intervals',
#     yaxis_title='Power in kW',
#     template='plotly_white',
#     legend_title='Legend'
# )

#  # Save the interactive plot as an HTML file
# html_file_path = f'a_load_summary.html'
# fig.write_html(html_file_path)

# print(f"Plot for load summary saved as HTML at {html_file_path}")
# #****************************************************************************************************************
# # # check for whole dataset
# # import iesopt
# # import pandas as pd

# # # Load the household results
# # # Convert to pandas DataFrame and save as CSV
# # df_household = model.results.to_pandas(orientation='wide')
# # df_household.to_csv('debug_household.csv', index=False)

# # print("CSV file has been saved successfully!")
# #****************************************************************************************************************