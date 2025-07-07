# uv run 5_imbalance_settelment_monthly.py
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import numpy as np
from datetime import datetime, timedelta
import os
pio.kaleido.scope.mathjax = None
# Define model names and paths
models = {
    "Base": "results/base_model/base_200hh/combined_household_data.csv",
    "TOU": "results/tou_model/run_200hh/combined_household_data.csv",
    "MPD": "results/mpd_model/run_200hh/combined_household_data.csv",
    "MPD-50": "results/mpd-50_model/run_200hh/combined_household_data.csv",
    "MPD-50-inc": "results/mpd-50_inc/run_200hh/combined_household_data.csv",
    "MPD-50-inc-4kW": "results/mpd-50_inc_4kW/run_200hh/combined_household_data.csv"
}

# Output folder setup
output_folder = "C:\\Users\\ScheiblS\\Documents\\Repositories\\dyn-grid-tariffs\\results\\a_combined_results"
os.makedirs(output_folder, exist_ok=True)

# Initialize storage dictionaries
schedule = {}
grid_load = {}

# Process each model
for model_name, path in models.items():
    print(f"Processing {model_name}...")

    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"Failed to read {model_name}: {e}")
        continue

    # Ensure column exists
    if "grid_load" not in df.columns:
        print(f"Skipping {model_name}, missing 'grid_load'")
        continue

    # Store grid load
    grid_load[model_name] = df["grid_load"]

    # Generate time series for full year
    start_date = datetime(2021, 1, 1, 0, 0)
    date_times = [start_date + timedelta(minutes=15 * i) for i in range(len(grid_load[model_name]))]

    # Create DataFrame with grid load and time
    net_load_lv_grid_df = pd.DataFrame({'grid_load': grid_load[model_name], 'date_time': date_times})
    net_load_lv_grid_df['day_of_week'] = net_load_lv_grid_df['date_time'].dt.dayofweek
    net_load_lv_grid_df['day_type'] = net_load_lv_grid_df['day_of_week'].apply(lambda x: 'Weekday' if x < 5 else ('Saturday' if x == 5 else 'Sunday'))    # Add month number to DataFrame
    net_load_lv_grid_df['month'] = net_load_lv_grid_df['date_time'].dt.month
    
    # Group data by month
    monthly_dfs = {}
    for month in range(1, 13):
        monthly_dfs[month] = net_load_lv_grid_df[net_load_lv_grid_df['month'] == month]
    
    # Function to filter data by day type
    def separate_by_day_type(month_df):
        return {
            "Weekday": month_df[month_df['day_type'] == 'Weekday']['grid_load'],
            "Saturday": month_df[month_df['day_type'] == 'Saturday']['grid_load'],
            "Sunday": month_df[month_df['day_type'] == 'Sunday']['grid_load']
        }
        
    # Split monthly data by day type
    monthly_data = {}
    for month in range(1, 13):
        monthly_data[month] = separate_by_day_type(monthly_dfs[month])    # Function to calculate representative daily profile
    def calculate_representative_day(data):
        daily_data = data.values.reshape(-1, 96)
        return pd.DataFrame(daily_data.mean(axis=0), columns=['grid_load'])

    # Compute representative days for each month and day type
    representative_days = {}
    for month, day_types in monthly_data.items():
        representative_days[month] = {}
        for day_type, data in day_types.items():
            if not data.empty:  # Check if there's data for this day type in this month
                representative_days[month][day_type] = calculate_representative_day(data)
            else:
                # Handle empty data case (might happen for some months without specific day types)
                print(f"Warning: No data for {day_type} in month {month}")
                # Use a fallback (e.g., create a profile of zeros or use a neighboring month)
                representative_days[month][day_type] = pd.DataFrame([0] * 96, columns=['grid_load'])

    # Construct full-year schedule
    full_year_schedule = pd.DataFrame(index=net_load_lv_grid_df.index, columns=["scheduled_load"])
    
    # Fill schedule based on monthly patterns
    for i, row in net_load_lv_grid_df.iterrows():
        month = row["date_time"].month
        day_type = row["day_type"]
        
        # Calculate time of day index (0-95 for the 96 15-min intervals in a day)
        time_of_day_idx = (row["date_time"].hour * 4) + (row["date_time"].minute // 15)
        
        # Assign the appropriate profile value
        full_year_schedule.at[i, "scheduled_load"] = representative_days[month][day_type].iloc[time_of_day_idx].values[0]

    # Store schedule for each model
    schedule[model_name] = full_year_schedule
# uv run 5_imbalance_settelment_monthly.py
#***************************************************************************************************
# calulate imbalance settlement costs
# Load imbalance price data
# imbalance_path = "Imbalance_202101010000-202201010000_utc.csv"
# time_col = "Imbalance settlement period (UTC)"
# imbalance_path = "Imbalance_202101010000-202201010000.csv"
imbalance_path = "Imbalance_202101010000-202201010000v2.csv"
imbalance_df = pd.read_csv(imbalance_path, sep=';')

# Define column names
price_col = "+ Imbalance Price [EUR/MWh] - SCA|AT"
# time_col = "Imbalance settlement period (CET/CEST)"
imbalance_price_mwh = imbalance_df[price_col]
imbalance_price_mwh = imbalance_price_mwh.to_frame()

# # Extract relevant data once
# imbalance_price_mwh = imbalance_df[[time_col, price_col]].drop_duplicates(subset=[time_col])

# # Check for missing values
# missing_values = imbalance_price_mwh[price_col].isna().sum()
# print(f"Missing values in imbalance price: {missing_values}")

# # Fill missing values with mean
# imbalance_price_mwh[price_col].fillna(imbalance_price_mwh[price_col].mean(), inplace=True)

# Remove extreme outliers using IQR (ensuring realistic pricing)
Q1 = imbalance_price_mwh[price_col].quantile(0.25)
Q3 = imbalance_price_mwh[price_col].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 3 * IQR
upper_bound = Q3 + 3 * IQR

imbalance_price_mwh.loc[
    (imbalance_price_mwh[price_col] < lower_bound) | (imbalance_price_mwh[price_col] > upper_bound), 
    price_col
] = imbalance_price_mwh[price_col].mean()

print(f"Final shape of imbalance price data: {imbalance_price_mwh.shape}")

# Convert EUR/MWh to €/kWh
imbalance_price = imbalance_price_mwh[price_col] / 1000

# imbalance_cost = {}
# difference_energy = {}
# for model_name in models.keys():
#     imbalance_before = ((grid_load[model_name] - schedule[model_name]["scheduled_load"]) / 4) * imbalance_price
#     print(f"difference:{((grid_load[model_name] - schedule[model_name]["scheduled_load"]) / 4).sum()}")
#     imbalance_cost[model_name] = (imbalance_before).sum()
#     print(f"tell index: {grid_load[model_name].index}")
#     print(f"tell index: {schedule[model_name]["scheduled_load"].index}")
#     print(grid_load[model_name].describe())  
#     print(schedule[model_name]["scheduled_load"].describe())
#     # print(grid_load[model_name].index.equals(imbalance_price.index))  # Should be True
#     # print(f"Imbalance cost for {model_name}: €{imbalance_cost[model_name]:,.2f}")
#     # print(f"sum of residual laod {grid_load[model_name]}")
#     # print(f"sum of scheduled load: {schedule[model_name]['scheduled_load'].sum()}")

imbalance_cost = {}
difference_energy = {}
for model_name in models.keys():
    schedule_series_kwh = schedule[model_name]["scheduled_load"] / 4
    grid_load_series_kwh = grid_load[model_name] / 4
    dif = (grid_load_series_kwh - schedule_series_kwh) 
    dif_sum = dif.sum()
    imbalance = dif * imbalance_price
    total_imbalance = imbalance.sum()
    imbalance_cost[model_name] = total_imbalance
    
# Print results
print("\nImbalance Costs Per Model:")
for model, cost in imbalance_cost.items():
    print(f"{model}: €{cost:,.2f}")

# uv run 5_imbalance_settelment_monthly.py
#***************************************************************************************************
# visualize schedule vs actual load and imbalance settlement costs
# Define output folder
output_folder = "C:\\Users\\ScheiblS\\Documents\\Repositories\\dyn-grid-tariffs\\results\\a_combined_results"
os.makedirs(output_folder, exist_ok=True)

for model_name in models.keys():
    print(f"Processing visualization for {model_name}...")

    fig = go.Figure()

    # Add schedule (Red Line)
    fig.add_trace(go.Scatter(
        x=schedule[model_name].index,
        y=schedule[model_name]["scheduled_load"],
        mode="lines",
        name="Schedule",
        line=dict(color="red", width=2)
    ))

    # Add residual load (Blue Dashed Line)
    fig.add_trace(go.Scatter(
        x=grid_load[model_name].index,
        y=grid_load[model_name],
        mode="lines",
        name="Residual Load",
        line=dict(color="blue", width=2, dash="dash")
    ))    # Add imbalance settlement cost (Secondary Y-Axis) - using original EUR/MWh values
    fig.add_trace(go.Scatter(
        x=imbalance_price.index,
        y=imbalance_price,
        mode="lines",
        name="Imbalance Settlement Cost",
        line=dict(color="green", width=2),
        yaxis="y2"
    ))

    # Configure layout for dual y-axes
    fig.update_layout(
        title=f"Schedule vs Actual Load - {model_name}",
        xaxis_title="Time Interval",
        yaxis=dict(title="Load [kW]", showgrid=True),
        yaxis2=dict(title="Imbalance Settlement Cost [EUR/MWh]", overlaying="y", side="right", showgrid=False),
        template="plotly_white",
        font=dict(color="black"),
        showlegend=True,
        height=600
    )

    # Save plot as HTML
    html_file_path = os.path.join(output_folder, f"imbalance_schedule_vs_actual_{model_name}.html")
    fig.write_html(html_file_path)

    print(f"Plot saved as HTML: {html_file_path}")

#****************************************************************************************************
# uv run 5_imbalance_settelment_monthly.py
# scheduled load visualization
import os
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

# Define output folder
output_folder = "C:\\Users\\ScheiblS\\Documents\\Repositories\\dyn-grid-tariffs\\results\\a_combined_results"
os.makedirs(output_folder, exist_ok=True)

for model_name in models.keys():
    print(f"Processing visualization for {model_name}...")

    fig = go.Figure()

    # Add schedule (Red Line)
    fig.add_trace(go.Scatter(
        x=schedule[model_name].index,
        y=schedule[model_name]["scheduled_load"],
        mode="lines",
        name="Schedule",
        line=dict(color="red", width=0.2)
    ))

    # Add monthly vertical dividers
    for month in range(2, 13):
        month_start = pd.Timestamp(f"2021-{month:02d}-01 00:00")
        fig.add_vline(x=month_start, line_width=1, line_dash="dash", line_color="gray", opacity=0.5)

    # Configure layout
    fig.update_layout(
        xaxis_title="Time Interval",
        yaxis=dict(title="Load [kW]", showgrid=True),
        xaxis=dict(
            tickformat="%b",  # Format x-axis to show months (Jan, Feb, etc.)
            tickangle=90,    # Rotate labels for better readability
            showgrid=True
        ),
        template="plotly_white",
        font=dict(color="black"),
        showlegend=False,
        height=400,
        margin=dict(l=20, r=20, t=20, b=20)
    )

    # Save plot as HTML
    html_file_path = os.path.join(output_folder, f"scheduled_load_{model_name}.html")
    fig.write_html(html_file_path)

    # Save plot as PDF
    pdf_file_path = os.path.join(output_folder, f"scheduled_load_{model_name}.pdf")
    fig.write_image(pdf_file_path, format="pdf")

    print(f"Plot saved as:\n- HTML: {html_file_path}\n- PDF: {pdf_file_path}")

#****************************************************************************************************
print("All visualizations saved successfully.")
