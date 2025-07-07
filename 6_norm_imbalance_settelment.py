import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import numpy as np
import os
pio.kaleido.scope.mathjax = None
# uv run 5_imbalance_settelment.py
# Normalized imbalance settlement costs for each model
#*************************************************************************************************************************
# Define model names and paths
models = {
    "Base": "results/base_model/base_200hh/combined_household_data.csv",
    "TOU": "results/tou_model/run_200hh/combined_household_data.csv",
    "MPD": "results/mpd_model/run_200hh/combined_household_data.csv",
    "MPD-50": "results/mpd-50_model/run_200hh/combined_household_data.csv",
    "MPD-50-inc": "results/mpd-50_inc/run_200hh/combined_household_data.csv",
    "MPD-50-inc-4kW": "results/mpd-50_inc_4kW/run_200hh/combined_household_data.csv"
}
#****************************************************************************************************************************
# Define colors for models
model_colors = {
    "Base": ("blue", "solid"),
    "TOU": ("red", "solid"),
    "MPD": ("green", "solid"),
    "MPD-50": ("orange", "dash"),
    "MPD-50-inc": ("purple", "dash"),
    "MPD-50-inc-4kW": ("cyan", "dash")
}
# uv run 6_norm_imbalance_settelment.py
# Load imbalance price data
# imbalance_path = "Imbalance_202101010000-202201010000_utc.csv"
imbalance_path = "Imbalance_202101010000-202201010000v2.csv"
# imbalance_path = "Imbalance_202101010000-202201010000.csv"
try:
    imbalance_df = pd.read_csv(imbalance_path, sep=';')
except Exception as e:
    raise RuntimeError(f"Failed to load imbalance price file: {e}")

price_col = "+ Imbalance Price [EUR/MWh] - SCA|AT"
# time_col = "Imbalance settlement period (UTC)"
time_col = "Imbalance settlement period (CET/CEST)"

# Remove duplicate timestamps while keeping other columns
imbalance_df_clean = imbalance_df.drop_duplicates(subset=[time_col])
print(f"Cleaned imbalance data shape: {imbalance_df_clean.shape}")
# Extract price column
imbalance_mhw = imbalance_df_clean[price_col]
imbalance_price = imbalance_mhw / 1000
# print(f"Imbalance price data shape: {imbalance_price.shape}")
# print(f"Imbalance price data head:\n{imbalance_price.head()}")
imbalance_costs = dict()
household_annual_cons = pd.DataFrame()
cost_per_households = pd.DataFrame()
schedule = pd.DataFrame()
residual_load = pd.DataFrame()
variance_hhs = dict()
variance_timestep = dict()	

# Loop over each model/scenario
for model_name, path in models.items():
    print(f"Processing {model_name}...")

    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"Failed to read {model_name}: {e}")
        continue
    # Dictionary to store total costs and annual consumption

    # # Select relevant columns (ensuring they exist in df)
    # cons_data = [f'el_market_exp_value_{i}' for i in range(1, 201) if f'el_market_exp_value_{i}' in df.columns]
    # consumption_data = df[cons_data]
    # # Create a new DataFrame that stores the sum for each row across 200 columns
    # df_summed = consumption_data.sum(axis=0)
    # household_annual_cons_w[model_name] = df_summed / 4

    # Get the relevant columns
    # cols_1_80 = [f'net_load_hh_{i}' for i in range(1, 81) if f'net_load_hh_{i}' in df.columns]
    cols_1_200 = [f'el_market_exp_value_{i}' for i in range(1, 201) if f'el_market_exp_value_{i}' in df.columns]
    combined_cols = cols_1_200

    if not combined_cols:
        print(f"No valid columns in {model_name}")
        continue
    # household_totals = 0
    # normalized_data = 0
    # normalized_mean = 0
    # cost_per_household = 0
    # total_cost= 0
    # imbalance_cost = 0
    # difference_kWh = 0
    # difference = 0
    # schedule_norm_df = 0
    # schedule_norm = 0
    household_total_kW = 0
    household_totals = 0
    
    data = df[combined_cols]
    residual_load[model_name] = data.sum(axis=1)
    variance_hh = data.var(axis=0)
    variance_hhs[model_name] = variance_hh
    # print(f"Varia of household loads: {variance_hh.shape}")
    variance_timesteps = data.var(axis=1)
    variance_timestep[model_name] = variance_timesteps
    # print(f"Va of time steps: {variance_timesteps.shape}")
    # Annual consumption per household
    household_array = data.sum(axis=0)
    household_totals = household_array.values
    # print(household_totals)  # This will return only the numerical values

    # print(f'household_totals:{household_totals.max()}')
    # print(f'household_totals:{household_totals.shape}')
    household_total_kW = household_totals.sum()
    household_annual_cons[model_name] = household_totals / 4
    # Normalize and rebuild schedule
    normalized_data = data.div(household_totals)
    normalized_mean = normalized_data.mean(axis=1)
    schedule[model_name] = normalized_mean.mul(household_total_kW)
    # print(f'normalized_mean:{normalized_mean.shape}')
    # Ensure normalized_mean is (35040,) and household_totals is (200,)
    schedule_norm = normalized_mean.values[:, np.newaxis] * household_totals[np.newaxis, :]
    # Convert to DataFrame
    # print(f'data:{data.head()}')
    # print(f'schedule_norm_df:{schedule_norm_df.head()}') 
    difference = data - schedule_norm
    # print(f'difference:{difference.head()}')
    # print(f'difference_kW:{difference_kW.head()}')
    difference_kWh = difference / 4 
    # print(f'difference_kW_og:{difference_kWh.shape}')
    imbalance_cost = difference_kWh.mul(imbalance_price, axis=0)
    # print(f'imbalance_cost:{imbalance_price.shape}')
    # print(f'imbalance_cost:{imbalance_cost.shape}')
    cost_per_household = imbalance_cost.sum(axis=0)
    cost_per_households[model_name] = cost_per_household
    # print(f'cost_per_household:{cost_per_household.shape}')
    print(f'cost_per_household:{cost_per_household.head()}')
    total_cost = cost_per_household.sum()
    imbalance_costs[model_name] = total_cost

# Print results
print("\nNormalized Imbalance Settlement Costs (€):")
for model, cost in imbalance_costs.items():
    print(f"{model}: €{cost:,.2f}")
# uv run 6_norm_imbalance_settelment.py
#************************************************************************************************************************
# varianz shown
# varianz der zeitschritte

# Plot monthly variance of timesteps for each model
import calendar
import os
import plotly.graph_objects as go
from datetime import datetime

model_colors_wol = {
    "Base": "blue",
    "TOU": "red",
    "MPD": "green",
    "MPD-50": "orange",
    "MPD-50-inc": "purple",
    "MPD-50-inc-4kW": "cyan"
}

# Define output folder
output_folder = "C:\\Users\\ScheiblS\\Documents\\Repositories\\dyn-grid-tariffs\\results\\a_combined_results"
os.makedirs(output_folder, exist_ok=True)

# Define month names
month_names = list(calendar.month_name)[1:13]  # Skip the empty string at index 0
# Print yearly variance for each model
for model_name in variance_timestep.keys():
    yearly_variance = variance_timestep[model_name].mean()  # Compute variance for the whole year
    print(f"Yearly variance of time steps for {model_name}: {yearly_variance:.4f}")
# Loop through each model
for model_name in variance_timestep.keys():
    # Extract the variance data
    variance_data = variance_timestep[model_name]
    
    # Generate timestamps for the year (same as in the main processing logic)
    times = pd.date_range(start='2021-01-01', end='2021-12-31 23:45:00', freq='15min')
    months = times.month
    
    # Create a dataframe with the timestep variances and corresponding months
    timestep_df = pd.DataFrame({'variance': variance_data, 'month': months})
    
    # Group by month and calculate mean variance for each month
    monthly_mean_variance = timestep_df.groupby('month')['variance'].mean()
    
    # Create figure
    fig = go.Figure()
    
    # Add bar chart
    # fig.add_trace(go.Bar(
    #     x=month_names,
    #     y=monthly_mean_variance.values,
    #     marker_color=model_colors_wol[model_name]  # Correct way to set bar colors
    # ))
    fig.add_trace(go.Bar(
        x=month_names,
        y=monthly_mean_variance.values,
        marker_color=model_colors_wol[model_name],  # Correct way to set bar colors
        text=monthly_mean_variance.values,  # Add labels inside bars
        texttemplate='%{text:.2f}',  # Format labels
        textposition='inside',  # Position text inside the bars
        textfont=dict(size=12, color="black")  # Adjust font size and color
    ))
    
    # Update layout
    fig.update_layout(
        yaxis_title="Monthly mean variance",
        template="plotly_white",
        font=dict(color="black"),
        height=300,
        margin=dict(l=40, r=40, t=20, b=40)
    )
    
    # Save as HTML and PDF
    html_file_path = os.path.join(output_folder, f"variance_timesteps_monthly_{model_name}.html")
    pdf_file_path = os.path.join(output_folder, f"variance_timesteps_monthly_{model_name}.pdf")
    
    fig.write_html(html_file_path)
    fig.write_image(pdf_file_path, format='pdf')
    
    print(f"Monthly variance plot for {model_name} saved as HTML: {html_file_path}")
    print(f"Monthly variance plot for {model_name} saved as PDF: {pdf_file_path}")
#uv run 6_norm_imbalance_settelment.py
#*************************************************************************************************************************
import os
import plotly.graph_objects as go

# Define output folder
output_folder = "C:\\Users\\ScheiblS\\Documents\\Repositories\\dyn-grid-tariffs\\results\\a_combined_results"
os.makedirs(output_folder, exist_ok=True)

# Compute yearly mean variance for each model (rounded to 2 decimal places)
yearly_variances = {model_name: round(variance_timestep[model_name].mean(), 2) for model_name in variance_timestep.keys()}

# Create figure
fig = go.Figure()

# Add bar chart with labels
fig.add_trace(go.Bar(
    x=list(yearly_variances.keys()),  # Model names on X-axis
    y=list(yearly_variances.values()),  # Mean variance on Y-axis
    marker_color=[model_colors_wol[model_name] for model_name in yearly_variances.keys()],  # Assign correct colors
    text=[f"{value:.2f}" for value in yearly_variances.values()],  # Text labels with 2 decimal places
    textposition="auto",  # Automatically position text inside bars
    textfont_size=16  # Increase text font size (adjust as needed)
))

# Update layout
fig.update_layout(
    yaxis_title="Mean variance <br> across the year",
    template="plotly_white",
    font=dict(color="black"),
    height=300,
    width=400,
    margin=dict(l=40, r=40, t=20, b=40)
)

# Save as HTML and PDF
html_file_path = os.path.join(output_folder, "variance_timesteps_yearly.html")
pdf_file_path = os.path.join(output_folder, "variance_timesteps_yearly.pdf")

fig.write_html(html_file_path)
fig.write_image(pdf_file_path, format="pdf")

print(f"Yearly variance bar chart saved as HTML: {html_file_path}")
print(f"Yearly variance bar chart saved as PDF: {pdf_file_path}")
#************************************************************************************************************************
#see difference run against imbalance costs
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import os

# Define output folder
output_folder = "C:\\Users\\ScheiblS\\Documents\\Repositories\\dyn-grid-tariffs\\results\\a_combined_results"
os.makedirs(output_folder, exist_ok=True)

models_list = ["Base", "TOU", "MPD", "MPD-50", "MPD-50-inc", "MPD-50-inc-4kW"]
group_colors = {
    "FCG1": "blue", "FCG2": "green", "FCG3": "purple",
    "FCG4": "orange", "FCG5": "cyan", "IFCG": "magenta"
}

consumer_ranges = {
    "FCG1": (0, 20), "FCG2": (20, 50), "FCG3": (50, 80),
    "FCG4": (80, 110), "FCG5": (110, 160), "IFCG": (160, 200)
}

for model_name in models_list:
    fig = make_subplots(rows=1, cols=2, shared_xaxes=True, specs=[[{}, {"secondary_y": True}]])

    # --- Scatter Plot ---
    for group, (start, end) in consumer_ranges.items():
        fig.add_trace(go.Scatter(
            x=household_annual_cons[model_name].iloc[start:end],
            y=cost_per_households[model_name].iloc[start:end],
            mode="lines",
            marker=dict(color=group_colors[group], size=5),
            name=group
        ), row=1, col=1)

    # Add imbalance settlement cost (Secondary Y-Axis)
    fig.add_trace(go.Scatter(
        x=imbalance_mhw.index,
        y=imbalance_mhw,
        mode="lines",
        name="Imbalance Settlement Cost",
        line=dict(color="green", width=2)
    ), row=1, col=2, secondary_y=True)

    # Configure layout
    fig.update_layout(
        title=f"Consumer Group Differences vs Imbalance Costs - {model_name}",
        xaxis_title="Household Annual Consumption",
        yaxis_title="Cost per Household (€)",
        yaxis2=dict(title="Imbalance Settlement Cost [EUR/MWh]", overlaying="y", side="right", showgrid=False),
        template="plotly_white",
        font=dict(color="black"),
        showlegend=True,
        height=600
    )

    # Save plot
    html_file_path = os.path.join(output_folder, f"difference_vs_balance_cost_{model_name}.html")
    fig.write_html(html_file_path)
    print(f"Plot saved as HTML: {html_file_path}")

#************************************************************************************************************************
# see data run against schedule
import plotly.graph_objects as go
import os
# Define output folder
output_folder = "C:\\Users\\ScheiblS\\Documents\\Repositories\\dyn-grid-tariffs\\results\\a_combined_results"
os.makedirs(output_folder, exist_ok=True)
print(type(models))  # Should return <class 'dict'>
# Loop through each model scenario
for model_name, path in models.items():
    print(f"Processing {model_name}...")

    # Create figure
    fig = go.Figure()

    # Add schedule (Red Line)
    fig.add_trace(go.Scatter(
        x=schedule.index,  # Assuming index represents the time interval
        y=schedule[model_name],
        mode="lines",
        name="Schedule",
        line=dict(color="red", width=2)
    ))

    # Add residual load (Blue Dashed Line)
    fig.add_trace(go.Scatter(
        x=residual_load.index,  # Assuming index represents the time interval
        y=residual_load[model_name],
        mode="lines",
        name="Residual Load",
        line=dict(color="blue", width=2, dash="dash")
    ))

    # Add imbalance settlement cost (Secondary Y-Axis)
    fig.add_trace(go.Scatter(
        x=imbalance_mhw.index,  # Assuming index represents the time interval
        y=imbalance_mhw,  # Ensure correct indexing
        mode="lines",
        name="Imbalance Settlement Cost",
        line=dict(color="green", width=2),
        yaxis="y2"  # Assign to second y-axis
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

    # Save individual plot for each model
    html_file_path = os.path.join(output_folder, f"a_schedule_vs_actual_{model_name}.html")
    fig.write_html(html_file_path)

    print(f"Plot saved as HTML: {html_file_path}")
#uv run 5_imbalance_settelment.py
#************************************************************************************************************************
# a bar chart imbalance settlement costs
# Define colors for models
import plotly.graph_objects as go
import os
model_colors_wol = {
    "Base": "blue",
    "TOU": "red",
    "MPD": "green",
    "MPD-50": "orange",
    "MPD-50-inc": "purple",
    "MPD-50-inc-4kW": "cyan"
}

# Create figure
fig = go.Figure()

# Add bars for each model and annotations
for model_name in imbalance_costs.keys():
    value = imbalance_costs[model_name]
    fig.add_trace(go.Bar(
        x=[model_name],  # Model names as categories
        y=[value],  # Cost values
        marker=dict(color=model_colors_wol[model_name]),  # Assign correct color
        name=model_name
    ))

    # Add annotation (place text slightly above the bar)
    fig.add_annotation(
        x=model_name,
        y=value + (max(imbalance_costs.values()) * 0.02),  # Offset for visibility
        text=f"{value:.0f}€/a",
        showarrow=False,
        font=dict(size=12, color="black"),
        align="center"
    )

fig.update_layout(
    yaxis_title="Total imbalance <br> settlement costs [€/a]",
    template="plotly_white",
    font=dict(color="black"),
    showlegend=False,  # Hide redundant legend
    height=300,  # Adjust figure size for better readability
    margin=dict(l=40, r=40, t=10, b=40),  # Reduce empty space
    yaxis=dict(range=[0, max(imbalance_costs.values()) * 1.05], showgrid=False),  # Fixed lower limit at 80K, no grid
    xaxis=dict(showgrid=False)  # Disable gridlines on x-axis as well
)

# Define file paths
output_folder = "C:\\Users\\ScheiblS\\Documents\\Repositories\\dyn-grid-tariffs\\results\\a_combined_results"
os.makedirs(output_folder, exist_ok=True)

pdf_file_path = os.path.join(output_folder, "a_imbalance_bar_chart.pdf")
fig.write_image(pdf_file_path, format='pdf')

html_file_path = os.path.join(output_folder, "a_imbalance_bar_chart.html")
fig.write_html(html_file_path)

print(f"Bar chart saved as PDF: {pdf_file_path}")
print(f"Bar chart saved as HTML: {html_file_path}")
#  uv run 5_imbalance_settelment.py
#************************************************************************************************************************
# imbalance_combined_box_scatter_for all scenarios
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import os
from plotly.subplots import make_subplots

# Define models and colors
models = ["Base", "TOU", "MPD", "MPD-50", "MPD-50-inc", "MPD-50-inc-4kW"]
group_colors = {
    "FCG1": "blue", "FCG2": "green", "FCG3": "purple",
    "FCG4": "orange", "FCG5": "cyan", "IFCG": "magenta"
}

# Ensure output directory exists
output_folder = "results/a_combined_results/"
os.makedirs(output_folder, exist_ok=True)

for model_name in models:
    # Create subplot (Boxplot takes 35% space, Scatter takes 65%)
    fig = make_subplots(
        rows=1, cols=2, 
        column_widths=[0.20, 0.65], 
        shared_yaxes=True,  # Unify the y-axis grid
        subplot_titles=("Distribution of <br> imbalance settlement <br> costs", "Impact of annual consumption on <br> imbalance settlement costs")
    )

    consumer_ranges = {
        "FCG1": (0, 20), "FCG2": (20, 50), "FCG3": (50, 80),
        "FCG4": (80, 110), "FCG5": (110, 160), "IFCG": (160, 200)
    }

    fig.add_trace(go.Box(
        y=cost_per_households[model_name],
        name=model_name,
        boxpoints="outliers",
        jitter=0.3,
        fillcolor="rgba(255,255,255,0)",
        showlegend=False
    ), row=1, col=1)

    # --- Scatter Plot ---
    for group, (start, end) in consumer_ranges.items():
        fig.add_trace(go.Scatter(
            x=household_annual_cons[model_name].iloc[start:end],
            y=cost_per_households[model_name].iloc[start:end],
            mode="markers",
            marker=dict(color=group_colors[group], size=5),
            name=group
        ), row=1, col=2)
    # Find the column with the minimum value
    min_column = household_annual_cons.min().idxmin()
    # Find the row index where the minimum value occurs in that column
    min_row = household_annual_cons[min_column].idxmin()
    # Find the actual minimum value
    min_value = household_annual_cons[min_column].min()
    print(f"Minimum value: {min_value} in column '{min_column}', row {min_row}")
        # Define tick values and labels for readability
    tick_vals = [-8000, -4000, 0, 4000, 8000, 12000, 16000]
    tick_labels = ["-8K", "-4K", "0", "4K", "8K", "12K", "16K"]  # Custom labels for better readability

    # Define tick values and labels for x-axis
    # --- Update Layout ---
    fig.update_layout(
        template="plotly_white",
        font=dict(color="black"),
        xaxis2=dict(
            title="Sum of consumption and feed-in [kWh/a]",
            tickmode="array",
            tickvals=tick_vals,
            ticktext=tick_labels,
        ),
        yaxis=dict(title="Imbalance settlement costs [€/a]", showgrid=True),  # Enable shared y-axis grid
        yaxis2=dict(showticklabels=False),  # Disable right-side y-axis labels
        legend=dict(title="Consumer Groups", font=dict(color="black")),
        height=300,
        margin=dict(l=40, r=40, t=60, b=40)
    )

    # --- Save as PDF ---
    pdf_path = os.path.join(output_folder, f"imbalance_combined_box_scatter_{model_name}.pdf")
    pio.write_image(fig, pdf_path, format="pdf")

    print(f"Subplot PDF saved: {pdf_path}")

print("All subplot PDFs successfully generated!")

# uv run 5_imbalance_settelment.py
# import pandas as pd
# import plotly.graph_objects as go
# import os

# # Define models and colors
# models = ["Base", "TOU", "MPD", "MPD-50", "MPD-50-inc", "MPD-50-inc-4kW"]
# group_colors = {
#     "FCG1": "blue", "FCG2": "green", "FCG3": "purple",
#     "FCG4": "orange", "FCG5": "cyan", "IFCG": "magenta"
# }

# # Ensure output directory exists
# output_folder = "results/a_combined_results/"
# os.makedirs(output_folder, exist_ok=True)

# # Loop through models and create separate boxplots
# for model_name in models:
#     fig = go.Figure()

#     # Assign ranges for consumer groups
#     consumer_ranges = {
#         "FCG1": (0, 20), "FCG2": (20, 50), "FCG3": (50, 80),
#         "FCG4": (80, 110), "FCG5": (110, 160), "IFCG": (160, 200)
#     }

#     # Add boxplot for the model
#     fig.add_trace(go.Box(
#         y=cost_per_households[model_name],
#         name=model_name,
#         boxpoints="outliers",  # Show outliers separately
#         # marker=dict(color="black"),
#         jitter=0.3,
#         fillcolor="rgba(255,255,255,0)"  # Remove fill color
#     ))

# import random

# random.seed(42)  # If using Python's random module
# np.random.seed(42)  # For NumPy's random generation

# x_jittered = np.random.normal(loc=0, scale=0.1, size=len(cost_per_households[model_name])) 

# for group, (start, end) in consumer_ranges.items():
#     fig.add_trace(go.Scatter(
#         x=x_jittered,
#         y=cost_per_households[model_name].iloc[start:end],
#         mode='markers',
#         marker=dict(color=group_colors[group], size=5, opacity=0.7),
#         name=group,
#         showlegend=False  # Prevent repeated legends for same group
#     ))

#     # # Add jittered scatter points for each consumer group
#     # for group, (start, end) in consumer_ranges.items():
#     #     fig.add_trace(go.Scatter(
#     #         x=[model_name] * (end - start),  # Align jittered points under the boxplot
#     #         y=cost_per_households[model_name].iloc[start:end],
#     #         mode='markers',
#     #         marker=dict(color=group_colors[group], size=5, opacity=0.7),
#     #         name=group
#     #     ))

#     # Update layout for better visualization
#     fig.update_layout(
#         yaxis_title="Grid Cost [€/a]",
#         template="plotly_white",
#         font=dict(color="black"),
#         boxmode="group",  # Group boxplots together
#         legend=dict(title="Consumer Groups", font=dict(color="black")),
#         # width=300,  # Set width in pixels (~7 inches)
#         # height=300  # Adjust height if needed
#     )

#     # Save plots as PDF and HTML
#     pdf_file_path = os.path.join(output_folder, f"a_imbalance_boxplot_{model_name}.pdf")
#     html_file_path = os.path.join(output_folder, f"a_imbalance_boxplot_{model_name}.html")

#     fig.write_image(pdf_file_path, format='pdf')
#     fig.write_html(html_file_path)

#     print(f"Boxplot saved:\n - {pdf_file_path}\n - {html_file_path}")

# print("All boxplots successfully generated!")


# uv run 4_combined_result_handling.py
#*************************************************************************************************************************
# scatter an boxplot within two separate figures
# import pandas as pd
# import numpy as np
# import plotly.graph_objects as go
# import plotly.io as pio
# import os
# from plotly.subplots import make_subplots

# # Define models and colors
# models = ["Base", "TOU", "MPD", "MPD-50", "MPD-50-inc", "MPD-50-inc-4kW"]
# group_colors = {
#     "FCG1": "blue", "FCG2": "green", "FCG3": "purple",
#     "FCG4": "orange", "FCG5": "cyan", "IFCG": "magenta"
# }

# # Ensure output directory exists
# output_folder = "results/a_combined_results/"
# os.makedirs(output_folder, exist_ok=True)

# for model_name in models:
#     # Create subplot figure
#     fig = make_subplots(rows=1, cols=2, subplot_titles=("Boxplot", "Scatter Plot"))

#     consumer_ranges = {
#         "FCG1": (0, 20), "FCG2": (20, 50), "FCG3": (50, 80),
#         "FCG4": (80, 110), "FCG5": (110, 160), "IFCG": (160, 200)
#     }

#     # --- Add Boxplot ---
#     fig.add_trace(go.Box(
#         y=cost_per_households[model_name],
#         name=model_name,
#         boxpoints="outliers",
#         jitter=0.3,
#         fillcolor="rgba(255,255,255,0)"
#     ), row=1, col=1)

#     # --- Add Scatter Plot ---
#     np.random.seed(42)
#     x_jittered = np.random.normal(loc=0, scale=0.1, size=len(cost_per_households[model_name]))

#     for group, (start, end) in consumer_ranges.items():
#         fig.add_trace(go.Scatter(
#             x=x_jittered[start:end],
#             y=cost_per_households[model_name].iloc[start:end],
#             mode="markers",
#             marker=dict(color=group_colors[group], size=5, opacity=0.7),
#             name=group,
#             showlegend=False
#         ), row=1, col=2)

#     # --- Update Layout ---
#     fig.update_layout(
#         title=f"Boxplot & Scatter Plot for {model_name}",
#         xaxis_title="Jittered X Values",
#         yaxis_title="Grid Cost [€/a]",
#         template="plotly_white"
#     )

#     # --- Save Subplot as PDF ---
#     pdf_path = os.path.join(output_folder, f"combined_subplot_{model_name}.pdf")
#     pio.write_image(fig, pdf_path, format="pdf")

#     print(f"Subplot PDF saved: {pdf_path}")

# print("All subplot PDFs successfully generated!")
# uv run 4_combined_result_handling.py
#*************************************************************************************************************************
# grid cost analysis (scatter plots for all scenarios)
import plotly.graph_objects as go

# Create a new figure
fig = go.Figure()

fig.add_trace(go.Scatter(
    y=cost_per_households["MPD"],
    x=household_annual_cons["MPD"],  # Handle NaN values
    mode='markers',
    marker=dict(color='green', size=5),
    name="MPD"
))
print(f'cost_per_households:{cost_per_households["MPD"].head()}')
fig.add_trace(go.Scatter(
    y=cost_per_households["TOU"],
    x=household_annual_cons["TOU"],
    mode='markers',
    marker=dict(color='red', size=5),
    name="TOU"
))

fig.add_trace(go.Scatter(
    y=cost_per_households["MPD-50"],
    x=household_annual_cons["MPD-50"],  # Handle NaN values
    mode='markers',
    marker=dict(color='orange', size=5),
    name="MPD-50"
))

# Add data for each scenario with correct colors
fig.add_trace(go.Scatter(
    y=cost_per_households["MPD-50-inc"],
    x=household_annual_cons["MPD-50-inc"],
    mode='markers',
    marker=dict(color='purple', size=5),
    name="MPD-50-inc"
))

# Add data for each scenario with correct colors
fig.add_trace(go.Scatter(
    y=cost_per_households["MPD-50-inc-4kW"],
    x=household_annual_cons["MPD-50-inc-4kW"],
    mode='markers',
    marker=dict(color='cyan', size=5),
    name="MPD-50-inc-4kW"
))

# Add data for each scenario with correct colors
fig.add_trace(go.Scatter(
    y=cost_per_households["Base"],
    x=household_annual_cons["Base"],
    mode='markers',
    marker=dict(color='blue', size=5),
    name="Base"
))

# Define tick values and labels
tick_vals = list(range(0, int(max(household_annual_cons["Base"])) + 1000, 1000))
tick_labels = [f"{val:,}".replace(",", " ") for val in tick_vals]  # Format numbers with space as a separator

fig.update_layout(
    xaxis=dict(
        title="Annual consumption [kWh/a]",
        tickmode="array",
        tickvals=tick_vals,
        ticktext=tick_labels,
        tickangle=-90
    ),
    yaxis=dict(
        title="Imbalance settlement cost [€/a]",
        tickmode="array",
        tickvals=list(range(0, int(max(cost_per_households["Base"])) + 200, 200))  # Set tick marks every 200 €
    ),
    template="plotly_white",
    font=dict(color="black"),
    legend=dict(title="Legend", font=dict(color="black"))
)

# Save the plot as PDF and HTML
pdf_file_path = "C:\\Users\\ScheiblS\\Documents\\Repositories\\dyn-grid-tariffs\\results\\a_combined_results\\a_imbalance_scatter.pdf"
fig.write_image(pdf_file_path, format='pdf')
print(f"Scatterplot has been saved as '{pdf_file_path}'")

html_file_path = "C:\\Users\\ScheiblS\\Documents\\Repositories\\dyn-grid-tariffs\\results\\a_combined_results\\a_imbalance_scatter.html"
fig.write_html(html_file_path)
print(f"Scatterplot has been saved as '{html_file_path}'")

# uv run 5_imbalance_settelment.py
# *************************************************************************************************************************
# a_imbalance_scatter per scenario
import pandas as pd
import plotly.graph_objects as go
import os

# Define models and colors
models = ["Base", "TOU", "MPD", "MPD-50", "MPD-50-inc", "MPD-50-inc-4kW"]
group_colors = {
    "FCG1": "blue", "FCG2": "green", "FCG3": "purple",
    "FCG4": "orange", "FCG5": "cyan", "IFCG": "magenta"
}

# Ensure output directory exists
output_folder = "results/a_combined_results/"
os.makedirs(output_folder, exist_ok=True)

# Loop through models and create separate scatter plots
for model_name in models:
    fig = go.Figure()

    # Assign ranges for consumer groups
    consumer_ranges = {
        "FCG1": (0, 20), "FCG2": (20, 50), "FCG3": (50, 80),
        "FCG4": (80, 110), "FCG5": (110, 160), "IFCG": (160, 200)
    }

    # Add traces for each consumer group
    for group, (start, end) in consumer_ranges.items():
        fig.add_trace(go.Scatter(
            x=household_annual_cons[model_name].iloc[start:end],
            y=cost_per_households[model_name].iloc[start:end],
            mode='markers',
            marker=dict(color=group_colors[group], size=5),
            name=group
        ))

    # Define tick values and labels for better readability
    tick_vals = list(range(0, int(max(household_annual_cons[model_name])) + 1000, 1000))
    tick_labels = [f"{val:,}".replace(",", " ") for val in tick_vals]  # Space as thousands separator

    # Update layout for clear visualization
    fig.update_layout(
        title=f"Annual Consumption vs. Imbalance Cost ({model_name})",
        xaxis_title="Annual Consumption [kWh/a]",
        yaxis_title="Grid Cost [€/a]",
        template="plotly_white",
        font=dict(color="black"),
        xaxis=dict(
            tickmode="array",
            tickvals=tick_vals,
            ticktext=tick_labels,
            tickangle=-90
        ),
        legend=dict(title="Consumer Groups", font=dict(color="black"))
    )

    # Save plots as PDF and HTML
    pdf_file_path = os.path.join(output_folder, f"a_imbalance_scatter_{model_name}.pdf")
    html_file_path = os.path.join(output_folder, f"a_imbalance_scatter_{model_name}.html")

    fig.write_image(pdf_file_path, format='pdf')
    fig.write_html(html_file_path)

    print(f"Scatter plot saved:\n - {pdf_file_path}\n - {html_file_path}")

print("All scatter plots successfully generated!")
# uv run 4_combined_result_handling.py
#*************************************************************************************************************************