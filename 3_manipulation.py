# uv run 3_manipulation.py
import pandas as pd
import numpy as np

# Define run name and paths
# model = 'base_model'
# run_name = 'base_200hh'
# model = 'tou_model'
# run_name = 'run_200hh'
# model = 'mpd_model'
# run_name = 'run_200hh'
# model = 'mpd-50_model'
# run_name = 'run_200hh'
# model = 'mpd-50_inc'
# run_name = 'run_200hh'
# model = 'mpd-50_inc_4kW'
# run_name = 'run_200hh'
# model = 'cap_sub'
# run_name = 'run_200hh'
model = 'cap_sub_ext'
run_name = 'run_200hh'

#*******************************************************************************************************************************************

input_file_path = f"results/{model}/{run_name}/combined_household_data.csv"  # Folder containing the CSV files
# Load TOU tariff data using row index
tou_tariff = pd.read_csv("files/grid_tariff_tou.csv")
# ******************************************************************************************************************************************
# variability metrics
output_file_path = f"results/{model}/{run_name}/a_single_value_{model}.csv"
try:
    # Load the combined CSV file
    combined_timeseries = pd.read_csv(input_file_path)

    # general load and generation metrics
    max_load = round(combined_timeseries['net_load_lv_grid'].max(), 2)
    min_load = round(combined_timeseries['net_load_lv_grid'].min(), 2)
    median_load = round(combined_timeseries['net_load_lv_grid'].median(), 2)
    mean_load = round(combined_timeseries['net_load_lv_grid'].mean(), 2)  # Mean load metric
    first_quantile = round(combined_timeseries['net_load_lv_grid_abs'].quantile(0.25), 2)
    third_quantile = round(combined_timeseries['net_load_lv_grid_abs'].quantile(0.75), 2)

    inflexible_load = int(combined_timeseries['demand'].sum() / 4)  # Total demand
    heat_sh = int(combined_timeseries['heat_unit'].sum() / 4)  # Total heat unit
    heat_water = int(combined_timeseries['water_unit'].sum() / 4)  # Total water unit
    ev_unit = int(combined_timeseries['ev_unit'].sum() / 4)  # Total EV unit
    total_consumption = int(combined_timeseries['ev_cum'].sum() / 4)  # Total consumption
    grid_load = int(combined_timeseries['grid_load'].sum() / 4)  # Total household load
    residual_load_lv_grid = int(combined_timeseries['net_load_lv_grid_abs'].sum() / 4)  # Integral of net load

    pv_sum = int(combined_timeseries['pv_generation'].sum()/4) # Total PV generation
    surplus = int(combined_timeseries['pv_surplus'].sum()/4)  # Total surplus
    surplus_lv_grid = int(combined_timeseries['pv_surplus_lv_grid'].sum()/4)* -1 # Total surplus on LV grid
    
    bat_sum = int(combined_timeseries['bat_charge'].sum()/4)  # Total battery charge

    # Initialize net load summation row-wise
    net_load_fr = combined_timeseries.filter(like="net_load_hh").sum(axis=1)
    # Iterate over indices and sum corresponding values row-wise
    for i in range(81, 201):  
        net_load_fr += combined_timeseries.filter(like=f"el_market_exp_value_{i}").sum(axis=1)  
    # Insert calculated net load column into DataFrame
    combined_timeseries["net_load_fr"] = net_load_fr  # Direct assignment ensures correct shape
    # Adjust total net load after insertion
    net_load_value = net_load_fr.abs().sum() / 4  # Ensures correct total calculation
 
    # Aggregated variability metrics
    # Define column lists dynamically
    columns_of_interest_2 = [col for i in range(1, 81) if (col := f'net_load_hh_{i}') in combined_timeseries.columns]
    columns_of_interest_1 = [col for i in range(81, 201) if (col := f'el_market_exp_value_{i}') in combined_timeseries.columns]
    # Select the relevant columns
    data = combined_timeseries[columns_of_interest_1 + columns_of_interest_2]
    global_mean = data.values.mean()
    global_std = round(data.values.std(), 2) #shown
    mean_values = data.mean(axis=0)
    mean_values_time = data.mean(axis=1)
    std_deviation = data.std(axis=0)
    cv = (std_deviation / mean_values) * 100
    weighted_cv = round((cv * mean_values).sum() / mean_values.sum(), 2) # shown
    average_load_curve = data.mean(axis=1)
    absolute_deviation = round(data.sub(average_load_curve, axis=0).abs().values.mean(), 2) # shown
    absolute_global_deviation = np.abs(data.values - global_mean).mean()
    load_variability_index = round((absolute_global_deviation / global_mean) * 100, 2) # shown

    print(mean_values.shape)
    print(mean_values_time.shape)
    # Complementary cumulative distribution function - load duration curve
    # Sort the 'net_load_lv_grid_abs' column in descending order
    sorted_grid_load = combined_timeseries['net_load_lv_grid_abs'].sort_values(ascending=False).reset_index(drop=True)
    # Calculate thresholds for 5% and 80% based on sorted values
    x_5_percent = int(0.05 * len(sorted_grid_load))
    x_80_percent = int(0.80 * len(sorted_grid_load))
    # Split the load values into lower (base) and upper (peak) segments
    total_load_values = sorted_grid_load.sum()
    lower_load_values = sorted_grid_load.iloc[x_80_percent:].sum()  # Values from 80% onwards (base load)
    upper_load_values = sorted_grid_load.iloc[:x_5_percent].sum()  # Values up to 5% (peak load)
    # Calculate Base Load (%) and Peak Load (%)
    base_load_percentage = round((lower_load_values / total_load_values) * 100, 2)
    peak_load_percentage = round((upper_load_values / total_load_values) * 100, 2)

    # Create a single DataFrame with all metrics
    df_single_values = pd.DataFrame({
        'snapshot': [
            'Descriptive Statistics',  # Header
            'Objective value', 'Max Load', 'Min Load', 'Median load', 'Mean load', 'First quantile', 'Third quantile', 'Inflexible load',
            'Space heating', 'Domestic hot water', 'EV consumption', 'Total consumption', 'Grid load', 'Residual load - LV grid', 'Total PV generation', 'Total surplus HH',
            'Total surpuls LV grid', 'Stored energy in BESS',
            'Global standard deviation', 'Aggregate coefficient of variation', 'Mean absolute deviation', 
            'Load variability index', 'Base load (%)', 'Peak load (%)', 'Load LV grid'
        ],
        'value': [
            None,  # Empty value for the header
            None, max_load, min_load, median_load, mean_load, first_quantile, third_quantile, inflexible_load,
            heat_sh, heat_water, ev_unit, total_consumption, grid_load ,residual_load_lv_grid, pv_sum, surplus, surplus_lv_grid, bat_sum,
            global_std, weighted_cv, absolute_deviation, load_variability_index, base_load_percentage, peak_load_percentage, net_load_value
        ]
    })

    # Save the updated DataFrame to a CSV file
    df_single_values.to_csv(output_file_path, index=False)
    print(f"Single value metrics saved to {output_file_path}")

except Exception as e:
    print(f"Error while processing the file: {e}")

# uv run 3_manipulation.py
#*******************************************************************************************************************************************
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import plotly.io as pio
pio.kaleido.scope.mathjax = None  # Disable MathJax (optional)
#********************************************************************************************************************************************

# uv run 3_manipulation.py
#*******************************************************************************************************************************************
# PV utilization pie chart
# depicting the variabbility metrics
# Calculated segments
pv_sum_new = pv_sum-(pv_sum * 0.03)
own_consumption = pv_sum_new - surplus
utilization_lv = surplus - surplus_lv_grid
not_utilized = surplus_lv_grid

# Labels and values
labels = ["Own consumption", "Utilization in LV", "not utilized"]
values = [own_consumption, utilization_lv, not_utilized]
colors = ["rgb(255,217,47)", "rgb(166,216,84)", "rgb(128,177,211)"] # yellow, green, blue

# Creating pie chart
fig = go.Figure(data=[go.Pie(labels=labels, values=values, 
                             textinfo='label+percent',  # Includes absolute values
                             hoverinfo='label+value',
                             textfont=dict(size=14),
                             hole=0.4,  # For donut chart
                             marker=dict(colors=colors),
                             textposition="outside")])  # Places labels outside for horizontal alignment)])

# Save the resulting plot as PDF
pdf_file_path = f"C:/Users/ScheiblS/Documents/Repositories/dyn-grid-tariffs/results/{model}/{run_name}/a_pie_chart_pv_{model}.pdf"
pio.write_image(fig, pdf_file_path, format='pdf')
print(f"Pie chart has been saved as '{pdf_file_path}'")

# Save the resulting plot as HTML for interactivity
html_file_path = f"C:/Users/ScheiblS/Documents/Repositories/dyn-grid-tariffs/results/{model}/{run_name}/a_pie_chart_pv_{model}.html"
fig.write_html(html_file_path)
print(f"Pie chart has been saved as '{html_file_path}'")
# uv run 3_manipulation.py
#*******************************************************************************************************************************************
# Define metric values
metrics = ["Global Std Dev", "Coefficient of Variation", "Mean Absolute Deviation", "Load Variability Index"]
values = [global_std, weighted_cv, absolute_deviation, load_variability_index]

# Create a Plotly bar chart
fig = go.Figure()

fig.add_trace(go.Bar(
    x=metrics,
    y=values,
    marker_color=['blue', 'orange', 'green', 'red']
))

fig.update_layout(
    title="Comparison of Variability Metrics",
    xaxis_title="Metrics",
    yaxis_title="Value",
    template="plotly_white"
)

# Save the resulting plot as PDF
pdf_file_path = f"C:/Users/ScheiblS/Documents/Repositories/dyn-grid-tariffs/results/{model}/{run_name}/a_bar_chart_var_metrics_{model}.pdf"
pio.write_image(fig, pdf_file_path, format='pdf')
print(f"Bar chart has been saved as '{pdf_file_path}'")

# Save the resulting plot as HTML for interactivity
html_file_path = f"C:/Users/ScheiblS/Documents/Repositories/dyn-grid-tariffs/results/{model}/{run_name}/a_bar_chart_var_metrics_{model}.html"
fig.write_html(html_file_path)
print(f"Bar chart has been saved as '{html_file_path}'")

# uv run 3_manipulation.py
# ******************************************************************************************************************************************
# #sensitivity analysis threshold: arithmetic mean of the twelve monthly peak
# Ensure the DataFrame index is numeric
columns_of_interest = [col for col in combined_timeseries.columns if col.startswith('el_market_exp_value_')]
data = combined_timeseries[columns_of_interest]
data.reset_index(drop=True, inplace=True)  # Reset index to make it numeric

# Select columns of interest (starting with 'el_market_exp_value_')
columns_of_interest = [col for col in data.columns if col.startswith('el_market_exp_value_')]
data = data[columns_of_interest]

# Ensure the DataFrame index is aligned with datetime in 15-minute intervals
datetime_index = pd.date_range(start="2021-01-01 00:00", end="2021-12-31 23:45", freq="15min")
data['datetime'] = datetime_index[:len(data)]  # Assign the datetime index to match the rows
data.set_index('datetime', inplace=True)

# Step 1: Group rows by months (using the datetime index) and calculate the monthly peaks
monthly_groups = data.resample('ME').max()  # Resample to monthly frequency and take max values for each column

# Step 2: Calculate the arithmetic mean of the twelve monthly peaks for each column
monthly_peak_means = monthly_groups.mean()  # Compute the arithmetic mean of monthly peaks per column

# Step 3: Create boxplots for sensitivity analysis
fig = go.Figure()

# Add the first boxplot for all means with the mean line, outlier markers, and data points
fig.add_trace(
    go.Box(
        y=monthly_peak_means,
        boxpoints='all',  # Include all data points
        jitter=0.3,
        pointpos=0,
        marker=dict(color='red', size=2, symbol='circle'),  # Smaller data points
        name='All households',
        boxmean=True  # Display mean line
    )
)

# Select data for additional boxplots based on specific ranges
selected_means_fc1 = monthly_peak_means.iloc[:20]  # grid_load_exp_value_1 to grid_load_exp_value_20
selected_means_fc2 = monthly_peak_means.iloc[20:50]  # grid_load_exp_value_21 to grid_load_exp_value_50
selected_means_fc3 = monthly_peak_means.iloc[50:80]  # grid_load_exp_value_51 to grid_load_exp_value_80
selected_means_fc4 = monthly_peak_means.iloc[80:110]  # grid_load_exp_value_81 to grid_load_exp_value_110
selected_means_fc5 = monthly_peak_means.iloc[110:160]  # grid_load_exp_value_111 to grid_load_exp_value_160
selected_means_ifcg = monthly_peak_means.iloc[160:200]  # grid_load_exp_value_161 to grid_load_exp_value_200

# Add boxplots for selected ranges with data points
fig.add_trace(go.Box(y=selected_means_fc1, boxpoints='all', jitter=0.3, pointpos=0,
                     marker=dict(color='blue', size=2), name='FCG1', boxmean=True))
fig.add_trace(go.Box(y=selected_means_fc2, boxpoints='all', jitter=0.3, pointpos=0,
                     marker=dict(color='green', size=2), name='FCG2', boxmean=True))
fig.add_trace(go.Box(y=selected_means_fc3, boxpoints='all', jitter=0.3, pointpos=0,
                     marker=dict(color='purple', size=2), name='FCG3', boxmean=True))
fig.add_trace(go.Box(y=selected_means_fc4, boxpoints='all', jitter=0.3, pointpos=0,
                     marker=dict(color='orange', size=2), name='FCG4', boxmean=True))
fig.add_trace(go.Box(y=selected_means_fc5, boxpoints='all', jitter=0.3, pointpos=0,
                     marker=dict(color='cyan', size=2), name='FCG5', boxmean=True))
fig.add_trace(go.Box(y=selected_means_ifcg, boxpoints='all', jitter=0.3, pointpos=0,
                     marker=dict(color='magenta', size=2), name='IFCG', boxmean=True))

# Update layout for the plot
fig.update_layout(
    yaxis_title="Arithmetic means of <br> monthly peaks [kW]",
    yaxis=dict(range=[0, None]),
    template="plotly_white",
    showlegend=False,
    font=dict(color="black"),
    margin = dict(l=20, r=20, t=20, b=20),  # Adjust margins for better visibility
    height=300,  # Set height for better visibility
)
# Save the resulting plot as PDF for compatibility
pdf_file_path = f"C:/Users/ScheiblS/Documents/Repositories/dyn-grid-tariffs/results/{model}/{run_name}/a_boxplot_monthly_peak_means_{model}.pdf"
pio.write_image(fig, pdf_file_path, format='pdf') 
print(f"Boxplot has been saved as 'a_boxplot_monthly_peak_means.pdf'")

# Save the resulting plot as HTML for interactivity
html_file_path = f"C:/Users/ScheiblS/Documents/Repositories/dyn-grid-tariffs/results/{model}/{run_name}/a_boxplot_monthly_peak_means_{model}.html"
fig.write_html(html_file_path)
print(f"Boxplot has been saved as 'a_boxplot_monthly_peak_means.html'")
# uv run 3_manipulation.py

# ******************************************************************************************************************************************
# test the statement of boxplot with household 85
import plotly.graph_objects as go

# Create a new Plotly figure
fig = go.Figure()

# Define the columns to be plotted
columns_to_plot = {
    'el_market_exp_value_85': 'blue',
    'demand_exp_value_85': 'red',
    'ev_unit_exp_in_electricity_85': 'green',
    'heat_unit_exp_in_electricity_85': 'violet',
    'water_unit_exp_in_electricity_85': 'orange'
}

# Loop through each column and add it to the figure if it exists
for column_name, color in columns_to_plot.items():
    if column_name in combined_timeseries.columns:
        fig.add_trace(go.Scatter(
            x=combined_timeseries.index,  # Use the DataFrame index for the x-axis
            y=combined_timeseries[column_name],  # The specified column
            mode='lines',  # Line plot
            name=column_name,
            line=dict(color=color)
        ))
    else:
        print(f"Warning: Column '{column_name}' does not exist in combined_timeseries.")

# Customize layout
fig.update_layout(
    title='Time Series Plot for Selected Columns',
    xaxis_title='Time',
    yaxis_title='Value',
    template='plotly_white',
    showlegend=True,
    font=dict(color="black")
)

# Save the interactive plot to an HTML file
output_html_path = f'results\\{model}\\{run_name}\\a_el_market_exp_value_85_plot.html'
fig.write_html(output_html_path)

print(f"Plot saved as an interactive HTML file at {output_html_path}")
# uv run 3_manipulation.py
# ******************************************************************************************************************************************
#net grid_load graph
import plotly.graph_objects as go
    # Create the plot for 'net_load_lv_grid'
fig = go.Figure()

# # Add 'net_load_lv_grid' data to the plot
# fig.add_trace(go.Scatter(x=combined_timeseries.index, y=combined_timeseries['net_load_lv_grid'], mode='lines', name='Net load on low-voltage grid', line=dict(color='red')
# ))
# Add 'net_load_lv_grid' data to the plot
fig.add_trace(go.Scatter(x=combined_timeseries.index, y=combined_timeseries['ev_unit'], mode='lines', name='Net load on low-voltage grid', line=dict(color='red')
))

# Add metrics as annotations
metrics_text = (f"Max Load: {max_load:.2f} kW<br>"f"Min Load: {min_load:.2f} kW<br>"f"Residual load on the LV grid: {residual_load_lv_grid:.2f} kWh"
)
fig.add_annotation(
    text=metrics_text, xref="paper", yref="paper", x=1, y=1, showarrow=False, align="left", font=dict(size=12, color="black"), bgcolor="lightyellow", bordercolor="black")

# Customize the layout of the plot
fig.update_layout(
    title='Net load LV grid',
    xaxis_title='15-minute intervals',
    yaxis_title='Power in kW',
    template='plotly_white',
    legend_title='Legend',
    font=dict(color="black")
)

# Save the interactive plot as an HTML file
html_file_path = f'results\\{model}\\{run_name}\\a_net_load_{model}.html'
fig.write_html(html_file_path)

print(f"Plot for net load on low-voltage grid saved as HTML at {html_file_path}")

# uv run 3_manipulation.py
#******************************************************************************************************************************************
#combined graph with all powers
import plotly.graph_objects as go

# Create the figure
fig = go.Figure()

# Add demand with light orange fill
if 'demand' in combined_timeseries.columns:
    fig.add_trace(go.Scatter(
        x=combined_timeseries.index, y=combined_timeseries['demand'],
        mode='lines', name='Infl. consumption', line=dict(color='orange'),
        fill='tozeroy', fillcolor='rgba(255, 165, 0, 0.3)'
    ))

# Add space heating with reddish-orange fill
if 'space_heating_cum' in combined_timeseries.columns:
    fig.add_trace(go.Scatter(
        x=combined_timeseries.index, y=combined_timeseries['space_heating_cum'],
        mode='lines', name='Space heating', line=dict(color='violet'),
        fill='tonexty', fillcolor='rgba(138, 43, 226, 0.3)'
    ))

# Add domestic hot water with violet fill
if 'dhw_cum' in combined_timeseries.columns:
    fig.add_trace(go.Scatter(
        x=combined_timeseries.index, y=combined_timeseries['dhw_cum'],
        mode='lines', name='DHW', line=dict(color='purple'),
        fill='tonexty', fillcolor='rgba(255, 99, 71, 0.3)'
    ))

# Add EV charging with red fill
if 'ev_cum' in combined_timeseries.columns:
    fig.add_trace(go.Scatter(
        x=combined_timeseries.index, y=combined_timeseries['ev_cum'],
        mode='lines', name='BEV charging', line=dict(color='red'),
        fill='tonexty', fillcolor='rgba(220, 20, 60, 0.3)'
    ))

# Add BESS charging
if 'bat_cum' in combined_timeseries.columns:
    fig.add_trace(go.Scatter(
        x=combined_timeseries.index, y=combined_timeseries["bat_cum"],
        mode="none", name="BESS charging",
        fill="tonexty",  fillcolor="rgba(173, 216, 230, 0.3)",  # Light blue with 30% opacity
        fillpattern=dict(shape=".")
    ))

# Add BESS discharging
if 'bat_dis_cum' in combined_timeseries.columns:
    fig.add_trace(go.Scatter(
        x=combined_timeseries.index, y=combined_timeseries["bat_dis_cum"],
        mode="none", name="BESS discharging",
        fill="tonexty", fillcolor="rgba(144, 238, 144, 0.3)",  # Light green with 30% opacity
        fillpattern=dict(shape="x")  # Correct way to add a pattern
    ))

# Add PV generation with yellow fill for Schnittmenge
if 'pv_generation' in combined_timeseries.columns:
    fig.add_trace(go.Scatter(
        x=combined_timeseries.index, y=combined_timeseries['pv_generation'],
        mode='lines', name='PV generation',
        fill='tozeroy', fillcolor='rgba(255, 255, 0, 0.3)',
        line=dict(color='yellow')
    ))

# Add net load on low-voltage grid with a finer dashed line
if 'net_load_lv_grid' in combined_timeseries.columns:
    fig.add_trace(go.Scatter(
        x=combined_timeseries.index, y=combined_timeseries['net_load_lv_grid'],
        mode='lines', name='Residual load',
        line=dict(color='blue', dash='dot')  # Change 'dash' to 'dot' or another style
    ))

if 'net_load_fr' in combined_timeseries.columns:
    fig.add_trace(go.Scatter(
        x=combined_timeseries.index, y=combined_timeseries['net_load_fr'],
        mode='lines', name='Net load FR',
        line=dict(color='green', dash='dot')  # Change 'dash' to 'dot' or another style
    ))

# Customize the layout of the plot
fig.update_layout(
    title='Load summary',
    xaxis_title='15-minute intervals',
    yaxis_title='Power in kW',
    template='plotly_white',
    legend_title='Legend',
    font=dict(color="black")
)

 # Save the interactive plot as an HTML file
html_file_path = f'results\\{model}\\{run_name}\\a_load_summary_{model}.html'
fig.write_html(html_file_path)

print(f"Plot for load summary saved as HTML at {html_file_path}")
# uv run 3_manipulation.py
#******************************************************************************************************************************************
# graph of represenative day (max load day)
import plotly.graph_objects as go
import pandas as pd

# Extract the day with the maximum net load on the low-voltage grid
if 'net_load_lv_grid' in combined_timeseries.columns:
    max_day_idx = combined_timeseries['net_load_lv_grid'].idxmax()
    day_start_idx = max_day_idx - (max_day_idx % 96)
    max_day_data = combined_timeseries.iloc[day_start_idx:day_start_idx + 96]

# Calculate the correct date from row index
days_passed = day_start_idx // 96  # Number of full days since start of 2021
corrected_date = pd.to_datetime("2021-01-01") + pd.Timedelta(days=days_passed)

# Generate date and time labels for the intervals (assuming 15-minute intervals in a day)
time_intervals = pd.date_range(start=f"{corrected_date.date()} 00:00", freq="15T", periods=96)

# Create a new figure for the specified day
fig = go.Figure()

# Add demand with light orange fill
if 'demand' in max_day_data.columns:
    fig.add_trace(go.Scatter(
        x=time_intervals, y=max_day_data['demand'],
        mode='lines', name='Infl. consumption', line=dict(color='orange'),
        fill='tozeroy', fillcolor='rgba(255, 165, 0, 0.3)'
    ))

# Add space heating with reddish-orange fill
if 'space_heating_cum' in max_day_data.columns:
    fig.add_trace(go.Scatter(
        x=time_intervals, y=max_day_data['space_heating_cum'],
        mode='lines', name='Space heating', line=dict(color='violet'),
        fill='tonexty', fillcolor='rgba(138, 43, 226, 0.3)'
    ))

# Add domestic hot water with violet fill
if 'dhw_cum' in max_day_data.columns:
    fig.add_trace(go.Scatter(
        x=time_intervals, y=max_day_data['dhw_cum'],
        mode='lines', name='DHW', line=dict(color='purple'),
        fill='tonexty', fillcolor='rgba(255, 99, 71, 0.3)'
    ))

# Add EV charging with red fill
if 'ev_cum' in max_day_data.columns:
    fig.add_trace(go.Scatter(
        x=time_intervals, y=max_day_data['ev_cum'],
        mode='lines', name='BEV charging', line=dict(color='red'),
        fill='tonexty', fillcolor='rgba(220, 20, 60, 0.3)'
    ))

# Add BESS charging
if 'bat_cum' in max_day_data.columns:
    fig.add_trace(go.Scatter(
        x=time_intervals, y=max_day_data["bat_cum"],
        mode="none", name="BESS charging",
        fill="tonexty",  fillcolor="rgba(173, 216, 230, 0.3)",  # Light blue with 30% opacity
        fillpattern=dict(shape=".")
    ))

# Add BESS discharging
if 'bat_dis_cum' in max_day_data.columns:
    fig.add_trace(go.Scatter(
        x=time_intervals, y=max_day_data["bat_dis_cum"],
        mode="none", name="BESS discharging",
        fill="tonexty", fillcolor="rgba(144, 238, 144, 0.3)",  # Light green with 30% opacity
        fillpattern=dict(shape="x")  # Correct way to add a pattern
    ))

# Add PV generation with yellow fill for Schnittmenge
if 'pv_generation' in max_day_data.columns:
    fig.add_trace(go.Scatter(
        x=time_intervals, y=max_day_data['pv_generation'],
        mode='lines', name='PV generation',
        fill='tozeroy', fillcolor='rgba(255, 255, 0, 0.3)',
        line=dict(color='yellow')
    ))

# Add net load on low-voltage grid with a finer dashed line
if 'net_load_lv_grid' in max_day_data.columns:
    fig.add_trace(go.Scatter(
        x=time_intervals, y=max_day_data['net_load_lv_grid'],
        mode='lines', name='Residual load',
        line=dict(color='blue', dash='dot')  # Change 'dash' to 'dot' or another style
    ))

# Customize the layout of the plot
fig.update_layout(
    title=f'Load summary for {corrected_date.date()} (day with max residual load on LV grid)',
    xaxis_title='Time intervals',
    yaxis_title='Power in kW',
    template='plotly_white',
    legend_title='Legend',
    showlegend=True,
    font=dict(color="black")
)

# Fix the x-axis to show only hourly labels
fig.update_xaxes(
    tickformat='%H:%M',  # Hourly format
    dtick=3600000  # One hour in milliseconds
)

# Save the interactive plot as an HTML file
html_file_path = f'results\\{model}\\{run_name}\\a_max_day_{model}.html'
fig.write_html(html_file_path)

# Save the plot as a PNG file
pdf_file_path = f'results\\{model}\\{run_name}\\a_max_day_{model}.pdf'
fig.write_image(pdf_file_path)

print(f"Plot for load summary saved as HTML at {html_file_path}")
print(f"Plot for load summary saved as PNG at {pdf_file_path}")
# uv run 3_manipulation.py
#******************************************************************************************************************************************
# graph of represenative day (min load day)
import plotly.graph_objects as go
import pandas as pd

# Extract the day with the minimum net load on the low-voltage grid
if 'net_load_lv_grid' in combined_timeseries.columns:
    min_day_idx = combined_timeseries['net_load_lv_grid'].idxmin()  # Find index of minimum net load
    day_start_idx = min_day_idx - (min_day_idx % 96)  # Align to start of the day
    min_day_data = combined_timeseries.iloc[day_start_idx:day_start_idx + 96]

# Calculate the correct date from row index
days_passed = day_start_idx // 96  # Number of full days since start of 2021
corrected_date = pd.to_datetime("2021-01-01") + pd.Timedelta(days=days_passed)

# Generate date and time labels for the intervals (assuming 15-minute intervals in a day)
time_intervals = pd.date_range(start=f"{corrected_date.date()} 00:00", freq="15T", periods=96)

# Create a new figure for the specified day
fig = go.Figure()

# Add space heating with reddish-orange fill
if 'space_heating_cum' in min_day_data.columns:
    fig.add_trace(go.Scatter(
        x=time_intervals, y=min_day_data['space_heating_cum'],
        mode='lines', name='Space heating', line=dict(color='violet'),
         fillcolor='rgba(138, 43, 226, 0.3)' #fill='tonexty',
    ))

# Add domestic hot water with violet fill
if 'dhw_cum' in min_day_data.columns:
    fig.add_trace(go.Scatter(
        x=time_intervals, y=min_day_data['dhw_cum'],
        mode='lines', name='DHW', line=dict(color='purple'),
         fillcolor='rgba(255, 99, 71, 0.3)' #fill='tonexty',
    ))

# Add demand with light orange fill
if 'demand' in min_day_data.columns:
    fig.add_trace(go.Scatter(
        x=time_intervals, y=min_day_data['demand'],
        mode='lines', name='Infl. consumption', line=dict(color='orange'),
        fill='tozeroy', fillcolor='rgba(255, 165, 0, 0.3)'
    ))

# Add EV charging with red fill
if 'ev_cum' in min_day_data.columns:
    fig.add_trace(go.Scatter(
        x=time_intervals, y=min_day_data['ev_cum'],
        mode='lines', name='BEV charging', line=dict(color='red'),
        fill='tonexty', fillcolor='rgba(220, 20, 60, 0.3)'
    ))

# Add BESS charging
if 'bat_cum' in max_day_data.columns:
    fig.add_trace(go.Scatter(
        x=time_intervals, y=min_day_data["bat_cum"],
        mode="none", name="BESS charging",
        fill="tonexty",  fillcolor="rgba(173, 216, 230, 0.3)",  # Light blue with 30% opacity
        fillpattern=dict(shape=".")
    ))

# Add BESS discharging
if 'bat_dis_cum' in max_day_data.columns:
    fig.add_trace(go.Scatter(
        x=time_intervals, y=min_day_data["bat_dis_cum"],
        mode="none", name="BESS discharging",
        fill="tonexty", fillcolor="rgba(144, 238, 144, 0.3)",  # Light green with 30% opacity
        fillpattern=dict(shape="x")  # Correct way to add a pattern
    ))

# Add PV generation with yellow fill for Schnittmenge
if 'pv_generation' in min_day_data.columns:
    fig.add_trace(go.Scatter(
        x=time_intervals, y=min_day_data['pv_generation'],
        mode='lines', name='PV generation',
        fill='tozeroy', fillcolor='rgba(255, 255, 0, 0.3)',
        line=dict(color='yellow')
    ))

# Add net load on low-voltage grid with a finer dashed line
if 'net_load_lv_grid' in min_day_data.columns:
    fig.add_trace(go.Scatter(
        x=time_intervals, y=min_day_data['net_load_lv_grid'],
        mode='lines', name='Residual load',
        line=dict(color='blue', dash='dot')  # Change 'dash' to 'dot' or another style
    ))

# Customize the layout of the plot
fig.update_layout(
    title=f'Load summary for {corrected_date.date()} (day with min residual load on LV grid)',
    xaxis_title='Time intervals',
    yaxis_title='Power in kW',
    template='plotly_white',
    legend_title='Legend',
    showlegend=True,
    font=dict(color="black")
)

# Fix the x-axis to show only hourly labels
fig.update_xaxes(
    tickformat='%H:%M',  # Hourly format
    dtick=3600000  # One hour in milliseconds
)
# Save the interactive plot as an HTML file
html_file_path = f'results\\{model}\\{run_name}\\a_min_day_{model}.html'
fig.write_html(html_file_path)

# Save the plot as a PNG file
pdf_file_path = f'results\\{model}\\{run_name}\\a_min_day_{model}.pdf'
fig.write_image(pdf_file_path)

print(f"Plot for load summary saved as HTML at {html_file_path}")
print(f"Plot for load summary saved as PNG at {pdf_file_path}")

# uv run 3_manipulation.py
#******************************************************************************************************************************************
# graph of represenative for Base day self chosen day.
# import plotly.graph_objects as go
# import pandas as pd

# # Function to generate and display plot based on user-specified date
# def generate_plot_base(desired_year, desired_month, desired_day, df_combined, tariff):
#     # Define dataset structure (assuming full year, 96 intervals/day)
#     days_per_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]  # Adjust for leap years
#     rows_per_day = 96

#     chosen_date = f"{desired_year}-{desired_month:02d}-{desired_day:02d}"

#     # Adjust for leap years
#     if desired_year % 4 == 0:
#         days_per_month[1] = 29

#     # Calculate starting row index
#     month_offset = sum(days_per_month[:desired_month - 1]) * rows_per_day
#     day_start_idx = month_offset + (desired_day - 1) * rows_per_day
#     day_end_idx = day_start_idx + rows_per_day

#     # Extract data for the selected day
#     selected_day_data = df_combined.iloc[day_start_idx:day_end_idx]

#     # Generate time labels (15-min intervals)
#     time_intervals = pd.date_range(
#         start=f"{desired_year}-{desired_month:02d}-{desired_day:02d} 00:00",
#         freq="15T", periods=rows_per_day
#     )
#     selected_day_data.index = time_intervals

#     ticktxt = str(tariff * 100)

#     # Define plot figure
#     fig = go.Figure()
    
#     # Add demand (primary y-axis)
#     if 'demand' in selected_day_data.columns:
#         fig.add_trace(go.Scatter(
#             x=time_intervals, y=selected_day_data["demand"],
#             mode="lines", name="Infl. consumption",
#             line=dict(color="orange"),
#             fill="tozeroy", fillcolor="rgba(255, 165, 0, 0.3)"
#         ))

#     # Add space heating
#     if 'space_heating_cum' in selected_day_data.columns:
#         fig.add_trace(go.Scatter(
#             x=time_intervals, y=selected_day_data["space_heating_cum"],
#             mode="lines", name="Space heating",
#             line=dict(color="violet"),
#             fill="tonexty", fillcolor="rgba(138, 43, 226, 0.3)"
#         ))

#     # Add DHW
#     if 'dhw_cum' in selected_day_data.columns:
#         fig.add_trace(go.Scatter(
#             x=time_intervals, y=selected_day_data["dhw_cum"],
#             mode="lines", name="DHW",
#             line=dict(color="purple"),
#             fill="tonexty", fillcolor="rgba(255, 99, 71, 0.3)"
#         ))

#     # Add BEV charging
#     if 'ev_cum' in selected_day_data.columns:
#         fig.add_trace(go.Scatter(
#             x=time_intervals, y=selected_day_data["ev_cum"],
#             mode="lines", name="BEV charging",
#             line=dict(color="red"),
#             fill="tonexty", fillcolor="rgba(220, 20, 60, 0.3)"
#         ))

#     # Add BESS charging
#     if 'bat_cum' in selected_day_data.columns:
#         fig.add_trace(go.Scatter(
#             x=time_intervals, y=selected_day_data["bat_cum"],
#             mode="none", name="BESS charging",
#             fill="tonexty",  fillcolor="rgba(173, 216, 230, 0.3)",  # Light blue with 30% opacity
#             fillpattern=dict(shape=".")
#         ))

#     # Add BESS discharging
#     if 'bat_dis_cum' in selected_day_data.columns:
#         fig.add_trace(go.Scatter(
#             x=time_intervals, y=selected_day_data["bat_dis_cum"],
#             mode="none", name="BESS discharging",
#             fill="tonexty", fillcolor="rgba(144, 238, 144, 0.3)",  # Light green with 30% opacity
#             fillpattern=dict(shape="x")  # Correct way to add a pattern
#         ))

#     # Add PV generation
#     if 'pv_generation' in selected_day_data.columns:
#         fig.add_trace(go.Scatter(
#             x=time_intervals, y=selected_day_data["pv_generation"],
#             mode="lines", name="PV generation",
#             fill="tozeroy", fillcolor="rgba(255, 255, 0, 0.3)",
#             line=dict(color="yellow")
#         ))

#     # Add net load (primary y-axis)
#     if 'net_load_lv_grid' in selected_day_data.columns:
#         fig.add_trace(go.Scatter(
#             x=time_intervals,
#             y=selected_day_data["net_load_lv_grid"],
#             mode="lines",
#             name="Residual load",
#             line=dict(color="blue", dash="dot")
#         ))

#     # Add Grid tariff (right y-axis, with area fill)
#     fig.add_trace(go.Scatter(
#         x=time_intervals,
#         y=[tariff] * len(time_intervals),  # Repeat tariff value for each interval
#         mode="lines",
#         name="Grid tariff (ct/kWh)",
#         line=dict(color="green"),
#         # fill="tozeroy",
#         # fillcolor="rgba(144, 238, 144, 0.4)",
#         yaxis="y2"
#     ))

#     # Update layout for dual y-axes
#     fig.update_layout(
#         title=f"Load summary and grid tariff - {chosen_date}",
#         xaxis_title="Time intervals",
#         yaxis=dict(
#             title="Net load on LV grid [kW]",
#             # range=[-120, 390],
#             side="left"
#         ),
#         yaxis2=dict(
#             title="Grid tariff [ct/kWh]",
#             # range=[price_min, price_max],  # Adjust lower bound
#             overlaying="y",
#             side="right",
#             tickvals=[tariff],
#             ticktext=[ticktxt],
#             tickfont=dict(color="green")  # Set tick color to green
#         ),
#         legend=dict(
#             title="Legend",
#             x=1.6,
#             y=1.0,  # Moved slightly higher
#             xanchor="right",
#             yanchor="top"
#         ),
#         template="plotly_white",
#         font=dict(color="black")
#     )

#     # Fix x-axis to hourly labels
#     fig.update_xaxes(tickformat="%H:%M", dtick=3600000)
    
#     # Save plots
#     html_file_path = f'results\\{model}\\{run_name}\\a_base_{chosen_date}_{model}.html'
#     fig.write_html(html_file_path)

#     pdf_file_path = f'results\\{model}\\{run_name}\\a_base_{chosen_date}_{model}.pdf'
#     fig.write_image(pdf_file_path)

# # Example usage
# generate_plot_base(2021, 6, 18, combined_timeseries, 0.104)
# generate_plot_base(2021, 11, 25, combined_timeseries, 0.104)
# generate_plot_base(2021, 10, 17, combined_timeseries, 0.104)
# uv run 3_manipulation.py
#******************************************************************************************************************************************
# graph of represenative for MPD scenario chosen day.
import plotly.graph_objects as go
import pandas as pd

# Function to generate and display plot based on user-specified date
def generate_plot_mpd(desired_year, desired_month, desired_day, df_combined):
    days_per_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]  
    rows_per_day = 96

    chosen_date = f"{desired_year}-{desired_month:02d}-{desired_day:02d}"

    if desired_year % 4 == 0:
        days_per_month[1] = 29

    month_offset = sum(days_per_month[:desired_month - 1]) * rows_per_day
    day_start_idx = month_offset + (desired_day - 1) * rows_per_day
    day_end_idx = day_start_idx + rows_per_day

    selected_day_data = df_combined.iloc[day_start_idx:day_end_idx]

    time_intervals = pd.date_range(
        start=f"{desired_year}-{desired_month:02d}-{desired_day:02d} 00:00",
        freq="15T", periods=rows_per_day
    )
    selected_day_data.index = time_intervals

    fig = go.Figure()
    
    if 'demand' in selected_day_data.columns:
        fig.add_trace(go.Scatter(
            x=time_intervals, y=selected_day_data["demand"],
            mode="lines", name="Infl. consumption",
            line=dict(color="orange"),
            fill="tozeroy", fillcolor="rgba(255, 165, 0, 0.3)"
        ))

    if 'space_heating_cum' in selected_day_data.columns:
        fig.add_trace(go.Scatter(
            x=time_intervals, y=selected_day_data["space_heating_cum"],
            mode="lines", name="Space heating",
            line=dict(color="violet"),
            fill="tonexty", fillcolor="rgba(138, 43, 226, 0.3)"
        ))

    if 'dhw_cum' in selected_day_data.columns:
        fig.add_trace(go.Scatter(
            x=time_intervals, y=selected_day_data["dhw_cum"],
            mode="lines", name="DHW",
            line=dict(color="purple"),
            fill="tonexty", fillcolor="rgba(255, 99, 71, 0.3)"
        ))

    if 'ev_cum' in selected_day_data.columns:
        fig.add_trace(go.Scatter(
            x=time_intervals, y=selected_day_data["ev_cum"],
            mode="lines", name="BEV charging",
            line=dict(color="red"),
            fill="tonexty", fillcolor="rgba(220, 20, 60, 0.3)"
        ))

    if 'bat_cum' in selected_day_data.columns:
        fig.add_trace(go.Scatter(
            x=time_intervals, y=selected_day_data["bat_cum"],
            mode="none", name="BESS charging",
            fill="tonexty", fillcolor="rgba(173, 216, 230, 0.3)"
        ))

    if 'bat_dis_cum' in selected_day_data.columns:
        fig.add_trace(go.Scatter(
            x=time_intervals, y=selected_day_data["bat_dis_cum"],
            mode="none", name="BESS discharging",
            fill="tonexty", fillcolor="rgba(144, 238, 144, 0.3)"
        ))

    if 'pv_generation' in selected_day_data.columns:
        fig.add_trace(go.Scatter(
            x=time_intervals, y=selected_day_data["pv_generation"],
            mode="lines", name="PV generation",
            fill="tozeroy", fillcolor="rgba(255, 255, 0, 0.3)",
            line=dict(color="yellow")
        ))

    if 'net_load_lv_grid' in selected_day_data.columns:
        fig.add_trace(go.Scatter(
            x=time_intervals,
            y=selected_day_data["net_load_lv_grid"],
            mode="lines",
            name="Net load",
            line=dict(color="blue", dash="dot")
        ))

    fig.update_layout(
        title=f"Load summary - {chosen_date}",
        xaxis_title="Time intervals",
        yaxis=dict(title="Net load on LV grid [kW]", side="left"),
        legend=dict(title="Legend", x=1.6, y=1.0, xanchor="right", yanchor="top"),
        template="plotly_white",
        font=dict(color="black")
    )

    fig.update_xaxes(tickformat="%H:%M", dtick=3600000)

    # Save plots
    html_file_path = f'results\\{model}\\{run_name}\\a_mpd_{chosen_date}_{model}.html'
    fig.write_html(html_file_path)

    pdf_file_path = f'results\\{model}\\{run_name}\\a_mpd_{chosen_date}_{model}.pdf'
    fig.write_image(pdf_file_path)

# Example usage
generate_plot_mpd(2021, 6, 16, combined_timeseries)
generate_plot_mpd(2021, 12, 2, combined_timeseries)
generate_plot_mpd(2021, 10, 17, combined_timeseries)

# uv run 3_manipulation.py
#******************************************************************************************************************************************
# graph of represenative for MPD-50 self chosen day.
import plotly.graph_objects as go
import pandas as pd

# Function to generate and display plot based on user-specified date
def generate_plot_base(desired_year, desired_month, desired_day, df_combined, tariff):
    # Define dataset structure (assuming full year, 96 intervals/day)
    days_per_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]  # Adjust for leap years
    rows_per_day = 96

    chosen_date = f"{desired_year}-{desired_month:02d}-{desired_day:02d}"

    # Adjust for leap years
    if desired_year % 4 == 0:
        days_per_month[1] = 29

    # Calculate starting row index
    month_offset = sum(days_per_month[:desired_month - 1]) * rows_per_day
    day_start_idx = month_offset + (desired_day - 1) * rows_per_day
    day_end_idx = day_start_idx + rows_per_day

    # Extract data for the selected day
    selected_day_data = df_combined.iloc[day_start_idx:day_end_idx]

    # Generate time labels (15-min intervals)
    time_intervals = pd.date_range(
        start=f"{desired_year}-{desired_month:02d}-{desired_day:02d} 00:00",
        freq="15T", periods=rows_per_day
    )
    selected_day_data.index = time_intervals

    ticktxt = str(tariff * 100)

    # Define plot figure
    fig = go.Figure()
    
    # Add demand (primary y-axis)
    if 'demand' in selected_day_data.columns:
        fig.add_trace(go.Scatter(
            x=time_intervals, y=selected_day_data["demand"],
            mode="lines", name="Infl. consumption",
            line=dict(color="orange"),
            fill="tozeroy", fillcolor="rgba(255, 165, 0, 0.3)"
        ))

    # Add space heating
    if 'space_heating_cum' in selected_day_data.columns:
        fig.add_trace(go.Scatter(
            x=time_intervals, y=selected_day_data["space_heating_cum"],
            mode="lines", name="Space heating",
            line=dict(color="violet"),
            fill="tonexty", fillcolor="rgba(138, 43, 226, 0.3)"
        ))

    # Add DHW
    if 'dhw_cum' in selected_day_data.columns:
        fig.add_trace(go.Scatter(
            x=time_intervals, y=selected_day_data["dhw_cum"],
            mode="lines", name="DHW",
            line=dict(color="purple"),
            fill="tonexty", fillcolor="rgba(255, 99, 71, 0.3)"
        ))

    # Add BEV charging
    if 'ev_cum' in selected_day_data.columns:
        fig.add_trace(go.Scatter(
            x=time_intervals, y=selected_day_data["ev_cum"],
            mode="lines", name="BEV charging",
            line=dict(color="red"),
            fill="tonexty", fillcolor="rgba(220, 20, 60, 0.3)"
        ))

    # Add BESS charging
    if 'bat_cum' in selected_day_data.columns:
        fig.add_trace(go.Scatter(
            x=time_intervals, y=selected_day_data["bat_cum"],
            mode="none", name="BESS charging",
            fill="tonexty",  fillcolor="rgba(173, 216, 230, 0.3)",  # Light blue with 30% opacity
            fillpattern=dict(shape=".")
        ))

    # Add BESS discharging
    if 'bat_dis_cum' in selected_day_data.columns:
        fig.add_trace(go.Scatter(
            x=time_intervals, y=selected_day_data["bat_dis_cum"],
            mode="none", name="BESS discharging",
            fill="tonexty", fillcolor="rgba(144, 238, 144, 0.3)",  # Light green with 30% opacity
            fillpattern=dict(shape="x")  # Correct way to add a pattern
        ))

    # Add PV generation
    if 'pv_generation' in selected_day_data.columns:
        fig.add_trace(go.Scatter(
            x=time_intervals, y=selected_day_data["pv_generation"],
            mode="lines", name="PV generation",
            fill="tozeroy", fillcolor="rgba(255, 255, 0, 0.3)",
            line=dict(color="yellow")
        ))

    # Add net load (primary y-axis)
    if 'net_load_lv_grid' in selected_day_data.columns:
        fig.add_trace(go.Scatter(
            x=time_intervals,
            y=selected_day_data["net_load_lv_grid"],
            mode="lines",
            name="Residual load",
            line=dict(color="blue", dash="dot")
        ))

    # Add Grid tariff (right y-axis, with area fill)
    fig.add_trace(go.Scatter(
        x=time_intervals,
        y=[tariff] * len(time_intervals),  # Repeat tariff value for each interval
        mode="lines",
        name="Grid tariff (ct/kWh)",
        line=dict(color="green"),
        yaxis="y2"
    ))

    # Update layout for dual y-axes
    fig.update_layout(
        title=f"Load summary and grid tariff - {chosen_date}",
        xaxis_title="Time intervals",
        yaxis=dict(
            title="Net load on LV grid [kW]",
            side="left"
        ),
        yaxis2=dict(
            title="Grid tariff [ct/kWh]",
            # range=[price_min, price_max],  # Adjust lower bound
            overlaying="y",
            side="right",
            tickvals=[tariff],
            ticktext=[ticktxt],
            tickfont=dict(color="green")  # Set tick color to green
        ),
        legend=dict(
            title="Legend",
            x=1.6,
            y=1.0,  # Moved slightly higher
            xanchor="right",
            yanchor="top"
        ),
        template="plotly_white",
        font=dict(color="black")
    )

    # Fix x-axis to hourly labels
    fig.update_xaxes(tickformat="%H:%M", dtick=3600000)
    
    # Save plots
    html_file_path = f'results\\{model}\\{run_name}\\a_mpd-50_{chosen_date}_{model}.html'
    fig.write_html(html_file_path)

    pdf_file_path = f'results\\{model}\\{run_name}\\a_mpd-50_{chosen_date}_{model}.pdf'
    fig.write_image(pdf_file_path)

# Example usage
generate_plot_base(2021, 6, 16, combined_timeseries, 0.057)
generate_plot_base(2021, 12, 2, combined_timeseries, 0.057)
generate_plot_base(2021, 10, 17, combined_timeseries, 0.057)

# uv run 3_manipulation.py
#******************************************************************************************************************************************
# graph of most energy consumption on one day Base Scenario & MPD-50 (including grid tariff)
import plotly.graph_objects as go
import pandas as pd
from plotly.subplots import make_subplots

def generate_base_comparison_plots(df_combined, tariff, model2):
    
    # Extract the day with the maximum and minimum net load on the low-voltage grid
    if 'net_load_lv_grid' in df_combined.columns:
        max_day_idx = df_combined['net_load_lv_grid'].idxmax()
        min_day_idx = df_combined['net_load_lv_grid'].idxmin()

        day_start_max_idx = max_day_idx - (max_day_idx % 96)
        day_start_min_idx = min_day_idx - (min_day_idx % 96)

        max_day_data = df_combined.iloc[day_start_max_idx:day_start_max_idx + 96]
        min_day_data = df_combined.iloc[day_start_min_idx:day_start_min_idx + 96]

    days_passed_max = day_start_max_idx // 96
    days_passed_min = day_start_min_idx // 96

    corrected_date_max = pd.to_datetime("2021-01-01") + pd.Timedelta(days=days_passed_max)
    corrected_date_min = pd.to_datetime("2021-01-01") + pd.Timedelta(days=days_passed_min)

    time_intervals_max = pd.date_range(start=f"{corrected_date_max.date()} 00:00", freq="15T", periods=96)
    time_intervals_min = pd.date_range(start=f"{corrected_date_min.date()} 00:00", freq="15T", periods=96)

    # Energy calculations
    energy_max_day = max_day_data['net_load_lv_grid'].sum() / 4
    energy_min_day = min_day_data['net_load_lv_grid'].sum() / 4

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=[f"{corrected_date_max.date()}", f"{corrected_date_min.date()}"],
        shared_yaxes=True,
        specs=[[{"secondary_y": True}, {"secondary_y": True}]],
        horizontal_spacing=0.05  # Default is 0.2 — try 0.05 or even 0.03
    )

    color_map = {
        "demand": ("line", "orange", "Demand", None, "rgba(255, 165, 0, 0.3)", "tozeroy", None),
        "space_heating_cum": ("line", "violet", "Space heating", None, "rgba(138, 43, 226, 0.3)", "tonexty", None),
        "dhw_cum": ("line", "purple", "DHW", None, "rgba(255, 99, 71, 0.3)", "tonexty", None),
        "ev_cum": ("line", "red", "BEV char.", None, "rgba(220, 20, 60, 0.3)", "tonexty", None),
        "bat_cum": (None, None, "BESS char.", None, "rgba(173, 216, 230, 0.3)", "tonexty", "."),
        "bat_dis_cum": (None, None, "BESS dischar.", None, "rgba(144, 238, 144, 0.3)", "tonexty", "x"),
        "pv_generation": ("line", "yellow", "PV generation", None, "rgba(255, 255, 0, 0.3)", "tozeroy", None),
        "net_load_lv_grid": ("line", "blue", "Net load", "dot", None, None, None)
    }

    def add_traces(fig, data, time_labels, col, show_legend):
        for key, (_, color, display_name, dash, fillcolor, fillmode, shape) in color_map.items():
            if key in data.columns:
                fig.add_trace(go.Scatter(
                    x=time_labels,
                    y=data[key],
                    mode="lines",  # Always use lines
                    name=display_name,
                    line=dict(color=color or "grey", dash=dash or "solid"),
                    fill=fillmode or None,
                    fillcolor=fillcolor or None,
                    fillpattern=dict(shape=shape) if shape else None,
                    showlegend=show_legend
                ), row=1, col=col, secondary_y=False)

    # Add traces
    add_traces(fig, max_day_data, time_intervals_max, 1, show_legend=True)
    add_traces(fig, min_day_data, time_intervals_min, 2, show_legend=False)

    # Add grid tariff line (secondary y-axis)
    fig.add_trace(go.Scatter(
        x=time_intervals_max,
        y=[tariff] * len(time_intervals_max),
        mode="lines",
        name="Grid tariff",
        line=dict(color="green"),
        showlegend=True
    ), row=1, col=1, secondary_y=True)

    fig.add_trace(go.Scatter(
        x=time_intervals_min,
        y=[tariff] * len(time_intervals_min),
        mode="lines",
        name="Grid tariff",
        line=dict(color="green"),
        showlegend=False
    ), row=1, col=2, secondary_y=True)
    
    # Layout with legend below
    fig.update_layout(
        title="Comparison of maximal and minimal consumption days",
        font=dict(color="black"),
        template="plotly_white",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.4,  # Adjust vertical position as needed
            xanchor="center",
            x=0.2,
            title=None
        ),
        margin=dict(b=120)  # Add extra bottom margin for legend
    )

    # Y-axes
    fig.update_yaxes(
        title_text="Net load on LV grid [kW]",
        row=1, col=1,
        secondary_y=False
    )
    fig.update_yaxes(
        title_text="", tickvals=[], ticktext=[],
        row=1, col=1, secondary_y=True
    )
    fig.update_yaxes(
        title_text="Grid tariff [ct/kWh]",
        tickvals=[tariff],
        ticktext=[str(tariff * 100)],
        tickfont=dict(color="green"),
        row=1, col=2,
        secondary_y=True
    )

    # X-axes
    fig.update_xaxes(tickformat="%H:%M", dtick=10800000, tickangle=90, row=1, col=1)
    fig.update_xaxes(tickformat="%H:%M", dtick=10800000, tickangle=90, row=1, col=2)

    fig.add_annotation(
        text=f"Max day: {energy_max_day:.2f} kWh<br>Min day: {energy_min_day:.2f} kWh",
        xref="paper", yref="paper",
        x=1.05, y=-0.3,
        align="left",
        showarrow=False,
        font=dict(size=12),
        borderpad=0
    )

    # Save
    fig.write_html(f'results\\{model}\\{run_name}\\a_{model2}_comparison_load_days.html')
    fig.write_image(f'results\\{model}\\{run_name}\\a_{model2}_comparison_load_days.pdf')
    print("Comparison of maximal and minimal consumption days saved as PDF and HTML.")

# Example usage
generate_base_comparison_plots(combined_timeseries, 0.104, "base")
generate_base_comparison_plots(combined_timeseries, 0.057, "mpd-50")

#******************************************************************************************************************************************
# graph of most energy consumption on one day MPD (no grid tariff)
import plotly.graph_objects as go
import pandas as pd
from plotly.subplots import make_subplots

def generate_base_comparison_plots(df_combined, tariff, model2):
    # Extract the day with the maximum and minimum net load on the low-voltage grid
    if 'net_load_lv_grid' in df_combined.columns:
        max_day_idx = df_combined['net_load_lv_grid'].idxmax()
        min_day_idx = df_combined['net_load_lv_grid'].idxmin()

        day_start_max_idx = max_day_idx - (max_day_idx % 96)
        day_start_min_idx = min_day_idx - (min_day_idx % 96)

        max_day_data = df_combined.iloc[day_start_max_idx:day_start_max_idx + 96]
        min_day_data = df_combined.iloc[day_start_min_idx:day_start_min_idx + 96]

    days_passed_max = day_start_max_idx // 96
    days_passed_min = day_start_min_idx // 96

    corrected_date_max = pd.to_datetime("2021-01-01") + pd.Timedelta(days=days_passed_max)
    corrected_date_min = pd.to_datetime("2021-01-01") + pd.Timedelta(days=days_passed_min)

    time_intervals_max = pd.date_range(start=f"{corrected_date_max.date()} 00:00", freq="15T", periods=96)
    time_intervals_min = pd.date_range(start=f"{corrected_date_min.date()} 00:00", freq="15T", periods=96)

    # Energy calculations
    energy_max_day = max_day_data['net_load_lv_grid'].sum() / 4
    energy_min_day = min_day_data['net_load_lv_grid'].sum() / 4

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=[f"{corrected_date_max.date()}", f"{corrected_date_min.date()}"],
        shared_yaxes=True,
        specs=[[{"secondary_y": True}, {"secondary_y": True}]],
        horizontal_spacing=0.05
    )

    color_map = {
        "demand": ("line", "orange", "Demand", None, "rgba(255, 165, 0, 0.3)", "tozeroy", None),
        "space_heating_cum": ("line", "violet", "Space heating", None, "rgba(138, 43, 226, 0.3)", "tonexty", None),
        "dhw_cum": ("line", "purple", "DHW", None, "rgba(255, 99, 71, 0.3)", "tonexty", None),
        "ev_cum": ("line", "red", "BEV char.", None, "rgba(220, 20, 60, 0.3)", "tonexty", None),
        "bat_cum": (None, None, "BESS char.", None, "rgba(173, 216, 230, 0.3)", "tonexty", "."),
        "bat_dis_cum": (None, None, "BESS dischar.", None, "rgba(144, 238, 144, 0.3)", "tonexty", "x"),
        "pv_generation": ("line", "yellow", "PV generation", None, "rgba(255, 255, 0, 0.3)", "tozeroy", None),
        "net_load_lv_grid": ("line", "blue", "Net load", "dot", None, None, None)
    }

    def add_traces(fig, data, time_labels, col, show_legend):
        for key, (_, color, display_name, dash, fillcolor, fillmode, shape) in color_map.items():
            if key in data.columns:
                fig.add_trace(go.Scatter(
                    x=time_labels,
                    y=data[key],
                    mode="lines",
                    name=display_name,
                    line=dict(color=color or "grey", dash=dash or "solid"),
                    fill=fillmode or None,
                    fillcolor=fillcolor or None,
                    fillpattern=dict(shape=shape) if shape else None,
                    showlegend=show_legend
                ), row=1, col=col, secondary_y=False)

    # Add traces
    add_traces(fig, max_day_data, time_intervals_max, 1, show_legend=True)
    add_traces(fig, min_day_data, time_intervals_min, 2, show_legend=False)

    # Conditionally add grid tariff line
    if tariff > 0:
        fig.add_trace(go.Scatter(
            x=time_intervals_max,
            y=[tariff] * len(time_intervals_max),
            mode="lines",
            name="Grid tariff",
            line=dict(color="green"),
            showlegend=True
        ), row=1, col=1, secondary_y=True)

        fig.add_trace(go.Scatter(
            x=time_intervals_min,
            y=[tariff] * len(time_intervals_min),
            mode="lines",
            name="Grid tariff",
            line=dict(color="green"),
            showlegend=False
        ), row=1, col=2, secondary_y=True)

    # Layout
    fig.update_layout(
        title="Comparison of maximal and minimal consumption days",
        font=dict(color="black"),
        template="plotly_white",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.4,
            xanchor="center",
            x=0.2,
            title=None
        ),
        margin=dict(b=120)
    )

    # Y-axes
    fig.update_yaxes(
        title_text="Net load on LV grid [kW]",
        row=1, col=1,
        secondary_y=False
    )
    fig.update_yaxes(
        title_text="", tickvals=[], ticktext=[],
        row=1, col=1, secondary_y=True
    )
    if tariff > 0:
        fig.update_yaxes(
            title_text="Grid tariff [ct/kWh]",
            tickvals=[tariff],
            ticktext=[str(tariff * 100)],
            tickfont=dict(color="green"),
            row=1, col=2,
            secondary_y=True
        )
    else:
        fig.update_yaxes(
            title_text="",
            tickvals=[],
            ticktext=[],
            row=1, col=2,
            secondary_y=True
        )

    # X-axes with vertical labels
    fig.update_xaxes(tickformat="%H:%M", dtick=10800000, tickangle=90, row=1, col=1)
    fig.update_xaxes(tickformat="%H:%M", dtick=10800000, tickangle=90, row=1, col=2)

    # Annotation
    fig.add_annotation(
        text=f"Max day: {energy_max_day:.2f} kWh<br>Min day: {energy_min_day:.2f} kWh",
        xref="paper", yref="paper",
        x=1.05, y=-0.3,
        align="left",
        showarrow=False,
        font=dict(size=12),
        borderpad=0
    )

    # Save
    fig.write_html(f'results\\{model}\\{run_name}\\a_{model2}_comparison_load_days.html')
    fig.write_image(f'results\\{model}\\{run_name}\\a_{model2}_comparison_load_days.pdf')
    print(f"Comparison for '{model2}' saved as PDF and HTML.")

generate_base_comparison_plots(combined_timeseries, 0.0, "mpd")  # No grid cost shown
# uv run 3_manipulation.py
#******************************************************************************************************************************************
#LDC load duration curve
import plotly.graph_objects as go

# Create a new figure for the Load Duration Curve (LDC)
fig_ldc = go.Figure()

# Sort the 'grid_load' column in descending order for LDC
if 'net_load_lv_grid' in combined_timeseries.columns:
    sorted_grid_load = combined_timeseries['net_load_lv_grid'].sort_values(ascending=False).reset_index(drop=True)
    fig_ldc.add_trace(go.Scatter(x=sorted_grid_load.index, y=sorted_grid_load, mode='lines', name='Load Duration Curve', line=dict(color='black')
    ))

# Add vertical lines for 5% and 80% of the x-axis
x_5_percent = int(0.05 * len(sorted_grid_load))
x_80_percent = int(0.80 * len(sorted_grid_load))

fig_ldc.add_shape(type="line", x0=x_5_percent, y0=sorted_grid_load.min(), x1=x_5_percent, y1=sorted_grid_load.max(), line=dict(color="blue", dash="dash")
)
fig_ldc.add_shape(type="line", x0=x_80_percent, y0=sorted_grid_load.min(), x1=x_80_percent, y1=sorted_grid_load.max(), line=dict(color="blue", dash="dash")
)

# Add annotations for circled numbers
max_load = sorted_grid_load.max()

# Add annotations for circled numbers with adjusted positions and smaller size
lower_factor_1 = 0.2  # Lowering factor for number ①
lower_factor_others = 0.45  # Lowering factor for numbers ② and ③

fig_ldc.add_trace(go.Scatter(
    x=[int(0.025 * len(sorted_grid_load)), int(0.45 * len(sorted_grid_load)), int(0.85 * len(sorted_grid_load))],
    y=[max_load * lower_factor_1, max_load * lower_factor_others, max_load * lower_factor_others],  # Adjusted positions
    mode='text', 
    text=['①', '②', '③'], 
    textfont=dict(size=20, color='black', family='Arial Black'),  # Smaller font size for circled numbers
    showlegend=False
))

# Combine the two metric texts into one annotation
combined_metrics_text = (
    f"1: Peak<br>2: Intermediate<br>3: Base<br>"
)

# Add the combined metrics annotation to the plot
fig_ldc.add_annotation(
    text=combined_metrics_text,
    xref="paper", yref="paper",
    x=1, y=1,  # Adjust the position as needed
    showarrow=False,
    align="left",
    font=dict(size=18, color="black"),
    bgcolor="white",
    bordercolor="black"
)

# Customize the layout for the LDC plot
fig_ldc.update_layout(
    xaxis_title='Sorted 15-min intervals (highest to lowest load)', yaxis_title='Grid load [kW]', template='plotly_white',
    showlegend=False, font=dict(size=16, color="black"),
    margin=dict(l=20, r=20, t=20, b=20),
    height= 300
)

# Save the Load Duration Curve as an HTML file
html_file_path_ldc = f'results\\{model}\\{run_name}\\a_load_duration_curve_{model}.html'
fig_ldc.write_html(html_file_path_ldc)
# Save the Load Duration Curve as a PNG file

pdf_file_path_ldc = f'results\\{model}\\{run_name}\\a_load_duration_curve_{model}.pdf'
fig_ldc.write_image(pdf_file_path_ldc)

print(f"Load Duration Curve saved as PDF at {pdf_file_path_ldc}")
print(f"Load Duration Curve saved as HTML at {html_file_path_ldc}")

#******************************************************************************************************************************************
# uv run 3_manipulation.py
# Stacked graph for net_load of each household on max power day
# Extract the day with the maximum net load on the low-voltage grid
if 'net_load_lv_grid' in combined_timeseries.columns:
    max_day_idx = combined_timeseries['net_load_lv_grid'].idxmax()
    day_start_idx = max_day_idx - (max_day_idx % 96)  # Align to start of the day
    max_day_data = combined_timeseries.iloc[day_start_idx:day_start_idx + 96]

    # Calculate the correct date from row index
    days_passed = day_start_idx // 96  # Number of full days since start of 2021
    corrected_date = pd.to_datetime("2021-01-01") + pd.Timedelta(days=days_passed)

    # Generate time intervals for plotting
    time_intervals = pd.date_range(start=f"{corrected_date.date()} 00:00", freq="15T", periods=96)
    max_day_data.index = time_intervals  # Assign time labels for plotting

    # Define the number of households for each dataset
    num_households_net_load = 80
    num_households_market_exp = 120  # 81 to 200

    # Select household columns for the chosen day
    net_load_data = max_day_data[[f"net_load_hh_{i}" for i in range(1, num_households_net_load + 1)]]
    market_exp_data = max_day_data[[f"el_market_exp_value_{i}" for i in range(81, num_households_market_exp + 81)]]

    # Define color gradient cycling logic
    colors = []
    color_groups = ["pink", "red", "blue", "violet", "green"]
    for i in range(num_households_net_load + num_households_market_exp):
        colors.append(color_groups[(i // 10) % len(color_groups)])

    # Create Plotly figure
    fig = go.Figure()

    # Add stacked traces for net_load households (1-80)
    fig.add_trace(go.Scatter(
        x=net_load_data.index,
        y=net_load_data[f"net_load_hh_1"],
        fill='tozeroy',
        mode='lines',
        line=dict(width=0.5, color=colors[1]),
        name=f"Household 1 (Net Load)",
        stackgroup='one'
    ))

    # Add stacked traces for net_load households (1-80)
    for i in range(2, num_households_net_load + 1):
        fig.add_trace(go.Scatter(
            x=net_load_data.index,
            y=net_load_data[f"net_load_hh_{i}"],
            fill='tonexty',
            mode='lines',
            line=dict(width=0.5, color=colors[i - 1]),
            name=f"Household {i} (Net Load)",
            stackgroup='one'
        ))

    # Add stacked traces for el_market_exp households (81-200)
    for i in range(81, num_households_market_exp + 81):
        fig.add_trace(go.Scatter(
            x=market_exp_data.index,
            y=market_exp_data[f"el_market_exp_value_{i}"],
            fill='tonexty',
            mode='lines',
            line=dict(width=0.5, color=colors[i - 1]),
            name=f"Household {i} (Market Exp)",
            stackgroup='one'
        ))

        # Add net load (primary y-axis)
    if 'net_load_lv_grid' in max_day_data.columns:
        fig.add_trace(go.Scatter(
            x=time_intervals,
            y=max_day_data["net_load_lv_grid"],
            mode="lines",
            name="Residual load",
            line=dict(color="blue", dash="dot")
        ))

    # Configure layout
    fig.update_layout(
        title=f"Stacked Area Plot - Net Load on {corrected_date.date()}",
        xaxis_title="Time Intervals (15-min resolution)",
        yaxis_title="Value [kWh]",
        showlegend=False,
        template="plotly_white",
        font=dict(color="black")
    )

    # Save as HTML
    fig_indivdual_net_load = f'results\\{model}\\{run_name}\\a_stacked_individual_max_net_load_{model}.html'
    fig.write_html(fig_indivdual_net_load)

    print(f"Saved as {fig_indivdual_net_load}")

else:
    print("Error: 'net_load_lv_grid' column not found in dataset")
#******************************************************************************************************************************************
# uv run 3_manipulation.py
# Stacked graph for net_load of each household on min power day
# Extract the day with the minimum net load on the low-voltage grid
if 'net_load_lv_grid' in combined_timeseries.columns:
    min_day_idx = combined_timeseries['net_load_lv_grid'].idxmin()  # Find index of minimum net load
    day_start_idx = min_day_idx - (min_day_idx % 96)  # Align to start of the day
    min_day_data = combined_timeseries.iloc[day_start_idx:day_start_idx + 96]

    # Calculate the correct date from row index
    days_passed = day_start_idx // 96  # Number of full days since start of 2021
    corrected_date = pd.to_datetime("2021-01-01") + pd.Timedelta(days=days_passed)

    # Generate date and time labels for the intervals (assuming 15-minute intervals in a day)
    time_intervals = pd.date_range(start=f"{corrected_date.date()} 00:00", freq="15T", periods=96)
    min_day_data.index = time_intervals  # Assign time labels for plotting

    # Define the number of households for each dataset
    num_households_net_load = 80
    num_households_market_exp = 120  # 81 to 200

    # Select household columns for the chosen day
    net_load_data = min_day_data[[f"net_load_hh_{i}" for i in range(1, num_households_net_load + 1)]]
    market_exp_data = min_day_data[[f"el_market_exp_value_{i}" for i in range(81, num_households_market_exp + 81)]]

    # Define color gradient cycling logic
    colors = []
    color_groups = ["pink", "red", "blue", "violet", "green"]
    for i in range(num_households_net_load + num_households_market_exp):
        colors.append(color_groups[(i // 10) % len(color_groups)])

    # Create Plotly figure
    fig = go.Figure()

    # Add stacked traces for net_load households (1-80)
    fig.add_trace(go.Scatter(
        x=net_load_data.index,
        y=net_load_data[f"net_load_hh_1"],
        fill='tozeroy',
        mode='lines',
        line=dict(width=0.5, color=colors[1]),
        name=f"Household 1 (Net Load)",
        stackgroup='one'
    ))

    # Add stacked traces for net_load households (1-80)
    for i in range(2, num_households_net_load + 1):
        fig.add_trace(go.Scatter(
            x=net_load_data.index,
            y=net_load_data[f"net_load_hh_{i}"],
            fill='tonexty',
            mode='lines',
            line=dict(width=0.5, color=colors[i - 1]),
            name=f"Household {i} (Net Load)",
            stackgroup='one'
        ))

    # Add stacked traces for el_market_exp households (81-200)
    for i in range(81, num_households_market_exp + 81):
        fig.add_trace(go.Scatter(
            x=market_exp_data.index,
            y=market_exp_data[f"el_market_exp_value_{i}"],
            fill='tonexty',
            mode='lines',
            line=dict(width=0.5, color=colors[i - 1]),
            name=f"Household {i} (Market Exp)",
            stackgroup='one'
        ))

        # Add net load (primary y-axis)
    if 'net_load_lv_grid' in min_day_data.columns:
        fig.add_trace(go.Scatter(
            x=time_intervals,
            y=min_day_data["net_load_lv_grid"],
            mode="lines",
            name="Residual load",
            line=dict(color="blue", dash="dot")
        ))

    # Configure layout
    fig.update_layout(
        title=f"Stacked Area Plot - Net Load on {corrected_date.date()}",
        xaxis_title="Time Intervals (15-min resolution)",
        yaxis_title="Value [kWh]",
        showlegend=False,
        template="plotly_white",
        font=dict(color="black")
    )

    # Save as HTML
    fig_indivdual_net_load = f'results\\{model}\\{run_name}\\a_stacked_individual_min_net_load_{model}.html'
    fig.write_html(fig_indivdual_net_load)

    print(f"Saved as {fig_indivdual_net_load}")

else:
    print("Error: 'net_load_lv_grid' column not found in dataset")
#******************************************************************************************************************************************
# uv run 3_manipulation.py
# Stacked graph für wunsch datum
# Extract the day with the minimum net load on the low-voltage grid
import pandas as pd
import plotly.graph_objects as go

# Hardcoded desired date (Changeable in future)
desired_day = 24
desired_month = 12
desired_year = 2021

# Define dataset structure (assuming full year, 96 intervals/day)
days_per_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]  # Adjust for leap years
rows_per_day = 96

# Calculate starting row index
month_offset = sum(days_per_month[:desired_month - 1]) * rows_per_day  # Sum rows from previous months
day_start_idx = month_offset + (desired_day - 1) * rows_per_day
day_end_idx = day_start_idx + rows_per_day  # Capture full day (96 rows)

# Load dataset
combined_timeseries = pd.read_csv(input_file_path)

# Extract data for the selected day
selected_day_data = combined_timeseries.iloc[day_start_idx:day_end_idx]

# Generate time labels (15-min steps)
time_intervals = pd.date_range(start=f"{desired_year}-{desired_month:02d}-{desired_day:02d} 00:00",
                               freq="15T", periods=rows_per_day)
selected_day_data.index = time_intervals  # Assign time labels for plotting

# Define number of households
num_households_net_load = 80
num_households_market_exp = 120  # 81 to 200

# Select household columns
net_load_data = selected_day_data[[f"net_load_hh_{i}" for i in range(1, num_households_net_load + 1)]]
market_exp_data = selected_day_data[[f"el_market_exp_value_{i}" for i in range(81, num_households_market_exp + 81)]]

# Define color gradient cycling logic
color_groups = ["pink", "red", "blue", "violet", "green"]
colors = [color_groups[(i // 10) % len(color_groups)] for i in range(num_households_net_load + num_households_market_exp)]

# Create Plotly figure
fig = go.Figure()

# Add stacked traces for net_load households (1-80)
fig.add_trace(go.Scatter(
    x=time_intervals,
    y=net_load_data[f"net_load_hh_1"],
    fill='tozeroy',
    mode='lines',
    line=dict(width=0.5, color=colors[0]),
    name=f"Household 1 (Net Load)",
    stackgroup='one'
))

# Add remaining stacked traces for net_load households (2-80)
for i in range(2, num_households_net_load + 1):
    fig.add_trace(go.Scatter(
        x=time_intervals,
        y=net_load_data[f"net_load_hh_{i}"],
        fill='tonexty',
        mode='lines',
        line=dict(width=0.5, color=colors[i - 1]),
        name=f"Household {i} (Net Load)",
        stackgroup='one'
    ))

# Add stacked traces for el_market_exp households (81-200)
for i in range(81, num_households_market_exp + 81):
    fig.add_trace(go.Scatter(
        x=time_intervals,
        y=market_exp_data[f"el_market_exp_value_{i}"],
        fill='tonexty',
        mode='lines',
        line=dict(width=0.5, color=colors[(i - 81)]),
        name=f"Household {i} (Market Exp)",
        stackgroup='one'
    ))

# Configure layout
fig.update_layout(
    title=f"Stacked Area Plot - Net Load on {desired_day:02d}-{desired_month:02d}-{desired_year}",
    xaxis_title="Time Intervals (15-min resolution)",
    yaxis_title="Value [kWh]",
    showlegend=False,
    template="plotly_white",
    font=dict(color="black")
)

# Save as HTML
fig_individual_net_load = f'results\\{model}\\{run_name}\\a_stacked_individual_net_load_{model}_{desired_day:02d}-{desired_month:02d}-{desired_year}.html'
fig_individual_net_pdf = f'results\\{model}\\{run_name}\\a_stacked_individual_net_load_{model}_{desired_day:02d}-{desired_month:02d}-{desired_year}.pdf'
fig.write_html(fig_individual_net_load)
fig.write_image(fig_individual_net_pdf)
print(f"Saved as {fig_individual_net_load}")
#******************************************************************************************************************************************
# uv run 3_manipulation.py
# Stacked graph for net_load of each household on max power day
# # Extract the day with the maximum net load on the low-voltage grid
# # Create Plotly figure
# fig = go.Figure()

# if 'bat_charge' in combined_timeseries.columns:
#     fig.add_trace(go.Scatter(x=combined_timeseries.index, y=combined_timeseries['bat_charge'], mode='lines', name='Bat charge', line=dict(color='red'), fill='tozeroy',))

# if 'bat_cum' in combined_timeseries.columns:
#     fig.add_trace(go.Scatter(x=combined_timeseries.index, y=combined_timeseries['bat_cum'], mode='lines', name='bat_cum', line=dict(color='blue'), fill='tozeroy',))

# if 'bat_discharge_cum' in combined_timeseries.columns:
#     fig.add_trace(go.Scatter(x=combined_timeseries.index, y=combined_timeseries['bat_discharge_cum'], mode='lines', name='bbat_discharge_cum', line=dict(color='blue'), fill='tozeroy',))

# # Configure layout
# fig.update_layout(
#     title=f"Stacked Area Plot - Net Load on {corrected_date.date()}",
#     xaxis_title="Time Intervals (15-min resolution)",
#     yaxis_title="Value [kWh]",
#     showlegend=False,
#     template="plotly_white"
# )

# # Save as HTML
# fig_indivdual_net_load = f'results\\{model}\\{run_name}\\a_stacked_individual_max_net_load_{model}.html'
# fig.write_html(fig_indivdual_net_load)

# print(f"Saved as {fig_indivdual_net_load}")
# #******************************************************************************************************************************************
# uv run 3_manipulation.py 
# # average for mpd scenario
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import pandas as pd

fig = go.Figure()

# Select a household
household_num = 24
household_column = f'el_market_exp_value_{household_num}'
houdehold_column_1 = f'net_load_hh_{household_num}'

# Check if the household data is available
if household_column in combined_timeseries.columns:
    datetime_index = pd.date_range(start="2021-01-01 00:00", end="2021-12-31 23:45", freq="15min")
    datetime_index = datetime_index[:len(combined_timeseries)]

    # Add household data
    fig.add_trace(go.Scatter(
        x=datetime_index, 
        y=combined_timeseries[household_column], 
        mode='lines', 
        name='Residual load', 
        line=dict(color='blue', dash='dash'),
        opacity=0.8
    ))

    # Add horizontal "mean peak" line using a Scatter to ensure legend entry
    fig.add_trace(go.Scatter(
        x=[datetime_index.min(), datetime_index.max()],
        y=[0.33, 0.33],
        mode='lines',
        name='Arithmetic mean <br> of monthly peaks',
        line=dict(color='red', width=2, dash='solid')
    ))

    # Monthly dividers
    for month in range(2, 13):
        month_start = pd.Timestamp(f"2021-{month:02d}-01 00:00")
        fig.add_vline(x=month_start, line_width=1, line_dash="dash", line_color="gray", opacity=0.5)

    # Layout
    fig.update_layout(
        xaxis_title='Time intervals',
        yaxis_title='Load [kW]',
        template='plotly_white',
        legend_title='Legend',
        font=dict(color="black"),
        margin= dict(r=20, t=20, b=20, l=20),
        height= 300
    )

    # Safely ensure 0.33 is included as a tick
    fig.update_yaxes(
        tickvals=[0.0, 0.1, 0.2, 0.3, 0.33, 0.4, 0.5, 0.6],
        tickformat=".2f"
    )

    # X-axis ticks
    fig.update_xaxes(
        tickformat='%b',
        dtick="M1",
        ticklabelmode="period"
    )

    # Save files
    html_file_path = f'results\\{model}\\{run_name}\\a_yearly_net_load_mpd_comparison_{model}.html'
    fig.write_html(html_file_path)

    pdf_file_path = f'results\\{model}\\{run_name}\\a_yearly_net_load_mpd_comparison_{model}.pdf'
    fig.write_image(pdf_file_path)

    print(f"Yearly net load comparison plot saved as HTML at {html_file_path}")
    print(f"Yearly net load comparison plot saved as PDF at {pdf_file_path}")
else:
    print(f"Household {household_num} data not found in the dataset.")
#******************************************************************************************************************************************
# # uv run 3_manipulation.py

from plotly.subplots import make_subplots
import plotly.graph_objects as go
import pandas as pd

fig = go.Figure()

# Select a household
household_num = 74
household_column = f'el_market_exp_value_{household_num}'
household_column_1 = f'net_load_hh_{household_num}'

# Check if household data is available
if household_column in combined_timeseries.columns and household_column_1 in combined_timeseries.columns:
    datetime_index = pd.date_range(start="2021-01-01 00:00", end="2021-12-31 23:45", freq="15min")
    datetime_index = datetime_index[:len(combined_timeseries)]

    # Add residual load time series
    fig.add_trace(go.Scatter(
        x=datetime_index, 
        y=combined_timeseries[household_column], 
        mode='lines', 
        name='Residual Load',
        line=dict(color='blue', dash='dash'),
        opacity=0.8
    ))

    # Add net load time series
    fig.add_trace(go.Scatter(
        x=datetime_index, 
        y=combined_timeseries[household_column_1], 
        mode='lines', 
        name='Net Load',
        line=dict(color='green', dash='solid'),
        opacity=0.8
    ))

    # Monthly dividers
    for month in range(2, 13):
        month_start = pd.Timestamp(f"2021-{month:02d}-01 00:00")
        fig.add_vline(x=month_start, line_width=1, line_dash="dash", line_color="gray", opacity=0.5)

    # Layout adjustments
    fig.update_layout(
        xaxis_title='Time Intervals',
        yaxis_title='Load [kW]',  # Keep original y-axis labels
        template='plotly_white',
        legend_title='Legend',
        font=dict(color="black")
    )

    # X-axis formatting (monthly ticks)
    fig.update_xaxes(
        tickformat='%b',
        dtick="M1",
        ticklabelmode="period"
    )

    # Save files
    html_file_path = f'results\\{model}\\{run_name}\\a_yearly_net_load_comparison_{model}.html'
    fig.write_html(html_file_path)

    pdf_file_path = f'results\\{model}\\{run_name}\\a_yearly_net_load_comparison_{model}.pdf'
    fig.write_image(pdf_file_path)

    print(f"Yearly net load comparison plot saved as HTML at {html_file_path}")
    print(f"Yearly net load comparison plot saved as PDF at {pdf_file_path}")
else:
    print(f"Household {household_num} data not found in the dataset.")

# uv run 3_manipulation.py
#******************************************************************************************************************************************
# Function to generate graph on a household level
def generate_tou_hh_comparison_plots(desired_year, desired_month, desired_day, df_combined, tariff, household_num):
    # Define dataset structure (assuming full year, 96 intervals/day)
    days_per_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]  # Adjust for leap years
    rows_per_day = 96

    chosen_date = f"{desired_year}-{desired_month:02d}-{desired_day:02d}"

    # Adjust for leap years
    if desired_year % 4 == 0:
        days_per_month[1] = 29

    # Calculate starting row index
    month_offset = sum(days_per_month[:desired_month - 1]) * rows_per_day
    day_start_idx = month_offset + (desired_day - 1) * rows_per_day
    day_end_idx = day_start_idx + rows_per_day

    # Extract data for the selected day
    selected_day_data = df_combined.iloc[day_start_idx:day_end_idx]

    # Load TOU tariff data
    tou_tariff_values = tariff.iloc[day_start_idx:day_end_idx]['grid_tariff']

    # Generate time labels (15-min intervals)
    time_intervals = pd.date_range(
        start=f"{desired_year}-{desired_month:02d}-{desired_day:02d} 00:00",
        freq="15T", periods=rows_per_day
    )
    selected_day_data.index = time_intervals

    household_column = f'net_load_hh_{household_num}'
    household_column_1 = f'demand_exp_value_{household_num}'
    # household_column_2 = f'heat_unit_exp_in_electricity_{household_num}'
    # household_column_3 = f'water_unit_exp_in_electricity_{household_num}'
    household_column_4 = f'ev_unit_exp_in_electricity_{household_num}'
    household_column_5 = f'bat_charge_var_flow_{household_num}'
    household_column_6 = f'bat_discharge_var_flow_{household_num}'
    household_column_7 = f'pv_generation_exp_value_{household_num}'

    bess_dis = selected_day_data[household_column_1] + selected_day_data[household_column_4] + selected_day_data[household_column_5] - selected_day_data[household_column_6]
    bess_dis_plus =  bess_dis  + selected_day_data[household_column_6]

    pv_plus = selected_day_data[household_column] + selected_day_data[household_column_7]
    # uv run 3_manipulation.py

    # Define plot figure
    fig = go.Figure()
    
    # Add demand (primary y-axis)
    fig.add_trace(go.Scatter(
        x=time_intervals, y=selected_day_data[household_column_1],
        mode="lines", name="Infl. consumption",
        line=dict(color="orange", width=0.5),
        fill="tozeroy", fillcolor="rgba(255, 165, 0, 0.3)", stackgroup='one'
    ))

    # # Add space heating
    # if 'space_heating_cum' in selected_day_data.columns:
    #     fig.add_trace(go.Scatter(
    #         x=time_intervals, y=selected_day_data[household_column_2],
    #         mode="lines", name="Space heating",
    #         line=dict(color="violet"),
    #         fill="tonexty", fillcolor="rgba(138, 43, 226, 0.3)", stackgroup='one'
    #     ))

    # # Add DHW
    # if 'dhw_cum' in selected_day_data.columns:
    #     fig.add_trace(go.Scatter(
    #         x=time_intervals, y=selected_day_data[household_column_3],
    #         mode="lines", name="DHW",
    #         line=dict(color="purple"),
    #         fill="tonexty", fillcolor="rgba(255, 99, 71, 0.3)", stackgroup='one'
    #     ))

    if 'ev_cum' in selected_day_data.columns:
        fig.add_trace(go.Scatter(
            x=time_intervals, y=selected_day_data[household_column_4],
            mode="none", name="BEV charging",
            fill="tonexty", fillcolor="rgba(220, 20, 60, 0.3)", stackgroup='one'
        ))
# uv run 3_manipulation.py
    # Add BESS charging
    if 'bat_cum' in selected_day_data.columns:
        fig.add_trace(go.Scatter(
            x=time_intervals,
            y=selected_day_data[household_column_5],
            mode="none",
            name="BESS charging",
            fill="tonexty",
            fillpattern=dict(shape="+"),
            fillcolor="rgba(0,0,0,0)",
            stackgroup='one'
        ))

    # Add net load (primary y-axis)
    if 'net_load_lv_grid' in selected_day_data.columns:
        fig.add_trace(go.Scatter(
            x=time_intervals,
            y=selected_day_data[household_column],
            mode="lines",
            name="Residual load",
            line=dict(color="blue", dash="dot"),
            # fill="tozeroy", fillcolor="rgba(0, 0, 255, 0.2)"
        ))
#uv run 3_manipulation.py
# Add PV generation
    if 'pv_generation' in selected_day_data.columns:
        fig.add_trace(go.Scatter(
            x=time_intervals, y=pv_plus,
            mode="none", name="PV generation",
            fill="tonexty", fillcolor="rgba(255, 255, 0, 0.4)"
        ))

    # Add final adjusted load trace
    fig.add_trace(go.Scatter(
        x=time_intervals,
        y=bess_dis,
        mode="lines",
        showlegend=False,
        line=dict(color="rgba(0, 0, 255, 0)", width=2)
    ))

    # Add final adjusted load trace
    fig.add_trace(go.Scatter(
        x=time_intervals,
        y=bess_dis_plus,
        mode="none",
        name="BESS discharging",
        fillcolor="rgba(144, 238, 144, 0)",
        fillpattern=dict(shape="x"),
        fill="tonexty"
    ))

    fig.add_trace(go.Scatter(
        x=time_intervals,
        y=tou_tariff_values,  # Repeat tariff value for each interval
        mode="lines",
        name="Consump.-based <br> grid tariff (ct/kWh)",
        line=dict(color="green"),
        # fill="tozeroy",
        # fillcolor="rgba(144, 238, 144, 0.4)",
        yaxis="y2"
    ))

    # Update layout for dual y-axes
    fig.update_layout(
        title=f"Load profile of household 24 on {chosen_date}",
        xaxis_title="Time intervals",
        yaxis=dict(
            title="Load [kW]",
            # range=[-120, 390],
            side="left"
        ),
        yaxis2=dict(
            title="Consump.-based <br> grid tariff [ct/kWh]",
            # range=[price_min, price_max],  # Adjust lower bound
            overlaying="y",
            side="right",
            tickvals=[0.053, 0.105, 0.158],
            ticktext=["5.3", "10.5", "15.8"],
            tickfont=dict(color="green"),
            showgrid=False,
            gridcolor="rgba(0,0,0,0)"
        ),
        legend=dict(
            title="Legend",
            x=1.7,
            y=1.0,  # Moved slightly higher
            xanchor="right",
            yanchor="top"
        ),
        template="plotly_white",
        font=dict(color="black"),
        margin=dict(r=20, t=30, b=20, l=20),
        height= 300
    )

    # Fix x-axis to hourly labels
    fig.update_xaxes(tickformat="%H:%M", dtick=10800000, tickangle=90)
    
    # Save plots
    html_file_path = f'results\\{model}\\{run_name}\\a_single_tou_hh_{chosen_date}_{model}.html'
    fig.write_html(html_file_path)

    pdf_file_path = f'results\\{model}\\{run_name}\\a_single_tou_hh_{chosen_date}_{model}.pdf'
    fig.write_image(pdf_file_path)

# Example usage
generate_tou_hh_comparison_plots(2021, 12, 15, combined_timeseries, tou_tariff, 24)

# uv run 3_manipulation.py
#******************************************************************************************************************************************
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

def generate_hh_comparison_subplots(date1, date2, df_combined, household_num, model, run_name):
    def extract_day_data(year, month, day):
        days_per_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        if year % 4 == 0:
            days_per_month[1] = 29
        rows_per_day = 96
        month_offset = sum(days_per_month[:month - 1]) * rows_per_day
        day_start_idx = month_offset + (day - 1) * rows_per_day
        day_end_idx = day_start_idx + rows_per_day
        time_intervals = pd.date_range(f"{year}-{month:02d}-{day:02d} 00:00", freq="15T", periods=rows_per_day)
        selected = df_combined.iloc[day_start_idx:day_end_idx].copy()
        selected.index = time_intervals
        return selected, time_intervals

    def add_household_traces(fig, col, selected_day_data, time_intervals, show_legend):
        col_map = {
            'net_load': f'net_load_hh_{household_num}',
            'consumption': f'demand_exp_value_{household_num}',
            'bev': f'ev_unit_exp_in_electricity_{household_num}',
            'bat_charge': f'bat_charge_var_flow_{household_num}',
            'bat_discharge': f'bat_discharge_var_flow_{household_num}',
            'pv': f'pv_generation_exp_value_{household_num}'
        }

        bess_dis = (
            selected_day_data[col_map['consumption']] +
            selected_day_data[col_map['bev']] +
            selected_day_data[col_map['bat_charge']] -
            selected_day_data[col_map['bat_discharge']]
        )
        bess_dis_plus = bess_dis + selected_day_data[col_map['bat_discharge']]
        pv_plus = selected_day_data[col_map['net_load']] + selected_day_data[col_map['pv']]

        fig.add_trace(go.Scatter(
            x=time_intervals, y=selected_day_data[col_map['consumption']],
            mode="lines", name="Infl. consumption",
            line=dict(color="orange", width=0.5),
            fill="tozeroy", fillcolor="rgba(255, 165, 0, 0.3)",
            stackgroup=f"group{col}", legendgroup="group", showlegend=show_legend
        ), row=1, col=col)

        if 'ev_cum' in selected_day_data.columns:
            fig.add_trace(go.Scatter(
                x=time_intervals, y=selected_day_data[col_map['bev']],
                mode="none", name="BEV charging",
                fill="tonexty", fillcolor="rgba(179,179,179, 0.8)",
                stackgroup=f"group{col}", legendgroup="group", showlegend=show_legend
            ), row=1, col=col)

        if 'bat_cum' in selected_day_data.columns:
            fig.add_trace(go.Scatter(
                x=time_intervals, y=selected_day_data[col_map['bat_charge']],
                mode="none", name="BESS charging",
                fill="tonexty", fillpattern=dict(shape="+"), fillcolor="rgba(0,0,0,0)",
                stackgroup=f"group{col}", legendgroup="group", showlegend=show_legend
            ), row=1, col=col)

        if 'net_load_lv_grid' in selected_day_data.columns:
            fig.add_trace(go.Scatter(
                x=time_intervals, y=selected_day_data[col_map['net_load']],
                mode="lines", name="Residual load",
                line=dict(color="blue", dash="dot"),
                legendgroup="group", showlegend=show_legend
            ), row=1, col=col)

        if 'pv_generation' in selected_day_data.columns:
            fig.add_trace(go.Scatter(
                x=time_intervals, y=pv_plus,
                mode="none", name="PV generation",
                fill="tonexty", fillcolor="rgba(255, 255, 0, 0.3)", #0.2
                legendgroup="group", showlegend=show_legend
            ), row=1, col=col)

        fig.add_trace(go.Scatter(
            x=time_intervals, y=bess_dis,
            mode="lines", showlegend=False,
            line=dict(color="rgba(0, 0, 255, 0)", width=2)
        ), row=1, col=col)

        fig.add_trace(go.Scatter(
            x=time_intervals, y=bess_dis_plus,
            mode="none", name="BESS discharging",
            fill="tonexty", fillcolor="rgba(144, 238, 144, 0)", fillpattern=dict(shape="x"),
            legendgroup="group", showlegend=show_legend
        ), row=1, col=col)

    fig = make_subplots(
        rows=1, cols=2,
        shared_yaxes=True,
        subplot_titles=[
            f"Household {household_num} on {date1[0]}-{date1[1]:02d}-{date1[2]:02d}",
            f"Household {household_num} on {date2[0]}-{date2[1]:02d}-{date2[2]:02d}"
        ],
        horizontal_spacing=0.02  # Reduced spacing between subplots
    )

    day1_data, time1 = extract_day_data(*date1)
    add_household_traces(fig, 1, day1_data, time1, show_legend=True)

    day2_data, time2 = extract_day_data(*date2)
    add_household_traces(fig, 2, day2_data, time2, show_legend=False)

    tick_every_3_hours = 10800000  # 3 hours in ms

    fig.update_xaxes(title_text="Time intervals", tickformat="%H:%M", dtick=tick_every_3_hours, tickangle=90, row=1, col=1)
    fig.update_xaxes(title_text="Time intervals", tickformat="%H:%M", dtick=tick_every_3_hours, tickangle=90, row=1, col=2)
    fig.update_yaxes(title_text="Load [kW]", row=1, col=1)

    # Final layout
    fig.update_layout(
        height=400,
        width=900,
        margin=dict(l=10, r=10, t=30, b=10),
        template="plotly_white",
        font=dict(color="black"),
        legend=dict(
            title="Legend",
            orientation="v",
            x=1.02, y=1,
            xanchor="left", yanchor="top",
            bgcolor="rgba(255,255,255,0)",
            borderwidth=0,
            font=dict(size=10)
        ),
    )

    # Save
    out_base = f"a_comparison_hh_{household_num}_{date1[0]}{date1[1]:02d}{date1[2]:02d}_vs_{date2[0]}{date2[1]:02d}{date2[2]:02d}_{model}"
    html_path = f"results\\{model}\\{run_name}\\{out_base}.html"
    pdf_path = f"results\\{model}\\{run_name}\\{out_base}.pdf"
    fig.write_html(html_path)
    fig.write_image(pdf_path)

    print(f"Saved comparison plots:\n- HTML: {html_path}\n- PDF: {pdf_path}")

generate_hh_comparison_subplots(
    date1=(2021, 12, 15),
    date2=(2021, 6, 16),
    df_combined=combined_timeseries,
    household_num=24,
    model=model,
    run_name=run_name
)
# uv run 3_manipulation.py
#******************************************************************************************************************************************
# graph of represenative for Base day self chosen day.
import plotly.graph_objects as go
import pandas as pd

def generate_plot_base(desired_year, desired_month, desired_day, df_combined):
    days_per_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    rows_per_day = 96

    chosen_date = f"{desired_year}-{desired_month:02d}-{desired_day:02d}"

    if desired_year % 4 == 0:
        days_per_month[1] = 29

    month_offset = sum(days_per_month[:desired_month - 1]) * rows_per_day
    day_start_idx = month_offset + (desired_day - 1) * rows_per_day
    day_end_idx = day_start_idx + rows_per_day

    selected_day_data = df_combined.iloc[day_start_idx:day_end_idx]

    time_intervals = pd.date_range(
        start=f"{desired_year}-{desired_month:02d}-{desired_day:02d} 00:00",
        freq="15T", periods=rows_per_day
    )
    selected_day_data.index = time_intervals


    bess_dis = selected_day_data['ev_cum'] + selected_day_data['bat_charge']
    bess_dis_plus = bess_dis - selected_day_data['bat_discharge']
    pv_plus = selected_day_data['net_load_lv_grid'] + selected_day_data['pv_generation']

    fig = go.Figure()

    # Order and visual styling based on depiction
    fig.add_trace(go.Scatter(
        x=time_intervals, y=selected_day_data["ev_cum"], # "demand"
        mode="lines", name="Consumption",
        line=dict(color="red", width=0.5),
        fill="tozeroy", fillcolor="rgba(251,128,114, 0.3)", stackgroup="group" #  255, 165, 0,
    ))
    fig.add_trace(go.Scatter(
        x=time_intervals, y=selected_day_data["bat_charge"],
        mode="none", name="BESS charging",
        fill="tonexty", fillcolor="rgba(0, 0, 0, 0)",
        fillpattern=dict(shape="+"), stackgroup="group"
    ))
    fig.add_trace(go.Scatter(
        x=time_intervals, y=selected_day_data["net_load_lv_grid"],
        mode="lines", name="Residual load",
        line=dict(color="blue", dash="dot"),
    ))
    fig.add_trace(go.Scatter(
        x=time_intervals, y=pv_plus,
        mode="none", name="PV generation",
        line=dict(color="yellow"),
        fill="tonexty", fillcolor="rgba(255, 255, 0, 0.3)"
    ))
    fig.add_trace(go.Scatter(
        x=time_intervals, y=bess_dis,
        mode="lines", showlegend=False,
        line=dict(color="rgba(0, 0, 255, 0)", width=2)
    ))
    fig.add_trace(go.Scatter(
        x=time_intervals, y=bess_dis_plus,
        mode="none", name="BESS discharging",
        fill="tonexty", fillcolor="rgba(0, 0, 0, 0)",
        fillpattern=dict(shape="x")
    ))

    fig.update_layout(
        title=dict(
            text=f"{chosen_date}",
            x=0.5,
            xanchor="center"
        ),
        xaxis_title="Time intervals",
        yaxis=dict(
            title="Load [kW]",
            side="left"
        ),
        legend=dict(
            title="Legend",
            x=1.6,
            y=1.0,
            xanchor="right",
            yanchor="top"
        ),
        template="plotly_white",
        font=dict(color="black"),
        margin=dict(l=10, r=10, t=30, b=10),
        height=300,
    )

    fig.update_xaxes(tickformat="%H:%M", dtick=10800000, tickangle=90)

    # Save plots
    html_file_path = f'results\\{model}\\{run_name}\\a_base_{chosen_date}_{model}.html'
    fig.write_html(html_file_path)

    pdf_file_path = f'results\\{model}\\{run_name}\\a_base_{chosen_date}_{model}.pdf'
    fig.write_image(pdf_file_path)

generate_plot_base(2021, 10, 17, combined_timeseries)
generate_plot_base(2021, 12, 2, combined_timeseries)
# uv run 3_manipulation.py
#******************************************************************************************************************************************
# graph of represenative for Base day self chosen day. ( subplots)
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

def extract_day_data(year, month, day, df):
    days_per_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    rows_per_day = 96

    if year % 4 == 0:
        days_per_month[1] = 29

    offset = sum(days_per_month[:month - 1]) * rows_per_day
    start_idx = offset + (day - 1) * rows_per_day
    end_idx = start_idx + rows_per_day

    time_index = pd.date_range(f"{year}-{month:02d}-{day:02d} 00:00", freq="15T", periods=rows_per_day)
    day_data = df.iloc[start_idx:end_idx].copy()
    day_data.index = time_index
    return day_data, time_index


def add_household_traces(fig, col, day_data, time_intervals, show_legend):
    # Prepare derived data
    bess_dis = day_data['ev_cum'] + day_data['bat_charge']
    bess_dis_plus = bess_dis - day_data['bat_discharge']
    pv_plus = day_data['net_load_lv_grid'] + day_data['pv_generation']

    fig.add_trace(go.Scatter(
        x=time_intervals, y=day_data["ev_cum"],
        mode="lines", name="Consumption",
        line=dict(color="red", width=0.5),
        fill="tozeroy", fillcolor="rgba(251,128,114,0.3)",
        stackgroup="group", showlegend=show_legend
    ), row=1, col=col)

    fig.add_trace(go.Scatter(
        x=time_intervals, y=day_data["bat_charge"],
        mode="none", name="BESS charging",
        fill="tonexty", fillcolor="rgba(0, 0, 0, 0)",
        fillpattern=dict(shape="+"), stackgroup="group", showlegend=show_legend
    ), row=1, col=col)

    fig.add_trace(go.Scatter(
        x=time_intervals, y=day_data["net_load_lv_grid"],
        mode="lines", name="Residual load",
        line=dict(color="blue", dash="dot"), showlegend=show_legend
    ), row=1, col=col)

    fig.add_trace(go.Scatter(
        x=time_intervals, y=pv_plus,
        mode="none", name="PV generation",
        line=dict(color="yellow"),
        fill="tonexty", fillcolor="rgba(255,255,0,0.3)", showlegend=show_legend
    ), row=1, col=col)

    fig.add_trace(go.Scatter(
        x=time_intervals, y=bess_dis,
        mode="lines", showlegend=False,
        line=dict(color="rgba(0, 0, 255, 0)", width=2)
    ), row=1, col=col)

    fig.add_trace(go.Scatter(
        x=time_intervals, y=bess_dis_plus,
        mode="none", name="BESS discharging",
        fill="tonexty", fillcolor="rgba(0, 0, 0, 0)",
        fillpattern=dict(shape="x"), showlegend=show_legend
    ), row=1, col=col)


def generate_lv_comparison_subplots(date1, date2, df):
    day1_data, time1 = extract_day_data(*date1, df)
    day2_data, time2 = extract_day_data(*date2, df)
    chosen_date = f"{desired_year}-{desired_month:02d}-{desired_day:02d}"
    fig = make_subplots(
        rows=1, cols=2,
        shared_yaxes=True,
        subplot_titles=[
            f"{date1[0]}-{date1[1]:02d}-{date1[2]:02d}",
            f"{date2[0]}-{date2[1]:02d}-{date2[2]:02d}"
        ],
        horizontal_spacing=0.03
    )

    add_household_traces(fig, 1, day1_data, time1, show_legend=True)
    add_household_traces(fig, 2, day2_data, time2, show_legend=False)

    fig.update_layout(
        xaxis=dict(tickformat="%H:%M", dtick=10800000, tickangle=90),
        xaxis2=dict(tickformat="%H:%M", dtick=10800000, tickangle=90),
        yaxis=dict(title="Load [kW]"),
        font=dict(color="black"),
        template="plotly_white",
        legend=dict(
            title="Legend",
            x=1.05,
            y=1,
            xanchor="left",
            yanchor="top"
        ),
        margin=dict(l=20, r=20, t=20, b=20),
        height=300,
        width=900
    )

    html_file_path = f'results\\{model}\\{run_name}\\a_base_lv_{chosen_date}_{model}.html'
    fig.write_html(html_file_path)

    pdf_file_path = f'results\\{model}\\{run_name}\\a_base_lv_{chosen_date}_{model}.pdf'
    fig.write_image(pdf_file_path)
    fig.write_image("comparison_plot.pdf")

# Example usage:
generate_lv_comparison_subplots(
    date1=(2021, 12, 2),
    date2=(2021, 6, 16),
    df=combined_timeseries
)
# uv run 3_manipulation.py
#******************************************************************************************************************************************
# from plotly.subplots import make_subplots
# import pandas as pd

# def extract_day_data(year, month, day, df, tariff_data):
#     days_per_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
#     rows_per_day = 96

#     if year % 4 == 0:
#         days_per_month[1] = 29

#     offset = sum(days_per_month[:month - 1]) * rows_per_day
#     start_idx = offset + (day - 1) * rows_per_day
#     end_idx = start_idx + rows_per_day

#     time_index = pd.date_range(f"{year}-{month:02d}-{day:02d} 00:00", freq="15T", periods=rows_per_day)
#     day_data = df.iloc[start_idx:end_idx].copy()
#     day_data.index = time_index
#     tariff_data = tariff_data.iloc[start_idx:end_idx].copy()
#     return day_data, time_index, tariff_data


# def add_household_traces(fig, col, day_data, time_intervals, tariff_data, show_legend):
#     # Prepare derived data
#     bess_dis = day_data['ev_cum'] + day_data['bat_charge']
#     bess_dis_plus = bess_dis - day_data['bat_discharge']
#     pv_plus = day_data['net_load_lv_grid'] + day_data['pv_generation']

#     fig.add_trace(go.Scatter(
#         x=time_intervals, y=day_data["ev_cum"],
#         mode="lines", name="Consumption",
#         line=dict(color="red", width=0.5),
#         fill="tozeroy", fillcolor="rgba(251,128,114,0.3)",
#         stackgroup="group", showlegend=show_legend
#     ), row=1, col=col)

#     fig.add_trace(go.Scatter(
#         x=time_intervals, y=day_data["bat_charge"],
#         mode="none", name="BESS charging",
#         fill="tonexty", fillcolor="rgba(0, 0, 0, 0)",
#         fillpattern=dict(shape="+"), stackgroup="group", showlegend=show_legend
#     ), row=1, col=col)

#     fig.add_trace(go.Scatter(
#         x=time_intervals, y=day_data["net_load_lv_grid"],
#         mode="lines", name="Residual load",
#         line=dict(color="blue", dash="dot"), showlegend=show_legend
#     ), row=1, col=col)

#     fig.add_trace(go.Scatter(
#         x=time_intervals, y=pv_plus,
#         mode="none", name="PV generation",
#         line=dict(color="yellow"),
#         fill="tonexty", fillcolor="rgba(255,255,0,0.3)", showlegend=show_legend
#     ), row=1, col=col)

#     fig.add_trace(go.Scatter(
#         x=time_intervals, y=bess_dis,
#         mode="lines", showlegend=False,
#         line=dict(color="rgba(0, 0, 255, 0)", width=2)
#     ), row=1, col=col)

#     fig.add_trace(go.Scatter(
#         x=time_intervals, y=bess_dis_plus,
#         mode="none", name="BESS discharging",
#         fill="tonexty", fillcolor="rgba(0, 0, 0, 0)",
#         fillpattern=dict(shape="x"), showlegend=show_legend
#     ), row=1, col=col)

#     fig.add_trace(go.Scatter(
#         x=time_intervals,
#         y=tariff_data,
#         mode="lines",
#         name="Consump.-based <br> grid tariff (ct/kWh)",
#         line=dict(color="green"),
#         showlegend=show_legend
#     ), row=1, col=col, secondary_y=True)  # <--- Ensure this!

# def generate_lv_comparison_subplots(date1, date2, df, tariff_data):
#     day1_data, time1, tariff1= extract_day_data(*date1, df, tariff_data)
#     day2_data, time2, tariff2 = extract_day_data(*date2, df, tariff_data)
#     chosen_date = f"{desired_year}-{desired_month:02d}-{desired_day:02d}"
#     fig = make_subplots(
#         rows=1, cols=2,
#         subplot_titles=[
#             f"{date1[0]}-{date1[1]:02d}-{date1[2]:02d}",
#             f"{date2[0]}-{date2[1]:02d}-{date2[2]:02d}"
#         ],
#         specs=[[{"secondary_y": True}, {"secondary_y": True}]],  # <--- Enable secondary y-axes!
#         horizontal_spacing=0.03
#     )

#     add_household_traces(fig, 1, day1_data, time1, tariff1, show_legend=True)
#     add_household_traces(fig, 2, day2_data, time2, tariff2, show_legend=False)

#     fig.update_layout(
#         # LEFT axis for subplot 1
#         yaxis=dict(
#             title=dict(text="Load [kW]"),
#             side="left",
#             showticklabels=True,
#             autorange=True
#         ),
#         # RIGHT axis for subplot 1
#         yaxis2=dict(
#             title=dict(text="Grid tariff (ct/kWh)", font=dict(color="green")),
#             overlaying="y",
#             side="right",
#             tickfont=dict(color="green"),
#             showticklabels=True,
#             showline=True,
#             linecolor="green",
#             autorange=True
#         ),
#         # LEFT axis for subplot 2
#         yaxis3=dict(
#             title=dict(text="Load [kW]"),
#             side="left",
#             showticklabels=True,
#             autorange=True
#         ),
#         # RIGHT axis for subplot 2
#         yaxis4=dict(
#             title=dict(text="Grid tariff (ct/kWh)", font=dict(color="green")),
#             overlaying="y3",
#             side="right",
#             tickfont=dict(color="green"),
#             showticklabels=True,
#             showline=True,
#             linecolor="green",
#             autorange=True
#         ),
#         font=dict(color="black"),
#         template="plotly_white",
#         legend=dict(
#             title="Legend",
#             x=1.05,
#             y=1,
#             xanchor="left",
#             yanchor="top"
#         ),
#         margin=dict(l=20, r=20, t=20, b=20),
#         height=300,
#         width=900
#     )

#     html_file_path = f'results\\{model}\\{run_name}\\a_tou_comparison_lv_{chosen_date}_{model}.html'
#     fig.write_html(html_file_path)

#     pdf_file_path = f'results\\{model}\\{run_name}\\a_tou_comparison_lv_{chosen_date}_{model}.pdf'
#     fig.write_image(pdf_file_path)
#     fig.write_image("comparison_plot.pdf")

# # Example usage:
# generate_lv_comparison_subplots(
#     date1=(2021, 12, 2),
#     date2=(2021, 6, 16),
#     df=combined_timeseries, tariff_data=tou_tariff
# )
# # uv run 3_manipulation.py
#******************************************************************************************************************************************
# fucntion to generate a graph on lv level TOU
# graph of represenative for TOU Model day self chosen day.
# Function to generate and display plot based on user-specified date
def generate_plot(desired_year, desired_month, desired_day, df_combined, df_tariff):
    # Define dataset structure (assuming full year, 96 intervals/day)
    days_per_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]  # Adjust for leap years
    rows_per_day = 96

    chosen_date = f"{desired_year}-{desired_month:02d}-{desired_day:02d}"

    # Adjust for leap years
    if desired_year % 4 == 0:
        days_per_month[1] = 29

    # Calculate starting row index
    month_offset = sum(days_per_month[:desired_month - 1]) * rows_per_day
    day_start_idx = month_offset + (desired_day - 1) * rows_per_day
    day_end_idx = day_start_idx + rows_per_day

    # Extract data for the selected day
    selected_day_data = df_combined.iloc[day_start_idx:day_end_idx]

    # Generate time labels (15-min intervals)
    time_intervals = pd.date_range(
        start=f"{desired_year}-{desired_month:02d}-{desired_day:02d} 00:00",
        freq="15T", periods=rows_per_day
    )
    selected_day_data.index = time_intervals

    # Load TOU tariff data
    tou_tariff_values = df_tariff.iloc[day_start_idx:day_end_idx]['grid_tariff']

    # Define plot figure
    fig = go.Figure()
    bess_dis = selected_day_data['ev_cum'] + selected_day_data['bat_charge']
    bess_dis_plus = bess_dis - selected_day_data['bat_discharge']
    pv_plus = selected_day_data['net_load_lv_grid'] + selected_day_data['pv_generation']

    fig.add_trace(go.Scatter(
        x=time_intervals, y=selected_day_data["ev_cum"],
        mode="lines", name="Consumption",
        line=dict(color="red", width=0.5),
        fill="tozeroy", fillcolor="rgba(251,128,114,0.3)",
        stackgroup="group", showlegend=True
    ))

    fig.add_trace(go.Scatter(
        x=time_intervals, y=selected_day_data["bat_charge"],
        mode="none", name="BESS charging",
        fill="tonexty", fillcolor="rgba(0, 0, 0, 0)",
        fillpattern=dict(shape="+"), stackgroup="group", showlegend=True
    ))

    fig.add_trace(go.Scatter(
        x=time_intervals, y=selected_day_data["net_load_lv_grid"],
        mode="lines", name="Residual load",
        line=dict(color="blue", dash="dot"), showlegend=True
    ))

    fig.add_trace(go.Scatter(
        x=time_intervals, y=pv_plus,
        mode="none", name="PV generation",
        line=dict(color="yellow"),
        fill="tonexty", fillcolor="rgba(255,255,0,0.3)", showlegend=True
    ))

    fig.add_trace(go.Scatter(
        x=time_intervals, y=bess_dis,
        mode="lines", showlegend=False,
        line=dict(color="rgba(0, 0, 255, 0)", width=2)
    ))

    fig.add_trace(go.Scatter(
        x=time_intervals, y=bess_dis_plus,
        mode="none", name="BESS discharging",
        fill="tonexty", fillcolor="rgba(0, 0, 0, 0)",
        fillpattern=dict(shape="x"), showlegend=True
    ))

    # Add TOU tariff (right y-axis, with area fill)
    fig.add_trace(go.Scatter(
        x=time_intervals,
        y=tou_tariff_values,
        mode="lines",
        name="Consump.-based <br> grid tariff (ct/kWh)",
        line=dict(color="green"),
        yaxis="y2"
    ))

    # Update layout for dual y-axes
    fig.update_layout(
        title=dict(text=f"{chosen_date}",x=0.5),
        xaxis_title="Time intervals",
        yaxis=dict(
            title="Load [kW]",
            side="left"
        ),
        yaxis2=dict(
            title="Consump.-based <br> grid tariff (ct/kWh)",
            overlaying="y",
            side="right",
            tickvals=[0.053, 0.105, 0.158],
            ticktext=["5.3", "10.5", "15.8"],
            tickfont=dict(color="green"),
            showgrid=False,  # **Ensure no grid lines for yaxis2**
            gridcolor="rgba(0,0,0,0)"  # Set tick color to green
        ),
        legend=dict(
            title="Legend",
            x=1.8,
            y=1.0,
            xanchor="right",
            yanchor="top"
        ),
        template="plotly_white",
        font=dict(color="black"),
        margin=dict(l=20, r=20, t=40, b=20),  # Adjust margins for better visibility
        height=300
    )

    # Fix x-axis to hourly labels
    fig.update_xaxes(tickformat="%H:%M", dtick=10800000, tickangle=90)
    
    # Save plots
    html_file_path = f'results\\{model}\\{run_name}\\a_tou_{chosen_date}_{model}.html'
    fig.write_html(html_file_path)

    pdf_file_path = f'results\\{model}\\{run_name}\\a_tou_{chosen_date}_{model}.pdf'
    fig.write_image(pdf_file_path)

# Example usage
generate_plot(2021, 10, 17, combined_timeseries, tou_tariff)
#*******************************************************************************************************************************************

def generate_plot(desired_year, desired_month, desired_day, df_combined, df_tariff):
    # Define dataset structure (assuming full year, 96 intervals/day)
    days_per_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]  # Adjust for leap years
    rows_per_day = 96

    chosen_date = f"{desired_year}-{desired_month:02d}-{desired_day:02d}"

    # Adjust for leap years
    if desired_year % 4 == 0:
        days_per_month[1] = 29

    # Calculate starting row index
    month_offset = sum(days_per_month[:desired_month - 1]) * rows_per_day
    day_start_idx = month_offset + (desired_day - 1) * rows_per_day
    day_end_idx = day_start_idx + rows_per_day

    # Extract data for the selected day
    selected_day_data = df_combined.iloc[day_start_idx:day_end_idx]

    # Generate time labels (15-min intervals)
    time_intervals = pd.date_range(
        start=f"{desired_year}-{desired_month:02d}-{desired_day:02d} 00:00",
        freq="15T", periods=rows_per_day
    )
    selected_day_data.index = time_intervals

    # Load TOU tariff data
    tou_tariff_values = df_tariff.iloc[day_start_idx:day_end_idx]['grid_tariff']

    # Define plot figure
    fig = go.Figure()
    bess_dis = selected_day_data['ev_cum'] + selected_day_data['bat_charge']
    bess_dis_plus = bess_dis - selected_day_data['bat_discharge']
    pv_plus = selected_day_data['net_load_lv_grid'] + selected_day_data['pv_generation']

    fig.add_trace(go.Scatter(
        x=time_intervals, y=selected_day_data["ev_cum"],
        mode="lines", name="Consumption",
        line=dict(color="red", width=0.5),
        fill="tozeroy", fillcolor="rgba(251,128,114,0.3)",
        stackgroup="group", showlegend=True
    ))

    fig.add_trace(go.Scatter(
        x=time_intervals, y=selected_day_data["bat_charge"],
        mode="none", name="BESS charging",
        fill="tonexty", fillcolor="rgba(0, 0, 0, 0)",
        fillpattern=dict(shape="+"), stackgroup="group", showlegend=True
    ))

    fig.add_trace(go.Scatter(
        x=time_intervals, y=selected_day_data["net_load_lv_grid"],
        mode="lines", name="Residual load",
        line=dict(color="blue", dash="dot"), showlegend=True
    ))

    fig.add_trace(go.Scatter(
        x=time_intervals, y=pv_plus,
        mode="none", name="PV generation",
        line=dict(color="yellow"),
        fill="tonexty", fillcolor="rgba(255,255,0,0.3)", showlegend=True
    ))

    fig.add_trace(go.Scatter(
        x=time_intervals, y=bess_dis,
        mode="lines", showlegend=False,
        line=dict(color="rgba(0, 0, 255, 0)", width=2)
    ))

    fig.add_trace(go.Scatter(
        x=time_intervals, y=bess_dis_plus,
        mode="none", name="BESS discharging",
        fill="tonexty", fillcolor="rgba(0, 0, 0, 0)",
        fillpattern=dict(shape="x"), showlegend=True
    ))

    # Add TOU tariff (right y-axis, with area fill)
    fig.add_trace(go.Scatter(
        x=time_intervals,
        y=tou_tariff_values,
        mode="lines",
        name="Consump.-based <br> grid tariff (ct/kWh)",
        line=dict(color="green"),
        yaxis="y2"
    ))

    fig.update_layout(
        title=dict(text=f"{chosen_date}", x=0.5),
        xaxis_title="Time intervals",
        yaxis=dict(
            title="Load [kW]",
            side="left",
            range=[-200, 800],
            tickvals=[-100,0, 200, 400, 600, 800],  # Explicitly set scale
        ),
        yaxis2=dict(
            overlaying="y",
            side="right",
            tickvals=[0.053, 0.105, 0.158],
            ticktext=["5.3", "10.5", "15.8"],
            tickfont=dict(color="green"),
            title=None,  # Remove axis title
            showticklabels=False,
            showgrid=False,  # **Ensure no grid lines for yaxis2**
            gridcolor="rgba(0,0,0,0)"  # Hide tick labels
        ),
        showlegend=False,
        template="plotly_white",
        font=dict(color="black"),
        margin=dict(l=20, r=20, t=40, b=20),  # Adjust margins for better visibility
        height=300,
        width=500
)

    # Fix x-axis to hourly labels
    fig.update_xaxes(tickformat="%H:%M", dtick=10800000, tickangle=90)
    
    # Save plots
    html_file_path = f'results\\{model}\\{run_name}\\a_tou_{chosen_date}_{model}.html'
    fig.write_html(html_file_path)

    pdf_file_path = f'results\\{model}\\{run_name}\\a_tou_{chosen_date}_{model}.pdf'
    fig.write_image(pdf_file_path)

generate_plot(2021, 11, 22, combined_timeseries, tou_tariff)
#*******************************************************************************************************************************************
def generate_plot(desired_year, desired_month, desired_day, df_combined, df_tariff):
    # Define dataset structure (assuming full year, 96 intervals/day)
    days_per_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]  # Adjust for leap years
    rows_per_day = 96

    chosen_date = f"{desired_year}-{desired_month:02d}-{desired_day:02d}"

    # Adjust for leap years
    if desired_year % 4 == 0:
        days_per_month[1] = 29

    # Calculate starting row index
    month_offset = sum(days_per_month[:desired_month - 1]) * rows_per_day
    day_start_idx = month_offset + (desired_day - 1) * rows_per_day
    day_end_idx = day_start_idx + rows_per_day

    # Extract data for the selected day
    selected_day_data = df_combined.iloc[day_start_idx:day_end_idx]

    # Generate time labels (15-min intervals)
    time_intervals = pd.date_range(
        start=f"{desired_year}-{desired_month:02d}-{desired_day:02d} 00:00",
        freq="15T", periods=rows_per_day
    )
    selected_day_data.index = time_intervals

    # Load TOU tariff data
    tou_tariff_values = df_tariff.iloc[day_start_idx:day_end_idx]['grid_tariff']

    # Define plot figure
    fig = go.Figure()
    bess_dis = selected_day_data['ev_cum'] + selected_day_data['bat_charge']
    bess_dis_plus = bess_dis - selected_day_data['bat_discharge']
    pv_plus = selected_day_data['net_load_lv_grid'] + selected_day_data['pv_generation']

    fig.add_trace(go.Scatter(
        x=time_intervals, y=selected_day_data["ev_cum"],
        mode="lines", name="Consumption",
        line=dict(color="red", width=0.5),
        fill="tozeroy", fillcolor="rgba(251,128,114,0.3)",
        stackgroup="group", showlegend=True
    ))

    fig.add_trace(go.Scatter(
        x=time_intervals, y=selected_day_data["bat_charge"],
        mode="none", name="BESS charging",
        fill="tonexty", fillcolor="rgba(0, 0, 0, 0)",
        fillpattern=dict(shape="+"), stackgroup="group", showlegend=True
    ))

    fig.add_trace(go.Scatter(
        x=time_intervals, y=selected_day_data["net_load_lv_grid"],
        mode="lines", name="Residual load",
        line=dict(color="blue", dash="dot"), showlegend=True
    ))

    fig.add_trace(go.Scatter(
        x=time_intervals, y=pv_plus,
        mode="none", name="PV generation",
        line=dict(color="yellow"),
        fill="tonexty", fillcolor="rgba(255,255,0,0.3)", showlegend=True
    ))

    fig.add_trace(go.Scatter(
        x=time_intervals, y=bess_dis,
        mode="lines", showlegend=False,
        line=dict(color="rgba(0, 0, 255, 0)", width=2)
    ))

    fig.add_trace(go.Scatter(
        x=time_intervals, y=bess_dis_plus,
        mode="none", name="BESS discharging",
        fill="tonexty", fillcolor="rgba(0, 0, 0, 0)",
        fillpattern=dict(shape="x"), showlegend=True
    ))

    # Add TOU tariff (right y-axis, with area fill)
    fig.add_trace(go.Scatter(
        x=time_intervals,
        y=tou_tariff_values,
        mode="lines",
        name="Consump.-based <br> grid tariff (ct/kWh)",
        line=dict(color="green"),
        yaxis="y2"
    ))

    fig.update_layout(
        title=dict(text=f"{chosen_date}", x=0.2),  # Shift title slightly to the left
        xaxis_title="Time intervals",
        yaxis=dict(
            showticklabels=False,  # Hide tick labels
            title=None,  # Remove axis title
            zeroline=False,  # Hide zero line
            showgrid=True,  # Ensure grid is visible
            tickvals=[-100, 0, 200, 400, 600, 800],  # Set grid positions
            range=[-200, 800]  # Explicitly set scale
        ),
        yaxis2=dict(
            title="Consump.-based <br> grid tariff (ct/kWh)",
            overlaying="y",
            side="right",
            tickvals=[0.053, 0.105, 0.158],
            ticktext=["5.3", "10.5", "15.8"],
            tickfont=dict(color="green"),
            showgrid=False,  # **Ensure no grid lines for yaxis2**
            gridcolor="rgba(0,0,0,0)"  # **Force grid invisibility**
        ),
        legend=dict(
            title="Legend",
            x=1.8,
            y=1.0,
            xanchor="right",
            yanchor="top"
        ),
        template="plotly_white",
        font=dict(color="black"),
        margin=dict(l=20, r=20, t=40, b=20),
        height=300,
        width=650
    )

    # Fix x-axis to hourly labels
    fig.update_xaxes(tickformat="%H:%M", dtick=10800000, tickangle=90)
    
    # Save plots
    html_file_path = f'results\\{model}\\{run_name}\\a_tou_{chosen_date}_{model}.html'
    fig.write_html(html_file_path)

    pdf_file_path = f'results\\{model}\\{run_name}\\a_tou_{chosen_date}_{model}.pdf'
    fig.write_image(pdf_file_path, scale=0.8)  # Try setting a lower scale


# Example usage
generate_plot(2021, 6, 18, combined_timeseries, tou_tariff)

#*******************************************************************************************************************************************
print("All plots generated successfully.")