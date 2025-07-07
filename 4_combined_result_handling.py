# uv run 4_combined_result_handling.py

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import numpy as np
pio.kaleido.scope.mathjax = None
import os

# Define model names and paths
models = {
    "Base": "results/base_model/base_200hh/combined_household_data.csv",
    "TOU": "results/tou_model/run_200hh/combined_household_data.csv",
    "MPD": "results/mpd_model/run_200hh/combined_household_data.csv",
    "MPD-50": "results/mpd-50_model/run_200hh/combined_household_data.csv",
    "MPD-50-inc": "results/mpd-50_inc/run_200hh/combined_household_data.csv",
    "MPD-50-inc-4kW": "results/mpd-50_inc_4kW/run_200hh/combined_household_data.csv"
}
otherfiles = {
    "summary": "files/household_summary.csv"}

output_file_path = "results/a_combined_results/"

# Load CSV files
dataframes = {name: pd.read_csv(path) for name, path in models.items()}
datafiles = {name: pd.read_csv(path) for name, path in otherfiles.items()}

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
#*****************************************************************************************************************************
# Create boxplot figure with all scenarios side by side
fig = go.Figure()

for model_name, df in dataframes.items():
    # Select relevant columns
    cols_1_80 = [f'net_load_hh_{i}' for i in range(1, 81) if f'net_load_hh_{i}' in df.columns]
    cols_81_200 = [f'el_market_exp_value_{i}' for i in range(81, 201) if f'el_market_exp_value_{i}' in df.columns]
    combined_cols = cols_1_80 + cols_81_200
    # Extract the color and line style from the dictionary
    color, line_style = model_colors[model_name]

    # Compute monthly peak mean
    data = df[combined_cols].copy()
    data.reset_index(drop=True, inplace=True)

    # Ensure datetime index matches 15-minute intervals
    datetime_index = pd.date_range(start="2021-01-01 00:00", end="2021-12-31 23:45", freq="15min")
    data["datetime"] = datetime_index[:len(data)]
    data.set_index("datetime", inplace=True)

    # Resample data to monthly peaks and calculate mean
    monthly_groups = data.resample("ME").max()
    monthly_peak_means = monthly_groups.mean()

    # Add boxplot for each scenario
    fig.add_trace(go.Box(
        y=monthly_peak_means,
        name=model_name,
        boxpoints="outliers",
        boxmean=True,
        line=dict(color=color)  # Display mean line
    ))

fig.update_layout(
    yaxis_title="Arithmetic means of <br> monthly peaks [kW]",
    template="plotly_white",
    font=dict(color="black"),
    showlegend=False,
    height=300,
    margin=dict(l=40, r=40, t=20, b=40)    # Assuming original height was 800, adjust accordingly
)

# Save figure as PDF and HTML
pdf_file_path = os.path.join(output_file_path, "a_boxplot_monthly_peak_means_all_models.pdf")
fig.write_image(pdf_file_path, format="pdf")

html_file_path = os.path.join(output_file_path, "a_boxplot_monthly_peak_means_all_models.html")
fig.write_html(html_file_path)

print(f"Boxplot saved as PDF: {pdf_file_path}")
print(f"Boxplot saved as HTML: {html_file_path}")
# uv run 4_combined_result_handling.py
#************************************************************************************************************************
# Create a new figure for Load Duration Curve (LDC)
fig_ldc = go.Figure()

# Iterate through models to add sorted curves and include max/min load values in the legend
max_loads = {}
min_loads = {}

for model_name, df in dataframes.items():
    if 'net_load_lv_grid' in df.columns:
        sorted_grid_load = df['net_load_lv_grid'].sort_values(ascending=False).reset_index(drop=True)
        max_loads[model_name] = sorted_grid_load.max()
        min_loads[model_name] = sorted_grid_load.min()

        # Extract the color and line style from the dictionary
        color, line_style = model_colors[model_name]
        
        fig_ldc.add_trace(go.Scatter(
            x=sorted_grid_load.index,
            y=sorted_grid_load,
            mode='lines',
            name=model_name,
            line=dict(color=color, dash=line_style)  # Apply colors & dash styles dynamically
        ))

# Add vertical lines for 5% and 80% of the x-axis
num_points = len(next(iter(dataframes.values()))['net_load_lv_grid'])
x_5_percent = int(0.05 * num_points)
x_80_percent = int(0.80 * num_points)

fig_ldc.add_shape(type="line", x0=x_5_percent, y0=-200, x1=x_5_percent, y1=700, line=dict(color="black", dash="dash"))
fig_ldc.add_shape(type="line", x0=x_80_percent, y0=-200, x1=x_80_percent, y1=700, line=dict(color="black", dash="dash"))

# Add circled numbers (Peak, Intermediate, Base) **ONLY ONCE**
lower_factors = [0.1, 0.4, 0.4]
fig_ldc.add_trace(go.Scatter(
    x=[int(0.025 * num_points), int(0.45 * num_points), int(0.85 * num_points)],
    y=[max(max_loads.values()) * lower_factors[0], max(max_loads.values()) * lower_factors[1], max(max_loads.values()) * lower_factors[2]],  
    mode='text', 
    text=['①', '②', '③'], 
    textfont=dict(size=20, color='black', family='Arial Black'),
    showlegend=False,
))

# Centered annotation for peak, intermediate, base loads (without border or bgcolor)
fig_ldc.add_annotation(
    text="①: Peak load<br>②: Intermediate load<br>③: Base load",
    xref="paper", yref="paper",
    x=0.5, y=0.9,  # Positioned towards center, slightly lower
    showarrow=False,
    align="left",
    font=dict(size=18, color="black")
)
# Customize the layout for the LDC plot
fig_ldc.update_layout(
    xaxis_title='Sorted 15-min intervals (highest to lowest load)',
    yaxis_title='Residual load on LV grid [kW]',
    template='plotly_white',
    showlegend=True,
    font=dict(color="black"),  # Ensuring all text is black
    margin=dict(l=20, r=20, t=20, b=20),  # Adjust margins for better visibility
    width=800,  # Set width to 800 pixels
    height=300  # Set height to 600 pixels
)

# Save the Load Duration Curve as an HTML file
html_file_path_ldc = f'{output_file_path}combined_ldc.html'
fig_ldc.write_html(html_file_path_ldc)

pdf_file_path_ldc = f'{output_file_path}combined_ldc.pdf'
fig_ldc.write_image(pdf_file_path_ldc)

print(f"Load Duration Curve saved as PDF at {pdf_file_path_ldc}")
print(f"Load Duration Curve saved as HTML at {html_file_path_ldc}")
# uv run 4_combined_result_handling.py
#****************************************************************************************************************
# generating table for masterthesis
# Initialize an empty DataFrame for combined results
combined_table = pd.DataFrame(columns=["Name", "Inflexible demand [kWh]", "SH [kWh]", "DHW [kWh]", "BEV [kWh]", "Flexibility [kWh]"])

base_model_df = dataframes["Base"]
summary_df = datafiles["summary"]

# Loop through 200 households and fill in the table row-wise
for i in range(1, 201):  # From HH1 to HH200
    name = f"HH{i}"

    demand = base_model_df[f"demand_exp_value_{i}"].sum() / 4 if f"demand_exp_value_{i}" in base_model_df.columns else 0
    SH = base_model_df[f"heat_unit_exp_in_electricity_{i}"].sum() / 4 if f"heat_unit_exp_in_electricity_{i}" in base_model_df.columns else 0
    DHW = base_model_df[f"water_unit_exp_in_electricity_{i}"].sum() / 4 if f"water_unit_exp_in_electricity_{i}" in base_model_df.columns else 0
    BEV = base_model_df[f"ev_unit_exp_in_electricity_{i}"].sum() / 4 if f"ev_unit_exp_in_electricity_{i}" in base_model_df.columns else 0

    # Flexibility calculation (assuming SH + DHW + BEV + Battery if enabled)
    bat = 6.9 if summary_df.iloc[i-1]["bat_enable"] == 1 else 0
    sh_flex = (summary_df.iloc[i-1]["heat_ub"] - summary_df.iloc[i-1]["heat_lb"]) / summary_df.iloc[i-1]["power_to_c"] if summary_df.iloc[i-1]["heat_enable"] == 1 else 0
    water_flex = summary_df.iloc[i-1]["water_vol"] * 0.0116 if summary_df.iloc[i-1]["heat_enable"] == 1 else 0
    ev_cap = summary_df.iloc[i-1]["ev_capacity"] if summary_df.iloc[i-1]["ev_enable"] == 1 else 0

    Flexibility = bat + sh_flex + water_flex + ev_cap

    demand, SH, DHW, BEV, Flexibility = map(lambda x: round(float(x)), [demand, SH, DHW, BEV, Flexibility])

    # Round values to integers
    row_values = [name, demand, SH, DHW, BEV, Flexibility]

    # Append row to DataFrame
    combined_table.loc[len(combined_table)] = row_values

# Define output file path
csv_file_path_ldc = f"{output_file_path}combined_table.csv"

# Save DataFrame to CSV
combined_table.to_csv(csv_file_path_ldc, index=False)

print(f"CSV file saved to: {csv_file_path_ldc}")

demand_mean = combined_table["Inflexible demand [kWh]"].mean()
print(f"Mean Inflexible demand: {demand_mean:.2f} kWh")
# uv run 4_combined_result_handling.py
#****************************************************************************************************************
# grid cost analysis (boxplots base scenario)

base_grid_df = pd.read_csv(r"C:\Users\ScheiblS\Documents\Repositories\dyn-grid-tariffs\results\base_model\base_200hh\combined_grid_cost.csv")

base_data = base_grid_df["grid_cost"].iloc[:200] + 57.6

# Step 3: Create boxplots for sensitivity analysis
fig = go.Figure()

# Add the first boxplot for all means with the mean line, outlier markers, and data points
fig.add_trace(
    go.Box(
        y=base_data,
        boxpoints='all',  # Include all data points
        jitter=0.3,
        pointpos=0,
        marker=dict(color='red', size=2, symbol='circle'),  # Smaller data points
        name='All means',
        boxmean=True  # Display mean line
    )
)

# Select data for additional boxplots based on specific ranges
selected_grid_cost_fc1 = base_data.iloc[:20]  # grid_load_exp_value_1 to grid_load_exp_value_20
selected_grid_cost_fc2 = base_data.iloc[20:50]  # grid_load_exp_value_21 to grid_load_exp_value_50
selected_grid_cost_fc3 = base_data.iloc[50:80]  # grid_load_exp_value_51 to grid_load_exp_value_80
selected_grid_cost_fc4 = base_data.iloc[80:110]  # grid_load_exp_value_81 to grid_load_exp_value_110
selected_grid_cost_fc5 = base_data.iloc[110:160]  # grid_load_exp_value_111 to grid_load_exp_value_160
selected_grid_cost_ifcg = base_data.iloc[160:200]  # grid_load_exp_value_161 to grid_load_exp_value_200

# Add boxplots for selected ranges with data points
fig.add_trace(go.Box(y=selected_grid_cost_fc1, boxpoints='all', jitter=0.3, pointpos=0,
                     marker=dict(color='blue', size=2), name='FCG1', boxmean=True))
fig.add_trace(go.Box(y=selected_grid_cost_fc2, boxpoints='all', jitter=0.3, pointpos=0,
                     marker=dict(color='green', size=2), name='FCG2', boxmean=True))
fig.add_trace(go.Box(y=selected_grid_cost_fc3, boxpoints='all', jitter=0.3, pointpos=0,
                     marker=dict(color='purple', size=2), name='FCG3', boxmean=True))
fig.add_trace(go.Box(y=selected_grid_cost_fc4, boxpoints='all', jitter=0.3, pointpos=0,
                     marker=dict(color='orange', size=2), name='FCG4', boxmean=True))
fig.add_trace(go.Box(y=selected_grid_cost_fc5, boxpoints='all', jitter=0.3, pointpos=0,
                     marker=dict(color='cyan', size=2), name='FCG5', boxmean=True))
fig.add_trace(go.Box(y=selected_grid_cost_ifcg, boxpoints='all', jitter=0.3, pointpos=0,
                     marker=dict(color='magenta', size=2), name='IFCG', boxmean=True))

# Update layout for the plot
fig.update_layout(
    yaxis_title="Grid cost [€/a]",
    yaxis=dict(range=[0, None]),
    template="plotly_white",
    showlegend=False,
    font=dict(color="black")
)

# Save the resulting plot as PDF for compatibility
pdf_file_path = "C:\\Users\\ScheiblS\\Documents\\Repositories\\dyn-grid-tariffs\\results\\a_combined_results\\a_boxplot_grid_cost_base.pdf"
pio.write_image(fig, pdf_file_path, format='pdf') 
print(f"Boxplot has been saved as 'a_boxplot_monthly_peak_means.pdf'")

# Save the resulting plot as HTML for interactivity
html_file_path = "C:\\Users\\ScheiblS\\Documents\\Repositories\\dyn-grid-tariffs\\results\\a_combined_results\\a_boxplot_grid_cost_base.html"
fig.write_html(html_file_path)
print(f"Boxplot has been saved as 'a_boxplot_monthly_peak_means.html'")
# uv run 4_combined_result_handling.py
# ****************************************************************************************************************
# scatter plot für grid cost im Base model
base_grid_df = pd.read_csv(r"C:\Users\ScheiblS\Documents\Repositories\dyn-grid-tariffs\results\base_model\base_200hh\combined_grid_cost.csv")
base_data = base_grid_df["grid_cost"].iloc[:200] + 57.6  # Begrenzung auf 200 Werte
print(f'Grid cost in Base is:{base_data.sum()}')

base_model_df = dataframes["Base"]

# Erstelle eine Liste für den Verbrauch der Haushalte
household_consumption_base = []

for i in range(1, 201):  # Von HH1 bis HH200
    consumption = base_model_df[f"el_market_exp_value_{i}"].sum() / 4 if f"el_market_exp_value_{i}" in base_model_df.columns else 0
    household_consumption_base.append(consumption)

# household_consumption_base = []
# for i in range(1, 201):  # From HH1 to HH200
#     if i <= 80:
#         col_name = f"net_load_hh_{i}"
#     else:
#         col_name = f"el_market_exp_value_{i}"

#     consumption = base_model_df[col_name].sum() / 4 if col_name in base_model_df.columns else 0
#     household_consumption_base.append(consumption)

# Erstelle eine Farbzuordnung basierend auf den Ranges
colors = ['blue'] * 20 + ['green'] * 30 + ['purple'] * 30 + ['orange'] * 30 + ['cyan'] * 50 + ['magenta'] * 40

# Scatter-Plot erstellen mit individuellen Farben und Legenden-Einträgen
fig = go.Figure()

# Füge die einzelnen Farbgruppen hinzu, damit sie in der Legende erscheinen
fig.add_trace(go.Scatter(
    x=household_consumption_base[:20],  # Erste 20 Haushalte (FCG1)
    y=base_data[:20],
    mode='markers',
    marker=dict(color='blue', size=5),
    name="FCG1"  # Legendenname
))

fig.add_trace(go.Scatter(
    x=household_consumption_base[20:50],  # 21-50 Haushalte (FCG2)
    y=base_data[20:50],
    mode='markers',
    marker=dict(color='green', size=5),
    name="FCG2"  # Legendenname
))

# Füge die restlichen Farbgruppen hinzu
fig.add_trace(go.Scatter(x=household_consumption_base[50:80], y=base_data[50:80], mode='markers',
                         marker=dict(color='purple', size=5), name="FCG3"))
fig.add_trace(go.Scatter(x=household_consumption_base[80:110], y=base_data[80:110], mode='markers',
                         marker=dict(color='orange', size=5), name="FCG4"))
fig.add_trace(go.Scatter(x=household_consumption_base[110:160], y=base_data[110:160], mode='markers',
                         marker=dict(color='cyan', size=5), name="FCG5"))
fig.add_trace(go.Scatter(x=household_consumption_base[160:200], y=base_data[160:200], mode='markers',
                         marker=dict(color='magenta', size=5), name="IFCG"))

tick_vals = [0, 4000, 8000, 12000, 16000]
tick_labels = ["0", "4K", "8K", "12K", "16K"]  # Tausendertrennzeichen als Leerzeichen

# Layout aktualisieren mit benutzerdefinierten Labels & Legende
fig.update_layout(
    xaxis_title="Annual consumption [kWh/a]",
    yaxis_title="Grid cost [€/a]",
    template="plotly_white",
    font=dict(color="black"),
    xaxis=dict(
        tickmode="array",
        tickvals=tick_vals,
        ticktext=tick_labels,
    ),
    legend=dict(title="Legend", font=dict(color="black")),
    margin=dict(l=20, r=20, t=20, b=20),  # Adjust margins for better visibility
    height=300,
    width= 400
)

# Speichern der Dateien
pdf_file_path = "C:\\Users\\ScheiblS\\Documents\\Repositories\\dyn-grid-tariffs\\results\\a_combined_results\\a_scatter_grid_cost_base.pdf"
fig.write_image(pdf_file_path, format='pdf')
print(f"Scatterplot has been saved as 'a_scatter_grid_cost_base.pdf'")

html_file_path = "C:\\Users\\ScheiblS\\Documents\\Repositories\\dyn-grid-tariffs\\results\\a_combined_results\\a_scatter_grid_cost_base.html"
fig.write_html(html_file_path)
print(f"Scatterplot has been saved as 'a_scatter_grid_cost_base.html'")
# uv run 4_combined_result_handling.py
#****************************************************************************************************************
# scatter plot für grid cost im TOU model
tou_grid_df = pd.read_csv(r"C:\Users\ScheiblS\Documents\Repositories\dyn-grid-tariffs\results\tou_model\run_200hh\combined_grid_cost.csv")
tou_data = tou_grid_df["grid_cost"].iloc[:200]  # Begrenzung auf 200 Werte
print(f'Grid cost in TOU is:{tou_data.sum()}')
tou_model_df = dataframes["TOU"]

# Erstelle eine Liste für den Verbrauch der Haushalte
household_consumption_tou = []

for i in range(1, 201):  # Von HH1 bis HH200
    consumption = tou_model_df[f"el_market_exp_value_{i}"].sum() / 4 if f"el_market_exp_value_{i}" in tou_model_df.columns else 0
    household_consumption_tou.append(consumption)

# Erstelle eine Farbzuordnung basierend auf den Ranges
colors = ['blue'] * 20 + ['green'] * 30 + ['purple'] * 30 + ['orange'] * 30 + ['cyan'] * 50 + ['magenta'] * 40

# Scatter-Plot erstellen mit individuellen Farben und Legenden-Einträgen
fig = go.Figure()

# Füge die einzelnen Farbgruppen hinzu, damit sie in der Legende erscheinen
fig.add_trace(go.Scatter(
    x=household_consumption_tou[:20],  # Erste 20 Haushalte (FCG1)
    y=tou_data[:20],
    mode='markers',
    marker=dict(color='blue', size=5),
    name="FCG1"  # Legendenname
))

fig.add_trace(go.Scatter(
    x=household_consumption_tou[20:50],  # 21-50 Haushalte (FCG2)
    y=tou_data[20:50],
    mode='markers',
    marker=dict(color='green', size=5),
    name="FCG2"  # Legendenname
))

# Füge die restlichen Farbgruppen hinzu
fig.add_trace(go.Scatter(x=household_consumption_tou[50:80], y=tou_data[50:80], mode='markers',
                         marker=dict(color='purple', size=5), name="FCG3"))
fig.add_trace(go.Scatter(x=household_consumption_tou[80:110], y=tou_data[80:110], mode='markers',
                         marker=dict(color='orange', size=5), name="FCG4"))
fig.add_trace(go.Scatter(x=household_consumption_tou[110:160], y=tou_data[110:160], mode='markers',
                         marker=dict(color='cyan', size=5), name="FCG5"))
fig.add_trace(go.Scatter(x=household_consumption_tou[160:200], y=tou_data[160:200], mode='markers',
                         marker=dict(color='magenta', size=5), name="IFCG"))

tick_vals = [0, 4000, 8000, 12000, 16000]
tick_labels = ["0", "4K", "8K", "12K", "16K"]

# Layout aktualisieren mit benutzerdefinierten Labels & Legende
fig.update_layout(
    xaxis_title="Annual consumption [kWh/a]",
    yaxis_title="Grid cost [€/a]",
    template="plotly_white",
    font=dict(color="black"),
    xaxis=dict(
        tickmode="array",
        tickvals=tick_vals,
        ticktext=tick_labels
    ),
    legend=dict(title="Legend", font=dict(color="black")),
    margin=dict(l=20, r=20, t=20, b=20),  # Adjust margins for better visibility
    height=300,
    width= 400
)

# Speichern der Dateien
pdf_file_path = "C:\\Users\\ScheiblS\\Documents\\Repositories\\dyn-grid-tariffs\\results\\a_combined_results\\a_scatter_grid_cost_tou.pdf"
fig.write_image(pdf_file_path, format='pdf')
print(f"Scatterplot has been saved as 'a_scatter_grid_cost_base.pdf'")

html_file_path = "C:\\Users\\ScheiblS\\Documents\\Repositories\\dyn-grid-tariffs\\results\\a_combined_results\\a_scatter_grid_cost_tou.html"
fig.write_html(html_file_path)
print(f"Scatterplot has been saved as 'a_scatter_grid_cost_base.html'")
# uv run 4_combined_result_handling.py
#****************************************************************************************************************
# scatter plot für grid cost im MPD model
mpd_model_df = dataframes["MPD"]

# Ensure the DataFrame index is numeric
data_mpd = mpd_model_df.copy()
data_mpd.reset_index(drop=True, inplace=True)  # Setzt Index zurück

# Filtere die relevanten Spalten (die mit 'el_market_exp_value_' beginnen)
columns_of_interest = [col for col in data_mpd.columns if col.startswith('el_market_exp_value_')]
data_mpd_2 = data_mpd[columns_of_interest]

# Erstelle einen datetime-Index für 15-minütige Intervalle
datetime_index = pd.date_range(start="2021-01-01 00:00", end="2021-12-31 23:45", freq="15min")
data_mpd_2['datetime'] = datetime_index[:len(data_mpd_2)]  # Passender datetime-Index
data_mpd_2.set_index('datetime', inplace=True)

# Gruppiere nach Monaten und berechne die monatlichen Spitzenwerte
monthly_groups = data_mpd_2.resample('ME').max()

# Berechne das arithmetische Mittel der monatlichen Spitzenwerte
monthly_peak_means = monthly_groups.mean()
mpd_data = monthly_peak_means * 70.93

monthly_peak_means.to_csv("C:\\Users\\ScheiblS\\Documents\\Repositories\\dyn-grid-tariffs\\results\\a_combined_results\\monthly_peak_means.csv", index=False)

print(f'Grid cost in MPD is:{mpd_data.sum()}')

# Erstelle eine Liste für den Verbrauch der Haushalte
household_consumption_mpd = []

for i in range(1, 201):  # Von HH1 bis HH200
    consumption = mpd_model_df[f"el_market_exp_value_{i}"].sum() / 4 if f"el_market_exp_value_{i}" in mpd_model_df.columns else 0
    household_consumption_mpd.append(consumption)

# Farbzuordnung automatisch aus Ranges ableiten
color_mapping = {
    range(0, 20): "blue",
    range(20, 50): "green",
    range(50, 80): "purple",
    range(80, 110): "orange",
    range(110, 160): "cyan",
    range(160, 200): "magenta"
}

# Farben zuweisen entsprechend der Haushaltsnummer
colors = [next((color for rng, color in color_mapping.items() if i in rng), "gray") for i in range(200)]

# Scatter-Plot erstellen mit individuellen Farben und Legenden-Einträgen
fig = go.Figure()

# Füge die einzelnen Farbgruppen hinzu, damit sie in der Legende erscheinen
fig.add_trace(go.Scatter(
    x=household_consumption_mpd[:20],  # Erste 20 Haushalte (FCG1)
    y=mpd_data[:20],
    mode='markers',
    marker=dict(color='blue', size=5),
    name="FCG1"  # Legendenname
))

fig.add_trace(go.Scatter(
    x=household_consumption_mpd[20:50],  # 21-50 Haushalte (FCG2)
    y=mpd_data[20:50],
    mode='markers',
    marker=dict(color='green', size=5),
    name="FCG2"  # Legendenname
))

fig.add_trace(go.Scatter(x=household_consumption_mpd[50:80], y=mpd_data[50:80], mode='markers',
                         marker=dict(color='purple', size=5), name="FCG3"))
fig.add_trace(go.Scatter(x=household_consumption_mpd[80:110], y=mpd_data[80:110], mode='markers',
                         marker=dict(color='orange', size=5), name="FCG4"))
fig.add_trace(go.Scatter(x=household_consumption_mpd[110:160], y=mpd_data[110:160], mode='markers',
                         marker=dict(color='cyan', size=5), name="FCG5"))
fig.add_trace(go.Scatter(x=household_consumption_mpd[160:200], y=mpd_data[160:200], mode='markers',
                         marker=dict(color='magenta', size=5), name="IFCG"))

tick_vals = [0, 4000, 8000, 12000, 16000]
tick_labels = ["0", "4K", "8K", "12K", "16K"]

# Layout aktualisieren mit benutzerdefinierten Labels & Legende
fig.update_layout(
    xaxis_title="Annual consumption [kWh/a]",
    yaxis_title="Grid cost [€/a]",
    template="plotly_white",
    font=dict(color="black"),
    xaxis=dict(
        tickmode="array",
        tickvals=tick_vals,
        ticktext=tick_labels
    ),
    legend=dict(title="Legend", font=dict(color="black")),
    margin=dict(l=20, r=20, t=20, b=20),  # Adjust margins for better visibility
    height=300,
    width= 400
)

# Speichern der Dateien
pdf_file_path = "C:\\Users\\ScheiblS\\Documents\\Repositories\\dyn-grid-tariffs\\results\\a_combined_results\\a_scatter_grid_cost_mpd.pdf"
fig.write_image(pdf_file_path, format='pdf')
print(f"Scatterplot has been saved as 'a_scatter_grid_cost_mpd.pdf'")

html_file_path = "C:\\Users\\ScheiblS\\Documents\\Repositories\\dyn-grid-tariffs\\results\\a_combined_results\\a_scatter_grid_cost_mpd.html"
fig.write_html(html_file_path)
print(f"Scatterplot has been saved as 'a_scatter_grid_cost_mpd.html'")
# uv run 4_combined_result_handling.py
#****************************************************************************************************************
# scatter plot für grid cost im MPD-50 model
mpd_grid_df = pd.read_csv(r"C:\Users\ScheiblS\Documents\Repositories\dyn-grid-tariffs\results\mpd-50_model\run_200hh\combined_grid_cost.csv")
mpd_data_ext = mpd_grid_df["grid_cost"].iloc[:200] # Begrenzung auf 200 Werte
mpd_data_df = pd.DataFrame(mpd_data_ext, columns=["grid_cost"])

mpd_model_df = dataframes["MPD-50"]
# Ensure the DataFrame index is numeric
data = mpd_model_df.copy()
data.reset_index(drop=True, inplace=True)  # Setzt Index zurück
# Filtere die relevanten Spalten (die mit 'el_market_exp_value_' beginnen)
columns_of_interest = [col for col in data.columns if col.startswith('el_market_exp_value_')]
data = data[columns_of_interest]
# Erstelle einen datetime-Index für 15-minütige Intervalle
datetime_index = pd.date_range(start="2021-01-01 00:00", end="2021-12-31 23:45", freq="15min")
data['datetime'] = datetime_index[:len(data)]  # Passender datetime-Index
data.set_index('datetime', inplace=True)
# Gruppiere nach Monaten und berechne die monatlichen Spitzenwerte
monthly_groups = data.resample('ME').max()
# Berechne das arithmetische Mittel der monatlichen Spitzenwerte
monthly_peak_means = monthly_groups.mean()
mpd_peaks_cost = monthly_peak_means * 35.47
mpd_peaks_cost_df = pd.DataFrame(mpd_peaks_cost, columns=["monthly_peak_cost"])

# Reset index and rename the column to match mpd_data_df's index
mpd_peaks_cost_df.reset_index(inplace=True)
mpd_peaks_cost_df.rename(columns={"index": "grid_cost"}, inplace=True)  # Adjust column name
# Convert index format to match mpd_data_df
mpd_peaks_cost_df.set_index(mpd_data_df.index, inplace=True)

# Perform row-wise addition
mpd_ext_grid_cost = mpd_data_df["grid_cost"] + mpd_peaks_cost_df["monthly_peak_cost"]
print(f'Grid cost in MPD-50 is:{mpd_ext_grid_cost.sum()}')
mpd_model_df2 = dataframes["MPD-50"]
# Erstelle eine Liste für den Verbrauch der Haushalte
household_consumption_mpd_ext = []
for i in range(1, 201):  # Von HH1 bis HH200
    consumption = mpd_model_df2[f"el_market_exp_value_{i}"].sum() / 4 if f"el_market_exp_value_{i}" in mpd_model_df2.columns else 0
    household_consumption_mpd_ext.append(consumption)

# Erstelle eine Farbzuordnung basierend auf den Ranges
colors = ['blue'] * 20 + ['green'] * 30 + ['purple'] * 30 + ['orange'] * 30 + ['cyan'] * 50 + ['magenta'] * 40

# Scatter-Plot erstellen mit individuellen Farben und Legenden-Einträgen
fig = go.Figure()

# Füge die einzelnen Farbgruppen hinzu, damit sie in der Legende erscheinen
fig.add_trace(go.Scatter(
    x=household_consumption_mpd_ext[:20],  # Erste 20 Haushalte (FCG1)
    y=mpd_ext_grid_cost[:20],
    mode='markers',
    marker=dict(color='blue', size=5),
    name="FCG1"  # Legendenname
))

fig.add_trace(go.Scatter(
    x=household_consumption_mpd_ext[20:50],  # 21-50 Haushalte (FCG2)
    y=mpd_ext_grid_cost[20:50],
    mode='markers',
    marker=dict(color='green', size=5),
    name="FCG2"  # Legendenname
))

# Füge die restlichen Farbgruppen hinzu
fig.add_trace(go.Scatter(x=household_consumption_mpd_ext[50:80], y=mpd_ext_grid_cost[50:80], mode='markers',
                         marker=dict(color='purple', size=5), name="FCG3"))
fig.add_trace(go.Scatter(x=household_consumption_mpd_ext[80:110], y=mpd_ext_grid_cost[80:110], mode='markers',
                         marker=dict(color='orange', size=5), name="FCG4"))
fig.add_trace(go.Scatter(x=household_consumption_mpd_ext[110:160], y=mpd_ext_grid_cost[110:160], mode='markers',
                         marker=dict(color='cyan', size=5), name="FCG5"))
fig.add_trace(go.Scatter(x=household_consumption_mpd_ext[160:200], y=mpd_ext_grid_cost[160:200], mode='markers',
                         marker=dict(color='magenta', size=5), name="IFCG"))

tick_vals = [0, 4000, 8000, 12000, 16000]
tick_labels = ["0", "4K", "8K", "12K", "16K"]

# Layout aktualisieren mit benutzerdefinierten Labels & Legende
fig.update_layout(
    xaxis_title="Annual consumption [kWh/a]",
    yaxis_title="Grid cost [€/a]",
    template="plotly_white",
    font=dict(color="black"),
    xaxis=dict(
        tickmode="array",
        tickvals=tick_vals,
        ticktext=tick_labels
    ),
    legend=dict(title="Legend", font=dict(color="black")),
    margin=dict(l=20, r=20, t=20, b=20),  # Adjust margins for better visibility
    height=300,
    width= 400
)

# Speichern der Dateien
pdf_file_path = "C:\\Users\\ScheiblS\\Documents\\Repositories\\dyn-grid-tariffs\\results\\a_combined_results\\a_scatter_grid_cost_mpd_ext.pdf"
fig.write_image(pdf_file_path, format='pdf')
print(f"Scatterplot has been saved as 'a_scatter_grid_cost_base.pdf'")

html_file_path = "C:\\Users\\ScheiblS\\Documents\\Repositories\\dyn-grid-tariffs\\results\\a_combined_results\\a_scatter_grid_cost_mpd_ext.html"
fig.write_html(html_file_path)
print(f"Scatterplot has been saved as 'a_scatter_grid_cost_mpd_ext.html'")
# uv run 4_combined_result_handling.py
#****************************************************************************************************************
# scatter plot für grid cost im MPD-50-inc model
mpd_inc_grid_df = pd.read_csv(r"C:\Users\ScheiblS\Documents\Repositories\dyn-grid-tariffs\results\mpd-50_model\run_200hh\combined_grid_cost.csv")
mpd_inc_data_ext = mpd_inc_grid_df["grid_cost"].iloc[:200] # Begrenzung auf 200 Werte
mpd_inc_data_df = pd.DataFrame(mpd_inc_data_ext, columns=["grid_cost"])

mpd_inc_model_df = dataframes["MPD-50-inc"]
# Ensure the DataFrame index is numeric
data = mpd_model_df.copy()
data.reset_index(drop=True, inplace=True)  # Setzt Index zurück
# Filtere die relevanten Spalten (die mit 'el_market_exp_value_' beginnen)
columns_of_interest = [col for col in data.columns if col.startswith('el_market_exp_value_')]
data = data[columns_of_interest]
# Erstelle einen datetime-Index für 15-minütige Intervalle
datetime_index = pd.date_range(start="2021-01-01 00:00", end="2021-12-31 23:45", freq="15min")
data['datetime'] = datetime_index[:len(data)]  # Passender datetime-Index
data.set_index('datetime', inplace=True)
# Gruppiere nach Monaten und berechne die monatlichen Spitzenwerte
monthly_groups = data.resample('ME').max()
# Berechne das arithmetische Mittel der monatlichen Spitzenwerte
monthly_peak_means = monthly_groups.mean()
# --- 4. Tarife definieren ---
rate_below = 20.09  # €/kW für bis zu 7.5 kW
rate_above = 40.18  # €/kW für alles über 7.5 kW
# --- 5. Aufteilen der Last in zwei Komponenten ---
avg_peak_below_7_5 = np.minimum(monthly_peak_means, 7.5)  # Maximal 7.5 kW
avg_peak_above_7_5 = np.maximum(monthly_peak_means - 7.5, 0)  # Überschuss über 7.5 kW
# --- 6. Kostenberechnung ---
cost_below = avg_peak_below_7_5 * rate_below
cost_above = avg_peak_above_7_5 * rate_above
mpd_inc_peaks_cost = cost_below + cost_above
mpd_inc_peaks_cost_df = pd.DataFrame(mpd_inc_peaks_cost, columns=["monthly_peak_cost"])
# Reset index and rename the column to match mpd_data_df's index
mpd_inc_peaks_cost_df.reset_index(inplace=True)
mpd_inc_peaks_cost_df.rename(columns={"index": "grid_cost"}, inplace=True)  # Adjust column name
# Convert index format to match mpd_data_df
mpd_inc_peaks_cost_df.set_index(mpd_inc_data_df.index, inplace=True)

# Perform row-wise addition
mpd_inc_ext_grid_cost = mpd_inc_data_df["grid_cost"] + mpd_inc_peaks_cost_df["monthly_peak_cost"]
print(f'Grid cost in MPD-50-inc is:{mpd_inc_ext_grid_cost.sum()}')
mpd_inc_model_df2 = dataframes["MPD-50-inc"]
# Erstelle eine Liste für den Verbrauch der Haushalte
household_consumption_mpd_inc = []
for i in range(1, 201):  # Von HH1 bis HH200
    consumption = mpd_inc_model_df2[f"el_market_exp_value_{i}"].sum() / 4 if f"el_market_exp_value_{i}" in mpd_inc_model_df2.columns else 0
    household_consumption_mpd_inc.append(consumption)

# Erstelle eine Farbzuordnung basierend auf den Ranges
colors = ['blue'] * 20 + ['green'] * 30 + ['purple'] * 30 + ['orange'] * 30 + ['cyan'] * 50 + ['magenta'] * 40

# Scatter-Plot erstellen mit individuellen Farben und Legenden-Einträgen
fig = go.Figure()

# Füge die einzelnen Farbgruppen hinzu, damit sie in der Legende erscheinen
fig.add_trace(go.Scatter(
    x=household_consumption_mpd_inc[:20],  # Erste 20 Haushalte (FCG1)
    y=mpd_inc_ext_grid_cost[:20],
    mode='markers',
    marker=dict(color='blue', size=5),
    name="FCG1"  # Legendenname
))

fig.add_trace(go.Scatter(
    x=household_consumption_mpd_inc[20:50],  # 21-50 Haushalte (FCG2)
    y=mpd_inc_ext_grid_cost[20:50],
    mode='markers',
    marker=dict(color='green', size=5),
    name="FCG2"  # Legendenname
))

# Füge die restlichen Farbgruppen hinzu
fig.add_trace(go.Scatter(x=household_consumption_mpd_inc[50:80], y=mpd_inc_ext_grid_cost[50:80], mode='markers',
                         marker=dict(color='purple', size=5), name="FCG3"))
fig.add_trace(go.Scatter(x=household_consumption_mpd_inc[80:110], y=mpd_inc_ext_grid_cost[80:110], mode='markers',
                         marker=dict(color='orange', size=5), name="FCG4"))
fig.add_trace(go.Scatter(x=household_consumption_mpd_inc[110:160], y=mpd_inc_ext_grid_cost[110:160], mode='markers',
                         marker=dict(color='cyan', size=5), name="FCG5"))
fig.add_trace(go.Scatter(x=household_consumption_mpd_inc[160:200], y=mpd_inc_ext_grid_cost[160:200], mode='markers',
                         marker=dict(color='magenta', size=5), name="IFCG"))

tick_vals = [0, 4000, 8000, 12000, 16000]
tick_labels = ["0", "4K", "8K", "12K", "16K"]

# Layout aktualisieren mit benutzerdefinierten Labels & Legende
fig.update_layout(
    xaxis_title="Annual consumption [kWh/a]",
    yaxis_title="Grid cost [€/a]",
    template="plotly_white",
    font=dict(color="black"),
    xaxis=dict(
        tickmode="array",
        tickvals=tick_vals,
        ticktext=tick_labels
    ),
    legend=dict(title="Legend", font=dict(color="black")),
    margin=dict(l=20, r=20, t=20, b=20),  # Adjust margins for better visibility
    height=300,
    width= 400
)

# Speichern der Dateien
pdf_file_path = "C:\\Users\\ScheiblS\\Documents\\Repositories\\dyn-grid-tariffs\\results\\a_combined_results\\a_scatter_grid_cost_mpd_inc.pdf"
fig.write_image(pdf_file_path, format='pdf')
print(f"Scatterplot has been saved as 'a_scatter_grid_cost_base.pdf'")

html_file_path = "C:\\Users\\ScheiblS\\Documents\\Repositories\\dyn-grid-tariffs\\results\\a_combined_results\\a_scatter_grid_cost_mpd_inc.html"
fig.write_html(html_file_path)
print(f"Scatterplot has been saved as 'a_scatter_grid_cost_mpd_ext.html'")
# uv run 4_combined_result_handling.py
#****************************************************************************************************************
# grid cost analysis (scatter plots for MPD-50-inc-4kW
# scatter plot für grid cost im MPD-50-inc model
mpd_inc_grid_df = pd.read_csv(r"C:\Users\ScheiblS\Documents\Repositories\dyn-grid-tariffs\results\mpd-50_model\run_200hh\combined_grid_cost.csv")
mpd_inc_data_ext = mpd_inc_grid_df["grid_cost"].iloc[:200] # Begrenzung auf 200 Werte
mpd_inc_data_df = pd.DataFrame(mpd_inc_data_ext, columns=["grid_cost"])

mpd_inc_model_df = dataframes["MPD-50-inc-4kW"]
# Ensure the DataFrame index is numeric
data = mpd_model_df.copy()
data.reset_index(drop=True, inplace=True)  # Setzt Index zurück
# Filtere die relevanten Spalten (die mit 'el_market_exp_value_' beginnen)
columns_of_interest = [col for col in data.columns if col.startswith('el_market_exp_value_')]
data = data[columns_of_interest]
# Erstelle einen datetime-Index für 15-minütige Intervalle
datetime_index = pd.date_range(start="2021-01-01 00:00", end="2021-12-31 23:45", freq="15min")
data['datetime'] = datetime_index[:len(data)]  # Passender datetime-Index
data.set_index('datetime', inplace=True)
# Gruppiere nach Monaten und berechne die monatlichen Spitzenwerte
monthly_groups = data.resample('ME').max()
# Berechne das arithmetische Mittel der monatlichen Spitzenwerte
monthly_peak_means = monthly_groups.mean()
# --- 4. Tarife definieren ---
rate_below = 17.87  # €/kW für bis zu 7.5 kW
rate_above = 35.75  # €/kW für alles über 7.5 kW
# --- 5. Aufteilen der Last in zwei Komponenten ---
avg_peak_below_7_5 = np.minimum(monthly_peak_means, 4)  # Maximal 7.5 kW
avg_peak_above_7_5 = np.maximum(monthly_peak_means - 4, 0)  # Überschuss über 7.5 kW
# --- 6. Kostenberechnung ---
cost_below = avg_peak_below_7_5 * rate_below
cost_above = avg_peak_above_7_5 * rate_above
mpd_inc_peaks_cost = cost_below + cost_above
mpd_inc_peaks_cost_df = pd.DataFrame(mpd_inc_peaks_cost, columns=["monthly_peak_cost"])
# Reset index and rename the column to match mpd_data_df's index
mpd_inc_peaks_cost_df.reset_index(inplace=True)
mpd_inc_peaks_cost_df.rename(columns={"index": "grid_cost"}, inplace=True)  # Adjust column name
# Convert index format to match mpd_data_df
mpd_inc_peaks_cost_df.set_index(mpd_inc_data_df.index, inplace=True)

# Perform row-wise addition
mpd_inc_4kw_grid_cost = mpd_inc_data_df["grid_cost"] + mpd_inc_peaks_cost_df["monthly_peak_cost"]
print(f'Grid cost in MPD-50-inc-4kW is:{mpd_inc_4kw_grid_cost.sum()}')
mpd_inc_model_df2 = dataframes["MPD-50-inc-4kW"]
# Erstelle eine Liste für den Verbrauch der Haushalte
household_consumption_mpd_inc_4kw = []
for i in range(1, 201):  # Von HH1 bis HH200
    consumption = mpd_inc_model_df2[f"el_market_exp_value_{i}"].sum() / 4 if f"el_market_exp_value_{i}" in mpd_inc_model_df2.columns else 0
    household_consumption_mpd_inc_4kw.append(consumption)

# Erstelle eine Farbzuordnung basierend auf den Ranges
colors = ['blue'] * 20 + ['green'] * 30 + ['purple'] * 30 + ['orange'] * 30 + ['cyan'] * 50 + ['magenta'] * 40

# Scatter-Plot erstellen mit individuellen Farben und Legenden-Einträgen
fig = go.Figure()

# Füge die einzelnen Farbgruppen hinzu, damit sie in der Legende erscheinen
fig.add_trace(go.Scatter(
    x=household_consumption_mpd_inc_4kw[:20],  # Erste 20 Haushalte (FCG1)
    y=mpd_inc_4kw_grid_cost[:20],
    mode='markers',
    marker=dict(color='blue', size=5),
    name="FCG1"  # Legendenname
))

fig.add_trace(go.Scatter(
    x=household_consumption_mpd_inc_4kw[20:50],  # 21-50 Haushalte (FCG2)
    y=mpd_inc_4kw_grid_cost[20:50],
    mode='markers',
    marker=dict(color='green', size=5),
    name="FCG2"  # Legendenname
))

# Füge die restlichen Farbgruppen hinzu
fig.add_trace(go.Scatter(x=household_consumption_mpd_inc_4kw[50:80], y=mpd_inc_4kw_grid_cost[50:80], mode='markers',
                         marker=dict(color='purple', size=5), name="FCG3"))
fig.add_trace(go.Scatter(x=household_consumption_mpd_inc_4kw[80:110], y=mpd_inc_4kw_grid_cost[80:110], mode='markers',
                         marker=dict(color='orange', size=5), name="FCG4"))
fig.add_trace(go.Scatter(x=household_consumption_mpd_inc_4kw[110:160], y=mpd_inc_4kw_grid_cost[110:160], mode='markers',
                         marker=dict(color='cyan', size=5), name="FCG5"))
fig.add_trace(go.Scatter(x=household_consumption_mpd_inc_4kw[160:200], y=mpd_inc_4kw_grid_cost[160:200], mode='markers',
                         marker=dict(color='magenta', size=5), name="IFCG"))

tick_vals = [0, 4000, 8000, 12000, 16000]
tick_labels = ["0", "4K", "8K", "12K", "16K"]

# Layout aktualisieren mit benutzerdefinierten Labels & Legende
fig.update_layout(
    xaxis_title="Annual consumption [kWh/a]",
    yaxis_title="Grid cost [€/a]",
    template="plotly_white",
    font=dict(color="black"),
    xaxis=dict(
        tickmode="array",
        tickvals=tick_vals,
        ticktext=tick_labels
    ),
    legend=dict(title="Legend", font=dict(color="black")),
    margin=dict(l=20, r=20, t=20, b=20),  # Adjust margins for better visibility
    height=300,
    width= 400
)

# Speichern der Dateien
pdf_file_path = "C:\\Users\\ScheiblS\\Documents\\Repositories\\dyn-grid-tariffs\\results\\a_combined_results\\a_scatter_grid_cost_mpd_inc_4kw.pdf"
fig.write_image(pdf_file_path, format='pdf')
print(f"Scatterplot has been saved as 'a_scatter_grid_cost_base.pdf'")

html_file_path = "C:\\Users\\ScheiblS\\Documents\\Repositories\\dyn-grid-tariffs\\results\\a_combined_results\\a_scatter_grid_cost_mpd_inc_4kw.html"
fig.write_html(html_file_path)
print(f"Scatterplot has been saved as 'a_scatter_grid_cost_mpd_ext.html'")

# uv run 4_combined_result_handling.py
#****************************************************************************************************************
# grid cost analysis (scatter plots for all scenarios)
import plotly.graph_objects as go

# Create a new figure
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=household_consumption_mpd,
    y=mpd_data,  # Handle NaN values
    mode='markers',
    marker=dict(color='green', size=5),
    name="MPD"
))

fig.add_trace(go.Scatter(
    x=household_consumption_tou,
    y=tou_data,
    mode='markers',
    marker=dict(color='red', size=5),
    name="TOU"
))

fig.add_trace(go.Scatter(
    x=household_consumption_mpd_ext,
    y=mpd_ext_grid_cost,  # Handle NaN values
    mode='markers',
    marker=dict(color='orange', size=5),
    name="MPD-50"
))

# Add data for each scenario with correct colors
fig.add_trace(go.Scatter(
    x=household_consumption_mpd_inc,
    y=mpd_inc_ext_grid_cost,
    mode='markers',
    marker=dict(color='purple', size=5),
    name="MPD-50-inc"
))

# Add data for each scenario with correct colors
fig.add_trace(go.Scatter(
    x=household_consumption_mpd_inc_4kw,
    y=mpd_inc_4kw_grid_cost,
    mode='markers',
    marker=dict(color='cyan', size=5),
    name="MPD-50-inc-4kW"
))

# Add data for each scenario with correct colors
fig.add_trace(go.Scatter(
    x=household_consumption_base,
    y=base_data,
    mode='markers',
    marker=dict(color='blue', size=5),
    name="Base"
))

# Define tick values and labels
tick_vals = [0, 4000, 8000, 12000, 16000]
tick_labels = ["0", "4K", "8K", "12K", "16K"]

fig.update_layout(
    xaxis=dict(
        title="Annual consumption [kWh/a]",
        tickmode="array",
        tickvals=tick_vals,
        ticktext=tick_labels,
    ),
    yaxis=dict(
        title="Grid cost [€/a]",
        tickmode="array",
        tickvals=list(range(0, int(max(base_data)) + 200, 200))  # Set tick marks every 200 €
    ),
    template="plotly_white",
    font=dict(color="black"),
    legend=dict(title="Legend", font=dict(color="black")),
    margin=dict(l=20, r=20, t=20, b=20),  # Adjust margins for better visibility
    height=300,
    width= 400
)

# Save the plot as PDF and HTML
pdf_file_path = "C:\\Users\\ScheiblS\\Documents\\Repositories\\dyn-grid-tariffs\\results\\a_combined_results\\a_combined_scatter_grid_cost.pdf"
fig.write_image(pdf_file_path, format='pdf')
print(f"Scatterplot has been saved as '{pdf_file_path}'")

html_file_path = "C:\\Users\\ScheiblS\\Documents\\Repositories\\dyn-grid-tariffs\\results\\a_combined_results\\a_combined_scatter_grid_cost.html"
fig.write_html(html_file_path)
print(f"Scatterplot has been saved as '{html_file_path}'")
# uv run 4_combined_result_handling.py
#****************************************************************************************************************
# grid cost analysis (scatter plots for Base, TOU, MPD, MPD-50)
import plotly.graph_objects as go

# Create a new figure
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=household_consumption_mpd,
    y=mpd_data,  # Handle NaN values
    mode='markers',
    marker=dict(color='green', size=5),
    name="MPD"
))

fig.add_trace(go.Scatter(
    x=household_consumption_tou,
    y=tou_data,
    mode='markers',
    marker=dict(color='red', size=5),
    name="TOU"
))

fig.add_trace(go.Scatter(
    x=household_consumption_mpd_ext,
    y=mpd_ext_grid_cost,  # Handle NaN values
    mode='markers',
    marker=dict(color='orange', size=5),
    name="MPD-50"
))

# # Add data for each scenario with correct colors
# fig.add_trace(go.Scatter(
#     x=household_consumption_mpd_inc,
#     y=mpd_inc_ext_grid_cost,
#     mode='markers',
#     marker=dict(color='purple', size=5),
#     name="MPD-50-inc"
# ))

# # Add data for each scenario with correct colors
# fig.add_trace(go.Scatter(
#     x=household_consumption_mpd_inc_4kw,
#     y=mpd_inc_4kw_grid_cost,
#     mode='markers',
#     marker=dict(color='cyan', size=5),
#     name="MPD-50-inc-4kW"
# ))

# Add data for each scenario with correct colors
fig.add_trace(go.Scatter(
    x=household_consumption_base,
    y=base_data,
    mode='markers',
    marker=dict(color='blue', size=5),
    name="Base"
))

# Define tick values and labels
tick_vals = [0, 4000, 8000, 12000, 16000]
tick_labels = ["0", "4K", "8K", "12K", "16K"]

fig.update_layout(
    xaxis=dict(
        title="Annual consumption [kWh/a]",
        tickmode="array",
        tickvals=tick_vals,
        ticktext=tick_labels,
    ),
    yaxis=dict(
        title="Grid cost [€/a]",
        tickmode="array",
        tickvals=list(range(0, 1600 + 200, 200)),  # Extending to 1600
        range=[0, 1600]  # Explicitly setting the axis range
    ),
    template="plotly_white",
    font=dict(color="black"),
    legend=dict(title="Legend", font=dict(color="black")),
    margin=dict(l=20, r=20, t=20, b=20),  # Adjust margins for better visibility
    height=300,
    width= 400
)

# Save the plot as PDF and HTML
pdf_file_path = "C:\\Users\\ScheiblS\\Documents\\Repositories\\dyn-grid-tariffs\\results\\a_combined_results\\a_combined_scatter_Base_TOU_MPD_grid_cost.pdf"
fig.write_image(pdf_file_path, format='pdf')
print(f"Scatterplot has been saved as '{pdf_file_path}'")

html_file_path = "C:\\Users\\ScheiblS\\Documents\\Repositories\\dyn-grid-tariffs\\results\\a_combined_results\\a_combined_scatter_Base_TOU_MPD_grid_cost.html"
fig.write_html(html_file_path)
print(f"Scatterplot has been saved as '{html_file_path}'")


# uv run 4_combined_result_handling.py
#****************************************************************************************************************
# grid cost analysis (scatter plots for MPD scnarios)
import plotly.graph_objects as go

# Create a new figure
fig = go.Figure()

# fig.add_trace(go.Scatter(
#     x=household_consumption_mpd,
#     y=mpd_data,  # Handle NaN values
#     mode='markers',
#     marker=dict(color='green', size=5),
#     name="MPD"
# ))

# fig.add_trace(go.Scatter(
#     x=household_consumption_tou,
#     y=tou_data,
#     mode='markers',
#     marker=dict(color='red', size=5),
#     name="TOU"
# ))

fig.add_trace(go.Scatter(
    x=household_consumption_mpd_ext,
    y=mpd_ext_grid_cost,  # Handle NaN values
    mode='markers',
    marker=dict(color='orange', size=5),
    name="MPD-50"
))

# Add data for each scenario with correct colors
fig.add_trace(go.Scatter(
    x=household_consumption_mpd_inc_4kw,
    y=mpd_inc_4kw_grid_cost,
    mode='markers',
    marker=dict(color='cyan', size=5),
    name="MPD-50-inc-4kW"
))

# Add data for each scenario with correct colors
fig.add_trace(go.Scatter(
    x=household_consumption_mpd_inc,
    y=mpd_inc_ext_grid_cost,
    mode='markers',
    marker=dict(color='purple', size=5),
    name="MPD-50-inc"
))

# # Add data for each scenario with correct colors
# fig.add_trace(go.Scatter(
#     x=household_consumption_base,
#     y=base_data,
#     mode='markers',
#     marker=dict(color='blue', size=5),
#     name="Base"
# ))

# Define tick values and labels
tick_vals = [0, 4000, 8000, 12000, 16000]
tick_labels = ["0", "4K", "8K", "12K", "16K"]

fig.update_layout(
    xaxis=dict(
        title="Annual consumption [kWh/a]",
        tickmode="array",
        tickvals=tick_vals,
        ticktext=tick_labels,
    ),
    yaxis=dict(
        title="Grid cost [€/a]",
        tickmode="array",
        tickvals=list(range(0, 1600 + 200, 200)),  # Extending to 1600
        range=[0, 1600]  # Explicitly setting the axis range
    ),
    template="plotly_white",
    font=dict(color="black"),
    legend=dict(title="Legend", font=dict(color="black")),
    margin=dict(l=20, r=20, t=20, b=20),  # Adjust margins for better visibility
    height=300,
    width= 450,
)

# Save the plot as PDF and HTML
pdf_file_path = "C:\\Users\\ScheiblS\\Documents\\Repositories\\dyn-grid-tariffs\\results\\a_combined_results\\a_combined_scatter_MPD_grid_cost.pdf"
fig.write_image(pdf_file_path, format='pdf')
print(f"Scatterplot has been saved as '{pdf_file_path}'")

html_file_path = "C:\\Users\\ScheiblS\\Documents\\Repositories\\dyn-grid-tariffs\\results\\a_combined_results\\a_combined_scatter_MPD_grid_cost.html"
fig.write_html(html_file_path)
print(f"Scatterplot has been saved as '{html_file_path}'")
# uv run 4_combined_result_handling.py
#****************************************************************************************************************
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

# Output paths
output_folder = "C:\\Users\\ScheiblS\\Documents\\Repositories\\dyn-grid-tariffs\\results\\a_combined_results\\"
pdf_path = os.path.join(output_folder, "a_pie_chart_self_cons.pdf")

# Define subplots with adjusted title size and reduced spacing
fig = make_subplots(
    rows=1, cols=2,
    specs=[[{'type': 'domain'}, {'type': 'domain'}]],
    subplot_titles=["Consumption", "Generation"],
    horizontal_spacing=0.03,
    font=dict(size=18) 
)

# Adjust subtitle font size
fig.update_annotations(font=dict(size=18))  

# Consumption Pie Chart (left)
fig.add_trace(go.Pie(
    labels=["Covered from PV", "Covered from Grid"],
    values=[337650, 1001254],
    textinfo='label+percent',
    hoverinfo='label+value',
    hole=0.4,
    showlegend=False,
    textfont=dict(size=14),
    marker=dict(colors=["rgb(255,215,47)","rgb(0, 181, 247)"])
), row=1, col=1)

# Generation Pie Chart (right)
fig.add_trace(go.Pie(
    labels=["PV own consumption", "Grid export"],
    values=[339286, 156300],
    textinfo='label+percent',
    hoverinfo='label+value',
    hole=0.4,
    showlegend=False,
    marker=dict(colors=["rgb(153,153,153)", "rgb(128,177,211)"])
), row=1, col=2)

# Update layout: adjust title font and margins
fig.update_layout(
    margin=dict(l=0, r=0, t=30, b=10),
    height=300,
    width=800,
)

# Export
fig.write_image(pdf_path, format='pdf')

print(f"Saved cropped pie charts:\n- PDF: {pdf_path}")

# uv run 4_combined_result_handling.py
#****************************************************************************************************************
# Data
labels = ["Covered by the grid", "Covered by PV generation"]
values = [1001, 338]
total_consumption = 1339
colors = ["rgb(179,179,179)", "rgb(255,217,47)"]  # Define colors

# Creating pie chart
fig = go.Figure(data=[go.Pie(labels=labels, values=values, 
                             textinfo='label+percent',
                             hoverinfo='label+value',
                             textfont=dict(size=14),
                             hole=0.4,  # Correct font size setting
                             showlegend = False,
                             marker=dict(colors=colors))])


# Tight layout with minimal margins
fig.update_layout(
    margin=dict(l=0, r=0, t=0, b=0),  # Remove all outer margins
    width=400,  # Adjust width
    height=400,  # Adjust height
)

# Saving the files
pdf_file_path = "C:\\Users\\ScheiblS\\Documents\\Repositories\\dyn-grid-tariffs\\results\\a_combined_results\\a_pie_chart_self_suff.pdf"
fig.write_image(pdf_file_path, format='pdf')
print(f"Pie chart has been saved as 'a_pie_chart_self_cons.pdf'")

html_file_path = "C:\\Users\\ScheiblS\\Documents\\Repositories\\dyn-grid-tariffs\\results\\a_combined_results\\a_pie_chart_self_suff.html"
fig.write_html(html_file_path)
print(f"Pie chart has been saved as 'a_pie_chart_self_cons.html'")
# uv run 4_combined_result_handling.py
#****************************************************************************************************************
# Bar chart of consumption
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

pio.kaleido.scope.mathjax = None  # Disable MathJax (optional)

# Define labels and values
labels = ["Infl. consumption", "Space heating", "DHW", "EV charging"]
values = [781, 84, 42, 431]
colors = ["rgb(254,217,166)", "rgb(231,138,195)", "rgb(254,136,177)", "rgb(251,180,174)"]  # Define colors for bars

# Create bar chart
fig = go.Figure()

fig.add_trace(go.Bar(
    x=labels, 
    y=values, 
    marker=dict(color=colors),
    text=values,  # Show values on bars
    textposition="outside",
    textfont=dict(size=18, color="black")  # Ensure black font color for bar labels
))

# Update layout for better readability
fig.update_layout(
    yaxis_title="Consumption [MWh]",
    template="plotly_white",
    font=dict(size=18, color="black"),  # Set overall font color to black
    xaxis=dict(
        tickfont=dict(size=16, color="black")  # Set x-axis labels to black
    ),
    yaxis=dict(
        tickfont=dict(size=16, color="black"),  # Set y-axis labels to black
        range=[0, max(values) * 1.2]  # Extend range for visibility
    ),
    margin=dict(l=20, r=20, t=20, b=50),  # Adjust margins for cleaner layout
    width=500,  # Increase width for better scaling
    height=500  # Adjust height
)

# Save the bar chart as PDF
pdf_file_path = "C:\\Users\\ScheiblS\\Documents\\Repositories\\dyn-grid-tariffs\\results\\a_combined_results\\a_bar_chart_consumption.pdf"
fig.write_image(pdf_file_path, format="pdf")
print(f"Bar chart has been saved as '{pdf_file_path}'")

# Save the bar chart as an interactive HTML file
html_file_path = "C:\\Users\\ScheiblS\\Documents\\Repositories\\dyn-grid-tariffs\\results\\a_combined_results\\a_bar_chart_consumption.html"
fig.write_html(html_file_path)
print(f"Bar chart has been saved as '{html_file_path}'")

# import pandas as pd
# import plotly.graph_objects as go
# import plotly.io as pio

# pio.kaleido.scope.mathjax = None  # Disable MathJax (optional)

# # Define labels and values
# labels = ["Inflexible load", "Space heating", "DHW", "EV charging"]
# values = [781, 84, 42, 431]
# colors = ["rgb(254,217,166)", "rgb(231,138,195)", "rgb(254,136,177)", "rgb(251,180,174)"]  # Define colors for bars

# # Create bar chart
# fig = go.Figure()

# fig.add_trace(go.Bar(
#     x=labels, 
#     y=values, 
#     marker=dict(color=colors),
#     text=values,  # Show values on bars
#     textposition="outside"
# ))

# # Update layout for better readability
# fig.update_layout(
#     xaxis_title="Category",
#     yaxis_title="Consumption [MWh]",
#     template="plotly_white",
#     font=dict(size=14),
#     yaxis=dict(range=[0, max(values) * 1.2]),  # Extend range for visibility
#         margin=dict(l=0, r=0, t=0, b=0),  # Remove all outer margins
#     width=300,  # Adjust width
#     height=400,  # Adjust height
# )

# # Save the bar chart as PDF
# pdf_file_path = "C:\\Users\\ScheiblS\\Documents\\Repositories\\dyn-grid-tariffs\\results\\a_combined_results\\a_bar_chart_consumption.pdf"
# fig.write_image(pdf_file_path, format="pdf")
# print(f"Bar chart has been saved as '{pdf_file_path}'")

# # Save the bar chart as an interactive HTML file
# html_file_path = "C:\\Users\\ScheiblS\\Documents\\Repositories\\dyn-grid-tariffs\\results\\a_combined_results\\a_bar_chart_consumption.html"
# fig.write_html(html_file_path)
# print(f"Bar chart has been saved as '{html_file_path}'")
# uv run 4_combined_result_handling.py
# Data
import plotly.graph_objects as go

# Data
models = ["Base", "TOU", "MPD", "MPD-50", "MPD-50-inc", "MPD-50-inc-4kW"]
costs = [115650, 92285, 68698, 91451, 77306, 79794]  # Rounded to zero decimals

model_colors = {
    "Base": "blue",
    "TOU": "red",
    "MPD": "green",
    "MPD-50": "orange",
    "MPD-50-inc": "purple",
    "MPD-50-inc-4kW": "cyan"
}

# Create bar chart with cost labels
fig = go.Figure()
for model, cost in zip(models, costs):
    fig.add_trace(go.Bar(
        x=[model],
        y=[cost],
        marker_color=model_colors[model],  # Using the toned-down color
        name=model,
        text=f"€{cost:,}",  # Adds cost values as text
        textposition='inside',  # Displays text above bars
        textfont_size=16 
    ))

# Layout customization
fig.update_layout(
    yaxis_title="Total grid cost across <br> the scenarios [€]",
    yaxis=dict(tickformat='d'),  # Removes decimal places
    showlegend=False,
    margin=dict(l=20, r=20, t=20, b=20),
    template="plotly_white",
    width=800,
    height=300  # Increased height to accommodate labels
)

# Save the bar chart as PDF
pdf_file_path = "C:\\Users\\ScheiblS\\Documents\\Repositories\\dyn-grid-tariffs\\results\\a_combined_results\\a_bar_chart_grid_cost.pdf"
fig.write_image(pdf_file_path, format="pdf")
print(f"Bar chart has been saved as '{pdf_file_path}'")