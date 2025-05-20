import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
from matplotlib.ticker import PercentFormatter

# Set style for better visuals
plt.style.use('seaborn-v0_8-whitegrid')
mpl.rcParams['font.family'] = 'DejaVu Sans'
mpl.rcParams['font.size'] = 11

# Load and prepare tourism data
visits_data = pd.read_csv('Number-of-visits.csv')

# The CSV has missing year values, so we need to fill them properly
# Forward fill the Year values to handle the way data is stored in the CSV
visits_data['Year'] = visits_data['Year'].ffill()  # Using ffill() instead of fillna(method='ffill')

# Drop any remaining rows with missing Year or Quarter values
visits_data = visits_data.dropna(subset=['Year', 'Quarter'])

# Fill any remaining NaN values with 0 in the 'Number of visits' column
visits_data['Number of visits'] = visits_data['Number of visits'].fillna(0)

# Convert Year to int for proper sorting
visits_data['Year'] = visits_data['Year'].astype(int)

# Create a mapping for quarters to ensure consistent ordering
quarter_mapping = {'I': 0, 'II': 1, 'III': 2, 'IV': 3}
visits_data['QuarterOrder'] = visits_data['Quarter'].map(quarter_mapping)
visits_data = visits_data.sort_values(['QuarterOrder', 'Year'])

# Let's reshape the data to properly group by quarter
quarters = ['I', 'II', 'III', 'IV']
quarter_names = ['Q1', 'Q2', 'Q3', 'Q4']
grouped_data = visits_data.pivot_table(
    index='Quarter', 
    columns='Year', 
    values='Number of visits', 
    aggfunc='first'
)

# Ensure quarters are in the correct order (I, II, III, IV)
grouped_data = grouped_data.reindex(quarters)

# Plot setup with proper dimensions
fig, ax = plt.subplots(figsize=(16, 10))

# Get quarters for x-axis positions
x = np.arange(len(quarters))
width = 0.07  # Width of bars - slightly narrower
years = sorted(visits_data['Year'].unique())

# Create colormap for years - using a more vibrant colormap
cmap = plt.cm.plasma  # Using plasma instead of viridis for more contrast
colors = [cmap(i/len(years)) for i in range(len(years))]

# Create a text to display the number of years shown
ax.text(0.02, 0.98, f"Including all {len(years)} years (2014-2025)", 
        transform=ax.transAxes, fontsize=12, fontweight='bold',
        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Plot each year as bars grouped by quarter
for i, year in enumerate(years):
    position = i - (len(years) - 1) / 2  # Center the bars
    values = []
    
    for q in quarters:
        if q in grouped_data.index and year in grouped_data.columns:
            value = grouped_data.loc[q, year] / 1000 if not pd.isna(grouped_data.loc[q, year]) else 0
        else:
            value = 0
        values.append(value)
    
    # Highlight pandemic years
    if year in [2020, 2021]:
        alpha = 1.0  # Make pandemic years stand out
        edgecolor = 'red'
        linewidth = 1.5
    else:
        alpha = 0.85
        edgecolor = None
        linewidth = 0
        
    bars = ax.bar(x + position * width, values, width, label=str(year), 
                 color=colors[i], alpha=alpha, edgecolor=edgecolor, linewidth=linewidth)
    
    # Add value labels on bars
    for j, v in enumerate(values):
        if v > 20:  # Only show labels for values above 20 (to reduce clutter)
            ax.text(x[j] + position * width, v + 5, f'{v:.0f}', 
                   ha='center', va='bottom', fontsize=7, rotation=90, 
                   color='black', fontweight='bold')

# Add horizontal grid lines only
ax.yaxis.grid(True, linestyle='--', alpha=0.7)
ax.xaxis.grid(False)

# Styling
ax.set_xlabel('Quarter', fontweight='bold', fontsize=14)
ax.set_ylabel('Number of Visitors (Thousands)', fontweight='bold', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(quarter_names, fontsize=13, fontweight='bold')

# Set title - now without extra padding
ax.set_title('International Tourist Arrivals in Georgia by Quarter (2014-2025)', 
            fontweight='bold', fontsize=16)

# Create a legend at the bottom of the chart instead of the top
# This avoids any overlap with the title
ax.legend(title='Year', 
          title_fontsize=13, 
          fontsize=11, 
          loc='upper center', 
          bbox_to_anchor=(0.5, -0.10),  # Position below the chart
          ncol=6, 
          frameon=True,
          fancybox=True,
          shadow=True)

# Remove spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(axis='both', which='major', labelsize=12)

# Add some annotation to highlight the COVID-19 impact
plt.annotate('COVID-19 Impact', 
             xy=(2, 46.7/1000), 
             xytext=(2.5, 200/1000),
             arrowprops=dict(facecolor='red', shrink=0.05, width=2, headwidth=8),
             fontsize=12, fontweight='bold', color='darkred')

# Adjust layout to make room for the legend at the bottom
plt.subplots_adjust(bottom=0.20, top=0.90)

# Show the plot
plt.show()

if __name__ == "__main__":
    print("Tourism trends visualization done. To see the Monte Carlo simulation, run tourism_derivatives.py")
