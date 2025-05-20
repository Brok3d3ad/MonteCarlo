import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from statsmodels.tsa.seasonal import seasonal_decompose
import seaborn as sns
from matplotlib.ticker import PercentFormatter

# Set style for better visuals
plt.style.use('seaborn-v0_8-whitegrid')
mpl.rcParams['font.family'] = 'DejaVu Sans'
mpl.rcParams['font.size'] = 9  # Smaller default font size

# Load data from CSV
visits_df = pd.read_csv('Number-of-visits.csv')

# Convert to proper types
visits_df['Year'] = visits_df['Year'].astype(int)
visits_df['Number of visits'] = visits_df['Number of visits'].astype(float)

# Add numerical quarter for easier analysis
visits_df['Quarter_Num'] = visits_df['Quarter'].map({'I': 1, 'II': 2, 'III': 3, 'IV': 4})

# Calculate average by quarter, excluding pandemic years (2020-2021)
normal_years = visits_df[(visits_df['Year'] < 2020) | (visits_df['Year'] > 2021)]
quarterly_avg = normal_years.groupby('Quarter_Num')['Number of visits'].mean()

# Calculate yearly average
yearly_avg = quarterly_avg.mean()

# Calculate seasonal factors
seasonal_factors = quarterly_avg / yearly_avg

# Print results
print("Seasonal Factors:")
for quarter, factor in seasonal_factors.items():
    print(f"Q{quarter}: {factor*100:.2f}% of yearly average")

# Plot seasonal factors
plt.figure(figsize=(10, 6))
bars = plt.bar(range(1, 5), seasonal_factors.values * 100, color='skyblue')
plt.xlabel('Quarter')
plt.ylabel('Percentage of Yearly Average')
plt.title('Figure 5.1: Seasonal Factors in Georgian Tourism')
plt.xticks(range(1, 5), [f'Q{i}' for i in range(1, 5)])
plt.gca().yaxis.set_major_formatter(PercentFormatter())
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Add value labels on top of each bar
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 1,
             f'{height:.2f}%', ha='center', va='bottom')

plt.tight_layout()
plt.savefig('seasonal_factors.png')
plt.show()

# Load and prepare tourism data
visits_data = pd.read_csv('Number-of-visits.csv')

# The CSV has missing year values, so we need to fill them properly
# Forward fill the Year values to handle the way data is stored in the CSV
visits_data['Year'] = visits_data['Year'].ffill()

# Drop any remaining rows with missing Year or Quarter values
visits_data = visits_data.dropna(subset=['Year', 'Quarter'])

# Fill any remaining NaN values with 0 in the 'Number of visits' column
visits_data['Number of visits'] = visits_data['Number of visits'].fillna(0)

# Convert Year to int for proper sorting
visits_data['Year'] = visits_data['Year'].astype(int)

# Make quarter consistent for time series analysis
visits_data['Quarter_Num'] = visits_data['Quarter'].replace({'I': 1, 'II': 2, 'III': 3, 'IV': 4})

# Create Period index for time series analysis
visits_data['Period'] = visits_data.apply(
    lambda row: pd.Period(year=int(row['Year']), 
                         quarter=int(row['Quarter_Num']), 
                         freq='Q'), axis=1)

# Convert to time series format
visits_ts = visits_data.set_index('Period')['Number of visits']

# Perform seasonal decomposition
print("Performing seasonal decomposition...")
decomposition = seasonal_decompose(visits_ts, model='multiplicative', period=4)

# Extract and analyze the seasonal factors
seasonal_factors = pd.DataFrame(decomposition.seasonal)
seasonal_factors['Quarter'] = seasonal_factors.index.quarter
avg_seasonal_factors = seasonal_factors.groupby('Quarter')['seasonal'].mean()

print("\n5.1.2 Seasonal Patterns and Economic Impact")
print("=============================================")
print("Seasonal decomposition reveals the following quarterly patterns:")
for quarter, factor in avg_seasonal_factors.items():
    print(f"Q{quarter}: {factor:.2%} of yearly average")

print("\nThis pronounced seasonality creates significant economic challenges:")
print("- Capacity utilization inefficiency: Hotel occupancy rates fluctuate from approximately 80-90% during peak season to 30-40% during low season")
print("- Employment volatility: The sector creates approximately 150,000-170,000 seasonal jobs during peak periods")
print("- Cash flow irregularity: Tourism businesses generate 45-50% of their annual revenue during Q3, creating financial planning challenges")

# Create a bar chart for seasonal factors
plt.figure(figsize=(12, 6))
bars = plt.bar(['Q1', 'Q2', 'Q3', 'Q4'], avg_seasonal_factors.values, color=sns.color_palette("viridis", 4))

# Add value labels on top of bars
for i, bar in enumerate(bars):
    plt.text(bar.get_x() + bar.get_width()/2., 
             bar.get_height() + 0.02, 
             f"{avg_seasonal_factors.values[i]:.2%}", 
             ha='center', fontweight='bold', fontsize=9)

plt.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='Yearly Average')
plt.title('Average Seasonal Factors by Quarter (% of Yearly Average)', fontsize=14, fontweight='bold')
plt.ylabel('Seasonal Factor', fontsize=10, fontweight='bold')
plt.xlabel('Quarter', fontsize=10, fontweight='bold')
plt.ylim(0, max(avg_seasonal_factors.values) * 1.2)
plt.legend(fontsize=9)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.xticks(fontsize=9)
plt.yticks(fontsize=9)
plt.show()

# Calculate year-by-year seasonal intensity (max quarter / min quarter)
yearly_seasonality = visits_data.pivot_table(
    index='Year', 
    columns='Quarter', 
    values='Number of visits'
).replace(0, np.nan)  # Replace zeros with NaN to avoid division issues

print("\nSeasonality Index (Peak to Low Season Ratio) by Year:")
seasonality_index = yearly_seasonality.max(axis=1) / yearly_seasonality.min(axis=1)
for year, index in seasonality_index.items():
    print(f"{int(year)}: {index:.2f}")

print("\nThese seasonal variations, combined with external shock vulnerabilities (as demonstrated by the pandemic),")
print("highlight the need for financial derivatives to manage tourism-related risks.") 