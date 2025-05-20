# Tourism Data Analysis Project

## Overview
This project analyzes tourism data in Georgia with a focus on seasonal patterns, trends, and financial risk management through derivatives. The analysis includes visualization of tourism trends, seasonal decomposition, and Monte Carlo simulation for pricing tourism occupancy options.

## Data
The project uses "Number-of-visits.csv" which contains quarterly tourist arrival data for Georgia from 2014 to 2025, including historical data and projections.

## Project Components

### 1. Tourism Trends Analysis (`tourism_trends.py`)
Visualizes international tourist arrivals in Georgia by quarter across multiple years, highlighting patterns and the impact of events like the COVID-19 pandemic.

**Output:** Interactive visualization of quarterly tourism trends with color-coded years.

### 2. Seasonal Analysis (`seasonal_analysis.py`)
Performs seasonal decomposition of tourism data to identify quarterly patterns and calculate seasonal factors. The analysis quantifies the economic impact of tourism seasonality on hotel occupancy, employment, and business revenue.

**Outputs:**
- `seasonal_factors.png`: Visualization of quarterly seasonal factors
- Detailed seasonal decomposition statistics

### 3. Tourism Derivatives (`tourism_derivatives.py`)
Implements a Monte Carlo simulation for pricing occupancy rate options that can help tourism businesses manage seasonal financial risks.

**Key features:**
- Custom Linear Congruential Generator for random number generation
- Mean-reverting stochastic process for occupancy rate modeling
- Option pricing for different strike levels and quarters
- Convergence analysis to validate simulation accuracy

**Outputs:**
- `option_prices.png`: Visualization of option prices by quarter
- `seasonal_vs_price.png`: Relationship between seasonal factors and option prices

## How to Run

1. Ensure you have the required dependencies installed:
   ```
   pip install pandas numpy matplotlib seaborn statsmodels
   ```

2. Place the "Number-of-visits.csv" data file in the project directory

3. Run each script individually:
   ```
   python tourism_trends.py
   python seasonal_analysis.py
   python tourism_derivatives.py
   ```

4. Review the generated visualizations and console output

## Dependencies
- Python 3.x
- pandas
- numpy
- matplotlib
- seaborn
- statsmodels

## Results
The analysis reveals significant seasonality in Georgian tourism, with summer (Q3) showing approximately 150% of the yearly average visitor numbers, while winter (Q1) drops to around 60%. This seasonality creates economic challenges:

- Fluctuating hotel occupancy rates (80-90% in peak season vs. 30-40% in low season)
- Seasonal employment volatility (150,000-170,000 seasonal jobs)
- Cash flow irregularity (45-50% of annual revenue in Q3)

The tourism derivatives component demonstrates how financial instruments could be used to hedge against these seasonal risks, with option pricing varying significantly by quarter. 