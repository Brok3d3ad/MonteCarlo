import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Linear Congruential Generator for random number generation
class LCG:
    def __init__(self, seed=12345, a=1664525, c=1013904223, m=2**32):
        """Initialize Linear Congruential Generator with parameters"""
        self.state = seed
        self.a = a
        self.c = c
        self.m = m
    
    def next(self):
        """Generate next integer in sequence"""
        self.state = (self.a * self.state + self.c) % self.m
        return self.state
    
    def random(self):
        """Generate random number in [0,1)"""
        return self.next() / self.m
    
    def normal(self, mean=0, std=1):
        """Generate normal random variable using Box-Muller transform"""
        u1 = self.random()
        u2 = self.random()
        
        # Box-Muller transform
        z0 = np.sqrt(-2.0 * np.log(u1)) * np.cos(2.0 * np.pi * u2)
        
        # Scale and shift to get desired mean and std
        return mean + std * z0

# Monte Carlo simulation for occupancy option pricing
def monte_carlo_occupancy_option(seasonal_factor, option_type, strike, 
                                notional, risk_free_rate, maturity,
                                n_paths, n_steps, seed=12345):
    """Price an occupancy rate option using Monte Carlo simulation"""
    # Initialize random number generator
    lcg = LCG(seed=seed)
    
    # Model parameters
    kappa = 3.2    # mean reversion rate
    theta = 0.0    # long-term mean (deseasonalized)
    sigma = 0.087  # volatility
    base_occupancy = 0.65  # base occupancy level
    
    # Time parameters
    dt = maturity / n_steps
    
    # Initialize paths
    paths = np.zeros((n_paths, n_steps + 1))
    paths[:, 0] = 0  # Start at equilibrium
    
    # Simulate paths
    for t in range(1, n_steps + 1):
        # Generate random shocks using LCG
        z = np.array([lcg.normal() for _ in range(n_paths)])
        
        # Mean-reverting step
        paths[:, t] = paths[:, t-1] * np.exp(-kappa * dt) + \
                     theta * (1 - np.exp(-kappa * dt)) + \
                     sigma * np.sqrt((1 - np.exp(-2 * kappa * dt)) / (2 * kappa)) * z
    
    # Calculate final occupancy rates using seasonal factor
    final_occupancy = base_occupancy * seasonal_factor + paths[:, -1]
    final_occupancy = np.clip(final_occupancy, 0, 1)  # Ensure occupancy is between 0-100%
    
    # Calculate payoffs
    if option_type.lower() == 'put':
        payoffs = np.maximum(0, strike - final_occupancy) * notional
    else:  # 'call'
        payoffs = np.maximum(0, final_occupancy - strike) * notional
    
    # Calculate present value
    option_price = np.mean(payoffs) * np.exp(-risk_free_rate * maturity)
    std_error = np.std(payoffs) / np.sqrt(n_paths) * np.exp(-risk_free_rate * maturity)
    
    return option_price, std_error

# Generate price table for different strikes and quarters
def generate_price_table(seasonal_factors, option_type, strikes, risk_free_rate):
    """Generate a table of option prices for different strikes and quarters"""
    price_matrix = np.zeros((len(strikes), 4))
    
    for i, strike in enumerate(strikes):
        for quarter in range(1, 5):
            # Set maturity to end of quarter (0.25 years per quarter)
            maturity = 0.25
            seasonal_factor = seasonal_factors[quarter-1]
            
            price, _ = monte_carlo_occupancy_option(
                seasonal_factor, option_type, strike, 1.0, 
                risk_free_rate, maturity, 10000, 50
            )
            price_matrix[i, quarter-1] = price * 100  # Convert to percentage
    
    return price_matrix

# Perform convergence analysis
def convergence_analysis(seasonal_factor, option_type, strike, 
                         notional, risk_free_rate, maturity,
                         path_counts, n_steps):
    """Analyze Monte Carlo convergence for different numbers of paths"""
    results = []
    
    for n_paths in path_counts:
        price, std_error = monte_carlo_occupancy_option(
            seasonal_factor, option_type, strike, notional,
            risk_free_rate, maturity, n_paths, n_steps
        )
        
        error_pct = (std_error / price) * 100 if price > 0 else float('nan')
        conf_interval = 1.96 * error_pct  # 95% confidence interval
        
        results.append({
            'n_paths': n_paths,
            'price': price,
            'std_error': std_error,
            'error_pct': error_pct,
            'conf_interval': conf_interval
        })
    
    return results

# Load and calculate seasonal factors from the CSV data
def calculate_seasonal_factors(exclude_pandemic=False):
    # Load data from CSV
    visits_df = pd.read_csv('Number-of-visits.csv')

    # Convert to proper types
    visits_df['Year'] = visits_df['Year'].astype(int)
    visits_df['Number of visits'] = visits_df['Number of visits'].astype(float)

    # Add numerical quarter for easier analysis
    visits_df['Quarter_Num'] = visits_df['Quarter'].map({'I': 1, 'II': 2, 'III': 3, 'IV': 4})

    # Calculate average by quarter
    if exclude_pandemic:
        # Exclude pandemic years (2020-2021) as in the original analysis
        normal_years = visits_df[(visits_df['Year'] < 2020) | (visits_df['Year'] > 2021)]
        quarterly_avg = normal_years.groupby('Quarter_Num')['Number of visits'].mean()
    else:
        # Include all years in the analysis
        quarterly_avg = visits_df.groupby('Quarter_Num')['Number of visits'].mean()

    # Calculate yearly average
    yearly_avg = quarterly_avg.mean()

    # Calculate seasonal factors
    seasonal_factors = quarterly_avg / yearly_avg
    
    return seasonal_factors.values

# Visualize option prices results
def plot_option_prices(price_matrix, strikes, seasonal_factors):
    # Visualize option prices by quarter
    plt.figure(figsize=(12, 8))
    
    # Plot for each strike
    for i, strike in enumerate(strikes):
        plt.plot(range(1, 5), price_matrix[i], marker='o', label=f"{strike*100:.0f}% Strike")
    
    plt.xlabel('Quarter')
    plt.ylabel('Option Price (% of Notional)')
    plt.title('Figure 5.2: Occupancy Put Option Prices by Quarter')
    plt.xticks(range(1, 5), ['Q1 (Winter)', 'Q2 (Spring)', 'Q3 (Summer)', 'Q4 (Autumn)'])
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig('option_prices.png')
    
    # Visualize relationship between seasonal factors and option prices
    plt.figure(figsize=(10, 6))
    
    # For 70% strike put option
    idx = 2  # Index for 70% strike
    plt.scatter(seasonal_factors * 100, price_matrix[idx], s=100, c='blue', alpha=0.7)
    
    for i, quarter in enumerate(['Q1', 'Q2', 'Q3', 'Q4']):
        plt.annotate(
            quarter, 
            (seasonal_factors[i] * 100, price_matrix[idx, i]),
            xytext=(5, 5),
            textcoords='offset points'
        )
    
    plt.xlabel('Seasonal Factor (% of Yearly Average)')
    plt.ylabel('Put Option Price (% of Notional)')
    plt.title('Figure 5.3: Relationship Between Seasonal Factors and 70% Strike Put Option Prices')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('seasonal_vs_price.png')

# Main execution
if __name__ == "__main__":
    # Calculate seasonal factors from data (including all years)
    seasonal_factors = calculate_seasonal_factors(exclude_pandemic=False)
    print("\nSeasonal Factors (All Years):")
    for i, factor in enumerate(seasonal_factors):
        print(f"Q{i+1}: {factor*100:.2f}% of yearly average")
    
    # For comparison, also calculate with pandemic years excluded
    seasonal_factors_excl = calculate_seasonal_factors(exclude_pandemic=True)
    print("\nSeasonal Factors (Excluding Pandemic Years):")
    for i, factor in enumerate(seasonal_factors_excl):
        print(f"Q{i+1}: {factor*100:.2f}% of yearly average")
    
    # Define strikes and risk-free rate
    strikes = [0.5, 0.6, 0.7, 0.8]  # 50%, 60%, 70%, 80% occupancy
    risk_free_rate = 0.04  # 4% annual rate
    
    # Generate price table for put options using all years
    print("\nOccupancy Put Option Prices (% of notional):")
    price_matrix = generate_price_table(seasonal_factors, 'put', strikes, risk_free_rate)
    
    # Format the table
    print("\nStrike Level | Winter (Q1) | Spring (Q2) | Summer (Q3) | Autumn (Q4)")
    print("-" * 65)
    
    for i, strike in enumerate(strikes):
        strike_pct = strike * 100
        row = f"{strike_pct:>11.0f}% | "
        for j in range(4):
            row += f"{price_matrix[i, j]:>11.2f}% | "
        print(row)
    
    # Convergence analysis for a 70% strike put option in Q1 (winter)
    print("\nConvergence Analysis for 70% strike put option (Q1):")
    path_counts = [1000, 10000, 100000]
    convergence_results = convergence_analysis(
        seasonal_factors[0], 'put', 0.7, 1.0, risk_free_rate, 0.25, path_counts, 50
    )
    
    print("\nNumber of Paths | Standard Error (% of Price) | 95% Confidence Interval")
    print("-" * 70)
    
    for result in convergence_results:
        print(f"{result['n_paths']:>14} | {result['error_pct']:>26.2f}% | ±{result['conf_interval']:>20.2f}%")
    
    # Plot the results
    plot_option_prices(price_matrix, strikes, seasonal_factors)
    
    print("\nAnalysis complete. Figures saved to 'option_prices.png' and 'seasonal_vs_price.png'.") 