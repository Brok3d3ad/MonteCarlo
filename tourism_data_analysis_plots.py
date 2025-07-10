"""
ტურიზმის მონაცემების ვიზუალიზაცია და ემპირიული პარამეტრების ანალიზი
მონაცემთა ვიზუალიზაცია Age.csv, Expenditure.csv, Satisfaction.csv ფაილებიდან
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import matplotlib.dates as mdates
import matplotlib.font_manager as fm
import os

# Georgian font configuration - Fixed for proper Georgian display
plt.rcParams['font.family'] = ['DejaVu Sans', 'sans-serif']
plt.rcParams['font.size'] = 12
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['axes.unicode_minus'] = False

# Configure matplotlib to use fonts that support Georgian Unicode
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

# Try to find and use Georgian fonts
georgian_fonts = [
    'Sylfaen',           # Windows Georgian font
    'BPG Nino Mtavruli', # Georgian font
    'DejaVu Sans',       # Fallback with Unicode support
    'Arial Unicode MS',  # Windows Unicode font
    'Noto Sans Georgian' # Google Noto Georgian
]

# Set the font property for all text
plt.rcParams['font.family'] = georgian_fonts
plt.rcParams['font.sans-serif'] = georgian_fonts

# Force matplotlib to rebuild font cache
try:
    fm._rebuild()
except:
    pass

# Set backend to support both display and saving
# Try to use interactive backend if available, fall back to Agg
try:
    plt.switch_backend('TkAgg')  # Interactive backend
    print("Using TkAgg backend for display")
except:
    try:
        plt.switch_backend('Qt5Agg')  # Alternative interactive backend
        print("Using Qt5Agg backend for display")
    except:
        plt.switch_backend('Agg')  # Non-interactive backend (save only)
        print("Using Agg backend (plots will be saved but not displayed)")

# Set style
sns.set_style("whitegrid")

def configure_georgian_fonts():
    """Configure matplotlib to properly display Georgian text"""
    import matplotlib.font_manager as fm
    
    # Force matplotlib to use system fonts that support Georgian
    # Windows typically has Sylfaen font which supports Georgian
    plt.rcParams['font.family'] = ['sans-serif']
    plt.rcParams['font.sans-serif'] = ['Sylfaen', 'DejaVu Sans', 'Arial Unicode MS', 'Liberation Sans', 'sans-serif']
    
    # Set text properties to handle Unicode properly
    plt.rcParams['text.usetex'] = False
    plt.rcParams['font.size'] = 12
    
    # Suppress font warnings
    import warnings
    warnings.filterwarnings('ignore', message='.*missing.*font.*')
    warnings.filterwarnings('ignore', message='.*Glyph.*missing.*')
    
    # Force font cache rebuild
    try:
        fm.fontManager.__init__()
    except:
        pass
    
    print("Georgian fonts configured with Unicode support")
    return True

# Configure Georgian fonts
configure_georgian_fonts()

# Create a wrapper function for text rendering
def render_georgian_text(text, ax, **kwargs):
    """Render Georgian text with proper font configuration"""
    try:
        # Use Sylfaen font specifically for Georgian text
        font_prop = fm.FontProperties(family='Sylfaen')
        if 'fontproperties' not in kwargs:
            kwargs['fontproperties'] = font_prop
        ax.text(text, **kwargs)
    except:
        # Fallback to default rendering
        ax.text(text, **kwargs)

def load_and_prepare_data():
    """მონაცემების ჩატვირთვა და მომზადება"""
    # Load data
    age_df = pd.read_csv('Age.csv')
    expenditure_df = pd.read_csv('Expenditure.csv')
    satisfaction_df = pd.read_csv('Satisfaction.csv')
    
    # Clean expenditure data (remove commas)
    expenditure_df['Total expenditure'] = expenditure_df['Total expenditure'].str.replace(',', '').astype(float)
    
    # Create datetime columns for better plotting
    def create_datetime(df):
        df['Date'] = pd.to_datetime(df['Year'].astype(str) + '-' + 
                                   df['Quarter'].map({'I': '01', 'II': '04', 'III': '07', 'IV': '10'}) + '-01')
        return df
    
    age_df = create_datetime(age_df)
    expenditure_df = create_datetime(expenditure_df)
    satisfaction_df = create_datetime(satisfaction_df)
    
    return age_df, expenditure_df, satisfaction_df

def calculate_empirical_parameters(age_df, expenditure_df, satisfaction_df):
    """ემპირიული პარამეტრების გამოთვლა"""
    covid_years = [2020, 2021]
    base_years = [2014, 2015, 2016, 2017, 2018, 2019, 2022, 2023, 2024]
    
    # Seasonal factors (excluding COVID years)
    normal_data = age_df[~age_df['Year'].isin(covid_years)]
    quarterly_avg = normal_data.groupby('Quarter')['Total'].mean()
    annual_avg = normal_data['Total'].mean()
    seasonal_factors = quarterly_avg / annual_avg
    
    # Growth rates
    yearly_totals = normal_data.groupby('Year')['Total'].sum()
    growth_rates = []
    years = sorted(yearly_totals.index)
    
    for i in range(1, len(years)):
        growth_rate = (yearly_totals[years[i]] - yearly_totals[years[i-1]]) / yearly_totals[years[i-1]]
        growth_rates.append(growth_rate)
    
    # Volatility by quarter
    visitor_volatility = {}
    expenditure_volatility = {}
    
    for quarter in ['I', 'II', 'III', 'IV']:
        # Visitor volatility
        quarter_data = age_df[(age_df['Quarter'] == quarter) & (~age_df['Year'].isin(covid_years))]['Total']
        if len(quarter_data) > 1:
            visitor_volatility[quarter] = quarter_data.std() / quarter_data.mean()
        else:
            visitor_volatility[quarter] = 0.10
        
        # Expenditure per visitor volatility
        expenditures_per_visitor = []
        for year in base_years:
            visitor_data = age_df[(age_df['Year'] == year) & (age_df['Quarter'] == quarter)]
            exp_data = expenditure_df[(expenditure_df['Year'] == year) & (expenditure_df['Quarter'] == quarter)]
            
            if not visitor_data.empty and not exp_data.empty:
                visitors = visitor_data['Total'].iloc[0]
                total_exp = exp_data['Total expenditure'].iloc[0]
                if visitors > 0:
                    exp_per_visitor = (total_exp * 1000000) / (visitors * 1000)
                    expenditures_per_visitor.append(exp_per_visitor)
        
        if len(expenditures_per_visitor) > 1:
            mean_exp = np.mean(expenditures_per_visitor)
            std_exp = np.std(expenditures_per_visitor)
            expenditure_volatility[quarter] = std_exp / mean_exp if mean_exp > 0 else 0.10
        else:
            expenditure_volatility[quarter] = 0.10
    
    return {
        'seasonal_factors': seasonal_factors,
        'growth_rates': growth_rates,
        'growth_years': years[1:],
        'visitor_volatility': visitor_volatility,
        'expenditure_volatility': expenditure_volatility
    }

def plot_historical_trends(age_df, expenditure_df, satisfaction_df):
    """ისტორიული ტრენდების ვიზუალიზაცია"""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('ტურიზმის ისტორიული მონაცემების ტრენდები (2014-2025)', fontsize=16, fontweight='bold')
    
    # 1. Visitor Numbers Over Time
    ax1 = axes[0, 0]
    quarters = ['I', 'II', 'III', 'IV']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for i, quarter in enumerate(quarters):
        quarter_data = age_df[age_df['Quarter'] == quarter]
        ax1.plot(quarter_data['Date'], quarter_data['Total'], 
                marker='o', label=f'{quarter} კვარტალი', color=colors[i], linewidth=2)
    
    ax1.set_title('ვიზიტორთა რაოდენობა კვარტლების მიხედვით (ათასებში)', fontweight='bold')
    ax1.set_xlabel('წელი')
    ax1.set_ylabel('ვიზიტორები (ათასი)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Highlight COVID period
    covid_start = pd.to_datetime('2020-01-01')
    covid_end = pd.to_datetime('2021-12-31')
    ax1.axvspan(covid_start, covid_end, alpha=0.2, color='red', label='COVID პერიოდი')
    
    # 2. Total Expenditure Over Time
    ax2 = axes[0, 1]
    for i, quarter in enumerate(quarters):
        quarter_data = expenditure_df[expenditure_df['Quarter'] == quarter]
        ax2.plot(quarter_data['Date'], quarter_data['Total expenditure'], 
                marker='s', label=f'{quarter} კვარტალი', color=colors[i], linewidth=2)
    
    ax2.set_title('მთლიანი დანახარჯები კვარტლების მიხედვით (მილიონი ლარი)', fontweight='bold')
    ax2.set_xlabel('წელი')
    ax2.set_ylabel('დანახარჯები (მილიონი ლარი)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.axvspan(covid_start, covid_end, alpha=0.2, color='red')
    
    # 3. Satisfaction Rates Over Time
    ax3 = axes[1, 0]
    satisfaction_df['Satisfaction_Rate'] = (satisfaction_df['Very satisfied'] + satisfaction_df['Satisfied']) / satisfaction_df['Total']
    
    for i, quarter in enumerate(quarters):
        quarter_data = satisfaction_df[satisfaction_df['Quarter'] == quarter]
        ax3.plot(quarter_data['Date'], quarter_data['Satisfaction_Rate'] * 100, 
                marker='^', label=f'{quarter} კვარტალი', color=colors[i], linewidth=2)
    
    ax3.set_title('კმაყოფილების დონე კვარტლების მიხედვით (%)', fontweight='bold')
    ax3.set_xlabel('წელი')
    ax3.set_ylabel('კმაყოფილების დონე (%)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.axvspan(covid_start, covid_end, alpha=0.2, color='red')
    
    # 4. Expenditure per Visitor
    ax4 = axes[1, 1]
    for i, quarter in enumerate(quarters):
        exp_per_visitor = []
        dates = []
        for _, row in expenditure_df[expenditure_df['Quarter'] == quarter].iterrows():
            visitor_row = age_df[(age_df['Year'] == row['Year']) & (age_df['Quarter'] == quarter)]
            if not visitor_row.empty:
                visitors = visitor_row['Total'].iloc[0]
                if visitors > 0:
                    exp_per_visitor.append((row['Total expenditure'] * 1000000) / (visitors * 1000))
                    dates.append(row['Date'])
        
        if exp_per_visitor:
            ax4.plot(dates, exp_per_visitor, marker='d', label=f'{quarter} კვარტალი', 
                    color=colors[i], linewidth=2)
    
    ax4.set_title('დანახარჯი ერთ ვიზიტორზე კვარტლების მიხედვით (ლარი)', fontweight='bold')
    ax4.set_xlabel('წელი')
    ax4.set_ylabel('დანახარჯი ერთ ვიზიტორზე (ლარი)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.axvspan(covid_start, covid_end, alpha=0.2, color='red')
    
    plt.tight_layout()
    
    # Save the plot
    filename = 'historical_tourism_trends.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"📊 შენახულია: {filename}")
    
    # Display the plot
    plt.show(block=False)
    plt.pause(0.1)

def plot_empirical_parameters(parameters):
    """ემპირიული პარამეტრების ვიზუალიზაცია"""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('ემპირიული პარამეტრები მონტე კარლოს სიმულაციისთვის', fontsize=16, fontweight='bold')
    
    quarters = ['I', 'II', 'III', 'IV']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    # 1. Seasonal Factors
    ax1 = axes[0, 0]
    seasonal_values = [parameters['seasonal_factors'][q] for q in quarters]
    bars1 = ax1.bar(quarters, seasonal_values, color=colors, alpha=0.8)
    ax1.set_title('სეზონური ფაქტორები კვარტლების მიხედვით', fontweight='bold')
    ax1.set_xlabel('კვარტალი')
    ax1.set_ylabel('სეზონური ფაქტორი')
    ax1.axhline(y=1, color='black', linestyle='--', alpha=0.5)
    
    # Add value labels on bars
    for bar, value in zip(bars1, seasonal_values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # 2. Growth Rates Over Time
    ax2 = axes[0, 1]
    ax2.plot(parameters['growth_years'], [r*100 for r in parameters['growth_rates']], 
             marker='o', linewidth=2, markersize=8, color='green')
    ax2.set_title('წლიური ზრდის ტემპი (COVID-ის გამორიცხვით)', fontweight='bold')
    ax2.set_xlabel('წელი')
    ax2.set_ylabel('ზრდის ტემპი (%)')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # Add median line
    median_growth = np.median(parameters['growth_rates']) * 100
    ax2.axhline(y=median_growth, color='red', linestyle='--', alpha=0.7, 
                label=f'მედიანა: {median_growth:.1f}%')
    ax2.legend()
    
    # 3. Visitor Volatility by Quarter
    ax3 = axes[1, 0]
    visitor_vol_values = [parameters['visitor_volatility'][q]*100 for q in quarters]
    bars3 = ax3.bar(quarters, visitor_vol_values, color=colors, alpha=0.8)
    ax3.set_title('ვიზიტორთა ცვალებადობა კვარტლების მიხედვით', fontweight='bold')
    ax3.set_xlabel('კვარტალი')
    ax3.set_ylabel('ცვალებადობა (%)')
    
    # Add value labels on bars
    for bar, value in zip(bars3, visitor_vol_values):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # 4. Expenditure Volatility by Quarter
    ax4 = axes[1, 1]
    exp_vol_values = [parameters['expenditure_volatility'][q]*100 for q in quarters]
    bars4 = ax4.bar(quarters, exp_vol_values, color=colors, alpha=0.8)
    ax4.set_title('დანახარჯების ცვალებადობა ერთ ვიზიტორზე კვარტლების მიხედვით', fontweight='bold')
    ax4.set_xlabel('კვარტალი')
    ax4.set_ylabel('ცვალებადობა (%)')
    
    # Add value labels on bars
    for bar, value in zip(bars4, exp_vol_values):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    
    # Save the plot
    filename = 'empirical_parameters.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"📊 შენახულია: {filename}")
    
    # Display the plot
    plt.show(block=False)
    plt.pause(0.1)

def plot_seasonal_patterns(age_df, expenditure_df, satisfaction_df):
    """სეზონური პატერნების ვიზუალიზაცია"""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('სეზონური პატერნების ანალიზი', fontsize=16, fontweight='bold')
    
    # Exclude COVID years for cleaner patterns
    covid_years = [2020, 2021]
    age_clean = age_df[~age_df['Year'].isin(covid_years)]
    expenditure_clean = expenditure_df[~expenditure_df['Year'].isin(covid_years)]
    satisfaction_clean = satisfaction_df[~satisfaction_df['Year'].isin(covid_years)]
    
    quarters = ['I', 'II', 'III', 'IV']
    
    # 1. Box plot of visitors by quarter
    ax1 = axes[0, 0]
    visitor_data = [age_clean[age_clean['Quarter'] == q]['Total'].values for q in quarters]
    bp1 = ax1.boxplot(visitor_data, labels=quarters, patch_artist=True)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    for patch, color in zip(bp1['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax1.set_title('ვიზიტორთა განაწილება კვარტლების მიხედვით', fontweight='bold')
    ax1.set_ylabel('ვიზიტორები (ათასი)')
    ax1.grid(True, alpha=0.3)
    
    # 2. Box plot of expenditure by quarter
    ax2 = axes[0, 1]
    exp_data = [expenditure_clean[expenditure_clean['Quarter'] == q]['Total expenditure'].values for q in quarters]
    bp2 = ax2.boxplot(exp_data, labels=quarters, patch_artist=True)
    for patch, color in zip(bp2['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax2.set_title('დანახარჯების განაწილება კვარტლების მიხედვით', fontweight='bold')
    ax2.set_ylabel('დანახარჯები (მილიონი ლარი)')
    ax2.grid(True, alpha=0.3)
    
    # 3. Average satisfaction by quarter
    ax3 = axes[1, 0]
    satisfaction_clean['Satisfaction_Rate'] = (satisfaction_clean['Very satisfied'] + satisfaction_clean['Satisfied']) / satisfaction_clean['Total']
    avg_satisfaction = satisfaction_clean.groupby('Quarter')['Satisfaction_Rate'].mean() * 100
    bars3 = ax3.bar(quarters, avg_satisfaction, color=colors, alpha=0.8)
    ax3.set_title('საშუალო კმაყოფილების დონე კვარტლების მიხედვით', fontweight='bold')
    ax3.set_ylabel('კმაყოფილების დონე (%)')
    ax3.set_ylim(75, 95)
    
    # Add value labels
    for bar, value in zip(bars3, avg_satisfaction):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.2,
                f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # 4. Expenditure per visitor by quarter
    ax4 = axes[1, 1]
    exp_per_visitor_by_quarter = []
    for quarter in quarters:
        exp_per_visitor = []
        for year in range(2014, 2025):
            if year in covid_years:
                continue
            visitor_data = age_df[(age_df['Year'] == year) & (age_df['Quarter'] == quarter)]
            exp_data = expenditure_df[(expenditure_df['Year'] == year) & (expenditure_df['Quarter'] == quarter)]
            
            if not visitor_data.empty and not exp_data.empty:
                visitors = visitor_data['Total'].iloc[0]
                total_exp = exp_data['Total expenditure'].iloc[0]
                if visitors > 0:
                    exp_per_visitor.append((total_exp * 1000000) / (visitors * 1000))
        
        exp_per_visitor_by_quarter.append(exp_per_visitor)
    
    bp4 = ax4.boxplot(exp_per_visitor_by_quarter, labels=quarters, patch_artist=True)
    for patch, color in zip(bp4['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax4.set_title('დანახარჯი ერთ ვიზიტორზე კვარტლების მიხედვით', fontweight='bold')
    ax4.set_ylabel('დანახარჯი ერთ ვიზიტორზე (ლარი)')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    filename = 'seasonal_patterns.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"📊 შენახულია: {filename}")
    
    # Display the plot
    plt.show(block=False)
    plt.pause(0.1)

def create_parameter_summary_table(parameters):
    """პარამეტრების ცხრილის შექმნა"""
    
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('tight')
    ax.axis('off')
    
    # Create table data
    quarters = ['I', 'II', 'III', 'IV']
    table_data = []
    
    for quarter in quarters:
        row = [
            f'{quarter} კვარტალი',
            f'{parameters["seasonal_factors"][quarter]:.3f}',
            f'{parameters["visitor_volatility"][quarter]*100:.1f}%',
            f'{parameters["expenditure_volatility"][quarter]*100:.1f}%'
        ]
        table_data.append(row)
    
    # Add summary row
    median_growth = np.median(parameters['growth_rates']) * 100
    table_data.append([
        'წლიური ზრდა',
        f'{median_growth:.1f}%',
        'ზრდის ტემპი',
        f'{np.std(parameters["growth_rates"])*100:.1f}%'
    ])
    
    # Create table
    table = ax.table(cellText=table_data,
                    colLabels=['კვარტალი', 'სეზონური ფაქტორი', 'ვიზიტორთა ცვალებადობა', 'დანახარჯების ცვალებადობა'],
                    cellLoc='center',
                    loc='center',
                    colWidths=[0.25, 0.25, 0.25, 0.25])
    
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 2)
    
    # Style the table
    for i in range(len(table_data) + 1):
        for j in range(4):
            if i == 0:  # Header row
                table[(i, j)].set_facecolor('#4CAF50')
                table[(i, j)].set_text_props(weight='bold', color='white')
            elif i == len(table_data):  # Summary row
                table[(i, j)].set_facecolor('#E8F5E8')
                table[(i, j)].set_text_props(weight='bold')
            else:
                table[(i, j)].set_facecolor('#F5F5F5')
    
    plt.title('ემპირიული პარამეტრების ცხრილი მონტე კარლოს სიმულაციისთვის', 
              fontsize=16, fontweight='bold', pad=20)
              
    # Save the plot
    filename = 'parameter_summary_table.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"📊 შენახულია: {filename}")
    
    # Display the plot
    plt.show(block=False)
    plt.pause(0.1)

def main():
    """მთავარი ფუნქცია"""
    print("მონაცემების ჩატვირთვა...")
    age_df, expenditure_df, satisfaction_df = load_and_prepare_data()
    
    print("ემპირიული პარამეტრების გამოთვლა...")
    parameters = calculate_empirical_parameters(age_df, expenditure_df, satisfaction_df)
    
    print("გრაფიკების შექმნა...")
    
    # 1. Historical trends
    print("- ისტორიული ტრენდების გრაფიკი")
    plot_historical_trends(age_df, expenditure_df, satisfaction_df)
    
    # 2. Empirical parameters
    print("- ემპირიული პარამეტრების გრაფიკი")
    plot_empirical_parameters(parameters)
    
    # 3. Seasonal patterns
    print("- სეზონური პატერნების გრაფიკი")
    plot_seasonal_patterns(age_df, expenditure_df, satisfaction_df)
    
    # 4. Parameter summary table
    print("- პარამეტრების ცხრილი")
    create_parameter_summary_table(parameters)
    
    print("\n🎉 ყველა გრაფიკი შენახულია PNG ფორმატში:")
    print("- historical_tourism_trends.png")
    print("- empirical_parameters.png")
    print("- seasonal_patterns.png")
    print("- parameter_summary_table.png")
    
    # Ensure all plots are displayed
    print("\n📺 გრაფიკების ჩვენება...")
    try:
        plt.show(block=True)  # Show all plots and wait
        print("✅ ყველა გრაფიკი ნაჩვენებია!")
    except:
        print("⚠️ გრაფიკების ჩვენება ვერ მოხერხდა, მაგრამ PNG ფაილები შენახულია")
    
    # Print key statistics
    print("\nძირითადი ემპირიული პარამეტრები:")
    print("-" * 50)
    print(f"მედიანური ზრდის ტემპი: {np.median(parameters['growth_rates'])*100:.1f}%")
    print(f"ზრდის ცვალებადობა: {np.std(parameters['growth_rates'])*100:.1f}%")
    print("\nსეზონური ფაქტორები:")
    for quarter in ['I', 'II', 'III', 'IV']:
        print(f"  {quarter} კვარტალი: {parameters['seasonal_factors'][quarter]:.3f}")
    
    print("\nვიზიტორთა ცვალებადობა კვარტლების მიხედვით:")
    for quarter in ['I', 'II', 'III', 'IV']:
        print(f"  {quarter} კვარტალი: {parameters['visitor_volatility'][quarter]*100:.1f}%")
    
    print("\nდანახარჯების ცვალებადობა კვარტლების მიხედვით:")
    for quarter in ['I', 'II', 'III', 'IV']:
        print(f"  {quarter} კვარტალი: {parameters['expenditure_volatility'][quarter]*100:.1f}%")

if __name__ == "__main__":
    main() 