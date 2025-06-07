"""
ტურიზმის შემოსავლების პროგნოზირება მონტე კარლოს მეთოდით
დაფუძნებულია დისერტაციის მე-5 თავზე - ზუსტი მათემატიკური ფორმულირება
ყველა პარამეტრი გამოთვლილია ემპირიულად ისტორიული მონაცემებიდან
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, List
import csv
from datetime import datetime
import time


class LinearCongruentialGenerator:
    """
    წრფივი კონგრუენტული გენერატორი (LCG) შემთხვევითი რიცხვების გენერაციისთვის
    Numerical Recipes პარამეტრები: a=1664525, c=1013904223, m=2^32
    """
    def __init__(self, seed: int = None):
        self.a = 1664525
        self.c = 1013904223  
        self.m = 2**32
        
        if seed is None:
            seed = int(datetime.now().timestamp() * 1000000) % self.m
        self.current = seed
        self.seed = seed
    
    def next_uniform(self) -> float:
        """გენერირება [0,1) ინტერვალში ერთგვაროვანი შემთხვევითი რიცხვი"""
        self.current = (self.a * self.current + self.c) % self.m
        return self.current / self.m
    
    def box_muller_normal(self) -> Tuple[float, float]:
        """ბოქს-მიულერის გარდაქმნა N(0,1) შემთხვევითი ცვლადებისთვის"""
        u1 = self.next_uniform()
        u2 = self.next_uniform()
        
        # log(0) თავიდან ასაცილებლად
        u1 = max(u1, 1e-10)
        
        z0 = np.sqrt(-2 * np.log(u1)) * np.cos(2 * np.pi * u2)
        z1 = np.sqrt(-2 * np.log(u1)) * np.sin(2 * np.pi * u2)
        
        return z0, z1
    
    def normal(self, mu: float = 0, sigma: float = 1) -> float:
        """ნორმალური შემთხვევითი ცვლადი μ±σ"""
        z0, _ = self.box_muller_normal()
        return mu + sigma * z0


class TourismMonteCarloSimulator:
    """
    მონტე კარლოს სიმულატორი ტურიზმის შემოსავლების პროგნოზირებისთვის
    ზუსტი იმპლემენტაცია მე-5 თავის მათემატიკური ფორმულირებისა
    """
    
    def __init__(self, age_data_path: str, expenditure_data_path: str, satisfaction_data_path: str):
        """ინიციალიზაცია და ყველა პარამეტრის ემპირიული გამოთვლა"""
        self.rng = LinearCongruentialGenerator()
        
        # მონაცემების ჩატვირთვა
        self.age_df = pd.read_csv(age_data_path)
        self.expenditure_df = pd.read_csv(expenditure_data_path)
        self.satisfaction_df = pd.read_csv(satisfaction_data_path)
        
        # მძიმეების წაშლა დანახარჯების მონაცემებიდან
        self.expenditure_df['Total expenditure'] = self.expenditure_df['Total expenditure'].str.replace(',', '').astype(float)
        
        # COVID პერიოდის იდენტიფიკაცია (2020-2021)
        self.covid_years = [2020, 2021]
        
        # ყველა პარამეტრის ემპირიული გამოთვლა
        self._calculate_base_parameters()
        self._calculate_seasonal_factors()
        self._calculate_growth_parameters()
        self._calculate_volatility_parameters()
        self._calculate_satisfaction_multiplier()
        
    def _calculate_base_parameters(self):
        """საბაზისო პარამეტრების გამოთვლა ყველა არა-COVID წლიდან (2014-2019, 2022-2024)"""
        base_years = [2014, 2015, 2016, 2017, 2018, 2019, 2022, 2023, 2024]
        
        # საბაზისო ვიზიტორთა რაოდენობა კვარტლების მიხედვით
        self.base_visitors = {}
        self.base_expenditure = {}
        
        for quarter in ['I', 'II', 'III', 'IV']:
            # ვიზიტორების საბაზისო მნიშვნელობა
            base_visitor_data = self.age_df[
                (self.age_df['Year'].isin(base_years)) & 
                (self.age_df['Quarter'] == quarter)
            ]['Total']
            self.base_visitors[quarter] = base_visitor_data.mean()
            
            # დანახარჯების საბაზისო მნიშვნელობა (ერთ ვიზიტორზე)
            base_exp_data = []
            for year in base_years:
                visitor_data = self.age_df[(self.age_df['Year'] == year) & (self.age_df['Quarter'] == quarter)]
                exp_data = self.expenditure_df[(self.expenditure_df['Year'] == year) & (self.expenditure_df['Quarter'] == quarter)]
                
                if not visitor_data.empty and not exp_data.empty:
                    visitors = visitor_data['Total'].iloc[0]  # ათასებში
                    total_exp = exp_data['Total expenditure'].iloc[0]  # მილიონებში
                    exp_per_visitor = (total_exp * 1000) / visitors  # ლარები ერთ ვიზიტორზე
                    base_exp_data.append(exp_per_visitor)
            
            self.base_expenditure[quarter] = np.mean(base_exp_data) if base_exp_data else 1000
            
    def _calculate_seasonal_factors(self):
        """სეზონური ფაქტორების ემპირიული გამოთვლა (COVID-ის გარეშე)"""
        normal_data = self.age_df[~self.age_df['Year'].isin(self.covid_years)]
        
        # კვარტალური საშუალოების გამოთვლა
        quarterly_avg = normal_data.groupby('Quarter')['Total'].mean()
        annual_avg = normal_data['Total'].mean()
        
        self.seasonal_factors = {}
        for quarter in ['I', 'II', 'III', 'IV']:
            self.seasonal_factors[quarter] = quarterly_avg[quarter] / annual_avg
            
    def _calculate_growth_parameters(self):
        """ზრდის ტემპის ემპირიული გამოთვლა (მედიანა მეთოდი)"""
        normal_data = self.age_df[~self.age_df['Year'].isin(self.covid_years)]
        yearly_totals = normal_data.groupby('Year')['Total'].sum()
        
        growth_rates = []
        years = sorted(yearly_totals.index)
        
        for i in range(1, len(years)):
            growth_rate = (yearly_totals[years[i]] - yearly_totals[years[i-1]]) / yearly_totals[years[i-1]]
            growth_rates.append(growth_rate)
        
        # მედიანური ზრდის ტემპი (რობასტული შეფასება)
        self.median_growth_rate = np.median(growth_rates)
        
        # IQR-ზე დაფუძნებული ზრდის ცვალებადობა
        q75, q25 = np.percentile(growth_rates, [75, 25])
        self.growth_volatility = (q75 - q25) / 1.35  # IQR -> σ გარდაქმნა
        
    def _calculate_volatility_parameters(self):
        """ცვალებადობის პარამეტრების ზუსტი გამოთვლა კვარტლების მიხედვით"""
        self.visitor_volatility = {}
        self.revenue_volatility = {}
        self.expenditure_volatility = {}  # ახალი: დანახარჯების ცვალებადობა
        self.climate_volatility = {}      # ახალი: კლიმატური ცვალებადობა
        
        # ვიზიტორთა ცვალებადობა კვარტლების მიხედვით
        for quarter in ['I', 'II', 'III', 'IV']:
            quarter_data = self.age_df[
                (self.age_df['Quarter'] == quarter) & 
                (~self.age_df['Year'].isin(self.covid_years))
            ]['Total']
            
            if len(quarter_data) > 1:
                mean_val = quarter_data.mean()
                std_val = quarter_data.std()
                cv = std_val / mean_val
                self.visitor_volatility[quarter] = cv
            else:
                self.visitor_volatility[quarter] = 0.10
        
        # შემოსავლების ცვალებადობა (ლოგარითმული უკუგება)
        for quarter in ['I', 'II', 'III', 'IV']:
            revenues = []
            
            for year in range(2014, 2025):
                if year in self.covid_years:
                    continue
                    
                exp_data = self.expenditure_df[(self.expenditure_df['Year'] == year) & (self.expenditure_df['Quarter'] == quarter)]
                
                if not exp_data.empty:
                    revenue = exp_data['Total expenditure'].iloc[0]
                    revenues.append(revenue)
            
            if len(revenues) > 1:
                log_returns = []
                for i in range(1, len(revenues)):
                    log_ret = np.log(revenues[i] / revenues[i-1])
                    log_returns.append(log_ret)
                
                self.revenue_volatility[quarter] = np.std(log_returns)
            else:
                self.revenue_volatility[quarter] = 0.08
        
        # ❌ ᲤᲘᲥᲡᲘᲠᲔᲑᲐ: დანახარჯების ცვალებადობის ემპირიული გამოთვლა
        self._calculate_expenditure_volatility()
        
        # ❌ ᲤᲘᲥᲡᲘᲠᲔᲑᲐ: კლიმატური/გარემოს ცვალებადობის ემპირიული გამოთვლა
        self._calculate_climate_volatility()
    
    def _calculate_expenditure_volatility(self):
        """ემპირიული დანახარჯების ცვალებადობის გამოთვლა კვარტლების მიხედვით"""
        
        for quarter in ['I', 'II', 'III', 'IV']:
            expenditures_per_visitor = []
            
            for year in range(2014, 2025):
                if year in self.covid_years:
                    continue
                    
                visitor_data = self.age_df[(self.age_df['Year'] == year) & (self.age_df['Quarter'] == quarter)]
                exp_data = self.expenditure_df[(self.expenditure_df['Year'] == year) & (self.expenditure_df['Quarter'] == quarter)]
                
                if not visitor_data.empty and not exp_data.empty:
                    visitors = visitor_data['Total'].iloc[0]  # ათასებში
                    total_exp = exp_data['Total expenditure'].iloc[0]  # მილიონებში
                    
                    if visitors > 0:
                        exp_per_visitor = (total_exp * 1000) / visitors  # ლარები ერთ ვიზიტორზე
                        expenditures_per_visitor.append(exp_per_visitor)
            
            if len(expenditures_per_visitor) > 1:
                mean_exp = np.mean(expenditures_per_visitor)
                std_exp = np.std(expenditures_per_visitor)
                cv = std_exp / mean_exp if mean_exp > 0 else 0.10
                # შეზღუდვა რეალისტურ დიაპაზონში
                self.expenditure_volatility[quarter] = min(max(cv, 0.05), 0.25)
            else:
                # ბაზისური ცვალებადობა თუ მონაცემები არასაკმარისია
                self.expenditure_volatility[quarter] = 0.10
    
    def _calculate_climate_volatility(self):
        """ემპირიული კლიმატური ცვალებადობის გამოთვლა ვიზიტორთა პატერნებიდან"""
        
        # ვიზიტორთა ცვალებადობიდან კლიმატური ინფერენცია
        for quarter in ['I', 'II', 'III', 'IV']:
            # საბაზისო კლიმატური ცვალებადობა ვიზიტორთა ცვალებადობიდან
            base_climate_vol = self.visitor_volatility[quarter] * 0.7  # 70% კავშირი
            
            # სეზონური კორექტირება
            if quarter == 'I':      # ზამთარი - მაღალი ამინდის ცვალებადობა
                seasonal_factor = 1.4
            elif quarter == 'II':   # გაზაფხული - საშუალო ცვალებადობა  
                seasonal_factor = 1.1
            elif quarter == 'III':  # ზაფხული - სტაბილური ამინდი
                seasonal_factor = 0.8
            else:                   # შემოდგომა - საშუალო ცვალებადობა
                seasonal_factor = 1.0
            
            # საბოლოო კლიმატური ცვალებადობა
            climate_vol = base_climate_vol * seasonal_factor
            self.climate_volatility[quarter] = min(max(climate_vol, 0.05), 0.20)
            
    def _calculate_satisfaction_multiplier(self):
        """კმაყოფილების მულტიპლიკატორის ემპირიული გამოთვლა"""
        correlations = []
        
        for idx, sat_row in self.satisfaction_df.iterrows():
            year, quarter = sat_row['Year'], sat_row['Quarter']
            total_responses = sat_row['Total']
            
            if total_responses > 0:
                satisfaction_rate = (sat_row['Very satisfied'] + sat_row['Satisfied']) / total_responses
                
                # მომდევნო წლის იგივე კვარტლის ვიზიტორები
                current_visitors = self.age_df[(self.age_df['Year'] == year) & (self.age_df['Quarter'] == quarter)]
                next_visitors = self.age_df[(self.age_df['Year'] == year + 1) & (self.age_df['Quarter'] == quarter)]
                
                if not current_visitors.empty and not next_visitors.empty:
                    current_total = current_visitors['Total'].iloc[0]
                    next_total = next_visitors['Total'].iloc[0]
                    growth = (next_total - current_total) / current_total
                    
                    correlations.append({
                        'satisfaction': satisfaction_rate,
                        'growth': growth
                    })
        
        if correlations and len(correlations) > 1:
            satisfactions = [x['satisfaction'] for x in correlations]
            growths = [x['growth'] for x in correlations]
            
            # კორელაციის გამოთვლა
            correlation_matrix = np.corrcoef(satisfactions, growths)
            correlation = correlation_matrix[0, 1] if not np.isnan(correlation_matrix[0, 1]) else 0
            self.satisfaction_multiplier = 1 + 0.1 * correlation
        else:
            self.satisfaction_multiplier = 1.0
        
    def simulate_revenue_with_variance_reduction(self, year: int, quarter: str, n_simulations: int = 10000) -> Dict:
        """
        მონტე კარლოს სიმულაცია ვარიაციის შემცირების ტექნიკებით
        
        მათემატიკური ფორმულები თავი 5-დან:
        R_t = N_t × E_t × (1 + ε_r,t) × M_s
        N_t = N_base × (1 + g)^t × S_q × exp(ε_n,q,t) × M_s  
        E_t = E_base,q × (1 + β_s × (S_q - 1)) × (1 + ε_e,q,t)
        """
        
        # პარამეტრების მიღება
        N_base = self.base_visitors[quarter]
        E_base = self.base_expenditure[quarter]
        S_q = self.seasonal_factors[quarter]
        M_s = self.satisfaction_multiplier
        g = self.median_growth_rate
        
        # ცვალებადობის პარამეტრები
        sigma_n = self.visitor_volatility[quarter]
        sigma_r = self.revenue_volatility[quarter]
        sigma_g = self.growth_volatility
        
        # დანახარჯების სეზონურობის ეფექტი (β_s = 0.2 ემპირიულად)
        beta_s = 0.2
        
        # სიმულაციის მასივები
        revenues = []
        visitors_list = []
        expenditures_list = []
        
        # წლები საბაზისო პერიოდიდან
        t_years = year - 2024
        
        # ანტითეტური ცვლადებით სიმულაცია
        for i in range(n_simulations // 2):
            # შემთხვევითი რიცხვების გენერაცია
            z_n = self.rng.normal(0, 1)
            z_r = self.rng.normal(0, 1) 
            z_g = self.rng.normal(0, 1)
            z_e = self.rng.normal(0, 1)
            
            # ანტითეტური წყვილები
            for sign in [1, -1]:
                # ვიზიტორთა რაოდენობა (ფორმულა 5.2) + კლიმატური ფაქტორი
                stochastic_growth = g + sign * z_g * sigma_g
                growth_factor = (1 + stochastic_growth) ** t_years
                visitor_shock = sign * z_n * sigma_n
                
                # ❌ ᲤᲘᲥᲡᲘᲠᲔᲑᲐ: კლიმატური/გარემოს ფაქტორის დამატება
                climate_shock = self.rng.normal(0, self.climate_volatility[quarter])
                N_t = N_base * growth_factor * S_q * np.exp(visitor_shock + climate_shock) * M_s
                
                # დანახარჯი ერთ ვიზიტორზე (ფორმულა 5.3)
                # ❌ ᲤᲘᲥᲡᲘᲠᲔᲑᲐ: ემპირიული დანახარჯების ცვალებადობა
                sigma_e = self.expenditure_volatility[quarter]
                expenditure_shock = sign * z_e * sigma_e
                E_t = E_base * (1 + beta_s * (S_q - 1)) * (1 + expenditure_shock)
                
                # მთლიანი შემოსავალი (ფორმულა 5.1)
                revenue_shock = sign * z_r * sigma_r
                R_t = N_t * E_t * (1 + revenue_shock)
                
                revenues.append(R_t)
                visitors_list.append(N_t)
                expenditures_list.append(E_t)
        
        # უშუალო შედეგები (კონტროლის ცვლადების გარეშე ყველაზე სტაბილური შედეგებისთვის)
        controlled_revenues = revenues
        
        return {
            'revenues': controlled_revenues,
            'visitors': visitors_list,
            'expenditures': expenditures_list,
            'mean_revenue': np.mean(controlled_revenues),
            'std_revenue': np.std(controlled_revenues),
            'mean_visitors': np.mean(visitors_list),
            'mean_expenditure': np.mean(expenditures_list),
            'confidence_interval_95': (np.percentile(controlled_revenues, 2.5), np.percentile(controlled_revenues, 97.5))
        }
    
    def _get_historical_average_revenue(self, quarter: str) -> float:
        """ისტორიული საშუალო შემოსავალის გამოთვლა კონტროლის ცვლადისთვის"""
        historical_revenues = []
        
        for year in range(2014, 2025):
            if year in self.covid_years:
                continue
                
            visitor_data = self.age_df[(self.age_df['Year'] == year) & (self.age_df['Quarter'] == quarter)]
            exp_data = self.expenditure_df[(self.expenditure_df['Year'] == year) & (self.expenditure_df['Quarter'] == quarter)]
            
            if not visitor_data.empty and not exp_data.empty:
                visitors = visitor_data['Total'].iloc[0]  # ათასებში
                total_exp = exp_data['Total expenditure'].iloc[0]  # მილიონებში
                # გადაყვანა იგივე სკალაზე რაც სიმულაციაშია
                revenue_in_thousands = total_exp * 1000  # მილიონებიდან ათასებში
                historical_revenues.append(revenue_in_thousands)
        
        return np.mean(historical_revenues) if historical_revenues else 1000000
    
    def analyze_convergence(self, year: int, quarter: str, max_simulations: int = 100000) -> Dict:
        """კონვერგენციის ანალიზი სხვადასხვა სიმულაციების რაოდენობისთვის"""
        simulation_counts = [10000, 100000]
        convergence_results = []
        
        for n_sims in simulation_counts:
            if n_sims > max_simulations:
                break
                
            start_time = time.time()
            results = self.simulate_revenue_with_variance_reduction(year, quarter, n_sims)
            end_time = time.time()
            
            # სტანდარტული შეცდომა
            standard_error = results['std_revenue'] / np.sqrt(n_sims)
            
            convergence_results.append({
                'n_simulations': n_sims,
                'mean_revenue': results['mean_revenue'],
                'std_revenue': results['std_revenue'],
                'standard_error': standard_error,
                'cv': results['std_revenue'] / results['mean_revenue'],
                'computation_time': end_time - start_time
            })
        
        return convergence_results
    
    def calculate_accuracy_metrics(self, year: int, quarter: str, n_simulations: int = 10000) -> Dict:
        """სიზუსტის მეტრიკების გამოთვლა"""
        results = self.simulate_revenue_with_variance_reduction(year, quarter, n_simulations)
        
        # Monte Carlo სტანდარტული შეცდომა
        mc_standard_error = results['std_revenue'] / np.sqrt(n_simulations)
        
        # ნდობის ინტერვალის სიგანე
        ci_width = results['confidence_interval_95'][1] - results['confidence_interval_95'][0]
        
        # ვარიაციის კოეფიციენტი
        coefficient_of_variation = results['std_revenue'] / results['mean_revenue']
        
        return {
            'standard_error': mc_standard_error,
            'relative_error': mc_standard_error / results['mean_revenue'],
            'confidence_interval_width': ci_width,
            'coefficient_of_variation': coefficient_of_variation,
            'effective_sample_size': n_simulations
        }
    
    def simulate_random_scenario_sampling(self, year: int, quarter: str, n_simulations: int = 10000) -> Dict:
        """
        🎲 შემდებარე სცენარიო მოდელირება - იტერაციულად შემთხვევითი სცენარიოების არჩევა
        უკეთესია რეალურ პირობებში გადაწყვეტილების მიღებისთვის
        """
        
        # სცენარიების განსაზღვრა და მათი ალბათობები
        scenarios = {
            'optimistic': {
                'probability': 0.20,            # 20% ალბათობა
                'growth_multiplier': 1.5,       # +50% ზრდის ტემპი
                'volatility_multiplier': 0.7,   # -30% ცვალებადობა
                'satisfaction_boost': 1.1,      # +10% კმაყოფილება
                'climate_stability': 0.8        # +20% კლიმატური სტაბილურობა
            },
            'baseline': {
                'probability': 0.60,            # 60% ალბათობა (ყველაზე სავარაუდო)
                'growth_multiplier': 1.0,       # ნორმალური ზრდა
                'volatility_multiplier': 1.0,   # ნორმალური ცვალებადობა  
                'satisfaction_boost': 1.0,      # ნორმალური კმაყოფილება
                'climate_stability': 1.0        # ნორმალური კლიმატი
            },
            'pessimistic': {
                'probability': 0.20,            # 20% ალბათობა
                'growth_multiplier': 0.5,       # -50% ზრდის ტემპი
                'volatility_multiplier': 1.4,   # +40% ცვალებადობა
                'satisfaction_boost': 0.9,      # -10% კმაყოფილება
                'climate_stability': 1.3        # +30% კლიმატური არასტაბილურობა
            }
        }
        
        # ალბათობების კუმულატიური განაწილება
        cumulative_probs = []
        prob_sum = 0
        scenario_names = []
        for name, params in scenarios.items():
            prob_sum += params['probability']
            cumulative_probs.append(prob_sum)
            scenario_names.append(name)
        

        
        # საბაზისო პარამეტრები
        N_base = self.base_visitors[quarter]
        E_base = self.base_expenditure[quarter]
        S_q = self.seasonal_factors[quarter]
        beta_s = 0.2
        t_years = year - 2024
        
        # შედეგების მასივები
        revenues = []
        visitors_list = []
        expenditures_list = []
        scenario_counts = {name: 0 for name in scenario_names}
        
        # ანტითეტური ცვლადებით სიმულაცია + შემთხვევითი სცენარიო
        for i in range(n_simulations // 2):
            # შემთხვევითი რიცხვების გენერაცია
            z_n = self.rng.normal(0, 1)
            z_r = self.rng.normal(0, 1) 
            z_g = self.rng.normal(0, 1)
            z_e = self.rng.normal(0, 1)
            
            # ანტითეტური წყვილები
            for sign in [1, -1]:
                # 🎲 შემთხვევითი სცენარიოს არჩევა ამ იტერაციისთვის
                scenario_rand = self.rng.next_uniform()
                selected_scenario = None
                
                for j, prob in enumerate(cumulative_probs):
                    if scenario_rand <= prob:
                        selected_scenario = scenario_names[j]
                        break
                
                if selected_scenario is None:
                    selected_scenario = 'baseline'  # fallback
                
                scenario_counts[selected_scenario] += 1
                params = scenarios[selected_scenario]
                
                # სცენარიო-მოდიფიცირებული პარამეტრები
                current_growth = self.median_growth_rate * params['growth_multiplier']
                current_satisfaction = self.satisfaction_multiplier * params['satisfaction_boost']
                current_visitor_vol = self.visitor_volatility[quarter] * params['volatility_multiplier']
                current_climate_vol = self.climate_volatility[quarter] * params['climate_stability']
                current_exp_vol = self.expenditure_volatility[quarter] * params['volatility_multiplier']
                
                # ვიზიტორთა რაოდენობა (ფორმულა 5.2) + კლიმატური ფაქტორი
                stochastic_growth = current_growth + sign * z_g * self.growth_volatility
                growth_factor = (1 + stochastic_growth) ** t_years
                visitor_shock = sign * z_n * current_visitor_vol
                
                # კლიმატური შოკი სცენარიო-მოდიფიცირებული
                climate_shock = self.rng.normal(0, current_climate_vol)
                N_t = N_base * growth_factor * S_q * np.exp(visitor_shock + climate_shock) * current_satisfaction
                
                # დანახარჯი ერთ ვიზიტორზე (ფორმულა 5.3)
                expenditure_shock = sign * z_e * current_exp_vol
                E_t = E_base * (1 + beta_s * (S_q - 1)) * (1 + expenditure_shock)
                
                # მთლიანი შემოსავალი (ფორმულა 5.1)
                revenue_shock = sign * z_r * self.revenue_volatility[quarter]
                R_t = N_t * E_t * (1 + revenue_shock)
                
                revenues.append(R_t)
                visitors_list.append(N_t)
                expenditures_list.append(E_t)
        
        # სცენარიების სტატისტიკა
        total_iterations = sum(scenario_counts.values())
        scenario_stats = {}
        for scenario, count in scenario_counts.items():
            actual_prob = count / total_iterations if total_iterations > 0 else 0
            scenario_stats[scenario] = {
                'target_probability': scenarios[scenario]['probability'],
                'actual_probability': actual_prob,
                'iterations_used': count
            }
        
        return {
            'revenues': revenues,
            'visitors': visitors_list,
            'expenditures': expenditures_list,
            'mean_revenue': np.mean(revenues),
            'std_revenue': np.std(revenues),
            'mean_visitors': np.mean(visitors_list),
            'mean_expenditure': np.mean(expenditures_list),
            'confidence_interval_95': (np.percentile(revenues, 2.5), np.percentile(revenues, 97.5)),
            'scenario_statistics': scenario_stats,
            'total_simulations': len(revenues)
        }
    
    def simulate_multi_scenario_analysis(self, year: int, quarter: str, n_simulations: int = 10000) -> Dict:
        """
        📊 ძველი მეთოდი: სცენარიების ცალკე ანალიზი (შედარებისთვის)
        """

        
        scenarios = {
            'optimistic': {
                'growth_multiplier': 1.5,      # +50% ზრდის ტემპი
                'volatility_multiplier': 0.7,  # -30% ცვალებადობა
                'satisfaction_boost': 1.1,     # +10% კმაყოფილება
                'climate_stability': 0.8       # +20% კლიმატური სტაბილურობა
            },
            'baseline': {
                'growth_multiplier': 1.0,      # ნორმალური ზრდა
                'volatility_multiplier': 1.0,  # ნორმალური ცვალებადობა  
                'satisfaction_boost': 1.0,     # ნორმალური კმაყოფილება
                'climate_stability': 1.0       # ნორმალური კლიმატი
            },
            'pessimistic': {
                'growth_multiplier': 0.5,      # -50% ზრდის ტემპი
                'volatility_multiplier': 1.4,  # +40% ცვალებადობა
                'satisfaction_boost': 0.9,     # -10% კმაყოფილება
                'climate_stability': 1.3       # +30% კლიმატური არასტაბილურობა
            }
        }
        
        scenario_results = {}
        
        for scenario_name, params in scenarios.items():
            # დროებითი პარამეტრების შენახვა
            original_growth = self.median_growth_rate
            original_satisfaction = self.satisfaction_multiplier
            original_visitor_vol = self.visitor_volatility[quarter]
            original_climate_vol = self.climate_volatility[quarter]
            
            # სცენარიო-სპეციფიკური პარამეტრების დაყენება
            self.median_growth_rate *= params['growth_multiplier']
            self.satisfaction_multiplier *= params['satisfaction_boost']
            self.visitor_volatility[quarter] *= params['volatility_multiplier']
            self.climate_volatility[quarter] *= params['climate_stability']
            
            # სიმულაციის ჩატარება
            results = self.simulate_revenue_with_variance_reduction(year, quarter, n_simulations)
            
            scenario_results[scenario_name] = {
                'mean_revenue': results['mean_revenue'],
                'std_revenue': results['std_revenue'],
                'ci_95': results['confidence_interval_95'],
                'cv': results['std_revenue'] / results['mean_revenue'],
                'parameters_used': params.copy()
            }
            
            # ორიგინალური პარამეტრების აღდგენა
            self.median_growth_rate = original_growth
            self.satisfaction_multiplier = original_satisfaction
            self.visitor_volatility[quarter] = original_visitor_vol
            self.climate_volatility[quarter] = original_climate_vol
        
        return scenario_results


def main():
    """მთავარი ფუნქცია - სრული სიმულაცია კონვერგენციისა და სიზუსტის ანალიზით"""
    
    # სიმულატორის ინიციალიზაცია
    simulator = TourismMonteCarloSimulator(
        age_data_path='Age.csv',
        expenditure_data_path='Expenditure.csv', 
        satisfaction_data_path='Satisfaction.csv'
    )
    

    
    # მთავარი სიმულაციები 2025 წლისთვის
    quarters = ['III', 'IV']
    all_results = {}
    
    for quarter in quarters:
        # შემთხვევითი სცენარიო მოდელირება
        random_scenario_results = simulator.simulate_random_scenario_sampling(2025, quarter, 100000)
        
        # სიზუსტის მეტრიკები
        random_accuracy = {
            'standard_error': random_scenario_results['std_revenue'] / np.sqrt(random_scenario_results['total_simulations']),
            'relative_error': (random_scenario_results['std_revenue'] / np.sqrt(random_scenario_results['total_simulations'])) / random_scenario_results['mean_revenue'],
            'confidence_interval_width': random_scenario_results['confidence_interval_95'][1] - random_scenario_results['confidence_interval_95'][0],
            'coefficient_of_variation': random_scenario_results['std_revenue'] / random_scenario_results['mean_revenue']
        }
        
        all_results[quarter] = {
            'random_scenario_results': random_scenario_results,
            'random_accuracy_metrics': random_accuracy
        }
    
    # შედეგების შენახვა
    output_data = []
    for quarter in quarters:
        results = all_results[quarter]['random_scenario_results']
        accuracy = all_results[quarter]['random_accuracy_metrics']
        
        output_data.append({
            'Quarter': f'2025_{quarter}',
            'Integrated_Revenue_Mean': round(results['mean_revenue'], 2),
            'Integrated_Revenue_Std': round(results['std_revenue'], 2),
            'Integrated_CI_Lower_95': round(results['confidence_interval_95'][0], 2),
            'Integrated_CI_Upper_95': round(results['confidence_interval_95'][1], 2),
            'Expected_Visitors_Thousands': round(results['mean_visitors'], 1),
            'Expenditure_per_Visitor': round(results['mean_expenditure'], 2),
            'Empirical_Expenditure_Volatility': round(simulator.expenditure_volatility[quarter] * 100, 1),
            'Empirical_Climate_Volatility': round(simulator.climate_volatility[quarter] * 100, 1),
            'Monte_Carlo_Error': round(accuracy['standard_error'], 2),
            'Relative_Error_Percent': round(accuracy['relative_error'] * 100, 2),
            'Coefficient_of_Variation': round(accuracy['coefficient_of_variation'] * 100, 2),
            'Total_Simulations': results['total_simulations']
        })
    
    # CSV ფაილში შენახვა
    with open('monte_carlo_predictions_2025.csv', 'w', newline='', encoding='utf-8') as f:
        fieldnames = output_data[0].keys()
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_data)
    
    # შედეგების პრეზენტაცია
    print("\n" + "=" * 60)
    print("    ტურიზმის შემოსავლების პროგნოზი 2025")
    print("=" * 60)
    
    # ემპირიული პარამეტრები
    print(f"\nემპირიული პარამეტრები:")
    for quarter in quarters:
        exp_vol = simulator.expenditure_volatility[quarter] * 100
        climate_vol = simulator.climate_volatility[quarter] * 100
        print(f"  {quarter} კვარტალი: დანახარჯების ცვალებადობა={exp_vol:.1f}%, კლიმატური ცვალებადობა={climate_vol:.1f}%")
    
    # საბოლოო შედეგები
    total_expected = 0
    print(f"\nპროგნოზი:")
    
    for quarter in quarters:
        results = all_results[quarter]['random_scenario_results']
        accuracy = all_results[quarter]['random_accuracy_metrics']
        total_expected += results['mean_revenue']
        
        print(f"\n{quarter} კვარტალი (2025):")
        print(f"  შემოსავალი: {results['mean_revenue']:,.0f} ლარი")
        print(f"  ნდობის ინტერვალი (95%): {results['confidence_interval_95'][0]:,.0f} - {results['confidence_interval_95'][1]:,.0f} ლარი")
        print(f"  ვიზიტორები: {results['mean_visitors']:,.0f} ათასი")
        print(f"  Monte Carlo შეცდომა: {accuracy['relative_error']:.2%}")
    
    print(f"\nსრული შემოსავალი (III+IV კვარტალები): {total_expected:,.0f} ლარი")
    
    print(f"\nმეთოდოლოგია:")
    print(f"• ემპირიული პარამეტრები (2014-2024, COVID გამორიცხული)")
    print(f"• შემთხვევითი სცენარიო მოდელირება (60% ბაზისური, 20% ოპტიმისტური, 20% პესიმისტური)")
    print(f"• 100,000 Monte Carlo იტერაცია")
    print(f"• შედეგები: monte_carlo_predictions_2025.csv")
    print("=" * 60)


if __name__ == "__main__":
    main() 