# %%
# PART F — Prep Data for Inequality Counterfactuals

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

RAW_DIR = r"E:/users_ra_local/fang_ray/Lifecycle-portfolio-allocation/raw"
OUT_DIR = r"E:/users_ra_local/fang_ray/Lifecycle-portfolio-allocation/outputs"
SCF_DIR = r"E:/users_ra_local/fang_ray/Lifecycle-portfolio-allocation/SCF Data/processed_data Goutham Gopalakrishna"

INCOME_GROUPS = fr"{OUT_DIR}/cps_income_groups_1968_2024.csv"
ADF_RESULTS = fr"{OUT_DIR}/cps_bcg_adf_results.csv"
BETA_RESULTS = fr"{OUT_DIR}/labor_income_betas.csv"
SCF_MERGED = fr"{OUT_DIR}/scf_cps_merged.csv"

# %%
# 1. Load and prepare CPS income data by group
income_data = pd.read_csv(INCOME_GROUPS)
adf_data = pd.read_csv(ADF_RESULTS)
beta_data = pd.read_csv(BETA_RESULTS)

adf_subset = adf_data[adf_data['L'] == 2][['incgroup', 'a3_coef']].copy()
risk_params = adf_subset.merge(beta_data[['incgroup', 'beta']], on='incgroup', how='inner')
risk_params.columns = ['Income_Group', 'a3_coef', 'beta']

print("Risk parameters by income group:")
print(risk_params)

# %%
# 2. Calculate mean income and income volatility by group
income_stats = income_data.groupby('incgroup')['ln_RLABINC'].agg([
    'mean', 'std', 'count'
]).reset_index()
income_stats.columns = ['Income_Group', 'mean_ln_income', 'std_ln_income', 'n_years']

group_params = income_stats.merge(risk_params, on='Income_Group', how='inner')

print("\nIncome statistics by group:")
print(group_params.round(4))

# %%
# 3. Construct synthetic income paths for ages 25-65
np.random.seed(42)  

ages = np.arange(25, 66)  
n_paths = 1000 
groups = group_params['Income_Group'].unique()

def simulate_income_path(mean_ln_inc, a3_coef, beta, std_ln_inc, ages, n_sims=1000):
    """
    ln(y_t) = μ + ρ * ln(y_{t-1}) + ε_t
    ρ = 1 + a3 (persistence parameter)
    """
    n_ages = len(ages)
    income_paths = np.zeros((n_sims, n_ages))
    
    rho = 1 + a3_coef 
    mu = mean_ln_inc * (1 - rho) 
    sigma = std_ln_inc * np.sqrt(1 - rho**2) 
    income_paths[:, 0] = np.random.normal(mean_ln_inc, std_ln_inc, n_sims)
    
    for t in range(1, n_ages):
        innovations = np.random.normal(0, sigma, n_sims)
        income_paths[:, t] = mu + rho * income_paths[:, t-1] + innovations
    
    return income_paths

all_income_paths = []

for _, group_data in group_params.iterrows():
    group = group_data['Income_Group']
    print(f"\nSimulating income paths for {group}...")
    
    paths = simulate_income_path(
        mean_ln_inc=group_data['mean_ln_income'],
        a3_coef=group_data['a3_coef'],
        beta=group_data['beta'],
        std_ln_inc=group_data['std_ln_income'],
        ages=ages,
        n_sims=n_paths
    )
    
    for sim_id in range(n_paths):
        for age_idx, age in enumerate(ages):
            all_income_paths.append({
                'Income_Group': group,
                'simulation_id': sim_id,
                'age': age,
                'ln_income': paths[sim_id, age_idx],
                'income': np.exp(paths[sim_id, age_idx])
            })

income_paths_df = pd.DataFrame(all_income_paths)

# %%
# 4. Load SCF data and construct risky share profiles by age and group
scf_data = pd.read_csv(SCF_MERGED)

scf_data['age_bin'] = pd.cut(scf_data['age'], 
                            bins=[0, 30, 35, 40, 45, 50, 55, 60, 65, 100],
                            labels=['25-30', '30-35', '35-40', '40-45', 
                                   '45-50', '50-55', '55-60', '60-65', '65+'])

portfolio_profiles = scf_data[scf_data['own_stock'] == 1].groupby(['Income_Group', 'age_bin']).agg({
    'risky_share': ['mean', 'std', 'count'],
    'own_stock': 'mean'  
}).reset_index()

portfolio_profiles.columns = ['Income_Group', 'age_bin', 'mean_risky_share', 'std_risky_share', 'n_obs', 'participation_rate']

participation_rates = scf_data.groupby(['Income_Group', 'age_bin'])['own_stock'].agg(['mean', 'count']).reset_index()
participation_rates.columns = ['Income_Group', 'age_bin', 'participation_rate', 'total_obs']

print("\nPortfolio profiles by age and income group:")
print(portfolio_profiles.head(10))

# %%
# 5. Construct synthetic portfolio paths
def get_portfolio_allocation(income_group, age):
    if age <= 30:
        age_bin = '25-30'
    elif age <= 35:
        age_bin = '30-35'
    elif age <= 40:
        age_bin = '35-40' 
    elif age <= 45:
        age_bin = '40-45'
    elif age <= 50:
        age_bin = '45-50'
    elif age <= 55:
        age_bin = '50-55'
    elif age <= 60:
        age_bin = '55-60'
    elif age <= 65:
        age_bin = '60-65'
    else:
        age_bin = '65+'

    portfolio_data = portfolio_profiles[
        (portfolio_profiles['Income_Group'] == income_group) & 
        (portfolio_profiles['age_bin'] == age_bin)
    ]
    
    participation_data = participation_rates[
        (participation_rates['Income_Group'] == income_group) & 
        (participation_rates['age_bin'] == age_bin)
    ]
    
    if len(portfolio_data) == 0 or len(participation_data) == 0:
        portfolio_data = portfolio_profiles[portfolio_profiles['Income_Group'] == income_group]
        participation_data = participation_rates[participation_rates['Income_Group'] == income_group]
    
    mean_risky_share = portfolio_data['mean_risky_share'].mean()
    participation_rate = participation_data['participation_rate'].mean()
        
    return participation_rate, mean_risky_share

all_portfolio_paths = []

for group in groups:    
    for sim_id in range(n_paths):
        for age in ages:
            participation_rate, risky_share = get_portfolio_allocation(group, age)
            
            participates = np.random.random() < participation_rate
            
            actual_risky_share = risky_share if participates else 0.0
            
            all_portfolio_paths.append({
                'Income_Group': group,
                'simulation_id': sim_id,
                'age': age,
                'participation_rate': participation_rate,
                'participates': participates,
                'risky_share': actual_risky_share,
                'risky_share_conditional': risky_share
            })

portfolio_paths_df = pd.DataFrame(all_portfolio_paths)

# %%
income_paths_df.to_csv(fr"{OUT_DIR}/income_paths_g.csv", index=False)
portfolio_paths_df.to_csv(fr"{OUT_DIR}/portfolio_paths_g.csv", index=False)
