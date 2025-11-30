# %%
import pandas as pd
import numpy as np 
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pyarrow

RAW_DIR      = r"E:/users_ra_local/fang_ray/Lifecycle-portfolio-allocation/raw"
OUT_DIR      = r"E:/users_ra_local/fang_ray/Lifecycle-portfolio-allocation/outputs"
SCF_DIR      = r"E:/users_ra_local/fang_ray/Lifecycle-portfolio-allocation/SCF Data/processed_data Goutham Gopalakrishna"

SCF_MERGED   = fr"{SCF_DIR}/all_merged.dta"
ADF_RESULTS  = fr"{OUT_DIR}/cps_bcg_adf_results.csv"
BETA_RESULTS = fr"{OUT_DIR}/labor_income_betas.csv"
GROUPS_CSV   = fr"{OUT_DIR}/cps_income_groups_1968_2024.csv"
# %%
# 1. Load SCF cleaned dataset and calculate proper risky share
scf_data = pd.read_stata(SCF_MERGED)
financial_assets = ['checking', 'saving', 'mma', 'call', 'cds', 'obmutf', 'nmmf', 'stocks', 'bond', 'equity']
scf_data['networth'] = scf_data[financial_assets].sum(axis=1)
scf_data['total_financial_assets'] = scf_data[financial_assets].sum(axis=1)
scf_data['risky_share_proper'] = scf_data['risky_asset'] / scf_data['total_financial_assets'].clip(lower=1)
scf_data['risky_share_proper'] = scf_data['risky_share_proper'].clip(0, 1)

scf_clean = scf_data[['age', 'income', 'networth', 'dum_stock', 'risky_share_proper', 'wgt', 'panel']].copy()
scf_clean.rename(columns={
    'income': 'labor_income',
    'dum_stock': 'own_stock', 
    'risky_share_proper': 'risky_share',
    'wgt': 'weights'
}, inplace=True)

scf_clean['YEAR'] = scf_clean['panel']
# %%
# 2. Assign SCF households to income percentiles 1-8
cut_probs = np.array([0.20, 0.40, 0.60, 0.80, 0.90, 0.95, 0.99])

def weighted_percentiles(values, weights, probs):
    values = np.asarray(values)
    weights = np.asarray(weights)
    
    mask = ~(np.isnan(values) | np.isnan(weights))
    values = values[mask]
    weights = weights[mask]
    
    if len(values) == 0:
        return np.full(len(probs), np.nan)
    
    sorter = np.argsort(values)
    v = values[sorter]
    w = weights[sorter]
    cum_w = np.cumsum(w)
    total_w = cum_w[-1]
    pct = cum_w / total_w
    return np.interp(probs, pct, v)

def assign_income_groups(df):
    df_with_groups = df.copy()
    df_with_groups['Income_Group'] = None
    
    for year in df['YEAR'].unique():
        year_data = df[df['YEAR'] == year].copy()
        year_data = year_data[year_data['labor_income'] > 0]
        
        if len(year_data) == 0:
            continue
            
        cuts = weighted_percentiles(
            year_data['labor_income'].values, 
            year_data['weights'].values, 
            cut_probs
        )
        
        bins = np.concatenate(([-np.inf], cuts, [np.inf]))
        labels = ["Group1", "Group2", "Group3", "Group4", "Group5", "Group6", "Group7", "Group8"]
        
        year_data['Income_Group'] = pd.cut(
            year_data['labor_income'],
            bins=bins,
            labels=labels,
            right=True,
            include_lowest=True
        )
        
        df_with_groups.loc[df_with_groups['YEAR'] == year, 'Income_Group'] = year_data['Income_Group']
    
    return df_with_groups

scf_with_groups = assign_income_groups(scf_clean)
scf_with_groups = scf_with_groups.dropna(subset=['Income_Group'])

# %%
# 3. Merge CPS group-level risk measures (β_g, a3_g, INDEX_g) into SCF by group
adf_results = pd.read_csv(ADF_RESULTS)
adf_results = adf_results[adf_results['L'] == 2][['incgroup', 'a3_coef']].copy()
adf_results.rename(columns={'incgroup': 'Income_Group', 'a3_coef': 'a3_g'}, inplace=True)

beta_results = pd.read_csv(BETA_RESULTS)
beta_results = beta_results[['incgroup', 'beta']].copy()
beta_results.rename(columns={'incgroup': 'Income_Group', 'beta': 'beta_g'}, inplace=True)

risk_measures = adf_results.merge(beta_results, on='Income_Group', how='outer')

risk_measures['INDEX_g'] = np.abs(risk_measures['beta_g']) + np.abs(risk_measures['a3_g'])

scf_cps_merged = scf_with_groups.merge(
    risk_measures[['Income_Group', 'a3_g', 'beta_g', 'INDEX_g']], 
    on='Income_Group', 
    how='left'
)

# %%
# 4. Save merged dataset
scf_cps_merged.to_csv(fr"{OUT_DIR}/scf_cps_merged.csv", index=False)
scf_cps_merged.to_stata(fr"{OUT_DIR}/scf_cps_merged.dta", write_index=False)

# %%
# PART E — Portfolio Regressions
reg_data = scf_cps_merged.copy()
reg_data['ln_income'] = np.log(reg_data['labor_income'].clip(lower=1))
reg_data['ln_networth'] = np.log(reg_data['networth'].clip(lower=1))

# Year fixed effects and controls
reg_data['year_fe'] = reg_data['YEAR'].astype('category')
controls = ['age', 'ln_income', 'ln_networth']
control_formula = ' + '.join(controls) + ' + C(year_fe)'

print(f"Regression sample size: {len(reg_data)}")
print(f"Owners sample size: {(reg_data['own_stock'] == 1).sum()}")

# 1. own_stock_i = α + θ * β_g(i) + controls + year FE
participation_formula = f"own_stock ~ beta_g + {control_formula}"
participation_model = smf.ols(participation_formula, data=reg_data).fit(cov_type='cluster', cov_kwds={'groups': reg_data['Income_Group']})

# 2. risky_share_i = α + θ * β_g(i) + controls + year FE (conditional on ownership)
owners_data = reg_data[reg_data['own_stock'] == 1].copy()
print(f"Owners risky_share range: [{owners_data['risky_share'].min():.3f}, {owners_data['risky_share'].max():.3f}]")

risky_share_formula = f"risky_share ~ beta_g + {control_formula}"
risky_share_model = smf.ols(risky_share_formula, data=owners_data).fit(cov_type='cluster', cov_kwds={'groups': owners_data['Income_Group']})

# Alternative specifications
participation_a3_formula = f"own_stock ~ a3_g + {control_formula}"
participation_a3_model = smf.ols(participation_a3_formula, data=reg_data).fit(cov_type='cluster', cov_kwds={'groups': reg_data['Income_Group']})

participation_index_formula = f"own_stock ~ INDEX_g + {control_formula}"
participation_index_model = smf.ols(participation_index_formula, data=reg_data).fit(cov_type='cluster', cov_kwds={'groups': reg_data['Income_Group']})

risky_share_a3_formula = f"risky_share ~ a3_g + {control_formula}"
risky_share_a3_model = smf.ols(risky_share_a3_formula, data=owners_data).fit(cov_type='cluster', cov_kwds={'groups': owners_data['Income_Group']})

risky_share_index_formula = f"risky_share ~ INDEX_g + {control_formula}"
risky_share_index_model = smf.ols(risky_share_index_formula, data=owners_data).fit(cov_type='cluster', cov_kwds={'groups': owners_data['Income_Group']})

# %%
# 4. Outputs
def extract_results(model, risk_var):
    return {
        'risk_variable': risk_var,
        'coef': model.params[risk_var],
        'se': model.bse[risk_var],
        'tstat': model.tvalues[risk_var],
        'pvalue': model.pvalues[risk_var],
        'n_obs': int(model.nobs),
        'r_squared': model.rsquared
    }

participation_results = [
    extract_results(participation_model, 'beta_g'),
    extract_results(participation_a3_model, 'a3_g'),
    extract_results(participation_index_model, 'INDEX_g')
]

risky_share_results = [
    extract_results(risky_share_model, 'beta_g'),
    extract_results(risky_share_a3_model, 'a3_g'),
    extract_results(risky_share_index_model, 'INDEX_g')
]

participation_df = pd.DataFrame(participation_results)
participation_df['regression_type'] = 'Participation'

risky_share_df = pd.DataFrame(risky_share_results)
risky_share_df['regression_type'] = 'Risky_Share'

all_results = pd.concat([participation_df, risky_share_df], ignore_index=True)

all_results.round(4).to_csv(fr"{OUT_DIR}/portfolio_regression_results.csv", index=False)

def create_regression_table(participation_results, risky_share_results):
    risk_labels = [r'$\beta_g$ (Labor Income Beta)', 
                   r'$a3_g$ (ADF Coefficient)', 
                   r'INDEX$_g$ (Combined Risk)']
    
    table_rows = []
    
    for i, (part_res, risky_res) in enumerate(zip(participation_results, risky_share_results)):
        part_coef_str = f"{part_res['coef']:.3f}"
        part_se_str = f"({part_res['se']:.3f})"
        
        risky_coef_str = f"{risky_res['coef']:.3f}"
        risky_se_str = f"({risky_res['se']:.3f})"
        
        if part_res['pvalue'] < 0.01:
            part_coef_str += "***"
        elif part_res['pvalue'] < 0.05:
            part_coef_str += "**"
        elif part_res['pvalue'] < 0.10:
            part_coef_str += "*"
            
        if risky_res['pvalue'] < 0.01:
            risky_coef_str += "***"
        elif risky_res['pvalue'] < 0.05:
            risky_coef_str += "**"
        elif risky_res['pvalue'] < 0.10:
            risky_coef_str += "*"
        
        table_rows.append({
            'Variable': risk_labels[i],
            'Participation_Coef': part_coef_str,
            'Participation_SE': part_se_str,
            'Risky_Share_Coef': risky_coef_str,
            'Risky_Share_SE': risky_se_str
        })
    
    return table_rows

table_data = create_regression_table(participation_results, risky_share_results)

latex_table = r"""
\begin{table}[htbp]
\centering
\caption{Portfolio Choice Regressions: Risk Measures and Investment Decisions}
\label{tab:portfolio_regressions}
\begin{tabular}{lcccc}
\toprule
 & \multicolumn{2}{c}{Stock Market Participation} & \multicolumn{2}{c}{Risky Asset Share} \\
\cmidrule(lr){2-3} \cmidrule(lr){4-5}
Risk Measure & Coefficient & Std. Error & Coefficient & Std. Error \\
\midrule
"""

for row in table_data:
    latex_table += f"{row['Variable']} & {row['Participation_Coef']} & {row['Participation_SE']} & {row['Risky_Share_Coef']} & {row['Risky_Share_SE']} \\\\\n"

part_n = participation_results[0]['n_obs']
risky_n = risky_share_results[0]['n_obs'] 

latex_table += r"""
\midrule
Observations & \multicolumn{2}{c}{""" + f"{part_n:,}" + r"""} & \multicolumn{2}{c}{""" + f"{risky_n:,}" + r"""} \\
Controls & \multicolumn{2}{c}{Yes} & \multicolumn{2}{c}{Yes} \\
Year FE & \multicolumn{2}{c}{Yes} & \multicolumn{2}{c}{Yes} \\
\bottomrule
\end{tabular}
\end{table}
"""

with open(fr"{OUT_DIR}/portfolio_regression_table.tex", 'w') as f:
    f.write(latex_table)

# %%
risk_vars = ['beta_g', 'a3_g', 'INDEX_g']

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Participation regression coefficients
part_coefs = [participation_results[i]['coef'] for i in range(3)]
part_se = [participation_results[i]['se'] for i in range(3)]

ax1.errorbar(range(3), part_coefs, yerr=1.96*np.array(part_se), 
             marker='o', capsize=5, capthick=2, markersize=8)
ax1.axhline(0, color='black', linestyle='--', alpha=0.5)
ax1.set_xticks(range(3))
ax1.set_xticklabels([r'$\beta_g$', r'$a3_g$', 'INDEX_g'])
ax1.set_ylabel('Coefficient')
ax1.set_title('Stock Market Participation\n(Hedging < 0, Double-down > 0)')
ax1.grid(True, alpha=0.3)

for i, (coef, se) in enumerate(zip(part_coefs, part_se)):
    if abs(coef) > 1.96 * se:  # Significant at 5%
        ax1.text(i, coef + 1.96*se + 0.05*max(part_coefs), '*', 
                ha='center', va='bottom', fontsize=14, fontweight='bold')

risky_coefs = [risky_share_results[i]['coef'] for i in range(3)]
risky_se = [risky_share_results[i]['se'] for i in range(3)]

max_coef = max([abs(c) for c in risky_coefs])
max_se = max(risky_se)

ax2.errorbar(range(3), risky_coefs, yerr=1.96*np.array(risky_se), 
             marker='s', capsize=5, capthick=2, markersize=8, color='orange')
ax2.axhline(0, color='black', linestyle='--', alpha=0.5)
ax2.set_xticks(range(3))
ax2.set_xticklabels([r'$\beta_g$', r'$a3_g$', 'INDEX_g'])
ax2.set_ylabel('Coefficient')
ax2.set_title('Risky Asset Share (Conditional on Participation)\n(Hedging < 0, Double-down > 0)')
ax2.grid(True, alpha=0.3)

for i, (coef, se) in enumerate(zip(risky_coefs, risky_se)):
    if abs(coef) > 1.96 * se: 
        ax2.text(i, coef + 1.96*se + 0.05*max(risky_coefs), '*', 
                ha='center', va='bottom', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig(fr"{OUT_DIR}/portfolio_regression_coefficients.png", dpi=300, bbox_inches='tight')
plt.show()
