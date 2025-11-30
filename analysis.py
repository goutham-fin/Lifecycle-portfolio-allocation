# %%
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf

# %%
RAW_DIR      = r"E:/users_ra_local/fang_ray/Lifecycle-portfolio-allocation/raw"
OUT_DIR      = r"E:/users_ra_local/fang_ray/Lifecycle-portfolio-allocation/outputs"

CPI_CSV      = fr"{RAW_DIR}/cpi_u.csv"
CRSP_CSV     = fr"{RAW_DIR}/crsp_value_weighted.csv"

CLEAN_PARQUET = fr"{OUT_DIR}/cps_00001_clean.parquet"
GROUPS_CSV    = fr"{OUT_DIR}/cps_income_groups_1968_2024.csv"
GROUPS_DTA    = fr"{OUT_DIR}/cps_income_groups_1968_2024.dta"

# %%
# Load market data
cpi = pd.read_csv(CPI_CSV)
crsp = pd.read_csv(CRSP_CSV)

# Prepare CPI data for later use
cpi["Year"] = cpi["Year"].astype(int)
annual_cpi = (
    cpi.groupby("Year")["Value"]
       .mean()
       .reset_index()
       .rename(columns={"Value": "CPI_U"})
)

BASE_YEAR = 2000
base_cpi = annual_cpi.loc[annual_cpi["Year"] == BASE_YEAR, "CPI_U"].iloc[0]
annual_cpi["CPI_REL"] = annual_cpi["CPI_U"] / base_cpi

# %%
# Load the pre-computed income group data from generate_8income_groups.py
print("Loading pre-computed income group data...")
group_means = pd.read_csv(GROUPS_CSV)
print(f"Available income groups: {sorted(group_means['incgroup'].unique())}")
print(f"Year range: {group_means['YEAR'].min()} - {group_means['YEAR'].max()}")

# Save to Stata format
group_means.to_stata(GROUPS_DTA, write_index=False)

# %%
crsp["DATE"] = pd.to_datetime(crsp["Year"])
crsp["YEAR"] = crsp["DATE"].dt.year

crsp["DIV_YIELD"] = (1 + crsp["VWRETD"]) / (1 + crsp["VWRETX"]) - 1

annual_div = (
    crsp.groupby("YEAR")["DIV_YIELD"]
         .apply(lambda x: (1 + x).prod() - 1)
         .reset_index()
         .rename(columns={"DIV_YIELD": "ANNUAL_DIV_RETURN"})
)

annual = annual_div.merge(
    annual_cpi[["Year", "CPI_REL"]],
    left_on="YEAR",
    right_on="Year",
    how="inner"
).drop(columns=["Year"])

annual = annual.sort_values("YEAR")

annual["DIV_INDEX_NOM"]  = (1 + annual["ANNUAL_DIV_RETURN"]).cumprod()
annual["DIV_INDEX_REAL"] = annual["DIV_INDEX_NOM"] / annual["CPI_REL"]
annual["real_div_t"]      = annual["DIV_INDEX_REAL"]
annual["log_div_t"]       = np.log(annual["real_div_t"])
# %%
# Merge group means with dividend data
div_panel = annual[["YEAR", "log_div_t"]].copy()

panel = group_means.merge(div_panel, on="YEAR", how="inner")
panel["y_gt"] = panel["ln_RLABINC"] - panel["log_div_t"]

panel = panel.sort_values(["incgroup", "YEAR"]).reset_index(drop=True)
panel["dy_gt"]   = panel.groupby("incgroup")["y_gt"].diff()
panel["t_index"] = panel["YEAR"] - panel["YEAR"].min()

Y_PANEL_CSV = fr"{OUT_DIR}/cps_income_dividend_panel.csv"
Y_PANEL_DTA = fr"{OUT_DIR}/cps_income_dividend_panel.dta"

panel.to_csv(Y_PANEL_CSV, index=False)

print("\nMerged panel head:")
print(panel.head())
# %%
# ADF Regression
def run_bcg_adf(panel, L):
    """
    Run Δy_t = a1 + a2 t + a3 y_{t-1} + Σ_{ℓ=1..L} Φ_ℓ Δy_{t-ℓ} + ε_t
    separately by incgroup, return DataFrame of a3, tstat, pvalue, etc.
    """
    df = panel.sort_values(["incgroup", "YEAR"]).copy()

    df["y_gt_lag"] = df.groupby("incgroup")["y_gt"].shift(1)
    df["dy_gt"]    = df.groupby("incgroup")["y_gt"].diff()

    for ell in range(1, L + 1):
        df[f"dy_lag{ell}"] = df.groupby("incgroup")["dy_gt"].shift(ell)

    cols_needed = ["dy_gt", "y_gt_lag"] + [f"dy_lag{ell}" for ell in range(1, L + 1)]
    df_reg = df.dropna(subset=cols_needed).copy()

    results = []

    for g in df_reg["incgroup"].unique():
        sub = df_reg[df_reg["incgroup"] == g]

        Y = sub["dy_gt"]
        X_dict = {
            "t_index": sub["t_index"],
            "y_lag":   sub["y_gt_lag"],
        }
        for ell in range(1, L + 1):
            X_dict[f"dy_lag{ell}"] = sub[f"dy_lag{ell}"]

        X = pd.DataFrame(X_dict)
        X = sm.add_constant(X)

        model = sm.OLS(Y, X).fit()

        results.append({
            "incgroup": g,
            "L":        L,
            "a3_coef":   model.params["y_lag"],
            "a3_tstat":  model.tvalues["y_lag"],
            "a3_pvalue": model.pvalues["y_lag"],
            "n_obs":     int(model.nobs),
            "r2":        model.rsquared,
        })

    return pd.DataFrame(results)

panel = panel.sort_values(["incgroup", "YEAR"]).reset_index(drop=True)

bcg_df = run_bcg_adf(panel, L=2)

BCG_CSV = fr"{OUT_DIR}/cps_bcg_adf_results.csv"
BCG_DTA = fr"{OUT_DIR}/cps_bcg_adf_results.dta"

bcg_df.to_csv(BCG_CSV, index=False)
bcg_df.to_stata(BCG_DTA, write_index=False)
# %%
# Table A
table_A = bcg_df[[
    "incgroup", "a3_coef", "a3_tstat", "a3_pvalue"
]].copy()

table_A["reject_5pct"] = (table_A["a3_pvalue"] < 0.05).astype(int)

table_A = table_A.rename(columns={
    "incgroup": "Group",
    "a3_coef": "$a_3$",
    "a3_tstat": "t-stat($a_3$)",
    "a3_pvalue": "p-value"
})

latex_path_A = fr"{OUT_DIR}/table_A_adf.tex"

table_A.to_latex(
    latex_path_A,
    index=False,
    float_format="%.3f",
    escape=False,
    column_format="c c c c c",
    caption="Baseline ADF Regression Results (Lag Length = 2, CRSP Dividends)",
    label="tab:adf_baseline"
)

# Table B

table_B_L1 = run_bcg_adf(panel, L=1)
table_B_L2 = run_bcg_adf(panel, L=2)
table_B_L3 = run_bcg_adf(panel, L=3)

table_B = pd.concat([table_B_L1, table_B_L2, table_B_L3], ignore_index=True)

table_B_export = table_B[[
    "incgroup", "L", "a3_coef", "a3_tstat", "a3_pvalue", "r2"
]].copy()

table_B_export = table_B_export.rename(columns={
    "incgroup": "Group",
    "L": "Lag $L$",
    "a3_coef": "$a_3$",
    "a3_tstat": "t-stat($a_3$)",
    "a3_pvalue": "p-value",
    "r2": "$R^2$"
})

latex_path_B = fr"{OUT_DIR}/table_B_adf_lag_sensitivity.tex"

table_B_export.to_latex(
    latex_path_B,
    index=False,
    float_format="%.3f",
    escape=False,
    column_format="c c c c c c",
    caption="Sensitivity of ADF Results to Lag Length $L$ (CRSP Dividends)",
    label="tab:adf_lag_sensitivity"
)
# %%
# Key Plots and Analysis
print("Generating key plots...")

# 1. Income trends over time
plt.figure(figsize=(12, 6))
group_order = ["Group1", "Group2", "Group3", "Group4", "Group5", "Group6", "Group7", "Group8"]
for g in group_order:
    subset = group_means[group_means["incgroup"] == g]
    plt.plot(subset["YEAR"], subset["ln_RLABINC"], label=g)

plt.title("Weighted Mean log(RLABINC) by Income Group, 1976–2024")
plt.xlabel("Year")
plt.ylabel("ln(RLABINC)")
plt.legend(title="Income Group")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 2. Labor-Dividend ratio by group
groups = sorted(panel["incgroup"].unique())
plt.figure(figsize=(12, 8))
for g in groups:
    subset = panel[panel["incgroup"] == g]
    plt.plot(subset["YEAR"], subset["y_gt"], label=f"{g}", linewidth=1.6)

plt.title("Labor–Dividend Log Ratio $y_{g,t}$ by Income Group")
plt.xlabel("Year")
plt.ylabel("$y_{g,t} = \\ln(RLABINC_g) - \\ln(DIV_t)$")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

# %%
# Final panel output
panel_clean = panel.sort_values(["incgroup", "YEAR"]).copy()

panel_clean["Dy_t"] = panel_clean.groupby("incgroup")["y_gt"].diff()
panel_clean["Dy_t_lag1"] = panel_clean.groupby("incgroup")["Dy_t"].shift(1)
panel_clean["Dy_t_lag2"] = panel_clean.groupby("incgroup")["Dy_t"].shift(2)

panel_clean = panel_clean.rename(columns={
    "y_gt": "y_t",
    "ln_RLABINC": "ln_RLABINC_t",
    "log_div_t": "log_div_t"
})

final_table = panel_clean[[
    "YEAR",
    "incgroup",
    "y_t",
    "Dy_t",
    "Dy_t_lag1",
    "Dy_t_lag2",
    "log_div_t",
    "ln_RLABINC_t"
]].dropna()

final_path = fr"{OUT_DIR}/cps_adf_final_panel.csv"
final_table.to_csv(final_path, index=False)

final_table.head()
# %%
# Panel ECM robustness
def run_panel_ecm(panel, L=2, top_label="Top1", hac_lags=None):
    """
    Pooled ECM with group fixed effects.
    """
    df = panel.sort_values(["incgroup", "YEAR"]).copy()

    df["y_gt_lag"] = df.groupby("incgroup")["y_gt"].shift(1)
    df["dy_gt"]    = df.groupby("incgroup")["y_gt"].diff()

    for j in range(1, L+1):
        df[f"dy_lag{j}"] = df.groupby("incgroup")["dy_gt"].shift(j)

    df["is_top"] = (df["incgroup"] == top_label).astype(int)

    df["y_gt_lag_top"] = df["y_gt_lag"] * df["is_top"]

    needed = ["dy_gt", "y_gt_lag", "t_index", "y_gt_lag_top"] + [f"dy_lag{j}" for j in range(1, L+1)]
    df_reg = df.dropna(subset=needed).copy()

    lag_terms = " + ".join([f"dy_lag{j}" for j in range(1, L+1)])
    formula = f"dy_gt ~ C(incgroup) + t_index + y_gt_lag + y_gt_lag_top"
    if L > 0:
        formula += " + " + lag_terms

    if hac_lags is None:
        hac_lags = L 

    res = smf.ols(formula, data=df_reg).fit(
        cov_type="HAC", cov_kwds={"maxlags": hac_lags}
    )

    wald = res.wald_test("y_gt_lag_top = 0")

    return res, wald

L_list = [1, 2, 3]

ecm_results = []

for L in L_list:
    res, wald = run_panel_ecm(panel, L=L, top_label="Group8", hac_lags=L)

    print("\n" + "="*70)
    print(f"PANEL ECM L={L}")
    print(res.summary())

    print("\nWald test H0: Top group has same a3 (y_gt_lag_top = 0)")
    print(wald)

    ecm_results.append({
        "L": L,
        "a3_common": res.params.get("y_gt_lag", np.nan),
        "a3_common_t": res.tvalues.get("y_gt_lag", np.nan),
        "a3_top_extra": res.params.get("y_gt_lag_top", np.nan),
        "a3_top_extra_t": res.tvalues.get("y_gt_lag_top", np.nan),
        "wald_stat": float(wald.statistic),
        "wald_pvalue": float(wald.pvalue),
        "n_obs": int(res.nobs)
    })

ecm_table = pd.DataFrame(ecm_results)
print("\nECM robustness summary:")
print(ecm_table)

ECM_CSV = fr"{OUT_DIR}/panel_ecm_robustness.csv"
ecm_table.to_csv(ECM_CSV, index=False)
print(f"\nSaved: {ECM_CSV}")
# %%
# Part B — Construct Short-Run Labor-Income Betas by Income Group
crsp_annual = (
    crsp.groupby("YEAR")
        .apply(lambda x: (1 + x["VWRETD"]).prod() - 1)
        .reset_index(name="annual_return")
)

print("Annual market returns (CRSP VW):")
print(crsp_annual.head(10))

group_means_sorted = group_means.sort_values(["incgroup", "YEAR"]).reset_index(drop=True)
group_means_sorted["dln_LABINC"] = group_means_sorted.groupby("incgroup")["ln_RLABINC"].diff()

labor_beta_panel = group_means_sorted.merge(
    crsp_annual, 
    on="YEAR", 
    how="inner"
)

def estimate_labor_beta(group_data):
    reg_data = group_data.dropna(subset=["dln_LABINC", "annual_return"])

    Y = reg_data["dln_LABINC"]
    X = sm.add_constant(reg_data["annual_return"])
    
    model = sm.OLS(Y, X).fit()
    
    return {
        "beta": model.params["annual_return"],
        "beta_tstat": model.tvalues["annual_return"],
        "beta_pvalue": model.pvalues["annual_return"],
        "alpha": model.params["const"],
        "r_squared": model.rsquared,
        "n_obs": int(model.nobs)
    }

beta_results = []

for group in labor_beta_panel["incgroup"].unique():
    group_data = labor_beta_panel[labor_beta_panel["incgroup"] == group]
    
    result = estimate_labor_beta(group_data)
    result["incgroup"] = group
    
    beta_results.append(result)

labor_beta_df = pd.DataFrame(beta_results)
labor_beta_df = labor_beta_df[["incgroup", "beta", "beta_tstat", "beta_pvalue", 
                               "alpha", "r_squared", "n_obs"]]

print("\nLabor Income Beta Results:")
print(labor_beta_df)

# Note: Enhanced analysis with SP500 comparison is performed in the section below

# Save CRSP labor beta results
LABOR_BETA_CSV = fr"{OUT_DIR}/labor_income_betas.csv"
labor_beta_df.to_csv(LABOR_BETA_CSV, index=False)

# %%
# Enhanced Analysis with 8 Income Groups - SP500 Comparison
print("Loading SP500 data for comparison analysis...")
sp500 = pd.read_csv(fr"{RAW_DIR}/sp500div.csv")

# Process SP500 data
sp500["YEAR"] = sp500["DATE"].str[:4].astype(int)
sp500_annual = (
    sp500.groupby("YEAR")["DIV_YIELD"]
         .apply(lambda x: (1 + x).prod() - 1)
         .reset_index()
         .rename(columns={"DIV_YIELD": "ANNUAL_DIV_RETURN"})
)

# Merge with CPI data and create real dividend index
sp500_annual = sp500_annual.merge(
    annual_cpi[["Year", "CPI_REL"]], left_on="YEAR", right_on="Year", how="inner"
).drop(columns=["Year"]).sort_values("YEAR")

sp500_annual["DIV_INDEX_REAL"] = (1 + sp500_annual["ANNUAL_DIV_RETURN"]).cumprod() / sp500_annual["CPI_REL"]
sp500_annual["log_div_t"] = np.log(sp500_annual["DIV_INDEX_REAL"])

# Create SP500 panel and run ADF tests
sp500_panel = group_means.merge(sp500_annual[["YEAR", "log_div_t"]], on="YEAR", how="inner")
sp500_panel["y_gt"] = sp500_panel["ln_RLABINC"] - sp500_panel["log_div_t"]
sp500_panel = sp500_panel.sort_values(["incgroup", "YEAR"]).reset_index(drop=True)
sp500_panel["dy_gt"] = sp500_panel.groupby("incgroup")["y_gt"].diff()
sp500_panel["t_index"] = sp500_panel["YEAR"] - sp500_panel["YEAR"].min()

sp500_adf = run_bcg_adf(sp500_panel, L=2)

# Save SP500 results
sp500_panel.to_csv(fr"{OUT_DIR}/cps_income_sp500_panel.csv", index=False)
sp500_adf.to_csv(fr"{OUT_DIR}/adf_sp500div.csv", index=False)

print("SP500 ADF Results:")
print(sp500_adf[["incgroup", "a3_coef", "a3_tstat", "a3_pvalue"]])

# CRSP vs SP500 ADF comparison plot
plt.figure(figsize=(12, 6))
x = np.arange(len(bcg_df))
width = 0.35

plt.bar(x - width/2, bcg_df["a3_coef"], width, label='CRSP', alpha=0.8)
plt.bar(x + width/2, sp500_adf["a3_coef"], width, label='SP500', alpha=0.8)

plt.xlabel('Income Group')
plt.ylabel('ADF Coefficient (a₃)')
plt.title('ADF Coefficients: CRSP vs SP500 Dividends (8 Income Groups)')
plt.xticks(x, bcg_df["incgroup"])
plt.legend()
plt.grid(True, alpha=0.3, axis='y')
plt.axhline(0, color='black', linewidth=0.8)
plt.tight_layout()

comparison_plot_path = fr"{OUT_DIR}/adf_comparison_crsp_vs_sp500.png"
plt.savefig(comparison_plot_path, dpi=300, bbox_inches='tight')
plt.show()

# %%
# Enhanced Labor Income Beta Analysis
print("Calculating labor income betas...")

# Load SP500 return data
sp500_data = pd.read_csv(fr"{RAW_DIR}/sp500div.csv")
sp500_data["YEAR"] = sp500_data["DATE"].str[:4].astype(int)

# Use dividend yield as return proxy for SP500
sp500_returns = (
    sp500_data.groupby("YEAR")["DIV_YIELD"]
             .apply(lambda x: (1 + x).prod() - 1)
             .reset_index()
             .rename(columns={"DIV_YIELD": "annual_return"})
)

# Calculate betas for both CRSP and SP500
beta_analyses = {"CRSP": crsp_annual, "SP500": sp500_returns}
all_beta_results = []

for data_source, return_data in beta_analyses.items():
    # Merge with labor income data
    beta_panel = group_means.merge(return_data, on="YEAR", how="inner")
    beta_panel = beta_panel.sort_values(["incgroup", "YEAR"]).reset_index(drop=True)
    beta_panel["dln_LABINC"] = beta_panel.groupby("incgroup")["ln_RLABINC"].diff()
    
    # Estimate betas for each income group
    for group in beta_panel["incgroup"].unique():
        group_data = beta_panel[beta_panel["incgroup"] == group]
        result = estimate_labor_beta(group_data.dropna(subset=["dln_LABINC", "annual_return"]))
        result["incgroup"] = group
        result["data_source"] = data_source
        all_beta_results.append(result)

# Save comprehensive results
all_beta_df = pd.DataFrame(all_beta_results)
all_beta_df.to_csv(fr"{OUT_DIR}/labor_income_betas_comprehensive.csv", index=False)

# Create comparison plot
plt.figure(figsize=(14, 6))
crsp_betas = all_beta_df[all_beta_df["data_source"] == "CRSP"]
sp500_betas = all_beta_df[all_beta_df["data_source"] == "SP500"]

x = np.arange(len(crsp_betas))
width = 0.35

plt.bar(x - width/2, crsp_betas["beta"], width, label='CRSP', alpha=0.8, color='steelblue')
plt.bar(x + width/2, sp500_betas["beta"], width, label='SP500', alpha=0.8, color='orange')

plt.xlabel('Income Group')
plt.ylabel('Labor Income Beta (β)')
plt.title('Labor Income Betas: CRSP vs SP500 Returns (8 Income Groups)')
plt.xticks(x, crsp_betas["incgroup"])
plt.legend()
plt.grid(True, alpha=0.3, axis='y')
plt.axhline(0, color='black', linewidth=0.8)
plt.tight_layout()
plt.show()

# %%
# Summary Statistics and Results
print("\n" + "="*80)
print("ANALYSIS RESULTS - 8 INCOME GROUPS")
print("="*80)

print("\nADF Test Results (CRSP Dividends):")
print(bcg_df[["incgroup", "a3_coef", "a3_tstat", "a3_pvalue"]].round(4))

crsp_betas = all_beta_df[all_beta_df["data_source"] == "CRSP"]
print(f"\nLabor Income Betas (CRSP Returns):")
print(crsp_betas[["incgroup", "beta", "beta_tstat", "beta_pvalue"]].round(4))

# Save summary statistics
with open(fr"{OUT_DIR}/summary_statistics_8groups.csv", 'w') as f:
    f.write("ADF Test Results (CRSP Dividends):\n")
    f.write(bcg_df[["incgroup", "a3_coef", "a3_tstat", "a3_pvalue"]].to_csv(index=False))
    f.write(f"\nLabor Income Betas:\n")
    f.write(all_beta_df[["data_source", "incgroup", "beta", "beta_tstat", "beta_pvalue"]].to_csv(index=False))

# %%
# Enhanced Visualization
# Combine ADF and Beta results for comprehensive display
combined_results = bcg_df.merge(
    labor_beta_df[["incgroup", "beta", "beta_pvalue"]], 
    on="incgroup", 
    suffixes=("_adf", "_beta")
)

# 4-panel comparison chart
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

# ADF coefficients
ax1.bar(combined_results["incgroup"], combined_results["a3_coef"], 
        color="steelblue", alpha=0.7, edgecolor="black")
ax1.axhline(0, color="black", linewidth=0.8)
ax1.set_title("ADF Coefficient ($a_3$)")
ax1.set_ylabel("$a_3$")
ax1.grid(True, alpha=0.3)

# ADF t-statistics with significance
adf_colors = ['red' if p < 0.05 else 'lightblue' for p in combined_results["a3_pvalue"]]
ax2.bar(combined_results["incgroup"], combined_results["a3_tstat"], 
        color=adf_colors, alpha=0.7, edgecolor="black")
ax2.axhline(-1.95, linestyle="--", color="red", label="5% Critical")
ax2.set_title("ADF t-statistics")
ax2.set_ylabel("t-stat($a_3$)")
ax2.legend()
ax2.grid(True, alpha=0.3)

# Labor Income Betas
ax3.bar(combined_results["incgroup"], combined_results["beta"], 
        color="darkorange", alpha=0.7, edgecolor="black")
ax3.axhline(0, color="black", linewidth=0.8)
ax3.set_title("Labor Income Beta ($β_g$)")
ax3.set_ylabel("$β_g$")
ax3.set_xlabel("Income Group")
ax3.grid(True, alpha=0.3)

# Beta significance
beta_colors = ['green' if p < 0.05 else 'lightgreen' for p in combined_results["beta_pvalue"]]
ax4.bar(combined_results["incgroup"], combined_results["beta"], 
        color=beta_colors, alpha=0.7, edgecolor="black")
ax4.axhline(0, color="black", linewidth=0.8)
ax4.set_title("Labor Income Beta (Green = Significant)")
ax4.set_ylabel("$β_g$")
ax4.set_xlabel("Income Group")
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Save combined results
combined_results.to_csv(fr"{OUT_DIR}/combined_adf_beta_results.csv", index=False)

# %%
# Income Group Verification (2020 sample)
full_df = pd.read_parquet(CLEAN_PARQUET)
sample_data = full_df[full_df["YEAR"] == 2020].copy()

if len(sample_data) > 0:
    plt.figure(figsize=(12, 6))
    
    group_sizes = sample_data.groupby("Income_Group")["ASECWT"].sum()
    group_shares = (group_sizes / group_sizes.sum() * 100)
    
    group_order = ["Group1", "Group2", "Group3", "Group4", "Group5", "Group6", "Group7", "Group8"]
    group_shares_ordered = group_shares.reindex(group_order)
    
    plt.bar(group_order, group_shares_ordered, color="lightcoral", alpha=0.7, edgecolor="black")
    plt.axhline(20, color="gray", linestyle="--", alpha=0.6)
    plt.axhline(10, color="gray", linestyle="--", alpha=0.6)
    plt.axhline(5, color="gray", linestyle="--", alpha=0.6)
    plt.axhline(1, color="gray", linestyle="--", alpha=0.6)
    plt.title("Income Group Shares (2020)")
    plt.ylabel("Share (%)")
    plt.xlabel("Income Group")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    print("Group shares in 2020:")
    for group, share in group_shares_ordered.items():
        print(f"{group}: {share:.1f}%")

# %%
print("\n" + "="*60)
print("ANALYSIS COMPLETE - 8 INCOME GROUPS")
print("="*60)
print("Key Results:")
print("  - ADF tests for unit root testing")
print("  - Labor income betas vs market returns") 
print("  - Panel ECM robustness checks")
print("  - CRSP vs SP500 comparison analysis")
print("="*60)
