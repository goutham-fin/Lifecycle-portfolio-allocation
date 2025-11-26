# %%
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

RAW_DIR = r"e:\users_ra_local\fang_ray\Lifecycle-portfolio-allocation\raw"
OUT_DIR = r"e:\users_ra_local\fang_ray\Lifecycle-portfolio-allocation\outputs"

GROUP_MEANS_CSV = fr"{OUT_DIR}/cps_income_groups_1968_2024.csv"
SP500DIV_CSV    = fr"{RAW_DIR}/sp500div.csv"
# %%
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

def run_all_adf(panel, L_list):
    """
    Run ADF tests for all lag lengths in L_list and return combined results
    """
    all_results = []
    for L in L_list:
        results = run_bcg_adf(panel, L)
        all_results.append(results)
    
    return pd.concat(all_results, ignore_index=True)

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

    wald = res.wald_test("y_gt_lag_top = 0", scalar=True)

    return res, wald

group_means = pd.read_csv(GROUP_MEANS_CSV)
sp = pd.read_csv(SP500DIV_CSV) 
sp = sp.sort_values("Year")
sp["log_div_t"] = np.log(sp["Dividend"])

panel = group_means.merge(sp[["Year", "log_div_t"]], left_on="YEAR", right_on="Year", how="inner")

panel["y_gt"] = panel["ln_RLABINC"] - panel["log_div_t"]
panel = panel.sort_values(["incgroup", "YEAR"]).reset_index(drop=True) 
panel["dy_gt"] = panel.groupby("incgroup")["y_gt"].diff()
panel["t_index"] = panel["YEAR"] - panel["YEAR"].min()

PANEL_SP500_CSV = fr"{OUT_DIR}/cps_income_sp500_panel.csv"
panel.to_csv(PANEL_SP500_CSV, index=False)
print(f"Saved panel: {PANEL_SP500_CSV}")

L_list = [1,2,3]

adf_table = run_all_adf(panel, L_list=L_list)
adf_table.to_csv(fr"{OUT_DIR}/adf_sp500div.csv", index=False)

ecm_results = []
for L in L_list:
    res, wald = run_panel_ecm(panel, L=L, top_label="Top1")
    ecm_results.append({
        "L": L,
        "a3_common": res.params["y_gt_lag"],
        "a3_top_extra": res.params["y_gt_lag_top"],
        "wald_pvalue": float(wald.pvalue)
    })

ecm_table = pd.DataFrame(ecm_results)
ecm_table.to_csv(fr"{OUT_DIR}/ecm_sp500div.csv", index=False)

print(adf_table)
print(ecm_table)
# %%
# Part B — Construct Short-Run Labor-Income Betas by Income Group

sp_full = pd.read_csv(SP500DIV_CSV)
if sp_full.shape[1] == 1:
    sp_full.columns = ["Dividend"]
    sp_full["Year"] = sp_full.index + sp_full.index.min()
else:
    sp_full.columns = ["Year", "Dividend"]

sp_full = sp_full.sort_values("Year")

sp_full["div_growth"] = sp_full["Dividend"].pct_change()

print("S&P 500 dividend data:")
print(sp_full.head(10))

group_means = pd.read_csv(GROUP_MEANS_CSV)
group_means_sorted = group_means.sort_values(["incgroup", "YEAR"]).reset_index(drop=True)
group_means_sorted["dln_LABINC"] = group_means_sorted.groupby("incgroup")["ln_RLABINC"].diff()

labor_beta_panel_sp500 = group_means_sorted.merge(
    sp_full[["Year", "div_growth"]], 
    left_on="YEAR", 
    right_on="Year", 
    how="inner"
)

print("\nLabor income growth and S&P 500 dividend growth panel:")
print(labor_beta_panel_sp500.head())

def estimate_labor_beta_sp500(group_data):
    reg_data = group_data.dropna(subset=["dln_LABINC", "div_growth"])
    
    if len(reg_data) < 5:
        return {
            "beta": np.nan,
            "beta_tstat": np.nan,
            "beta_pvalue": np.nan,
            "alpha": np.nan,
            "r_squared": np.nan,
            "n_obs": len(reg_data)
        }

    Y = reg_data["dln_LABINC"]
    X = sm.add_constant(reg_data["div_growth"])
    
    model = sm.OLS(Y, X).fit()
    
    return {
        "beta": model.params["div_growth"],
        "beta_tstat": model.tvalues["div_growth"],
        "beta_pvalue": model.pvalues["div_growth"],
        "alpha": model.params["const"],
        "r_squared": model.rsquared,
        "n_obs": int(model.nobs)
    }

beta_results_sp500 = []

for group in labor_beta_panel_sp500["incgroup"].unique():
    group_data = labor_beta_panel_sp500[labor_beta_panel_sp500["incgroup"] == group]
    
    result = estimate_labor_beta_sp500(group_data)
    result["incgroup"] = group
    
    beta_results_sp500.append(result)

labor_beta_sp500_df = pd.DataFrame(beta_results_sp500)
labor_beta_sp500_df = labor_beta_sp500_df[["incgroup", "beta", "beta_tstat", "beta_pvalue", 
                                           "alpha", "r_squared", "n_obs"]]

print("\nLabor Income Beta Results (S&P 500 Dividend):")
print(labor_beta_sp500_df)

LABOR_BETA_SP500_CSV = fr"{OUT_DIR}/labor_income_betas_sp500.csv"

labor_beta_sp500_df.to_csv(LABOR_BETA_SP500_CSV, index=False)

print(f"\nSaved S&P 500 labor income beta results to: {LABOR_BETA_SP500_CSV}")

# %%
# Bar Chart of β_g across income groups (S&P 500)

plt.figure(figsize=(10, 6))

group_order = ["Bottom50", "Middle40", "Top10", "Top1"]
labor_beta_sp500_df_sorted = labor_beta_sp500_df.set_index("incgroup").loc[group_order].reset_index()

bars = plt.bar(labor_beta_sp500_df_sorted["incgroup"], 
               labor_beta_sp500_df_sorted["beta"], 
               color=["lightcoral", "lightblue", "lightgreen", "gold"],
               edgecolor="black",
               linewidth=0.8)

for i, (bar, beta, tstat, pval) in enumerate(zip(bars, 
                                                 labor_beta_sp500_df_sorted["beta"],
                                                 labor_beta_sp500_df_sorted["beta_tstat"],
                                                 labor_beta_sp500_df_sorted["beta_pvalue"])):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
             f'{beta:.3f}',
             ha='center', va='bottom', fontweight='bold')
    
plt.axhline(0, color="black", linewidth=0.8, linestyle="-")
plt.title("Labor Income Beta by Income Group\n" + 
          r"$\Delta \ln(LABINC_{g,t}) = \alpha_g + \beta_g \cdot DIV\_GROWTH_t + \varepsilon_{g,t}$",
          fontsize=14)
plt.xlabel("Income Group", fontsize=12)
plt.ylabel(r"Labor Income Beta ($\beta_g$)", fontsize=12)
plt.grid(True, alpha=0.3, axis='y')


plt.tight_layout()
plt.show()

    
# %%
