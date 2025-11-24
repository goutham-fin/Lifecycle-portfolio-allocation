# %%
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf

# %%
RAW_DIR      = r"E:/users_ra_local/fang_ray/IPUMS-CPS/raw"
OUT_DIR      = r"E:/users_ra_local/fang_ray/IPUMS-CPS/outputs"

CPS_CSV      = fr"{RAW_DIR}/cps_00001.csv"
CPI_CSV      = fr"{RAW_DIR}/cpi_u.csv"
CRSP_CSV     = fr"{RAW_DIR}/crsp_value_weighted.csv"

CLEAN_PARQUET = fr"{OUT_DIR}/cps_00001_clean.parquet"
GROUPS_CSV    = fr"{OUT_DIR}/cps_income_groups_1968_2024.csv"
GROUPS_DTA    = fr"{OUT_DIR}/cps_income_groups_1968_2024.dta"

# %%
df  = pd.read_csv(CPS_CSV)
cpi = pd.read_csv(CPI_CSV)
crsp = pd.read_csv(CRSP_CSV)
df.to_parquet(fr"{OUT_DIR}/cps_00001_raw.parquet", index=False)

# %%
df = df[(df["AGE"] >= 25) & (df["AGE"] <= 60)]
df = df[df["UHRSWORKLY"] >= 20]
df = df[df["WKSWORK1"] >= 26]

keep = [
    "YEAR", "CPSIDP", "AGE",
    "INCWAGE", "INCBUS",
    "UHRSWORKLY", "WKSWORK1",
    "ASECWT", "EMPSTAT",
]
df = df[keep]

df["LABINC"] = df[["INCWAGE", "INCBUS"]].clip(lower=0).sum(axis=1)

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

df = df.merge(
    annual_cpi[["Year", "CPI_REL"]],
    left_on="YEAR",
    right_on="Year",
    how="left"
).drop(columns=["Year"])

df["RLABINC"] = df["LABINC"] / df["CPI_REL"]

df = df[df["RLABINC"] > 0]
# %%
cut_probs = np.array([0.50, 0.90, 0.99]) 

def weighted_percentiles(values, weights, probs):
    """Compute weighted percentiles for a 1D array."""
    values = np.asarray(values)
    weights = np.asarray(weights)
    sorter = np.argsort(values)
    v = values[sorter]
    w = weights[sorter]

    cum_w = np.cumsum(w)
    total_w = cum_w[-1]
    pct = cum_w / total_w

    return np.interp(probs, pct, v)

def assign_income_group(g):
    vals = g["RLABINC"].to_numpy()
    wts  = g["ASECWT"].to_numpy()

    cuts = weighted_percentiles(vals, wts, cut_probs)
    bins = np.concatenate(([-np.inf], cuts, [np.inf]))

    labels = ["Bottom50", "Middle40", "Top10", "Top1"]

    g["Income_Group"] = pd.cut(
        g["RLABINC"],
        bins=bins,
        labels=labels,
        right=True,
        include_lowest=True
    )

    return g

df = df.groupby("YEAR", group_keys=False).apply(assign_income_group)

# %%
df["ln_RLABINC"] = np.log(df["RLABINC"])

group_means = (
    df.groupby(["YEAR", "Income_Group"])
      .apply(lambda g: np.average(g["ln_RLABINC"], weights=g["ASECWT"]))
      .reset_index(name="ln_RLABINC")
      .rename(columns={"Income_Group": "incgroup"})
)

df.to_parquet(CLEAN_PARQUET, index=False)
group_means.to_csv(GROUPS_CSV, index=False)
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

latex_path_A = r"E:/users_ra_local/fang_ray/IPUMS-CPS/outputs/table_A_adf.tex"

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

latex_path_B = r"E:/users_ra_local/fang_ray/IPUMS-CPS/outputs/table_B_adf_lag_sensitivity.tex"

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
# Plotting weighted means over time
plt.figure(figsize=(12, 6))

group_order = ["Bottom50", "Middle40", "Top10", "Top1"]

for g in group_order:
    subset = group_means[group_means["incgroup"] == g]
    plt.plot(subset["YEAR"], subset["ln_RLABINC"], label=g)

plt.title("Weighted Mean log(RLABINC) by Income Bucket, 1976–2024")
plt.xlabel("Year")
plt.ylabel("ln(RLABINC)")
plt.legend(title="Income Bucket")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %%
# Plotting weighted shares over time for verification
share_check = (
    df.groupby(["YEAR", "Income_Group"])
      .apply(lambda g: g["ASECWT"].sum())
      .reset_index(name="group_weight")
)

total_weights = (
    df.groupby("YEAR")["ASECWT"].sum().reset_index(name="total_weight")
)

share_check = share_check.merge(total_weights, on="YEAR")
share_check["share"] = share_check["group_weight"] / share_check["total_weight"] * 100

plt.figure(figsize=(12, 6))

group_order = ["Bottom50", "Middle40", "Top10", "Top1"]

for g in group_order:
    subset = share_check[share_check["Income_Group"] == g]
    plt.plot(subset["YEAR"], subset["share"], label=g)

plt.axhline(50, color="gray", linestyle="dashed", linewidth=0.5)
plt.axhline(40, color="gray", linestyle="dashed", linewidth=0.5)
plt.axhline(10, color="gray", linestyle="dashed", linewidth=0.5)
plt.axhline(1,  color="gray", linestyle="dashed", linewidth=0.5)

plt.title("Income Bucket Weighted Shares over Time")
plt.xlabel("Year")
plt.ylabel("Share of Weighted Population (%)")
plt.legend(title="Income Bucket")
plt.tight_layout()
plt.show()
# %%
# Plotting dividend returns and log real dividend index
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

ax1.plot(annual["YEAR"], annual["ANNUAL_DIV_RETURN"] * 100,
         label="Nominal Dividend Return", linewidth=2)
ax1.set_title("Annual Nominal Dividend Returns")
ax1.set_xlabel("Year")
ax1.set_ylabel("Dividend Return (%)")
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(annual["YEAR"], annual["log_div_t"], linewidth=2)
ax2.set_title("Log Real Dividend Index")
ax2.set_xlabel("Year")
ax2.set_ylabel("log(real dividend index)")
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
# %%
# Plot y_{g,t}
groups = sorted(panel["incgroup"].unique())

plt.figure(figsize=(12, 8))
for g in groups:
    subset = panel[panel["incgroup"] == g]
    plt.plot(subset["YEAR"], subset["y_gt"], label=f"Group {g}", linewidth=1.6)

plt.title("Labor–Dividend Log Ratio $y_{g,t}$ by Income Group")
plt.xlabel("Year")
plt.ylabel("$y_{g,t} = \\ln(RLABINC_g) - \\ln(DIV_t)$")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()
# %%
# Plot Δy_{g,t}
groups = sorted(panel["incgroup"].unique())

for g in groups:
    subset = panel[panel["incgroup"] == g]
    plt.plot(subset["YEAR"], subset["dy_gt"], label=f"Group {g}", linewidth=1.2)

plt.title("First Difference of Labor–Dividend Ratio $\\Delta y_{g,t}$")
plt.xlabel("Year")
plt.ylabel("$\\Delta y_{g,t}$")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()
# %%
# Bar chart of ADF coefficients by income group
plt.figure(figsize=(10, 6))
plt.bar(bcg_df["incgroup"], bcg_df["a3_coef"], color="steelblue")

plt.axhline(0, color="black", linewidth=0.8)
plt.title("ADF Coefficient $a_3$ by Income Group (L = 2)")
plt.xlabel("Income Group")
plt.ylabel("$a_3$")
plt.tight_layout()
plt.show()
# %%
# Bar chart of ADF t-statistics by income group
plt.figure(figsize=(10, 6))
plt.bar(bcg_df["incgroup"], bcg_df["a3_tstat"], color="darkorange")

plt.axhline(-1.95, linestyle="--", color="red", label="5% critical (approx)")
plt.title("ADF t-statistic of $a_3$ by Income Group")
plt.xlabel("Income Group")
plt.ylabel("t-stat($a_3$)")
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

final_path = r"E:/users_ra_local/fang_ray/IPUMS-CPS/outputs/cps_adf_final_panel.csv"
final_table.to_csv(final_path, index=False)

final_table.head()
# %%
# Identify outliers
# Outlier: Δy > 0.10 for any income group
threshold = 0.10

outliers = (
    panel.loc[panel["dy_gt"].abs() > threshold,
              ["YEAR", "incgroup", "dy_gt"]]
    .sort_values(["YEAR", "incgroup"])
)

print("\nOutlier Δy_gt > 0.10:")
print(outliers)

outliers_wide = (
    panel.assign(abs_dy=lambda x: x["dy_gt"].abs())
          .pivot_table(index="YEAR",
                       columns="incgroup",
                       values="dy_gt")
)
outliers_wide = outliers_wide[outliers_wide.abs().max(axis=1) > threshold]

print(outliers_wide)

years_to_plot = sorted(outliers["YEAR"].unique())

for yr in years_to_plot:
    window = panel[(panel["YEAR"] >= yr - 3) & (panel["YEAR"] <= yr + 3)]

    fig, ax = plt.subplots(figsize=(8,4))
    for g in window["incgroup"].unique():
        sub = window[window["incgroup"] == g]
        ax.plot(sub["YEAR"], sub["dy_gt"], marker="o", label=g)

    ax.axhline(0.10, color="red", linestyle="--", alpha=0.6)
    ax.axhline(-0.10, color="red", linestyle="--", alpha=0.6)

    ax.set_title(f"Δy_gt Around Outlier Year {yr}")
    ax.set_ylabel("Δy_gt")
    ax.legend()
    plt.tight_layout()
    plt.show()
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
    res, wald = run_panel_ecm(panel, L=L, top_label="Top1", hac_lags=L)

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
