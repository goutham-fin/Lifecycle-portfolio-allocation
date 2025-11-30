# %%
import pandas as pd
import numpy as np 

# %%
OUT_DIR = r"E:/users_ra_local/fang_ray/Lifecycle-portfolio-allocation/outputs"
CLEAN_PARQUET = fr"{OUT_DIR}/cps_00001_clean.parquet"
GROUPS_CSV = fr"{OUT_DIR}/cps_income_groups_1968_2024.csv"

# Load cleaned data
df = pd.read_parquet(CLEAN_PARQUET) 

# %%
# Define income groups by year
cut_probs = np.array([0.20, 0.40, 0.60, 0.80, 0.90, 0.95, 0.99])

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
    wts = g["ASECWT"].to_numpy()
    cuts = weighted_percentiles(vals, wts, cut_probs)
    bins = np.concatenate(([-np.inf], cuts, [np.inf]))
    labels = ["Group1", "Group2", "Group3", "Group4", "Group5", "Group6", "Group7", "Group8"]
    g["Income_Group"] = pd.cut(g["RLABINC"], bins=bins, labels=labels, right=True, include_lowest=True)
    return g

df = df.groupby("YEAR", group_keys=False).apply(assign_income_group)

# %%
# Calculate weighted mean log-income per group
df["ln_RLABINC"] = np.log(df["RLABINC"])
group_means = (
    df.groupby(["YEAR", "Income_Group"])
      .apply(lambda g: np.average(g["ln_RLABINC"], weights=g["ASECWT"]))
      .reset_index(name="ln_RLABINC")
      .rename(columns={"Income_Group": "incgroup"})
)

# Save output
group_means.to_csv(GROUPS_CSV, index=False)


# %%
