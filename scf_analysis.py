# %%
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf

RAW_DIR      = r"E:/users_ra_local/fang_ray/Lifecycle-portfolio-allocation/raw"
OUT_DIR      = r"E:/users_ra_local/fang_ray/Lifecycle-portfolio-allocation/outputs"
SCF_DIR      = r"E:/users_ra_local/fang_ray/Lifecycle-portfolio-allocation/SCF Data"

CPS_CSV      = fr"{RAW_DIR}/cps_00001.csv"
CPI_CSV      = fr"{RAW_DIR}/cpi_u.csv"
CRSP_CSV     = fr"{RAW_DIR}/crsp_value_weighted.csv"
SP500DIV_CSV    = fr"{RAW_DIR}/sp500div.csv"

SCF_MERGED = fr"{SCF_DIR}/all_merged.dta"

CLEAN_PARQUET = fr"{OUT_DIR}/cps_00001_clean.parquet"
GROUPS_CSV    = fr"{OUT_DIR}/cps_income_groups_1968_2024.csv"
# %%
df = pd.read_stata(SCF_MERGED)
df.to_parquet(df, fr"{OUT_DIR}/all_merged.parquet")
# %%
