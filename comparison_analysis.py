import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

crsp_results = pd.read_csv("outputs/cps_bcg_adf_results.csv")
sp500_results = pd.read_csv("outputs/adf_sp500div.csv")

crsp_L2 = crsp_results[crsp_results["L"] == 2].copy()
sp500_L2 = sp500_results[sp500_results["L"] == 2].copy()

comparison = crsp_L2.merge(sp500_L2, on=["incgroup", "L"], suffixes=("_CRSP", "_SP500"))

for _, row in comparison.iterrows():
    group = row["incgroup"]
    p_crsp = row["a3_pvalue_CRSP"]
    p_sp500 = row["a3_pvalue_SP500"]
    
    sig_crsp = "Yes" if p_crsp < 0.05 else "No"
    sig_sp500 = "Yes" if p_sp500 < 0.05 else "No"
    
    print(f"{group:>10s}: CRSP = {sig_crsp:>3s} (p={p_crsp:.4f}), SP500 = {sig_sp500:>3s} (p={p_sp500:.4f})")

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

groups = comparison["incgroup"].tolist()
x_pos = np.arange(len(groups))

ax1.bar(x_pos - 0.2, comparison["a3_coef_CRSP"], 0.4, label="CRSP", alpha=0.7)
ax1.bar(x_pos + 0.2, comparison["a3_coef_SP500"], 0.4, label="SP500", alpha=0.7)
ax1.set_xlabel("Income Group")
ax1.set_ylabel("ADF Coefficient (a3)")
ax1.set_title("ADF Coefficients: CRSP vs SP500 Dividends")
ax1.set_xticks(x_pos)
ax1.set_xticklabels(groups, rotation=45)
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.axhline(0, color='black', linewidth=0.8)

ax2.bar(x_pos - 0.2, comparison["a3_tstat_CRSP"], 0.4, label="CRSP", alpha=0.7)
ax2.bar(x_pos + 0.2, comparison["a3_tstat_SP500"], 0.4, label="SP500", alpha=0.7)
ax2.set_xlabel("Income Group")
ax2.set_ylabel("t-statistic")
ax2.set_title("ADF t-statistics: CRSP vs SP500 Dividends")
ax2.set_xticks(x_pos)
ax2.set_xticklabels(groups, rotation=45)
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.axhline(-1.96, linestyle='--', color='red', alpha=0.7, label='5% Critical')
ax2.axhline(1.96, linestyle='--', color='red', alpha=0.7)

ax3.bar(x_pos - 0.2, comparison["a3_pvalue_CRSP"], 0.4, label="CRSP", alpha=0.7)
ax3.bar(x_pos + 0.2, comparison["a3_pvalue_SP500"], 0.4, label="SP500", alpha=0.7)
ax3.set_xlabel("Income Group")
ax3.set_ylabel("p-value (log scale)")
ax3.set_title("ADF p-values: CRSP vs SP500 Dividends")
ax3.set_xticks(x_pos)
ax3.set_xticklabels(groups, rotation=45)
ax3.set_yscale('log')
ax3.legend()
ax3.grid(True, alpha=0.3)
ax3.axhline(0.05, linestyle='--', color='red', alpha=0.7, label='5% Significance')

ax4.bar(x_pos - 0.2, comparison["r2_CRSP"], 0.4, label="CRSP", alpha=0.7)
ax4.bar(x_pos + 0.2, comparison["r2_SP500"], 0.4, label="SP500", alpha=0.7)
ax4.set_xlabel("Income Group")
ax4.set_ylabel("R-squared")
ax4.set_title("Model R-squared: CRSP vs SP500 Dividends")
ax4.set_xticks(x_pos)
ax4.set_xticklabels(groups, rotation=45)
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("outputs/adf_comparison_crsp_vs_sp500.png", dpi=300, bbox_inches='tight')
plt.show()