import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

PROJECT_ROOT = pd.Path if hasattr(pd, "Path") else None
from pathlib import Path
PROJECT_ROOT = Path("/mnt/nas/home/hiromi/src/emo2")

csv_path = PROJECT_ROOT / "v2/results/derived/v2_causal_pair_level.csv"
print("Loading data from", csv_path)
df = pd.read_csv(csv_path)
print(f"Total rows: {len(df)}")

# Primary Matched-Plain condition for Valence
df_primary = df[
    ((df["alignment"] == "base") & (df["format_condition"] == "plain"))
    | ((df["alignment"] == "inst") & (df["format_condition"] == "matched_plain"))
].copy()

print(f"Primary rows: {len(df_primary)}")

formula = "c_v_net_rand ~ C(family) + C(alignment) * C(task) * relative_depth"
term = "C(alignment)[T.inst]:C(task)[T.self]"

# 1. Standard OLS
print("\n--- 1. Standard OLS ---")
ols_fit = smf.ols(formula, data=df_primary).fit()
beta = ols_fit.params[term]
se = ols_fit.bse[term]
pval = ols_fit.pvalues[term]
print(f"OLS: beta = {beta:.6e}, se = {se:.6e}, p = {pval:.6f}")

# 2. Cluster-Robust SE (clustered by pair_id)
print("\n--- 2. OLS with Cluster-Robust SE (clustered by pair_id) ---")
ols_pair_cluster = smf.ols(formula, data=df_primary).fit(
    cov_type="cluster", cov_kwds={"groups": df_primary["pair_id"]}
)
beta_pair = ols_pair_cluster.params[term]
se_pair = ols_pair_cluster.bse[term]
pval_pair = ols_pair_cluster.pvalues[term]
print(f"Cluster (pair_id): beta = {beta_pair:.6e}, se = {se_pair:.6e}, p = {pval_pair:.6f}")

# 3. Cluster-Robust SE (clustered by family)
print("\n--- 3. OLS with Cluster-Robust SE (clustered by family) ---")
ols_fam_cluster = smf.ols(formula, data=df_primary).fit(
    cov_type="cluster", cov_kwds={"groups": df_primary["family"]}
)
beta_fam = ols_fam_cluster.params[term]
se_fam = ols_fam_cluster.bse[term]
pval_fam = ols_fam_cluster.pvalues[term]
print(f"Cluster (family): beta = {beta_fam:.6e}, se = {se_fam:.6e}, p = {pval_fam:.6f}")

# 4. HC3 Robust SE
print("\n--- 4. OLS with HC3 Robust SE ---")
ols_hc3 = smf.ols(formula, data=df_primary).fit(cov_type="HC3")
beta_hc3 = ols_hc3.params[term]
se_hc3 = ols_hc3.bse[term]
pval_hc3 = ols_hc3.pvalues[term]
print(f"HC3: beta = {beta_hc3:.6e}, se = {se_hc3:.6e}, p = {pval_hc3:.6f}")

# 5. MixedLM with groups="pair_id"
print("\n--- 5. MixedLM (groups=pair_id) ---")
try:
    lmm_fit = smf.mixedlm(formula, data=df_primary, groups=df_primary["pair_id"]).fit(method=["lbfgs", "cg"], maxiter=500)
    beta_lmm = lmm_fit.params[term]
    se_lmm = lmm_fit.bse[term]
    pval_lmm = lmm_fit.pvalues[term]
    group_var = lmm_fit.params.get("Group Var", np.nan)
    print(f"MixedLM: beta = {beta_lmm:.6e}, se = {se_lmm:.6e}, p = {pval_lmm:.6f}, Group Var = {group_var}")
    print(f"MixedLM converged: {lmm_fit.converged}")
except Exception as e:
    print("MixedLM error:", e)

# 6. Summary comparison
print("\n" + "="*70)
print(f"Target Term: {term}")
print(f"{'Method':<35} | {'Beta':<12} | {'SE':<12} | {'p-value':<10}")
print("-" * 70)
print(f"{'Standard OLS':<35} | {beta:<12.6e} | {se:<12.6e} | {pval:<10.6f}")
print(f"{'Cluster-Robust (pair_id)':<35} | {beta_pair:<12.6e} | {se_pair:<12.6e} | {pval_pair:<10.6f}")
print(f"{'Cluster-Robust (family)':<35} | {beta_fam:<12.6e} | {se_fam:<12.6e} | {pval_fam:<10.6f}")
print(f"{'HC3 Robust SE':<35} | {beta_hc3:<12.6e} | {se_hc3:<12.6e} | {pval_hc3:<10.6f}")
if 'beta_lmm' in locals():
    print(f"{'MixedLM (groups=pair_id)':<35} | {beta_lmm:<12.6e} | {se_lmm:<12.6e} | {pval_lmm:<10.6f}")
print("="*70)
